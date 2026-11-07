"""Bounded G4 CNN stage-1 topology and frozen structural equivalence probe."""
import copy

import torch

from model.research_config import digest
from model.VCASAHGCN import DatasetAwareSSConv, MultiDatasetVCASAHGCN, VCASAHGCNBackbone
from monitoring.dependency_groups import HEADS
from monitoring.interventions import ChannelIntervention, isolated_probe

from .channel_mapping import _slice_state, compact_state


# Mask after normalization: a zero input may have a nonzero BN output.
ANCHOR = "backbone.CNN_Branch.1.norm"
FIRST = "backbone.CNN_Branch.0"
SECOND = "backbone.CNN_Branch.1"


class _CompactBackbone(VCASAHGCNBackbone):
    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.stage1_width = width
        norm = self.CNN_Branch[1].norm.norm_type
        self.CNN_Branch[0] = DatasetAwareSSConv(64, width, 3, norm_type=norm, dataset_names=self.dataset_names)
        self.CNN_Branch[1] = DatasetAwareSSConv(width, 64, 5, norm_type=norm, dataset_names=self.dataset_names)

    def cnn_stage1(self, cnn_input, dataset_name=None):
        nchw = cnn_input.permute(2, 0, 1).unsqueeze(0)
        output = self.CNN_Branch[0](nchw, dataset_name)
        return output.squeeze(0).permute(1, 2, 0).reshape(-1, self.stage1_width)

    def cnn_stage2(self, pixels, image_shape, dataset_name=None):
        nchw = pixels.reshape(*image_shape, self.stage1_width).permute(2, 0, 1).unsqueeze(0)
        output = self.CNN_Branch[1](nchw, dataset_name)
        return output.squeeze(0).permute(1, 2, 0).reshape(-1, 64)


class _CompactModel(MultiDatasetVCASAHGCN):
    def export_config(self):
        raise ValueError("Compact G4 requires its own checkpoint schema")


def _template(reference, removed):
    if (type(reference) is not MultiDatasetVCASAHGCN or type(reference.backbone) is not VCASAHGCNBackbone
            or reference.backbone.interaction is not None or reference.backbone.edge_updater is not None):
        raise ValueError("Stock G4 with no intermediate interaction or learned edge updates required")
    first, second = reference.backbone.CNN_Branch
    if (type(first) is not DatasetAwareSSConv or type(second) is not DatasetAwareSSConv
            or second.norm.norm_type not in ("batch_norm", "domain_bn")):
        raise ValueError("Known CNN blocks and channel-independent stage-2 normalization required")
    if (not removed or len(set(removed)) != len(removed)
            or any(type(index) is not int or not 0 <= index < 128 for index in removed) or len(removed) == 128):
        raise ValueError("Unique stage-1 channels with at least one retained channel required")
    retained = [i for i in range(128) if i not in removed]
    cfg = reference.export_config()
    with torch.random.fork_rng(devices=[]):
        result = _CompactModel(cfg["class_counts"], **cfg["backbone_kwargs"])
        result.backbone = _CompactBackbone(len(retained), **cfg["backbone_kwargs"])
    result.training_config = copy.deepcopy(getattr(reference, "training_config", {}))
    result.to(next(reference.parameters())).eval()
    axes = [(FIRST + ".point_conv.weight", 0), (FIRST + ".depth_conv.weight", 0),
            (FIRST + ".depth_conv.bias", 0), (SECOND + ".point_conv.weight", 1)]
    norms = second.norm.norm.norms if second.norm.norm_type == "domain_bn" else {"": second.norm.norm}
    for name, bn in norms.items():
        prefix = SECOND + ".norm.norm" + (".norms." + name if name else "")
        target_bn = result.get_submodule(prefix)
        if (type(bn) is not torch.nn.BatchNorm2d or bn.num_features != 128
                or (bn.eps, bn.momentum, bn.affine, bn.track_running_stats) !=
                   (target_bn.eps, target_bn.momentum, target_bn.affine, target_bn.track_running_stats)):
            raise ValueError("Stock per-channel normalization contract required")
        axes.extend((prefix + "." + field, 0) for field in ("weight", "bias", "running_mean", "running_var"))
    return result, retained, axes


def _anchor(probe):
    found = [name for name, call in probe.manifest["calls"].items()
             if (call["spec"]["path"], call["call_index"], call["spec"]["layout"], call["spec"]["unit"])
             == (ANCHOR, 0, "NCHW", "pixel")]
    if len(found) != 1:
        raise ValueError("Frozen stage-2 normalized input probe anchor required")
    return found[0]


def _mapped_geometry(reference, removed):
    template, retained, axes = _template(reference, removed)
    state = _slice_state(reference.state_dict(), template.state_dict(), {axis: retained for axis in axes})
    template.load_state_dict(state, strict=True)
    return template, retained, axes


def compare_geometry(reference, forward, inputs, probe, removed):
    """Numerical engineering probe. It publishes no candidate acceptance or checkpoint."""
    anchor = _anchor(probe)
    template, retained, axes = _mapped_geometry(reference, removed)
    masked = isolated_probe(reference, lambda: copy.deepcopy(reference), forward, inputs, probe,
                            interventions=(ChannelIntervention(anchor, tuple(removed)),))
    compact = isolated_probe(template, lambda: copy.deepcopy(template), forward, inputs, probe)
    a, b = masked["outputs"], compact["outputs"]
    if set(a) != set(HEADS) or set(b) != set(HEADS):
        raise ValueError("Explicit three-head callback outputs required")
    differences, decisions = {}, {}
    for head in HEADS:
        if a[head].ndim != 2 or a[head].shape != b[head].shape or min(a[head].shape) < 1:
            raise ValueError("Matching nonempty [pixels, classes] outputs required")
        differences[head] = float((a[head] - b[head]).abs().max())
        decisions[head] = int((a[head].argmax(1) != b[head].argmax(1)).sum())
        torch.testing.assert_close(a[head], b[head], rtol=0., atol=1e-5)
        if decisions[head]:
            raise ValueError("Compact geometry changed frozen decisions")
    return {"removed": list(removed), "retained": retained, "affected_axes": [list(pair) for pair in axes],
            "before_parameters": sum(p.numel() for p in reference.parameters()),
            "after_parameters": sum(p.numel() for p in template.parameters()),
            "max_abs_by_head": differences, "decision_changes_by_head": decisions,
            "reference_state": masked["audit"]["reference_state"], "compact_state": compact["audit"]["reference_state"],
            "optimizer_updates": 0, "checkpoint_exported": False, "candidate_accepted": False}


def compact_from_passed_groups(reference, probe, plan, receipt, names, *, expected_receipt_sha256):
    """Construct a fresh compact G4 using a pinned, passing reversible trial."""
    if digest(receipt) != expected_receipt_sha256 or plan["probe_identity"] != probe.identity:
        raise ValueError("Exact pinned validation receipt and stage-1 probe required")
    anchor = _anchor(probe)
    groups = [g for g in plan["groups"] if g["name"] in names]
    if len(groups) != len(names) or any(g["sites"] != [{"observation": anchor, "channel": g["candidate_index"]}] for g in groups):
        raise ValueError("Only single-anchor stage-1 groups are supported")
    template, _, axes = _template(reference, [g["candidate_index"] for g in groups])
    entries = [{"group": g["name"], "observation": anchor, "channel": g["candidate_index"],
                "index": g["candidate_index"]} for g in groups]
    declaration = {"receipt_sha256": expected_receipt_sha256, "groups": list(names),
                   "axes": [{"tensor": name, "axis": axis, "remove": entries} for name, axis in axes]}
    state, evidence = compact_state(reference, template, plan, receipt, declaration, expected_sha256=digest(declaration))
    template.load_state_dict(state, strict=True)
    return template, evidence
