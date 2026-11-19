"""Load frozen WP05, WP07 and WP08 models without changing their graph semantics."""

from contextlib import nullcontext
from time import perf_counter

import torch

from evaluation.frozen_training_stages import json_pin, pinned, require
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN
from transfer.g3_training import state_hash
from transfer.native_scale_observation import forward
from transfer.region_message_attention import RegionMessageVCASAHGCN, profile_region_messages

BACKENDS = ("native", "multiscale", "aligned", "region_message")
HEADS = ("fused", "graph", "cnn")


def make_model(architecture, backend):
    require(set(architecture) in ({"format_version", "class_counts", "backbone_kwargs"},
                                 {"format_version", "class_counts", "backbone_kwargs", "training_config"})
            and architecture["format_version"] == 1 and backend in BACKENDS,
            "An exact registered architecture and supported backend are required")
    counts, kwargs = architecture["class_counts"], architecture["backbone_kwargs"]
    require(counts and all(type(value) is int and value > 1 for value in counts.values()),
            "Model head must have explicit positive class identities")
    multiscale = {"cnn_multiscale", "cnn_dilation", "graph_multiscale", "graph_radius"} & kwargs.keys()
    require(("region_attention" in kwargs) == (backend == "region_message")
            and bool(multiscale) == (backend == "multiscale")
            and (not multiscale or {"cnn_multiscale", "cnn_dilation"} <= multiscale),
            "Architecture disagrees with the declared backend")
    factory = {"native": MultiDatasetVCASAHGCN, "aligned": MultiDatasetVCASAHGCN,
               "multiscale": ConditionalMultiscaleVCASAHGCN, "region_message": RegionMessageVCASAHGCN}[backend]
    with torch.random.fork_rng(devices=[]):
        model = factory(counts, **kwargs).eval()
    b = model.backbone
    exchanged = backend in ("aligned", "region_message")
    require(b.graph_mode == b.graph_assignment_mode == b.graph_adjacency_mode == "learned"
            and (b.interaction is not None) == exchanged
            and (not exchanged or b.alignment.mode == "region")
            and b.graph_router is None and b.edge_updater is None,
            "Checkpoint uses an incompatible graph or interaction contract")
    model.training_config = architecture.get("training_config", {})
    model.backbone.adapter.cache_attention = False
    return model


def restore(architecture, backend, state, expected_state, *, device="cpu"):
    model = make_model(architecture, backend)
    reference = model.state_dict()
    require(state.keys() == reference.keys() and all(
        torch.is_tensor(value) and value.shape == reference[key].shape and value.dtype == reference[key].dtype
        and (not value.is_floating_point() or torch.isfinite(value).all().item())
        for key, value in state.items()), "Frozen tensors disagree with registered architecture or precision")
    model.load_state_dict(state, strict=True)
    require(state_hash(model) == expected_state, "Frozen state fingerprint differs")
    return model.to(device).eval().requires_grad_(False)


def load_export(root, export, *, device="cpu"):
    require(set(export) == {"backend", "weights", "metadata", "state"}, "Exact checkpoint export pins required")
    weights = pinned(root, export["weights"])
    metadata = pinned(root, export["metadata"])
    require(metadata == weights.with_suffix(weights.suffix + ".json"), "Weights and architecture are not sidecars")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    return restore(json_pin(root, export["metadata"]), export["backend"], state, export["state"], device=device)


def semantic_identity(model, backend):
    bone = model.backbone
    return dict(backend=backend, assignment=bone.graph_assignment_mode, adjacency=bone.graph_adjacency_mode,
                range_mode=getattr(model, "graph_multiscale", "off"), interaction=bone.interaction_config.interaction_mode,
                direction=bone.interaction_config.interaction_direction if bone.interaction is not None else None,
                attention=getattr(model, "region_attention", None))


@torch.no_grad()
def run_block(model, actual, producer, role, execution):
    """Run the existing native observation forward with exact bands, no truth or optimizer."""
    timing, message = {}, {}
    manager = profile_region_messages(message) if isinstance(model, RegionMessageVCASAHGCN) else nullcontext()
    execution.sync()
    start = perf_counter()
    with manager:
        output, graph = forward(model, actual, producer, role, execution=execution, timed=timing)
    execution.sync()
    stats = dict(elapsed=perf_counter()-start, graph_seconds=sum(timing.get("graph_seconds", [])),
                 message_seconds={name: sum(values) for name, values in message.items()},
                 message_trace=output[3].get("interaction", {}).get("region_message", {}))
    require(len(output) >= 4 and all(value.shape[0] == actual.receipt["pixels"] for value in output[:3]),
            "Backend did not emit the original three point heads")
    return output[:3], graph, stats
