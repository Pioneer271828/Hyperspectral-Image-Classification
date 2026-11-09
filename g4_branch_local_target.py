"""Parameter-matched branch-local capacity controls over original M6 targets.

The separate binding and required version buffer are checkpoint semantics;
ordinary legacy loaders must refuse enabled states rather than drop the control.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn

from g4_interaction_target import (KEY as BASE_KEY, POLICY as BASE_POLICY, configuration as base_configuration,
    initialize as base_initialize, operator_signature)
from joint_baseline import forward as legacy_forward
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import OrderedValidation
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.cross_branch_interaction import BidirectionalSelectiveInteraction, SelectiveMessage
from model.research_config import digest
from target_bn_calibration import (CONTEXT, GRAPH_POLICY, POLICY as BN_POLICY, calibrate_target,
    observation_id, verify_target_binding)
from target_initialization import state_id

POLICY = "g4_m6_branch_local_target_v1"
CALIBRATION = "g4_m6_branch_local_ordered_population_bn_v1"
KEY = "branch_local_target_binding"
CALIBRATION_KEY = "branch_local_target_calibration"
CONDITIONS = ("none", "gated", "branch_local")
PREFIX = "backbone.interaction."


class BranchLocalSelectiveInteraction(BidirectionalSelectiveInteraction):
    """Original gated transforms fed only by their own intermediate branch."""
    def __init__(self, channels=128, hidden_dim=32, strength=.1, chunk_size=16384):
        super().__init__(channels=channels, mode="gated", direction="both",
                         hidden_dim=hidden_dim, strength=strength, chunk_size=chunk_size)
        self.channels = channels
        self.register_buffer("protocol_version", torch.tensor(1, dtype=torch.int64))

    def forward_aligned(self, pixels, regions, assignment, alignment,
                        collect_stats=False, return_diagnostics=False):
        if (not torch.is_tensor(pixels) or pixels.ndim != 2 or not torch.is_tensor(regions) or regions.ndim != 2
                or not torch.is_tensor(assignment) or assignment.ndim != 2
                or assignment.shape != (pixels.shape[0], regions.shape[0])):
            raise ValueError("Local control requires declared pixel/region assignment shape")
        # Shape compatibility is checked; no Q values or alignment calls enter
        # either local transform. The graph backbone itself still uses original Q.
        return self(pixels, regions, collect_stats=collect_stats, return_diagnostics=return_diagnostics)

    def forward(self, pixels, regions, region_at_pixels=None, pixels_at_regions=None,
                collect_stats=False, return_diagnostics=False):
        if region_at_pixels is not None or pixels_at_regions is not None:
            raise ValueError("Branch-local control refuses external cross messages")
        if (self.mode != "gated" or self.direction != "both" or int(self.protocol_version) != 1
                or any(not torch.is_tensor(x) or x.ndim != 2 or x.shape[1] != self.channels
                       or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all()
                       for x in (pixels, regions))
                or pixels.dtype != regions.dtype or pixels.device != regions.device):
            raise ValueError("Finite matching branch-local feature matrices required")
        result, info = [], {}
        for label, module, value in (("pixel_local", self.graph_to_cnn, pixels),
                                     ("region_local", self.cnn_to_graph, regions)):
            updated, gate = module(value, value)
            result.append(updated)
            if collect_stats:
                info[label + "_gate_mean"] = gate.detach().mean() if gate.numel() else gate.new_zeros(())
                info[label + "_gate_std"] = gate.detach().std(unbiased=False) if gate.numel() else gate.new_zeros(())
            if return_diagnostics: info[label + "_gate"] = gate.detach()
        return result[0], result[1], info


def configuration(condition):
    if condition not in CONDITIONS: raise ValueError("Explicit branch-local capacity condition required")
    return {"matching_M6_condition": "none" if condition == "none" else "gated",
            "source_scope": "branch_local" if condition == "branch_local" else ("disabled" if condition == "none" else "original_cross_branch"),
            "parameter_matched": condition == "branch_local", "compute_budget_matched": False}


def local_operator(config):
    return BranchLocalSelectiveInteraction(channels=128, hidden_dim=config["interaction_hidden_dim"],
        strength=config["interaction_strength"], chunk_size=config["interaction_chunk_pixels"])

def schema(model):
    return {n: {"shape": list(v.shape), "dtype": str(v.dtype)} for n, v in model.state_dict().items()}


def initialize(target, method, condition, **kwargs):
    config = configuration(condition)
    model, base = base_initialize(target, method, config["matching_M6_condition"], **kwargs)
    previous = {n: v.detach().clone() for n, v in model.state_dict().items()}
    if condition == "branch_local":
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            control = local_operator(base["interaction_config"])
        state = control.state_dict()
        state.update(model.backbone.interaction.state_dict())
        control.load_state_dict(state, strict=True)
        model.backbone.interaction = control.to(next(model.parameters()).device)
    after = model.state_dict()
    expected_new = {PREFIX + "protocol_version"} if condition == "branch_local" else set()
    if set(after) - set(previous) != expected_new or any(not torch.equal(v, after[n]) for n, v in previous.items()):
        raise ValueError("Only declared local-control version buffer may extend matching M6 state")
    binding = {"policy": POLICY, "target": target, "method": method, "condition": condition,
        "configuration": config, "base_binding": copy.deepcopy(base), "state_schema": schema(model),
        "matching_M6_initial_state_sha256": base["initial_state_sha256"], "initial_state_sha256": state_id(model),
        "architecture_sha256": digest(model.export_config()), "operator_signature": operator_signature(model.backbone.interaction),
        "capacity_parameter_count": base["message_parameter_count"], "additional_trainable_parameters": 0,
        "additional_state_bytes": 8 if condition == "branch_local" else 0,
        "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
        "requires_local_aware_loader": condition == "branch_local", "optimizer_updates": 0,
        "training_executable": False, "query_labels_accessed": False,
        "export_kwargs_scope": "original_gated_shape_custom_branch_local_semantics_in_this_binding"}
    model.training_config[KEY] = copy.deepcopy(binding)
    model.eval(); verify_initialization_binding(model, binding)
    return model, binding


def verify_initialization_binding(model, expected):
    b = model.backbone; base = expected["base_binding"]; control = b.interaction
    config = configuration(expected["condition"]); local = expected["condition"] == "branch_local"
    if (expected.get("policy") != POLICY or expected["configuration"] != config or model.training_config.get(KEY) != expected
            or base.get("policy") != BASE_POLICY or base["target"] != expected["target"] or base["method"] != expected["method"]
            or expected["matching_M6_initial_state_sha256"] != base["initial_state_sha256"]
            or base["condition"] != config["matching_M6_condition"] or expected["additional_trainable_parameters"] != 0
            or expected["additional_state_bytes"] != (8 if local else 0) or expected["requires_local_aware_loader"] != local
            or expected["capacity_parameter_count"] != base["message_parameter_count"]
            or model.training_config.get(BASE_KEY) != base or model.training_config.get("target_initialization") != base["base_initialization"]
            or base["interaction_config"] != base_configuration(config["matching_M6_condition"])
            or asdict(b.interaction_config) != base["interaction_config"]
            or b.alignment.mode != base["interaction_config"]["alignment_mode"]
            or operator_signature(control) != expected["operator_signature"]
            or b.edge_updater is not None or b.coverage_modulation is not None or b.region_residual_mode != "none"
            or b.graph_mode != "static" or b.norm_type != "domain_bn"
            or any(tuple(h) != (expected["target"],) for h in (model.heads, model.hgnn_heads, model.cnn_heads))
            or digest(model.export_config()) != expected["architecture_sha256"] or schema(model) != expected["state_schema"]
            or {n: p.numel() for n, p in model.named_parameters() if p.requires_grad} != expected["trainable_parameters"]
            or expected["trainable_parameters"] != base["trainable_parameters"]):
        raise ValueError("Actual local-control/M6/source/architecture/permission identity differs")
    if local:
        if (type(control) is not BranchLocalSelectiveInteraction or control.channels != 128
                or getattr(control.forward, "__func__", None) is not BranchLocalSelectiveInteraction.forward
                or getattr(control.forward_aligned, "__func__", None) is not BranchLocalSelectiveInteraction.forward_aligned
                or int(control.protocol_version) != 1 or set(dict(control.named_buffers())) != {"protocol_version"}
                or any(type(m) is not SelectiveMessage or getattr(m.forward, "__func__", None) is not SelectiveMessage.forward
                       or getattr(m._chunk, "__func__", None) is not SelectiveMessage._chunk
                       for m in (control.graph_to_cnn, control.cnn_to_graph))):
            raise ValueError("Branch-local method/version/original message identity differs")
    elif operator_signature(control) != base["operator_signature"]:
        raise ValueError("Original matching M6 operator differs")
    return expected

def forward_observation(model, observation, *, expected_binding):
    verify_initialization_binding(model, expected_binding); observation_id(observation)
    return legacy_forward(model, expected_binding["target"], observation)


def load_calibrated(path, weight_sha256, sidecar_sha256, observation, *, expected_binding, source_state_sha256, device="cpu"):
    path = Path(path).resolve(strict=True); sidecar = Path(str(path) + ".json")
    pins = {str(path): weight_sha256, str(sidecar): sidecar_sha256}
    if any(file_hash(p) != sha for p, sha in pins.items()): raise ValueError("Pinned branch-local weight/sidecar pair required")
    meta = json.loads(sidecar.read_text())
    if meta["training_config"].get(KEY) != expected_binding: raise ValueError("Pinned explicit branch-local binding required")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = MultiDatasetVCASAHGCN(meta["class_counts"], **meta["backbone_kwargs"])
        if expected_binding["condition"] == "branch_local":
            model.backbone.interaction = local_operator(expected_binding["base_binding"]["interaction_config"])
    model.to(device); model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.training_config = copy.deepcopy(meta["training_config"]); model.eval()
    verify_calibrated(model, observation, expected_binding=expected_binding, source_state_sha256=source_state_sha256)
    if any(file_hash(p) != sha for p, sha in pins.items()): raise ValueError("Branch-local checkpoint pair changed during restore")
    return model


def calibrate(model, observation, *, expected_binding, source_state_sha256):
    binding = verify_initialization_binding(model, expected_binding)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Original FP32 ordered target calibration only")
    with OrderedValidation():
        result, receipt = calibrate_target(model, observation, mode=BN_POLICY, context=CONTEXT,
            graph_policy=GRAPH_POLICY, source_state_sha256=source_state_sha256)
    result.training_config[CALIBRATION_KEY] = {"policy": CALIBRATION, "binding_sha256": digest(binding),
        "legacy_receipt_sha256": digest(receipt), "arithmetic": "fp32_sorted_row_segment_sum_v1",
        "source_state_sha256": source_state_sha256, "observation": observation_id(observation),
        "state_sha256": state_id(result), "optimizer_updates": 0, "query_labels_accessed": False}
    verify_calibrated(result, observation, expected_binding=binding, source_state_sha256=source_state_sha256)
    return result, copy.deepcopy(result.training_config[CALIBRATION_KEY])


def verify_calibrated(model, observation, *, expected_binding, source_state_sha256):
    binding = verify_initialization_binding(model, expected_binding)
    legacy = verify_target_binding(model, observation, context=CONTEXT, graph_policy=GRAPH_POLICY,
                                   source_state_sha256=source_state_sha256)
    receipt = model.training_config.get(CALIBRATION_KEY, {})
    if (receipt.get("policy") != CALIBRATION or receipt.get("binding_sha256") != digest(binding)
            or receipt.get("legacy_receipt_sha256") != digest(legacy)
            or receipt.get("arithmetic") != "fp32_sorted_row_segment_sum_v1"
            or receipt.get("source_state_sha256") != source_state_sha256
            or receipt.get("state_sha256") != state_id(model) or receipt.get("observation") != observation_id(observation)
            or receipt.get("optimizer_updates") != 0 or receipt.get("query_labels_accessed") is not False
            or any(m.training for m in model.modules())):
        raise ValueError("Ordered branch-local-control target calibration receipt differs")
    return receipt





