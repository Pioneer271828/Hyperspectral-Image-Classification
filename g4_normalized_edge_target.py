"""Parameter-free original-M7 prior normalization over matching M6 controls.

The separate binding and required version buffer are checkpoint semantics;
ordinary legacy loaders must refuse enabled states rather than drop the control.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn

from g4_interaction_target import (KEY as BASE_KEY, configuration as base_configuration,
    initialize as base_initialize, operator_signature)
from joint_baseline import forward as legacy_forward
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import OrderedValidation
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from target_bn_calibration import (CONTEXT, GRAPH_POLICY, POLICY as BN_POLICY, calibrate_target,
    observation_id, verify_target_binding)
from target_initialization import state_id

POLICY = "g4_m7_prior_normalization_target_v1"
CALIBRATION = "g4_m7_prior_normalization_ordered_population_bn_v1"
KEY = "normalization_target_binding"
CALIBRATION_KEY = "normalization_target_calibration"
CONDITIONS = ("none", "row_normalized", "gated", "gated_row_normalized")
PREFIX = "backbone.edge_updater."


class CandidateRowNormalization(nn.Module):
    """Exactly the original candidate prior, including its tiny-mass floor."""
    def __init__(self, channels=128):
        super().__init__()
        if type(channels) is not int or channels <= 0: raise ValueError("Positive channel contract required")
        self.channels = channels; self.mode = "graph"; self.denominator_floor = 1e-12
        self.register_buffer("protocol_version", torch.tensor(1, dtype=torch.int64))

    def forward(self, regions, adjacency, aligned_pixels=None, collect_stats=False, return_diagnostics=False):
        if not torch.is_tensor(regions) or regions.ndim != 2 or not torch.is_tensor(adjacency):
            raise ValueError("Dense region matrix and adjacency tensor required")
        count = regions.shape[0]
        if (regions.ndim != 2 or count <= 0 or regions.shape[1] != self.channels
                or regions.dtype not in (torch.float32, torch.float64) or not torch.isfinite(regions).all()
                or adjacency.shape != (count, count) or adjacency.dtype != regions.dtype or adjacency.device != regions.device
                or self.mode != "graph" or self.denominator_floor != 1e-12 or int(self.protocol_version) != 1):
            raise ValueError("Declared finite normalization-only graph interface required")
        graph = adjacency.coalesce() if adjacency.is_sparse else adjacency.to_sparse().coalesce()
        values = graph.values()
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Original M7 finite nonnegative adjacency required")
        positive = values > 0
        indices, values = graph.indices()[:, positive], values[positive]
        mass = values.new_zeros(count).scatter_add(0, indices[0], values)
        isolated = (mass == 0).nonzero(as_tuple=True)[0]
        if isolated.numel():
            indices = torch.cat((indices, torch.stack((isolated, isolated))), dim=1)
            values = torch.cat((values, values.new_ones(isolated.numel())))
        prior = torch.sparse_coo_tensor(indices, values, (count, count)).coalesce()
        indices, values = prior.indices(), prior.values()
        mass = values.new_zeros(count).scatter_add(0, indices[0], values)
        weights = values / mass[indices[0]].clamp_min(self.denominator_floor)
        result = torch.sparse_coo_tensor(indices, weights, (count, count)).coalesce()
        info = {}
        if collect_stats:
            info.update(edge_update_l1=weights.detach().new_zeros(()), edge_update_max=weights.detach().new_zeros(()),
                edge_count=int(weights.numel()))
        if return_diagnostics:
            info.update(edge_indices=indices.detach(), edge_weights_before=weights.detach(), edge_weights_after=weights.detach())
        return result, info


def configuration(condition):
    if condition not in CONDITIONS: raise ValueError("Explicit normalization control condition required")
    return {"m6_condition": "gated" if condition.startswith("gated") else "none",
        "normalize": condition.endswith("row_normalized"), "floor": 1e-12,
        "location": "post_M6_pre_graph_stage2", "learned_M7": False}


def schema(model):
    return {n: {"shape": list(v.shape), "dtype": str(v.dtype)} for n, v in model.state_dict().items()}


def initialize(target, method, condition, **kwargs):
    config = configuration(condition)
    model, base = base_initialize(target, method, config["m6_condition"], **kwargs)
    previous = {n: v.detach().clone() for n, v in model.state_dict().items()}
    if config["normalize"]:
        model.backbone.edge_updater = CandidateRowNormalization().to(next(model.parameters()).device)
    after = model.state_dict()
    expected_new = {PREFIX + "protocol_version"} if config["normalize"] else set()
    if set(after) - set(previous) != expected_new or any(not torch.equal(v, after[n]) for n, v in previous.items()):
        raise ValueError("Only declared version buffer may extend matching M6 state")
    binding = {"policy": POLICY, "target": target, "method": method, "condition": condition,
        "configuration": config, "base_binding": copy.deepcopy(base), "state_schema": schema(model),
        "matching_M6_initial_state_sha256": base["initial_state_sha256"], "initial_state_sha256": state_id(model),
        "architecture_sha256": digest(model.export_config()), "normalization_parameter_count": 0,
        "matching_M6_parameter_count": base["message_parameter_count"], "additional_state_bytes": 8 if config["normalize"] else 0,
        "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
        "requires_normalization_aware_loader": config["normalize"], "optimizer_updates": 0,
        "training_executable": False, "query_labels_accessed": False,
        "export_kwargs_scope": "original_learned_M7_off_custom_normalization_in_this_binding"}
    model.training_config[KEY] = copy.deepcopy(binding)
    model.eval(); verify_initialization_binding(model, binding)
    return model, binding


def verify_initialization_binding(model, expected):
    b = model.backbone; base = expected["base_binding"]; control = b.edge_updater
    config = configuration(expected["condition"])
    if (expected.get("policy") != POLICY or expected["configuration"] != config or model.training_config.get(KEY) != expected
            or base["target"] != expected["target"] or base["method"] != expected["method"]
            or base["condition"] != config["m6_condition"] or expected["normalization_parameter_count"] != 0
            or expected["additional_state_bytes"] != (8 if config["normalize"] else 0)
            or expected["requires_normalization_aware_loader"] != config["normalize"]
            or expected["matching_M6_parameter_count"] != base["message_parameter_count"]
            or model.training_config.get(BASE_KEY) != base
            or model.training_config.get("target_initialization") != base["base_initialization"]
            or base["interaction_config"] != base_configuration(config["m6_condition"])
            or asdict(b.interaction_config) != base["interaction_config"]
            or b.alignment.mode != base["interaction_config"]["alignment_mode"]
            or operator_signature(b.interaction) != base["operator_signature"]
            or b.coverage_modulation is not None or b.region_residual_mode != "none" or b.graph_mode != "static"
            or b.norm_type != "domain_bn" or any(tuple(h) != (expected["target"],) for h in (model.heads, model.hgnn_heads, model.cnn_heads))
            or digest(model.export_config()) != expected["architecture_sha256"] or schema(model) != expected["state_schema"]
            or {n: p.numel() for n, p in model.named_parameters() if p.requires_grad} != expected["trainable_parameters"]
            or expected["trainable_parameters"] != base["trainable_parameters"]):
        raise ValueError("Actual normalization/M6/source/architecture/permission identity differs")
    if config["normalize"]:
        if (type(control) is not CandidateRowNormalization or getattr(control.forward, "__func__", None) is not CandidateRowNormalization.forward
                or control.channels != 128 or control.mode != "graph" or control.denominator_floor != 1e-12
                or int(control.protocol_version) != 1 or set(dict(control.named_buffers())) != {"protocol_version"}
                or list(control.parameters())):
            raise ValueError("Original parameter-free prior normalization identity differs")
    elif control is not None: raise ValueError("Disabled normalization must construct no operator")
    return expected


def forward_observation(model, observation, *, expected_binding):
    verify_initialization_binding(model, expected_binding); observation_id(observation)
    return legacy_forward(model, expected_binding["target"], observation)


def load_calibrated(path, weight_sha256, sidecar_sha256, observation, *, expected_binding, source_state_sha256, device="cpu"):
    path = Path(path).resolve(strict=True); sidecar = Path(str(path) + ".json")
    pins = {str(path): weight_sha256, str(sidecar): sidecar_sha256}
    if any(file_hash(p) != sha for p, sha in pins.items()): raise ValueError("Pinned normalization weight/sidecar pair required")
    meta = json.loads(sidecar.read_text())
    if meta["training_config"].get(KEY) != expected_binding: raise ValueError("Pinned explicit normalization binding required")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = MultiDatasetVCASAHGCN(meta["class_counts"], **meta["backbone_kwargs"])
        if configuration(expected_binding["condition"])["normalize"]: model.backbone.edge_updater = CandidateRowNormalization()
    model.to(device); model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.training_config = copy.deepcopy(meta["training_config"]); model.eval()
    verify_calibrated(model, observation, expected_binding=expected_binding, source_state_sha256=source_state_sha256)
    if any(file_hash(p) != sha for p, sha in pins.items()): raise ValueError("Normalization checkpoint pair changed during restore")
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
        raise ValueError("Ordered normalization-control target calibration receipt differs")
    return receipt



