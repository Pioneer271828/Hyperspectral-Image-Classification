"""Existing M6 messages over full NONE-source target initialization.

New message parameters initialize at target. No source-M6-pretraining claim,
target optimizer, campaign, query release or common-model changes.
"""
import copy
from dataclasses import asdict, replace

import torch

from joint_baseline import forward as legacy_forward
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import OrderedValidation
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import load_model_checkpoint
from model.interaction_config import InteractionConfig
from model.research_config import digest
from none_full_transfer import initialize_none_full_target
from target_bn_calibration import (CONTEXT, GRAPH_POLICY, POLICY as BN_POLICY, calibrate_target,
                                    observation_id, verify_target_binding)
from target_initialization import METHODS, state_id

POLICY = "g4_m6_target_initialization_v1"
CALIBRATION = "g4_m6_target_ordered_population_bn_v1"
KEY = "interaction_target_binding"
CALIBRATION_KEY = "interaction_target_calibration"
CONDITIONS = ("none", "add", "gated", "cnn_to_graph", "graph_to_cnn", "global")
PREFIX = "backbone.interaction."


def configuration(condition):
    if condition not in CONDITIONS:
        raise ValueError("Explicit existing M6 target condition required")
    config = InteractionConfig()
    if condition != "none":
        config = replace(config, interaction_mode="add" if condition == "add" else "gated",
            interaction_direction=condition if condition in ("cnn_to_graph", "graph_to_cnn") else "both",
            alignment_mode="global" if condition == "global" else "region")
    return asdict(config)


def operator_signature(module):
    """Bind non-tensor message semantics absent from a state-dictionary hash."""
    if module is None:
        return None
    attributes = {}
    for name in ("mode", "direction", "strength", "chunk_size", "eps", "normalized_shape", "elementwise_affine",
                 "in_features", "out_features", "approximate", "inplace"):
        if hasattr(module, name):
            value = getattr(module, name)
            attributes[name] = list(value) if isinstance(value, tuple) else value
    return {"type": type(module).__module__ + "." + type(module).__name__, "attributes": attributes,
            "children": {name: operator_signature(child) for name, child in module.named_children()}}


def initialize(target, method, condition, *, model_seed, head_seed, source_spec=None, device="cpu"):
    config = configuration(condition)
    if (method not in METHODS[:2] or type(model_seed) is not int or model_seed not in (64, 512, 2048)
            or type(head_seed) is not int or head_seed != model_seed + 1000):
        raise ValueError("Registered M6 scratch/full NONE-source roles and separate seeds required")
    base, record = initialize_none_full_target(target, method, model_seed=model_seed, head_seed=head_seed,
                                              source_spec=source_spec, device=device)
    if condition == "none":
        model = base
    else:
        kwargs = {**base.export_config()["backbone_kwargs"], **config}
        # Constructor runs on CPU; discarded common parameters cannot consume
        # caller RNG. The original M6 initialization/strength are retained.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(model_seed)
            model = MultiDatasetVCASAHGCN({target: DATASETS[target][2]}, **kwargs).to(device)
        original, state = base.state_dict(), model.state_dict()
        if (not set(original) <= set(state) or any(not n.startswith(PREFIX) for n in set(state) - set(original))
                or any(state[n].shape != v.shape or state[n].dtype != v.dtype for n, v in original.items())):
            raise ValueError("Only new M6 target tensors may extend the unchanged base state")
        state.update({n: v.detach().clone() for n, v in original.items()})
        model.load_state_dict(state, strict=True)
    parameters = {n: p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX)}
    binding = {"policy": POLICY, "target": target, "method": method, "condition": condition,
        "model_seed": model_seed, "head_seed": head_seed, "interaction_config": config,
        "operator_signature": operator_signature(model.backbone.interaction),
        "architecture_sha256": digest(model.export_config()), "base_initialization": copy.deepcopy(record),
        "base_state_sha256": state_id(base), "initial_state_sha256": state_id(model),
        "message_parameters": parameters, "message_parameter_count": sum(parameters.values()),
        "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
        "source": "accepted_full_none_source_target_initialized_M6", "optimizer_updates": 0,
        "query_labels_accessed": False, "training_executable": False, "full_target_scene_transductive": True}
    model.training_config = {"target_initialization": copy.deepcopy(record), KEY: copy.deepcopy(binding)}
    verify_initialization_binding(model, binding)
    return model.eval(), binding


def verify_initialization_binding(model, expected):
    backbone = model.backbone
    if (not isinstance(expected, dict) or expected.get("policy") != POLICY
            or model.training_config.get(KEY) != expected
            or model.training_config.get("target_initialization") != expected["base_initialization"]
            or expected["interaction_config"] != configuration(expected["condition"])
            or asdict(backbone.interaction_config) != expected["interaction_config"]
            or backbone.alignment.mode != expected["interaction_config"]["alignment_mode"]
            or operator_signature(backbone.interaction) != expected["operator_signature"]
            or (backbone.interaction is None) != (expected["condition"] == "none")
            or backbone.edge_updater is not None or backbone.coverage_modulation is not None
            or backbone.region_residual_mode != "none" or backbone.graph_mode != "static"
            or backbone.norm_type != "domain_bn" or tuple(model.heads) != (expected["target"],)
            or digest(model.export_config()) != expected["architecture_sha256"]):
        raise ValueError("Actual M6 target operator/alignment/architecture binding differs")
    parameters = {n: p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX)}
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if (parameters != expected["message_parameters"] or sum(parameters.values()) != expected["message_parameter_count"]
            or trainable != expected["trainable_parameters"]):
        raise ValueError("M6 target parameter/optimization permission differs")
    return expected


def forward_observation(model, observation, *, expected_binding, diagnostics=False):
    binding = verify_initialization_binding(model, expected_binding)
    observation_id(observation)
    if diagnostics:
        if binding["condition"] == "none":
            raise ValueError("Disabled M6 retains original forward return contract")
        return model(binding["target"], observation["net_input"], observation["Q"], observation["A"],
            wavelengths=observation["wavelengths"], fwhm=observation["fwhm"],
            superpixel_scale=observation["superpixel_scale"], return_graph_info=True, return_diagnostics=True)
    return legacy_forward(model, binding["target"], observation)


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
        raise ValueError("Ordered M6 target calibration receipt differs")
    return receipt


def load_calibrated(path, weight_sha256, sidecar_sha256, observation, *, expected_binding, source_state_sha256, device="cpu"):
    pins = {str(path): weight_sha256, str(path) + ".json": sidecar_sha256}
    if any(file_hash(p) != sha for p, sha in pins.items()):
        raise ValueError("Pinned M6 target weight/sidecar pair required")
    model = load_model_checkpoint(path, device=device)
    verify_calibrated(model, observation, expected_binding=expected_binding, source_state_sha256=source_state_sha256)
    if any(file_hash(p) != sha for p, sha in pins.items()):
        raise ValueError("M6 target checkpoint pair changed during restore")
    return model
