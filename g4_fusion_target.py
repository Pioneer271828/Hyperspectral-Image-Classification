"""Existing fixed M8 fusion/auxiliary contracts over accepted full NONE sources.

No new parameters, loss formula, learned fusion, optimizer or query interface.
"""
import copy

import torch

from joint_baseline import forward as legacy_forward
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import OrderedValidation
from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCNBackbone
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from none_full_transfer import initialize_none_full_target
from m4_full_transfer import initialize_full_target
from m4_transfer import MODES
from target_bn_calibration import (CONTEXT, GRAPH_POLICY, POLICY as BN_POLICY, calibrate_target,
                                    observation_id, verify_target_binding)
from target_initialization import METHODS, state_id
from target_training import loss_terms as legacy_loss_terms

POLICY = "g4_m8_target_initialization_v1"
CALIBRATION = "g4_m8_target_ordered_population_bn_v1"
KEY = "fusion_target_binding"
CALIBRATION_KEY = "fusion_target_calibration"
CONDITIONS = ("default", "aux_off", "alpha035")


def configuration(condition):
    if condition not in CONDITIONS:
        raise ValueError("Explicit existing M8 target condition required")
    return {"alph": .35 if condition == "alpha035" else .8,
            "cnn_loss_weight": 0. if condition == "aux_off" else 1.,
            "graph_branch_loss_weight": 0. if condition == "aux_off" else 1., "balanced": True}


def initialize(target, method, condition, *, model_seed, head_seed, source_spec=None, device="cpu", mode="none"):
    contract = configuration(condition)
    if (method not in METHODS[:2] or type(model_seed) is not int or model_seed not in (64, 512, 2048)
            or type(head_seed) is not int or head_seed != model_seed + 1000):
        raise ValueError("Declared M8 scratch/full NONE-source role and separate seeds required")
    if mode not in MODES:
        raise ValueError("Explicit accepted residual/capacity mode required")
    if mode == "none":
        base, record = initialize_none_full_target(target, method, model_seed=model_seed, head_seed=head_seed,
                                                  source_spec=source_spec, device=device)
    else:
        base, record = initialize_full_target(target, method, mode, model_seed=model_seed, head_seed=head_seed,
                                              source_spec=source_spec, device=device)
    if contract["alph"] == .8:
        model = base
    else:
        kwargs = {**base.export_config()["backbone_kwargs"], "alph": contract["alph"]}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(model_seed)
            model = MultiDatasetVCASAHGCN({target: DATASETS[target][2]}, **kwargs).to(device)
        before, after = base.state_dict(), model.state_dict()
        if (set(before) != set(after) or any(v.shape != after[n].shape or v.dtype != after[n].dtype
                                            for n, v in before.items())):
            raise ValueError("M8 configuration cannot add/change parameter tensors")
        model.load_state_dict(before, strict=True)
    if state_id(base) != state_id(model):
        raise ValueError("All original shared/fresh head/domain tensors must remain identical")
    binding = {"policy": POLICY, "target": target, "method": method, "condition": condition,
        "model_seed": model_seed, "head_seed": head_seed, "fusion_loss_contract": contract,
        "base_initialization": copy.deepcopy(record), "base_state_sha256": state_id(base),
        "initial_state_sha256": state_id(model), "architecture_sha256": digest(model.export_config()),
        "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
        "added_parameter_count": 0, "source": "accepted_full_none_source_fixed_M8_target_configuration",
        "optimizer_updates": 0, "training_executable": False, "query_labels_accessed": False,
        "full_target_scene_transductive": True}
    if mode != "none":
        binding.update(residual_mode=mode, combination_version=1,
                       source="accepted_full_m4_source_fixed_M8_target_configuration")
    model.training_config = {"target_initialization": copy.deepcopy(record), KEY: copy.deepcopy(binding)}
    verify_initialization_binding(model, binding)
    return model.eval(), binding


def residual_mode(binding):
    mode = binding.get("residual_mode", "none")
    if (mode not in MODES or binding["base_initialization"].get("m4_mode", "none") != mode
            or (mode != "none" and binding.get("combination_version") != 1)):
        raise ValueError("Residual mode and original source initialization differ")
    return mode


def verify_initialization_binding(model, expected):
    backbone = model.backbone
    if (not isinstance(expected, dict) or expected.get("policy") != POLICY
            or model.training_config.get(KEY) != expected
            or model.training_config.get("target_initialization") != expected["base_initialization"]
            or expected["fusion_loss_contract"] != configuration(expected["condition"])
            or backbone.alph != expected["fusion_loss_contract"]["alph"]
            or getattr(backbone.fuse_features, "__func__", None) is not VCASAHGCNBackbone.fuse_features
            or getattr(model.predict_features, "__func__", None) is not MultiDatasetVCASAHGCN.predict_features
            or getattr(model.fuse_and_predict, "__func__", None) is not MultiDatasetVCASAHGCN.fuse_and_predict
            or backbone.interaction_config.active or backbone.interaction is not None or backbone.edge_updater is not None
            or backbone.coverage_modulation is not None or backbone.region_residual_mode != residual_mode(expected)
            or backbone.graph_mode != "static" or backbone.norm_type != "domain_bn"
            or any(tuple(family) != (expected["target"],) for family in (model.heads, model.hgnn_heads, model.cnn_heads))
            or digest(model.export_config()) != expected["architecture_sha256"]
            or {n: p.numel() for n, p in model.named_parameters() if p.requires_grad} != expected["trainable_parameters"]):
        raise ValueError("Actual fixed M8 fusion/heads/architecture/permission binding differs")
    return expected


def forward_observation(model, observation, *, expected_binding, logits=False):
    binding = verify_initialization_binding(model, expected_binding)
    observation_id(observation)
    if logits:
        return model(binding["target"], observation["net_input"], observation["Q"], observation["A"],
            wavelengths=observation["wavelengths"], fwhm=observation["fwhm"],
            superpixel_scale=observation["superpixel_scale"], return_logits=True)
    return legacy_forward(model, binding["target"], observation)


def loss_terms(model, outputs, labels, config, *, expected_binding):
    """Consume permitted selected training rows using the original probability loss."""
    contract = verify_initialization_binding(model, expected_binding)["fusion_loss_contract"]
    if (config.model.alph != contract["alph"] or config.loss.cnn_loss_weight != contract["cnn_loss_weight"]
            or config.loss.graph_branch_loss_weight != contract["graph_branch_loss_weight"]
            or config.loss.balanced != contract["balanced"]):
        raise ValueError("Explicit M8 fusion/auxiliary loss configuration differs")
    classes = model.heads[expected_binding["target"]].out_features
    if (labels.ndim != 1 or labels.dtype != torch.int64 or len(outputs) != 3
            or any(not torch.is_tensor(v) or v.shape != (labels.numel(), classes) for v in outputs)):
        raise ValueError("Three selected-row probability matrices and integer labels required")
    if any(not torch.isfinite(v).all() or (v < 0).any() or (v > 1).any()
            or not torch.allclose(v.sum(-1), torch.ones_like(v.sum(-1)), rtol=1e-5, atol=1e-6) for v in outputs):
        raise ValueError("M8 retains three probability outputs, not logits supervision")
    return legacy_loss_terms(outputs, labels, config)



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
        raise ValueError("Ordered M8 target calibration receipt differs")
    return receipt


def load_calibrated(path, weight_sha256, sidecar_sha256, observation, *, expected_binding, source_state_sha256, device="cpu"):
    pins = {str(path): weight_sha256, str(path) + ".json": sidecar_sha256}
    if any(file_hash(p) != sha for p, sha in pins.items()):
        raise ValueError("Pinned M8 target weight/sidecar pair required")
    model = load_model_checkpoint(path, device=device)
    verify_calibrated(model, observation, expected_binding=expected_binding, source_state_sha256=source_state_sha256)
    if any(file_hash(p) != sha for p, sha in pins.items()):
        raise ValueError("M8 target checkpoint pair changed during restore")
    return model
