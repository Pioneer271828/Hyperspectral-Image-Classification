"""Opt-in final position fusion over accepted complete-NONE-source G4 targets.

No shared model/config/training changes. The standalone kernel's dictionary extra
state is deliberately replaced by tensor-only full-model state; its exact config
is mandatory in this adapter binding and sidecar instead.
"""
import copy
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from g4_fusion_target import (initialize as fixed_initialize, configuration as fixed_configuration,
                             KEY as BASE_KEY, residual_mode)
from joint_baseline import forward as legacy_forward
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCNBackbone
from model.research_config import digest
from position_gated_fusion import PositionGatedFusion, POLICY as KERNEL_POLICY, VERSION
from target_bn_calibration import (CONTEXT, GRAPH_POLICY, POLICY as BN_POLICY, calibrate_target,
    observation_id, verify_target_binding)
from target_initialization import state_id
from target_training import loss_terms as original_loss_terms

POLICY = "g4_m8_position_fusion_target_v1"
CALIBRATION = "g4_m8_position_fusion_ordered_population_bn_v1"
KEY = "position_fusion_target_binding"
CALIBRATION_KEY = "position_fusion_target_calibration"
CONDITIONS = ("fixed", "position_gate")
PREFIX = "backbone.position_fusion."
CORE_FORWARD = PositionGatedFusion.forward
ORIGINAL_FUSE = VCASAHGCNBackbone.fuse_features


class _BoundPositionGate(PositionGatedFusion):
    # nn.Module recognizes these exact default methods and emits no _extra_state.
    # The adapter binding below carries the immutable kernel configuration.
    get_extra_state = nn.Module.get_extra_state
    set_extra_state = nn.Module.set_extra_state

    def export_payload(self):
        raise ValueError("Bound G4 gate requires full-model paired checkpoint and explicit binding")


class _PositionFusionBackbone(VCASAHGCNBackbone):
    def fuse_features(self, cnn_features, graph_features):
        return self.position_fusion(cnn_features, graph_features)


BOUND_FUSE = _PositionFusionBackbone.fuse_features


def configuration(condition):
    if condition not in CONDITIONS:
        raise ValueError("Explicit fixed or position_gate condition required")
    enabled = condition == "position_gate"
    return {"condition": condition, "enabled": enabled, "added_trainable_parameters": 129 if enabled else 0,
        "kernel": {"policy": KERNEL_POLICY, "version": VERSION, "channels": 64, "enabled": enabled, "alph": .8},
        "fusion_loss_contract": fixed_configuration("default")}


def schema(model):
    return {n: {"shape": list(v.shape), "dtype": str(v.dtype)} for n, v in model.state_dict().items()}


def _install(model):
    if type(model.backbone) is not VCASAHGCNBackbone or hasattr(model.backbone, "position_fusion"):
        raise ValueError("Original unmodified backbone required before position-gate install")
    p = next(model.parameters())
    model.backbone.__class__ = _PositionFusionBackbone
    model.backbone.position_fusion = _BoundPositionGate(64, enabled=True, alph=.8, device=p.device, dtype=p.dtype)


def operator_signature(model, enabled):
    b = model.backbone
    if enabled:
        control = getattr(b, "position_fusion", None)
        if (type(b) is not _PositionFusionBackbone or getattr(b.fuse_features, "__func__", None) is not BOUND_FUSE
                or type(control) is not _BoundPositionGate or getattr(control.forward, "__func__", None) is not CORE_FORWARD
                or type(control).get_extra_state is not nn.Module.get_extra_state
                or type(control).set_extra_state is not nn.Module.set_extra_state
                or control.export_config() != configuration("position_gate")["kernel"]
                or set(dict(control.named_buffers())) != {"protocol_version"}
                or set(dict(control.named_parameters())) != {"gate_weight", "gate_bias"}):
            raise ValueError("Actual final position gate/method/configuration identity differs")
        # Validate original kernel tensor schema/version/finite FP32/64 values.
        state = control.state_dict()
        if (set(state) != {"gate_weight", "gate_bias", "protocol_version"}
                or state["protocol_version"].dtype != torch.int64 or state["protocol_version"].shape != torch.Size([])
                or state["protocol_version"].item() != VERSION
                or tuple(state["gate_weight"].shape) != (1, 128) or tuple(state["gate_bias"].shape) != (1,)
                or any(v.dtype != torch.float32 or not torch.isfinite(v).all() for n, v in state.items() if n != "protocol_version")):
            raise ValueError("Bound gate requires finite FP32 tensor-only kernel schema/version")
    elif type(b) is not VCASAHGCNBackbone or hasattr(b, "position_fusion") or getattr(b.fuse_features, "__func__", None) is not ORIGINAL_FUSE:
        raise ValueError("Fixed path must retain original backbone with no added operator")
    return {"backbone_type": type(b).__module__ + "." + type(b).__name__,
        "fuse_source_sha256": digest(inspect.getsource(BOUND_FUSE if enabled else ORIGINAL_FUSE)),
        "kernel_forward_source_sha256": digest(inspect.getsource(CORE_FORWARD)),
        "kernel_file_sha256": file_hash(Path(inspect.getfile(PositionGatedFusion)).resolve()),
        "tensor_only_bound_state": True, "enabled": enabled}


def initialize(target, method, condition, *, model_seed, head_seed, source_spec=None, device="cpu", mode="none"):
    config = configuration(condition)
    model, base = fixed_initialize(target, method, "default", model_seed=model_seed, head_seed=head_seed,
        source_spec=source_spec, device=device, mode=mode)
    before = {n: v.detach().clone() for n, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    if config["enabled"]: _install(model)
    if not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError("Final gate installation consumed CPU RNG")
    actual = model.state_dict(); shared = {n: v for n, v in actual.items() if not n.startswith(PREFIX)}
    if set(shared) != set(before) or any(not torch.equal(v, shared[n]) for n, v in before.items()):
        raise RuntimeError("Original shared, fresh head or target-domain tensors changed")
    added = {n: p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX)}
    if sum(added.values()) != config["added_trainable_parameters"]:
        raise RuntimeError("Declared scalar position-gate parameter budget differs")
    binding = {"policy": POLICY, "target": target, "method": method, "condition": condition,
        "model_seed": model_seed, "head_seed": head_seed, "config": config, "base_binding": copy.deepcopy(base),
        "base_state_sha256": base["initial_state_sha256"], "initial_state_sha256": state_id(model),
        "architecture_sha256": digest(model.export_config()), "state_schema": schema(model),
        "operator_signature": operator_signature(model, config["enabled"]),
        "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
        "added_parameter_count": sum(added.values()), "additional_state_bytes": 8 if config["enabled"] else 0,
        "requires_position_aware_loader": config["enabled"], "optimizer_updates": 0,
        "training_executable": False, "query_labels_accessed": False, "full_target_scene_transductive": True}
    model.training_config[KEY] = copy.deepcopy(binding)
    verify_initialization_binding(model, binding)
    return model.eval(), binding


def verify_initialization_binding(model, expected):
    if not isinstance(expected, dict) or expected.get("policy") != POLICY:
        raise ValueError("Explicit position-fusion target binding required")
    b = model.backbone; config = configuration(expected["condition"]); base = expected["base_binding"]
    if (model.training_config.get(KEY) != expected or model.training_config.get(BASE_KEY) != base
            or model.training_config.get("target_initialization") != base["base_initialization"]
            or expected["config"] != config or base["condition"] != "default"
            or base["fusion_loss_contract"] != config["fusion_loss_contract"]
            or expected["base_state_sha256"] != base["initial_state_sha256"]
            or base["target"] != expected["target"] or base["method"] != expected["method"]
            or base["model_seed"] != expected["model_seed"] or base["head_seed"] != expected["head_seed"]
            or b.alph != .8 or b.interaction is not None or b.edge_updater is not None or b.interaction_config.active
            or b.coverage_modulation is not None or b.region_residual_mode != residual_mode(base)
            or b.graph_mode != "static" or b.norm_type != "domain_bn"
            or getattr(b.forward_features, "__func__", None) is not VCASAHGCNBackbone.forward_features
            or getattr(model.predict_features, "__func__", None) is not MultiDatasetVCASAHGCN.predict_features
            or getattr(model.fuse_and_predict, "__func__", None) is not MultiDatasetVCASAHGCN.fuse_and_predict
            or any(tuple(h) != (expected["target"],) for h in (model.heads, model.hgnn_heads, model.cnn_heads))
            or digest(model.export_config()) != expected["architecture_sha256"] or schema(model) != expected["state_schema"]
            or operator_signature(model, config["enabled"]) != expected["operator_signature"]
            or expected["added_parameter_count"] != config["added_trainable_parameters"]
            or {n: p.numel() for n, p in model.named_parameters() if p.requires_grad} != expected["trainable_parameters"]):
        raise ValueError("Actual final fusion/source/architecture/permission identity differs")
    shared = {n: v for n, v in expected["trainable_parameters"].items() if not n.startswith(PREFIX)}
    if shared != base["trainable_parameters"]:
        raise ValueError("Original parameter permissions/budget changed")
    return expected


def forward_observation(model, observation, *, expected_binding, logits=False):
    binding = verify_initialization_binding(model, expected_binding); observation_id(observation)
    if logits:
        return model(binding["target"], observation["net_input"], observation["Q"], observation["A"],
            wavelengths=observation["wavelengths"], fwhm=observation["fwhm"],
            superpixel_scale=observation["superpixel_scale"], return_logits=True)
    return legacy_forward(model, binding["target"], observation)


def loss_terms(model, outputs, labels, config, *, expected_binding):
    contract = verify_initialization_binding(model, expected_binding)["config"]["fusion_loss_contract"]
    if (config.model.alph != contract["alph"] or config.loss.cnn_loss_weight != contract["cnn_loss_weight"]
            or config.loss.graph_branch_loss_weight != contract["graph_branch_loss_weight"] or config.loss.balanced != contract["balanced"]):
        raise ValueError("Original M8 three-head loss contract required")
    classes = model.heads[expected_binding["target"]].out_features
    if labels.dtype != torch.int64 or labels.ndim != 1 or len(outputs) != 3 or any(not torch.is_tensor(v) or v.shape != (labels.numel(), classes) for v in outputs):
        raise ValueError("Three selected-row probability matrices and integer labels required")
    if any(not torch.isfinite(v).all() or (v < 0).any() or (v > 1).any()
           or not torch.allclose(v.sum(-1), torch.ones_like(v.sum(-1)), rtol=1e-5, atol=1e-6) for v in outputs):
        raise ValueError("Original probability supervision required")
    return original_loss_terms(outputs, labels, config)


def calibrate(model, observation, *, expected_binding, source_state_sha256):
    binding = verify_initialization_binding(model, expected_binding)
    if any(p.dtype != torch.float32 for p in model.parameters()): raise ValueError("Original FP32 calibration only")
    with OrderedValidation():
        result, legacy = calibrate_target(model, observation, mode=BN_POLICY, context=CONTEXT,
            graph_policy=GRAPH_POLICY, source_state_sha256=source_state_sha256)
    result.training_config[CALIBRATION_KEY] = {"policy": CALIBRATION, "binding_sha256": digest(binding),
        "legacy_receipt": copy.deepcopy(legacy), "legacy_receipt_sha256": digest(legacy),
        "arithmetic": "fp32_sorted_row_segment_sum_v1", "source_state_sha256": source_state_sha256,
        "observation": observation_id(observation), "state_sha256": state_id(result),
        "optimizer_updates": 0, "query_labels_accessed": False}
    verify_calibrated(result, observation, expected_binding=binding, source_state_sha256=source_state_sha256)
    return result, copy.deepcopy(result.training_config[CALIBRATION_KEY])


def verify_calibrated(model, observation, *, expected_binding, source_state_sha256):
    binding = verify_initialization_binding(model, expected_binding)
    legacy = verify_target_binding(model, observation, context=CONTEXT, graph_policy=GRAPH_POLICY,
        source_state_sha256=source_state_sha256)
    receipt = model.training_config.get(CALIBRATION_KEY, {})
    if (receipt.get("policy") != CALIBRATION or receipt.get("binding_sha256") != digest(binding)
            or receipt.get("legacy_receipt") != legacy or receipt.get("legacy_receipt_sha256") != digest(legacy)
            or receipt.get("arithmetic") != "fp32_sorted_row_segment_sum_v1"
            or receipt.get("source_state_sha256") != source_state_sha256 or receipt.get("observation") != observation_id(observation)
            or receipt.get("state_sha256") != state_id(model) or receipt.get("optimizer_updates") != 0
            or receipt.get("query_labels_accessed") is not False or any(m.training for m in model.modules())):
        raise ValueError("Ordered position-fusion calibration receipt differs")
    return receipt


def load_calibrated(path, weight_sha256, sidecar_sha256, observation, *, expected_binding, source_state_sha256, device="cpu"):
    path = Path(path).resolve(strict=True); sidecar = Path(str(path) + ".json")
    pins = {str(path): weight_sha256, str(sidecar): sidecar_sha256}
    if any(file_hash(p) != sha for p, sha in pins.items()): raise ValueError("Pinned position-fusion weight/sidecar pair required")
    meta = json.loads(sidecar.read_text())
    if meta["training_config"].get(KEY) != expected_binding: raise ValueError("Explicit paired fusion binding required")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = MultiDatasetVCASAHGCN(meta["class_counts"], **meta["backbone_kwargs"])
        if configuration(expected_binding["condition"])["enabled"]: _install(model)
    model.to(device); model.training_config = copy.deepcopy(meta["training_config"]); model.eval()
    verify_initialization_binding(model, expected_binding)
    state = torch.load(path, map_location=device, weights_only=True)
    if any(not torch.is_tensor(v) or (v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())) for v in state.values()):
        raise ValueError("Finite FP32 tensor-only target checkpoint required")
    model.load_state_dict(state, strict=True)
    verify_calibrated(model, observation, expected_binding=expected_binding, source_state_sha256=source_state_sha256)
    if any(file_hash(p) != sha for p, sha in pins.items()): raise ValueError("Position-fusion pair changed during restore")
    return model
