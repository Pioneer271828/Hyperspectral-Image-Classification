"""E1 target initialization and full-observation population calibration.

Separate opt-in component; the accepted coverage-off target trainer/calibrator
remain frozen. No target optimizer, experiment runner or query release here.
"""
import copy
from contextlib import nullcontext
from dataclasses import asdict
import json

import torch

from joint_baseline import forward as legacy_forward
from joint_observation import _metadata_check
from loadData.dataset_registry import DATASETS, file_hash
from loadData.observation_sampling import BandMetadata
from m4_ordered_source import OrderedValidation
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import load_model_checkpoint
from model.coverage_config import CONDITIONS, coverage_signature
from model.research_config import digest
from none_full_transfer import initialize_none_full_target
from target_bn_calibration import CONTEXT, GRAPH_POLICY, observation_id, backbone_id
from target_initialization import METHODS, state_id

POLICY = "g4_coverage_target_initialization_v1"
CALIBRATION = "g4_coverage_target_population_bn_v1"
KEY = "coverage_target_binding"
CALIBRATION_KEY = "coverage_target_calibration"
CONTROLS = ("none", *CONDITIONS)
PREFIX = "backbone.coverage_modulation."


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def initialize(target, method, condition, *, metadata, model_seed, head_seed, source_spec=None, device="cpu"):
    """Copy the authenticated none initialization exactly; add only E1 state."""
    if (condition not in CONTROLS or method not in METHODS[:2]
            or type(model_seed) is not int or model_seed not in (64, 512, 2048)
            or type(head_seed) is not int or head_seed != model_seed + 1000
            or not isinstance(metadata, BandMetadata) or not all(metadata.valid)):
        raise ValueError("Explicit target coverage control and retained metadata required")
    base, record = initialize_none_full_target(target, method, model_seed=model_seed,
        head_seed=head_seed, source_spec=source_spec, device=device)
    original = base.state_dict()
    if condition == "none":
        model = base
    else:
        kwargs = {**base.export_config()["backbone_kwargs"], "coverage_mode": "affine",
                  "coverage_condition": condition}
        # New backbone/head initialization is discarded in favor of the exact
        # accepted base state. Fork RNG so enabling E1 cannot perturb training.
        with torch.random.fork_rng(devices=[]):
            model = MultiDatasetVCASAHGCN({target: DATASETS[target][2]}, **kwargs).to(device)
        current = model.state_dict()
        if set(current) - set(original) != {k for k in current if k.startswith(PREFIX)}:
            raise ValueError("Coverage may add only its own registered state")
        for name, value in original.items():
            if name not in current or (current[name].shape, current[name].dtype) != (value.shape, value.dtype):
                raise ValueError("Inherited target tensor differs")
            current[name] = value.detach().clone()
        model.load_state_dict(current, strict=True)
    extra = {n: p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX)}
    if sum(extra.values()) != (0 if condition == "none" else 194):
        raise ValueError("Fixed194 coverage capacity required")
    for name, value in original.items():
        if not torch.equal(value, model.state_dict()[name]):
            raise ValueError("Source/shared/fresh target state changed while adding coverage")
    binding = _plain({"policy": POLICY, "target": target, "method": method, "condition": condition,
        "model_seed": model_seed, "head_seed": head_seed, "metadata": asdict(metadata),
        "base_initial_state_sha256": state_id(base), "initial_state_sha256": state_id(model),
        "coverage_parameters": extra, "coverage_parameter_count": sum(extra.values()),
        "coverage_signature": coverage_signature(model.export_config()["backbone_kwargs"]),
        "operator_config": None if condition == "none" else model.backbone.coverage_modulation.export_config(),
        "geometry_config": None if condition == "none" else model.backbone.coverage_config.export_config(),
        "architecture_sha256": digest(model.export_config()), "base_initialization": record,
        "physical_calibration_authenticated": False, "full_target_scene_transductive": True,
        "optimizer_updates": 0, "query_labels_accessed": False, "training_executable": False})
    model.training_config = {"target_initialization": copy.deepcopy(record), KEY: copy.deepcopy(binding)}
    return model.eval(), binding


def verify_initialization_binding(model, expected):
    actual = getattr(model, "training_config", {}).get(KEY)
    if (not isinstance(expected, dict) or actual != expected or expected.get("policy") != POLICY
            or expected["condition"] not in CONTROLS or expected["method"] not in METHODS[:2]
            or expected["architecture_sha256"] != digest(model.export_config())
            or tuple(model.heads) != (expected["target"],)
            or getattr(model, "training_config", {}).get("target_initialization") != expected["base_initialization"]):
        raise ValueError("Target coverage architecture/initialization binding differs")
    actual_parameters = {n: p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX)}
    if actual_parameters != expected["coverage_parameters"]:
        raise ValueError("Target coverage parameter permission differs")
    module = model.backbone.coverage_modulation
    if (model.backbone.adapter.attention_mode != expected["coverage_signature"]["adapter_attention_mode"]
            or (None if module is None else module.export_config()) != expected["operator_config"]
            or (None if model.backbone.coverage_config is None else model.backbone.coverage_config.export_config()) != expected["geometry_config"]
            or (module is None) != (expected["condition"] == "none")):
        raise ValueError("Actual target coverage operator/geometry configuration differs")
    return expected


def _metadata(model, metadata):
    binding = verify_initialization_binding(model, model.training_config[KEY])
    if not isinstance(metadata, BandMetadata):
        raise ValueError("Bound coverage metadata required")
    full = BandMetadata(**binding["metadata"])
    try:
        positions = [full.band_ids.index(i) for i in metadata.band_ids]
    except ValueError as error:
        raise ValueError("View contains an unknown full-observation band") from error
    if positions != sorted(set(positions)) or full.subset(positions) != metadata:
        raise ValueError("View metadata quality/identity or ordered subset differs")
    return binding


def forward_observation(model, observation, metadata, *, diagnostics=False):
    """Only label-free actual observation fields; metadata cannot be upgraded."""
    observation_id(observation)
    binding = _metadata(model, metadata)
    item = {**observation, "bands": len(metadata.band_ids)}
    _metadata_check(item, metadata)
    if binding["condition"] == "none":
        if diagnostics:
            raise ValueError("Coverage-off keeps original forward/return contract")
        return legacy_forward(model, binding["target"], item)
    return model(binding["target"], observation["net_input"], observation["Q"], observation["A"],
        wavelengths=observation["wavelengths"], fwhm=observation["fwhm"],
        superpixel_scale=observation["superpixel_scale"], coverage_metadata=metadata,
        wavelength_unit=metadata.wavelength_unit, fwhm_unit=metadata.fwhm_unit or metadata.wavelength_unit,
        return_graph_info=diagnostics, return_diagnostics=diagnostics)


def calibrate(model, observation, metadata, *, expected_binding, source_state_sha256):
    """Adapt accepted population-BN math to explicitly bound E1 forward.

    Project-local adaptation of target_bn_calibration.calibrate_target. Its
    original coverage-off guard and all original files remain unchanged.
    """
    binding = verify_initialization_binding(model, expected_binding)
    _metadata(model, metadata)
    if metadata != BandMetadata(**binding["metadata"]):
        raise ValueError("Candidate calibration requires the full frozen observation")
    if (state_id(model) != source_state_sha256 or any(m.training for m in model.modules())
            or model.backbone.graph_mode != "static" or model.backbone.norm_type != "domain_bn"):
        raise ValueError("Fully eval pinned target population source required")
    identity = observation_id(observation)
    reference = next(model.parameters())
    if observation["net_input"].dtype != reference.dtype:
        raise ValueError("Population observation/model dtype differs")
    arithmetic = _arithmetic(model)
    candidate = copy.deepcopy(model)
    before = {n: t.detach().clone() for n, t in candidate.state_dict().items()}
    target = binding["target"]
    layers = {n: m for n, m in candidate.named_modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))}
    if (not layers or any(not n.endswith(".norms." + target) or not m.track_running_stats
            or m.running_mean is None or m.running_var is None for n, m in layers.items())):
        raise ValueError("Only declared target tracked population BN supported")
    seen, rows, changed, hooks = [], [], set(), []
    def make_hook(name):
        def hook(module, args):
            if name in seen:
                raise ValueError("Reused BN outside target population contract")
            x = args[0]
            if x.ndim not in (2, 3, 4) or not torch.isfinite(x).all():
                raise ValueError("Invalid target population")
            variance, mean = torch.var_mean(x, dim=(0, *range(2, x.ndim)), unbiased=False)
            if not torch.isfinite(variance).all() or not torch.isfinite(mean).all():
                raise ValueError("Nonfinite target moments")
            module.running_mean.copy_(mean); module.running_var.copy_(variance)
            seen.append(name); changed.update((name + ".running_mean", name + ".running_var"))
            rows.append({"layer": name, "population_units": "regions" if isinstance(module, torch.nn.BatchNorm1d) else "pixels",
                         "samples_per_channel": x.numel() // x.shape[1], "variance": "population_correction0"})
        return hook
    try:
        for name, bn in layers.items():
            hooks.append(bn.register_forward_pre_hook(make_hook(name)))
        moved = {k: v.to(reference.device) if torch.is_tensor(v) else v for k, v in observation.items()}
        with torch.no_grad(), (OrderedValidation() if reference.dtype == torch.float32 else nullcontext()):
            outputs = forward_observation(candidate, moved, metadata)
        if set(seen) != set(layers) or any(not torch.isfinite(v).all() for v in outputs[:3]):
            raise ValueError("Missing target BN execution or nonfinite output")
    finally:
        for hook in hooks:
            hook.remove()
    for name, value in candidate.state_dict().items():
        if name not in changed and not torch.equal(value, before[name]):
            raise RuntimeError("Population calibration changed protected state: " + name)
    if state_id(model) != source_state_sha256:
        raise RuntimeError("Original target model mutated")
    receipt = {"policy": CALIBRATION, "context": CONTEXT, "graph_policy": GRAPH_POLICY,
        "target": target, "binding_sha256": digest(binding), "metadata_sha256": metadata.fingerprint,
        "coverage_signature": coverage_signature(candidate.export_config()["backbone_kwargs"]),
        "source_state_sha256": source_state_sha256, "observation": identity, "layers": rows,
        "calibrated_buffers": sorted(changed), "parameters_affine_counters_unchanged": True,
        "state_sha256": state_id(candidate), "backbone_sha256": backbone_id(candidate),
        "arithmetic": arithmetic, "optimizer_updates": 0, "query_labels_accessed": False}
    candidate.training_config[CALIBRATION_KEY] = copy.deepcopy(receipt)
    return candidate.eval(), receipt


def verify_calibrated(model, observation, metadata, *, expected_binding, source_state_sha256):
    binding = verify_initialization_binding(model, expected_binding)
    _metadata(model, metadata)
    receipt = getattr(model, "training_config", {}).get(CALIBRATION_KEY, {})
    if (receipt.get("policy") != CALIBRATION or receipt.get("context") != CONTEXT
            or receipt.get("graph_policy") != GRAPH_POLICY or receipt.get("target") != binding["target"]
            or receipt.get("binding_sha256") != digest(binding)
            or receipt.get("metadata_sha256") != metadata.fingerprint
            or metadata != BandMetadata(**binding["metadata"])
            or receipt.get("source_state_sha256") != source_state_sha256
            or receipt.get("observation") != observation_id(observation)
            or receipt.get("state_sha256") != state_id(model) or receipt.get("backbone_sha256") != backbone_id(model)
            or receipt.get("coverage_signature") != coverage_signature(model.export_config()["backbone_kwargs"])
            or receipt.get("arithmetic") != _arithmetic(model)
            or any(m.training for m in model.modules())):
        raise ValueError("Calibrated coverage target state/observation/metadata binding differs")
    return receipt


def _arithmetic(model):
    parameter = next(model.parameters())
    if parameter.dtype == torch.float32:
        return "fp32_sorted_row_segment_sum_v1"
    if parameter.dtype == torch.float64 and parameter.device.type == "cpu":
        return "cpu_fp64_reference_only_no_checkpoint_dtype_claim"
    raise ValueError("Only FP32 population calibration or CPUFP64 reference supported")


def load_calibrated(path, weight_sha256, sidecar_sha256, observation, metadata, *, expected_binding,
                    source_state_sha256, device="cpu"):
    if file_hash(path) != weight_sha256 or file_hash(str(path) + ".json") != sidecar_sha256:
        raise ValueError("Calibrated target checkpoint pair differs")
    model = load_model_checkpoint(path, device=device)
    verify_calibrated(model, observation, metadata, expected_binding=expected_binding,
                      source_state_sha256=source_state_sha256)
    return model
