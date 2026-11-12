"""Compose an explicit low-rank convolution with independently admitted factor precision."""

from copy import deepcopy

import torch
from torch import nn

from compression.int8 import CalibratedMatrix, IntegerMatrix
from compression.int8_model import _ObserveTraining, _replace, calibrate_model, calibration_contract
from compression.low_rank import LowRankMatrix
from compression.low_rank_recovery import TARGET
from inference.light_cnn import validate_source
from loadData.spatial_tiles import tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.g3_training import _forward, state_hash
from transfer.observation_policy import scene_identity


ORDER = ("right", "left")
LAYOUTS = {"both": ("right", "left"), "right": ("right",), "left": ("left",)}
POLICY = "wp10_low_rank_factor_precision_v1"


class FactorOperators(nn.Module):
    """Apply the right factor, then the left factor with original bias."""
    def __init__(self, right, left):
        super().__init__()
        self.right = right
        self.left = left

    def forward(self, input):
        return self.left(self.right(input))


def _factorized(model):
    layer = model.get_submodule(TARGET)
    plan = getattr(model, "_low_rank_plan", None)
    if (type(model) is not MultiDatasetVCASAHGCN or type(layer) is not LowRankMatrix
            or layer.kind != "conv1x1" or not isinstance(plan, dict) or plan["ranks"] != {TARGET: layer.rank}
            or getattr(model, "_factor_precision", None) is not None or getattr(model, "_int8_plan", None) is not None
            or model.backbone.adapter.cache_attention or any(m.training for m in model.modules())
            or any(p.device.type != "cpu" or p.dtype != torch.float32 or not torch.isfinite(p).all()
                   for p in layer.parameters())):
        raise ValueError("Exact frozen FP32 low-rank source required")
    return layer


def dependencies(model):
    layer = _factorized(model)
    return {"right": dict(source=f"{TARGET}.right", operator=f"{TARGET}.right",
                          shape=[layer.rank, layer.in_features], bias=None, upstream="original_layer_input"),
            "left": dict(source=f"{TARGET}.left", operator=f"{TARGET}.left",
                         shape=[layer.out_features, layer.rank],
                         bias=None if layer.bias is None else f"{TARGET}.bias", upstream=f"{TARGET}.right")}


def contract(model, scene, record, layout, tile_ids, *, qat=False):
    if layout not in LAYOUTS or type(qat) is not bool or torch.backends.quantized.engine not in ("x86", "fbgemm"):
        raise ValueError("Explicit factor layout, QAT choice and CPU engine required")
    identity = scene_identity(scene, record)
    parts = dependencies(model)
    if (not isinstance(tile_ids, (list, tuple)) or not tile_ids
            or any(type(i) is not int or not 0 <= i < len(record["roles"]) for i in tile_ids)
            or len(set(tile_ids)) != len(tile_ids)
            or any(record["roles"][i]["role"] != "devtrain" for i in tile_ids)):
        raise ValueError("Factor observers require distinct registered devtrain blocks")
    inputs = [dict(tile_id=i, sha256=tensor_hash(scene.cube[tile_slice(scene.plan["tiles"][i])])) for i in tile_ids]
    return dict(policy=POLICY, sequence=list(ORDER), active=list(LAYOUTS[layout]), layout=layout, qat=qat,
                source_state=state_hash(model), factor_plan=deepcopy(model._low_rank_plan),
                architecture=model.export_config(), dependencies=parts, identity=identity, inputs=inputs,
                engine=torch.backends.quantized.engine, calibration_role="devtrain", optimizer_updates=0,
                left_observes="simulated_quantized_right" if qat and "right" in LAYOUTS[layout] else "floating_right")


def _conv(weight, bias=None):
    output, inputs = weight.shape
    with torch.random.fork_rng(devices=[]):
        module = nn.Conv2d(inputs, output, 1, bias=bias is not None)
    with torch.no_grad():
        module.weight.copy_(weight[:, :, None, None])
        if bias is not None:
            module.bias.copy_(bias)
    return module.eval()


def template(model, spec, phase):
    layer = _factorized(model)
    if (spec["source_state"] != state_hash(model) or spec["dependencies"] != dependencies(model)
            or spec["factor_plan"] != model._low_rank_plan or spec["architecture"] != model.export_config()
            or spec["sequence"] != list(ORDER) or spec["active"] != list(LAYOUTS[spec["layout"]])
            or spec["engine"] != torch.backends.quantized.engine):
        raise ValueError("Composition is bound to the original factorized source and exact dependency graph")
    pair = FactorOperators(_conv(layer.right), _conv(layer.left, layer.bias))
    for name in spec["active"]:
        calibration = CalibratedMatrix(getattr(pair, name), qat=spec["qat"])
        if phase == "integer":
            # Shape-only packing as in the existing INT8 checkpoint loader.
            calibration.weight_quant(calibration.layer.weight)
            calibration.batches_seen.fill_(1)
            calibration.freeze()
            calibration = calibration.convert()
        setattr(pair, name, calibration)
    result = deepcopy(model).eval()
    _replace(result, TARGET, pair.eval())
    return result


def observer_state(model, spec):
    pair = model.get_submodule(TARGET)
    return {name: {k: tensor_hash(v) for k, v in getattr(pair, name).state_dict().items()
                   if not k.startswith("layer.")} for name in spec["active"]}


def validate(model):
    plan = getattr(model, "_factor_precision", None)
    if not isinstance(plan, dict) or set(plan) != {"contract", "contract_hash", "phase", "observers", "observer_hash"}:
        raise ValueError("Explicit factor precision plan required")
    spec, phase = plan["contract"], plan["phase"]
    pair = model.get_submodule(TARGET)
    if (spec["policy"] != POLICY or digest(spec) != plan["contract_hash"] or spec["sequence"] != list(ORDER)
            or spec["engine"] != torch.backends.quantized.engine or spec["architecture"] != model.export_config()
            or spec["factor_plan"] != model._low_rank_plan or type(pair) is not FactorOperators
            or spec["active"] != list(LAYOUTS[spec["layout"]]) or phase not in ("ptq", "qat", "integer")
            or (phase != "integer" and (phase == "qat") != spec["qat"])
            or digest(plan["observers"]) != plan["observer_hash"] or set(plan["observers"]) != set(spec["active"])):
        raise ValueError("Factor precision plan or frozen observer identity differs")
    for name in ORDER:
        layer = getattr(pair, name)
        active = name in spec["active"]
        expected = nn.Conv2d if not active else IntegerMatrix if phase == "integer" else CalibratedMatrix
        if type(layer) is not expected:
            raise ValueError("Factor placement differs from the calibration dependency map")
        convolution = layer.kernel if expected is IntegerMatrix else layer.layer if active else layer
        out_width, in_width = spec["dependencies"][name]["shape"]
        bias = convolution.bias() if expected is IntegerMatrix else convolution.bias
        if ((convolution.out_channels, convolution.in_channels) != (out_width, in_width)
                or (bias is None) != (spec["dependencies"][name]["bias"] is None)):
            raise ValueError("Factor tensor shape or second-operator bias differs")
        if expected is IntegerMatrix:
            if (layer.kernel.weight().dtype != torch.qint8 or layer.input_scale.numel() != 1
                    or not torch.isfinite(layer.input_scale).all() or not float(layer.input_scale.item()) > 0 or not 0 <= int(layer.input_zero_point.item()) <= 127
                    or not 0 < layer.kernel.scale < float("inf") or not 0 <= layer.kernel.zero_point <= 255):
                raise ValueError("Invalid factor INT8 kernel or activation range")
        elif active and (layer.qat != spec["qat"] or not bool(layer.ranges_frozen)
                         or int(layer.batches_seen) != len(spec["inputs"])
                         or any(bool(q.observer_enabled[0]) or not bool(q.fake_quant_enabled[0])
                                for q in layer.quantizers())):
            raise ValueError("Factor observer is not frozen on all registered blocks")
    if phase != "integer" and observer_state(model, spec) != plan["observers"]:
        raise ValueError("Frozen per-factor observer changed")
    return deepcopy(plan)


def calibrate(model, scene, record, *, registration, expected_hash):
    if digest(registration) != expected_hash:
        raise ValueError("Factor calibration registration hash differs")
    ids = [row["tile_id"] for row in registration["inputs"]]
    spec = contract(model, scene, record, registration["layout"], ids, qat=registration["qat"])
    if spec != registration:
        raise ValueError("Frozen factor calibration role, image, source or dependency differs")
    result = template(model, spec, "qat" if spec["qat"] else "ptq")
    pair = result.get_submodule(TARGET)
    layers = {name: getattr(pair, name) for name in spec["active"]}
    for name, layer in layers.items():
        setattr(pair, name, _ObserveTraining(layer))
    with torch.no_grad():
        for tile_id in ids:
            _forward(result, scene, scene.plan["tiles"][tile_id])
    for name, layer in layers.items():
        setattr(pair, name, layer.freeze())
    observations = observer_state(result, spec)
    result._factor_precision = dict(contract=spec, contract_hash=expected_hash, phase="qat" if spec["qat"] else "ptq",
                                    observers=observations, observer_hash=digest(observations))
    result.eval()
    validate(result)
    if spec != contract(model, scene, record, spec["layout"], ids, qat=spec["qat"]):
        raise ValueError("Calibration changed the authorized input or original factors")
    return result


def deploy(model):
    plan = validate(model)
    if plan["phase"] == "integer":
        raise ValueError("The registered factors are already packed")
    result = deepcopy(model).eval()
    pair = result.get_submodule(TARGET)
    for name in plan["contract"]["active"]:
        setattr(pair, name, getattr(pair, name).convert())
    plan["phase"] = "integer"
    result._factor_precision = plan
    validate(result)
    return result.requires_grad_(False).eval()


def with_precision(model, scene, record, *, layout=None, tile_ids=(), qat=False, dense_paths=()):
    """Dispatch to the previous INT8 path or to an explicitly chosen factor layout."""
    if layout is not None and dense_paths:
        raise ValueError("Choose original INT8 or decomposition followed by factor INT8")
    if layout is not None:
        spec = contract(model, scene, record, layout, tile_ids, qat=qat)
        return calibrate(model, scene, record, registration=spec, expected_hash=digest(spec))
    if dense_paths:
        spec = calibration_contract(model, scene, record, dense_paths, tile_ids, qat=qat)
        return calibrate_model(model, scene, record, paths=dense_paths, contract=spec, expected_hash=digest(spec))
    if tile_ids or qat:
        raise ValueError("Disabled precision has no observer configuration")
    return deepcopy(model)


def fallback(dense, factors, candidates, accepted, *, low_rank_passed, expected_dense_state, expected_factor_state):
    """Each surviving subset must have its own guards under that exact upstream arithmetic."""
    validate_source(dense)
    _factorized(factors)
    if (state_hash(dense) != expected_dense_state or state_hash(factors) != expected_factor_state
            or dense.export_config() != factors.export_config()):
        raise ValueError("Registered dense or factor fallback source differs")
    if (type(low_rank_passed) is not bool or not set(accepted) <= set(candidates) or not set(candidates) <= set(LAYOUTS)
            or any(type(value) is not bool for value in accepted.values())):
        raise ValueError("Explicit registered candidate guards required")
    calibration = None
    for name, model in candidates.items():
        plan = validate(model)
        spec = plan["contract"]
        if (plan["phase"] != "integer" or spec["layout"] != name or spec["source_state"] != expected_factor_state):
            raise ValueError("Fallback candidate was calibrated under a different upstream layout")
        context = (spec["identity"], spec["inputs"], spec["qat"])
        if calibration is not None and calibration != context:
            raise ValueError("Fallback layouts must use the same calibration scope")
        calibration = context
    if not low_rank_passed:
        return "dense", deepcopy(dense)
    for name in ("both", "right", "left"):
        if accepted.get(name, False) and (name != "both" or all(accepted.get(part, False) for part in ORDER)):
            return name, deepcopy(candidates[name])
    return "float_factors", deepcopy(factors)
