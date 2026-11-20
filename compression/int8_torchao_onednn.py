"""Optional frozen WP10 oneDNN backend used by the TorchAO x86 lowering."""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from compression.int8 import _finite_cpu
from compression.int8_model import _replace, validate_plan
from compression.int8_torchao_reference import TORCHAO_COMMIT, TORCH_VERSION, TorchAOAffine, frozen_parameters
from compression.precision_cost import _state_identity
from compression.precision_recovery_audit import load_package
from model.research_config import digest
from research_runtime import external_directory
from transfer import matched_route_admission as admission, registered_development as dev


POLICY = "wp10-static-int8-onednn-frozen-v1"


class OneDNNPointConv(nn.Module):
    """Keep the exact packed WP10 affine contract and invoke the CPU U8/S8 kernel."""

    def __init__(self, native):
        super().__init__()
        p = frozen_parameters(native)
        if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled:
            raise ValueError("oneDNN must be enabled for the optional backend")
        self.in_features = native.in_features
        self.input_scale = float(p["input_scale"].item())
        self.input_zero = int(p["input_zero"].item())
        self.output_scale = float(p["output_scale"].item())
        self.output_zero = int(p["output_zero"].item())
        self.weight_scales = p["weight_scales"].float()
        self.weight_zero = torch.zeros_like(self.weight_scales, dtype=torch.int64)
        self.bias = None if p["bias"] is None else p["bias"].detach().clone()
        weight = p["weight"][:, :, None, None].contiguous()
        self.packed = torch.ops.onednn.qconv_prepack(weight, self.weight_scales, self.input_scale,
                                                    self.input_zero, [1, 1], [0, 0], [1, 1], 1, None)
        if self.packed.layout != torch._mkldnn:
            raise ValueError("oneDNN kernel failed to prepack integer weights")
        self.eval()

    def codes(self, x):
        _finite_cpu(x)
        if self.training or x.requires_grad or x.ndim != 4 or x.shape[1] != self.in_features:
            raise ValueError("Inference-only CPU NCHW point convolution required")
        return self.integer_conv(self.channels_last_codes(self.quantize_input(x)))

    def quantize_input(self, x):
        scale, zero = self.input_scale, self.input_zero
        # Keep the legacy reduced-range quantizer, including float multiplication/tie behavior.
        return torch.quantize_per_tensor(x.clamp(-zero*scale, (127-zero)*scale), scale, zero, torch.quint8)

    @staticmethod
    def channels_last_codes(quantized):
        return quantized.int_repr().contiguous(memory_format=torch.channels_last)

    def integer_conv(self, codes):
        return torch.ops.onednn.qconv2d_pointwise(codes, self.input_scale, self.input_zero, self.packed, self.weight_scales,
                    self.weight_zero, self.bias, [1, 1], [0, 0], [1, 1], 1,
                    self.output_scale, self.output_zero, torch.uint8, "none", [], "")

    def dequantize_output(self, codes):
        return (codes.to(torch.float32) - self.output_zero) * self.output_scale

    def forward(self, x):
        return self.dequantize_output(self.codes(x))


def convert_backend(native, oracle, *, enabled=False):
    if type(enabled) is not bool:
        raise ValueError("Boolean backend selection required")
    if not enabled:
        return native
    if type(oracle) is not TorchAOAffine or native.training:
        raise ValueError("Attested backend and original evaluation model required")
    plan = validate_plan(native)
    if plan["phase"] != "integer":
        raise ValueError("The backend consumes only a frozen native integer checkpoint")
    paths = plan["contract"]["paths"]
    # Build all replacements first. Failure leaves the original model unmodified.
    replacements = {path: OneDNNPointConv(native.get_submodule(path)) for path in paths}
    result = deepcopy(native)
    for path, operator in replacements.items():
        _replace(result, path, operator)
    return result.eval()


def register_backend(package, path, oracle, *, native=None):
    """Store only an authenticated native package reference and backend provenance."""
    native = load_package(package) if native is None else native
    plan = validate_plan(native)
    if (type(oracle) is not TorchAOAffine or plan["phase"] != "integer"
            or _state_identity(native) != package["state"]):
        raise ValueError("Attested backend and frozen integer plan required")
    path = external_directory(path)
    return dev._json_new(path, dict(policy=POLICY, package=package,
        calibration=digest(plan), torch=TORCH_VERSION, torchao=TORCHAO_COMMIT,
        quantized_engine=torch.backends.quantized.engine))


def load_backend(pin, oracle, *, enabled=False, native=None):
    value = admission.read(pin)
    if (value["policy"] != POLICY or value["torch"] != TORCH_VERSION or value["torchao"] != TORCHAO_COMMIT
            or value["quantized_engine"] != torch.backends.quantized.engine
            or set(value) != {"policy", "package", "calibration", "torch", "torchao", "quantized_engine"}):
        raise ValueError("Invalid optional backend sidecar")
    if type(enabled) is not bool or (enabled and type(oracle) is not TorchAOAffine):
        raise ValueError("Explicit attested backend or native execution required")
    if native is None:
        native = load_package(value["package"])
    else:
        if _state_identity(native) != value["package"]["state"]:
            raise ValueError("In-memory native state differs from the backend package")
    if digest(validate_plan(native)) != value["calibration"]:
        raise ValueError("Frozen calibration differs from registered backend")
    return convert_backend(native, oracle, enabled=enabled)
