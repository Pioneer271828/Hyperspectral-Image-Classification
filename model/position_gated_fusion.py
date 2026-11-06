"""Default-off scalar position gate over existing CNN/region features.

Independent project implementation of the sigmoid convex-mixture idea discussed in
Arevalo et al., arXiv:1702.01992v1, section 3.1. This is not their full GMU:
there are no additional modality projections/tanh and the gate is one scalar per
position, not one per channel. See the fixed-author-code/license review manifest.
The original model, fixed helper, heads, loss and checkpoint loader are unchanged.
"""
import copy
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from .branch_fusion import fixed_fusion

POLICY = "m8_position_scalar_fusion_kernel_v1"
VERSION = 1
DTYPES = (torch.float32, torch.float64)


def _options(channels, enabled, alph):
    if type(channels) is not int or channels <= 0:
        raise ValueError("channels must be a positive integer")
    if type(enabled) is not bool:
        raise ValueError("enabled must be an explicit bool")
    if isinstance(alph, bool) or not isinstance(alph, Real) or not math.isfinite(alph):
        raise ValueError("alph must be finite")
    if not 0 <= alph <= 1 or (enabled and not 0 < alph < 1):
        raise ValueError("fixed alph in [0,1]; enabled initial alph strictly in (0,1)")
    return channels, enabled, float(alph)


def _config(value):
    keys = {"policy", "version", "channels", "enabled", "alph"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Exact position-gate config required")
    if value["policy"] != POLICY or type(value["version"]) is not int or value["version"] != VERSION:
        raise ValueError("Unsupported position-gate policy/version")
    return _options(value["channels"], value["enabled"], value["alph"])


class PositionGatedFusion(nn.Module):
    """Fuse [N,C] features; opt-in trainable scalar gate has 2*C+1 parameters.

    Finite FP32/FP64 only. Gate values are mixture weights, not calibrated
    confidence or physical reliability. No cross-position statistics or RNG.
    The disabled path calls the original fixed_fusion for exact valid-input
    arithmetic. Both modes carry a required version marker in their own state.
    """

    def __init__(self, channels=64, *, enabled=False, alph=0.8, device=None, dtype=None):
        super().__init__()
        self.channels, self.enabled, self.alph = _options(channels, enabled, alph)
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in DTYPES:
            raise ValueError("Only FP32/FP64 are supported")
        self.register_buffer("protocol_version", torch.tensor(VERSION, device=device, dtype=torch.int64))
        if enabled:
            # Direct deterministic allocation: do not consume/replay global RNG.
            self.gate_weight = nn.Parameter(torch.zeros(1, 2 * channels, device=device, dtype=dtype))
            self.gate_bias = nn.Parameter(torch.full((1,), math.log(alph / (1 - alph)), device=device, dtype=dtype))
        else:
            self.register_parameter("gate_weight", None)
            self.register_parameter("gate_bias", None)

    def export_config(self):
        channels, enabled, alph = _options(self.channels, self.enabled, self.alph)
        return {"policy": POLICY, "version": VERSION, "channels": channels, "enabled": enabled, "alph": alph}

    @classmethod
    def from_config(cls, config, **kwargs):
        channels, enabled, alph = _config(config)
        return cls(channels, enabled=enabled, alph=alph, **kwargs)

    def get_extra_state(self):
        return self.export_config()

    def set_extra_state(self, state):
        _config(state)
        if state != self.export_config():
            raise ValueError("Position-gate config/state mismatch")

    def _validate_state(self, state):
        if not isinstance(state, dict):
            raise ValueError("State dictionary required")
        expected = {"protocol_version", "_extra_state"}
        if self.enabled:
            expected |= {"gate_weight", "gate_bias"}
        if set(state) != expected:
            raise ValueError("Exact position-gate state keys required")
        self.set_extra_state(state["_extra_state"])
        marker = state["protocol_version"]
        if not torch.is_tensor(marker) or marker.shape != torch.Size([]) or marker.dtype != torch.int64 or marker.item() != VERSION:
            raise ValueError("Required position-gate version marker")
        if self.enabled:
            for key, shape in (("gate_weight", (1, 2 * self.channels)), ("gate_bias", (1,))):
                value = state[key]
                if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype not in DTYPES:
                    raise ValueError("Invalid gate tensor schema")
                if value.dtype != self.gate_weight.dtype or not torch.isfinite(value).all():
                    raise ValueError("Gate state dtype/finite values must match destination")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True or assign is not False:
            raise ValueError("Strict copying restore required")
        # All keys, metadata, version and values checked before any parameter copy.
        self._validate_state(state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=False)

    def export_payload(self):
        value = {"config": self.export_config(), "state_dict": copy.deepcopy(self.state_dict())}
        self._validate_state(value["state_dict"])
        return value

    @classmethod
    def from_payload(cls, value, *, device="cpu", dtype=None):
        if not isinstance(value, dict) or set(value) != {"config", "state_dict"}:
            raise ValueError("Paired config and state required")
        channels, enabled, alph = _config(value["config"])
        state = value["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("State dictionary required")
        weight = state.get("gate_weight")
        source_dtype = weight.dtype if enabled and torch.is_tensor(weight) else torch.float32
        model = cls(channels, enabled=enabled, alph=alph, device=device, dtype=source_dtype)
        model.load_state_dict(state)
        if dtype is not None:
            if dtype not in DTYPES:
                raise ValueError("Only explicit FP32/FP64 conversion supported")
            model.to(dtype=dtype)
        return model

    def forward(self, cnn_features, graph_features, *, return_gate=False):
        self.export_config()
        if type(return_gate) is not bool:
            raise ValueError("return_gate must be bool")
        if self.protocol_version.dtype != torch.int64 or self.protocol_version.shape != torch.Size([]) or self.protocol_version.item() != VERSION:
            raise ValueError("Unsupported position-gate state version")
        if not all(torch.is_tensor(v) for v in (cnn_features, graph_features)):
            raise ValueError("Tensor branch features required")
        if cnn_features.ndim != 2 or cnn_features.shape != graph_features.shape or cnn_features.shape[1] != self.channels:
            raise ValueError("Fusion expects matching [positions,channels] features")
        if cnn_features.dtype not in DTYPES or graph_features.dtype != cnn_features.dtype or graph_features.device != cnn_features.device:
            raise ValueError("Matching FP32/FP64 dtype and device required")
        if not all(torch.isfinite(v).all() for v in (cnn_features, graph_features)):
            raise ValueError("Finite branch features required")
        if self.enabled:
            if tuple(self.gate_weight.shape) != (1, 2 * self.channels) or tuple(self.gate_bias.shape) != (1,):
                raise ValueError("Gate parameter schema changed")
            if any(p.dtype != cnn_features.dtype or p.device != cnn_features.device or not torch.isfinite(p).all()
                   for p in (self.gate_weight, self.gate_bias)):
                raise ValueError("Finite gate parameters must match feature dtype/device")
            scores = F.linear(torch.cat((cnn_features, graph_features), dim=1), self.gate_weight, self.gate_bias)
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite gate scores")
            gate = torch.sigmoid(scores)
            fused = gate * cnn_features + (1 - gate) * graph_features
        else:
            fused = fixed_fusion(cnn_features, graph_features, self.alph)
            gate = cnn_features.new_full((cnn_features.shape[0], 1), self.alph) if return_gate else None
        if not torch.isfinite(fused).all():
            raise ValueError("Nonfinite fused features")
        return (fused, gate) if return_gate else fused
