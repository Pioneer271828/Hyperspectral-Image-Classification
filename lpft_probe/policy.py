"""Independent two-stage Adam policy for a future registered G4 consumer.

No data loading, checkpoint I/O, calibration, scheduler or query access occurs.
Call start_phase only at a registered phase boundary. It creates fresh Adam
state, preserves model tensors, and freezes all stochastic/buffer-updating
layers in eval mode. A caller must not subsequently call model.train().
Interior-phase resume and production checkpoint integration are out of scope.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


HEAD_PREFIXES = ("heads.", "hgnn_heads.", "cnn_heads.")
BACKBONE_PREFIX = "backbone."
POLICY = "side_lpft_phase_policy_v1"


@dataclass(frozen=True)
class LPFTPlan:
    total_steps: int
    probe_steps: int
    head_lr: float
    backbone_lr: float
    weight_decay: float = 0.0

    def __post_init__(self):
        if (type(self.total_steps) is not int or type(self.probe_steps) is not int
                or not 0 < self.probe_steps < self.total_steps):
            raise ValueError("Two nonempty phases within one explicit total budget required")
        for name in ("head_lr", "backbone_lr", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Finite numeric optimizer options required")
        if not 0 < self.backbone_lr <= self.head_lr or self.weight_decay < 0:
            raise ValueError("Require 0 < backbone_lr <= head_lr and nonnegative decay")

    def phase(self, completed_steps):
        if type(completed_steps) is not int or not 0 <= completed_steps <= self.total_steps:
            raise ValueError("Completed update cursor outside the registered budget")
        if completed_steps == self.total_steps:
            return "complete"
        return "probe" if completed_steps < self.probe_steps else "finetune"

    def contract(self):
        return {"policy": POLICY, **asdict(self),
                "optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                "amsgrad": False, "transition_optimizer_state": "reset_all",
                "transition_model_state": "preserve_all_tensors",
                "forward_mode": "eval_all_layers_in_both_phases",
                "bn_running_statistics": "frozen_no_calibration_in_this_component",
                "bn_affine": "frozen_in_probe_trainable_in_finetune",
                "budget_includes_probe": True,
                "head_prefixes": list(HEAD_PREFIXES),
                "backbone_prefix": BACKBONE_PREFIX}


def _partition(model):
    # Reject tied aliases: name-based allowlists must have an unambiguous owner.
    parameters = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in parameters}) != len(parameters):
        raise ValueError("Aliased parameters require an explicit integration policy")
    heads, backbone = [], []
    for name, parameter in parameters:
        if not parameter.is_floating_point():
            raise ValueError("Only floating model parameters supported")
        if name.startswith(HEAD_PREFIXES):
            heads.append((name, parameter))
        elif name.startswith(BACKBONE_PREFIX):
            backbone.append((name, parameter))
        else:
            raise ValueError("Unknown parameter outside declared scopes: " + name)
    if not backbone or any(not any(n.startswith(prefix) for n, _ in heads)
                           for prefix in HEAD_PREFIXES):
        raise ValueError("Backbone and all three declared heads required")
    return heads, backbone


def phase_binding(model, plan, completed_steps):
    """Read-only binding of exact parameter names/shapes and next-update phase."""
    phase = plan.phase(completed_steps)
    heads, backbone = _partition(model)
    active = [] if phase == "complete" else heads + (backbone if phase == "finetune" else [])
    binding = {"contract": plan.contract(), "phase": phase,
               "parameter_schema": {n: {"shape": list(p.shape), "dtype": str(p.dtype)}
                                    for n, p in heads + backbone},
               "trainable_names": [n for n, _ in active],
               "trainable_elements": sum(p.numel() for _, p in active)}
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False)
    binding["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return binding


def start_phase(model, plan, completed_steps):
    """Create fresh Adam at update 0 or probe_steps, preserving tensor values.

    Explicitly reset head AND backbone optimizer history at the transition.
    The caller owns the authoritative update cursor and must discard the old
    optimizer. Calling within a phase or at completion is rejected before any
    mutation. This is a phase primitive, not a crash-safe training worker.
    """
    if type(completed_steps) is not int or completed_steps not in (0, plan.probe_steps):
        raise ValueError("Phase start requires an exact boundary cursor")
    binding = phase_binding(model, plan, completed_steps)
    heads, backbone = _partition(model)
    groups = [{"params": [p for _, p in heads], "lr": plan.head_lr,
               "lpft_scope": "heads"}]
    if binding["phase"] == "finetune":
        groups.append({"params": [p for _, p in backbone], "lr": plan.backbone_lr,
                       "lpft_scope": "backbone"})
    import torch
    optimizer = torch.optim.Adam(groups, weight_decay=plan.weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    active = set(binding["trainable_names"])
    for name, parameter in heads + backbone:
        parameter.grad = None
        parameter.requires_grad_(name in active)
    model.eval()
    return optimizer, binding
