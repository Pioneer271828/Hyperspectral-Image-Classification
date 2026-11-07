"""Opt-in single-device LP-FT update state; no dataset or file access.

The caller supplies a deterministic scalar-loss callback and authenticated input
hashes. Snapshots include Torch RNG only, not Python/NumPy/data-loader state.
Restore returns a new owned model; it never overwrites the supplied template.
External checkpoint hashes and atomic file publication belong to the consumer.
"""
import copy
import re

import torch

from .policy import _partition, phase_binding, start_phase


FORMAT = "side_lpft_runtime_v1"


def _identity(value):
    if (not isinstance(value, dict) or not value or
            any(not isinstance(k, str) or not k or not isinstance(v, str) or
                re.fullmatch(r"[0-9a-f]{64}", v) is None for k, v in value.items())):
        raise ValueError("Explicit input names and SHA256 values required")
    return dict(value)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _schema(model):
    result = {}
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Tensor-only model state required")
        result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    return result


def _device(model):
    devices = {v.device for v in model.state_dict().values()}
    if len(devices) != 1:
        raise ValueError("One device for all model parameters and buffers required")
    device = next(iter(devices))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or one CUDA device supported")
    return device


class LPFTRun:
    """Own a copied model, Adam state and the count of successful updates.

    Each update requires gradients for every active parameter. An exception
    poisons this instance; restart from the last externally saved snapshot.
    No automatic retry, extra update, scheduler or calibration is performed.
    """

    def __init__(self, model, plan, input_sha256):
        self.input_sha256 = _identity(input_sha256)
        self.plan = plan
        self.model = copy.deepcopy(model)
        self.device = _device(self.model)
        self.model_schema = _schema(self.model)
        self._buffers = _cpu(dict(self.model.named_buffers()))
        self.completed_steps = 0
        self._failed = False
        self.optimizer, _ = start_phase(self.model, self.plan, 0)

    def _binding(self):
        # At completion the retained optimizer belongs to the final phase.
        cursor = min(self.completed_steps, self.plan.total_steps - 1)
        return phase_binding(self.model, self.plan, cursor)

    def _phase_updates(self):
        return (self.completed_steps if self.completed_steps < self.plan.probe_steps
                else self.completed_steps - self.plan.probe_steps)

    def _validate_live(self):
        if self._failed:
            raise RuntimeError("Failed update: restore a saved snapshot into a new run")
        self.plan.phase(self.completed_steps)
        if _schema(self.model) != self.model_schema or _device(self.model) != self.device:
            raise ValueError("Model schema or device changed")
        if any(module.training for module in self.model.modules()):
            raise ValueError("LP-FT requires eval mode for every layer")
        active = set(self._binding()["trainable_names"])
        if any(p.requires_grad != (n in active) for n, p in self.model.named_parameters()):
            raise ValueError("Trainable parameter scope changed")
        if any(not torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError("Nonfinite model state")
        if any(not torch.equal(self._buffers[n], v.detach().cpu())
               for n, v in self.model.named_buffers()):
            raise ValueError("Frozen model buffer changed")

    def step(self, loss_function):
        self._validate_live()
        if self.completed_steps == self.plan.total_steps:
            raise ValueError("Registered update budget exhausted")
        try:
            self.optimizer.zero_grad(set_to_none=True)
            loss = loss_function(self.model)
            if (not isinstance(loss, torch.Tensor) or loss.ndim != 0 or
                    not loss.requires_grad or not torch.isfinite(loss).item()):
                raise ValueError("Finite differentiable scalar loss required")
            loss.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and (parameter.grad is None or
                                               not torch.isfinite(parameter.grad).all()):
                    raise ValueError("Missing/nonfinite gradient: " + name)
                if not parameter.requires_grad and parameter.grad is not None:
                    raise ValueError("Frozen parameter has a gradient: " + name)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.completed_steps += 1
            if self.completed_steps == self.plan.probe_steps:
                self.optimizer, _ = start_phase(self.model, self.plan, self.completed_steps)
            self._validate_live()
            return float(loss.detach())
        except Exception:
            self._failed = True
            raise

    def snapshot(self):
        self._validate_live()
        result = {"format": FORMAT, "contract": self.plan.contract(),
                  "input_sha256": self.input_sha256, "device": str(self.device),
                  "completed_steps": self.completed_steps, "binding": self._binding(),
                  "model_schema": self.model_schema, "model": self.model.state_dict(),
                  "optimizer": self.optimizer.state_dict(),
                  "torch_rng": torch.get_rng_state(),
                  "cuda_rng": (torch.cuda.get_rng_state(self.device)
                               if self.device.type == "cuda" else None)}
        result = _cpu(result)
        self._validate_optimizer(result["optimizer"])
        return result

    def _validate_optimizer(self, saved):
        boundary = 0 if self.completed_steps < self.plan.probe_steps else self.plan.probe_steps
        heads, backbone = _partition(self.model)
        groups = [{"params": [p for _, p in heads], "lr": self.plan.head_lr,
                   "lpft_scope": "heads"}]
        if boundary:
            groups.append({"params": [p for _, p in backbone], "lr": self.plan.backbone_lr,
                           "lpft_scope": "backbone"})
        reference = torch.optim.Adam(groups, weight_decay=self.plan.weight_decay,
                                     betas=(.9, .999), eps=1e-8, amsgrad=False)
        expected = reference.state_dict()
        if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
            raise ValueError("Malformed Adam state")
        if saved["param_groups"] != expected["param_groups"]:
            raise ValueError("Optimizer groups or options changed")
        ids = [i for g in expected["param_groups"] for i in g["params"]]
        parameters = [p for g in reference.param_groups for p in g["params"]]
        updates = self._phase_updates()
        if set(saved["state"]) != (set(ids) if updates else set()):
            raise ValueError("Optimizer state coverage does not match phase cursor")
        for index, parameter in zip(ids, parameters):
            if not updates:
                continue
            state = saved["state"][index]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Unexpected Adam fields")
            step = state["step"]
            if (not isinstance(step, torch.Tensor) or step.ndim != 0 or
                    not step.is_floating_point() or step.item() != updates):
                raise ValueError("Adam update count disagrees with phase cursor")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or
                        value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                    raise ValueError("Malformed Adam moment: " + key)
            if (state["exp_avg_sq"] < 0).any():
                raise ValueError("Negative Adam squared moment")

    @classmethod
    def restore(cls, template, plan, input_sha256, saved):
        """Validate fully before committing global Torch RNG; template unchanged.

        Only trusted, externally hash-verified snapshots may be supplied. Schema
        checks are consistency checks, not authentication of model values.
        Restore on the original device; cross-device equivalence is not claimed.
        """
        run = cls(template, plan, input_sha256)
        keys = {"format", "contract", "input_sha256", "device", "completed_steps",
                "binding", "model_schema", "model", "optimizer", "torch_rng", "cuda_rng"}
        if not isinstance(saved, dict) or set(saved) != keys:
            raise ValueError("Malformed runtime snapshot")
        if (saved["format"] != FORMAT or saved["contract"] != plan.contract() or
                saved["input_sha256"] != run.input_sha256 or saved["device"] != str(run.device) or
                saved["model_schema"] != run.model_schema):
            raise ValueError("Snapshot identity, plan, device or schema mismatch")
        plan.phase(saved["completed_steps"])
        run.completed_steps = saved["completed_steps"]
        if saved["binding"] != run._binding():
            raise ValueError("Snapshot parameter binding mismatch")
        model_state = saved["model"]
        if not isinstance(model_state, dict) or set(model_state) != set(run.model_schema):
            raise ValueError("Model tensor coverage mismatch")
        for name, spec in run.model_schema.items():
            tensor = model_state[name]
            if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != spec["shape"] or
                    str(tensor.dtype) != spec["dtype"] or not torch.isfinite(tensor).all()):
                raise ValueError("Invalid model tensor: " + name)
        run._validate_optimizer(saved["optimizer"])
        # Validate RNG bytes using independent generators, before global mutation.
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.set_state(saved["torch_rng"])
        if run.device.type == "cuda":
            generator = torch.Generator(device=run.device)
            generator.set_state(saved["cuda_rng"])
        elif saved["cuda_rng"] is not None:
            raise ValueError("Unexpected CUDA RNG for CPU snapshot")
        run.model.load_state_dict(model_state, strict=True)
        boundary = 0 if run.completed_steps < plan.probe_steps else plan.probe_steps
        run.optimizer, _ = start_phase(run.model, plan, boundary)
        run.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        run._validate_live()
        if run.device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], run.device)
        torch.set_rng_state(saved["torch_rng"])
        return run
