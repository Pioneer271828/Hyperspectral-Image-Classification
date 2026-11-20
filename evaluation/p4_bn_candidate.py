"""Candidate train-context population calibration on a private DomainBN clone."""
from copy import deepcopy

import torch

from evaluation.p4_bn_factors import GROUPS, Moments, fingerprint, groups, require
from research_runtime import tensor_hash
from transfer.g3_training import state_hash


SCHEMA = "wp05_p4_opt_in_devtrain_domainbn_candidate_v1"


def calibrate_private(model, contexts, forward, *, role, source_pins):
    """Recompute front-end moments in graph execution order, with no optimizer."""
    require(role == "devtrain" and len(contexts) == len(source_pins) > 0
            and all(isinstance(x, str) and len(x) == 64 for x in source_pins)
            and len(set(source_pins)) == len(source_pins), "Only individually pinned devtrain contexts may fit buffers")
    current = state_hash(model)
    private = deepcopy(model).eval()
    frozen_params = fingerprint(model)
    layers = dict(private.named_modules())
    organized = [name for name, group in groups(private).items() if group == "front"]
    require(len(organized) == 2 and all(name.startswith(GROUPS["front"]) for name in organized),
            "Only the two existing front-end DomainBN layers are eligible")
    moments = {}
    with torch.no_grad():
        for name in organized:
            layer = layers[name]
            stats = Moments()
            hook = layer.register_forward_pre_hook(stats.observe)
            try:
                for i, obs in enumerate(contexts):
                    stats.tile = i
                    forward(private, obs)
            finally:
                hook.remove()
            result = stats.compare(layer)
            layer.running_mean.copy_(torch.tensor(result["context_mean"], dtype=layer.running_mean.dtype))
            layer.running_var.copy_(torch.tensor(result["context_var"], dtype=layer.running_var.dtype))
            moments[name] = {"count": result["observations"], "mean": tensor_hash(layer.running_mean),
                             "variance": tensor_hash(layer.running_var)}
    require(fingerprint(private) == frozen_params and state_hash(model) == current,
            "Calibration changed original or trainable parameters")
    replacements = {f"{name}.{key}": getattr(layers[name], key).clone()
                    for name in organized for key in ("running_mean", "running_var")}
    proof = {"schema": SCHEMA, "role": role, "sources": list(source_pins), "base": current,
             "calibrated": state_hash(private), "statistics_population_passes": len(organized),
             "statistics_context_forwards": len(organized)*len(contexts),
             "statistics_buffer_writes": len(replacements), "optimizer_updates": 0,
             "moments": moments, "replacements": replacements}
    return private, proof


def restore_private(model, proof, source_pins):
    require(proof["schema"] == SCHEMA and proof["role"] == "devtrain" and proof["base"] == state_hash(model)
            and proof["sources"] == list(source_pins) and proof["optimizer_updates"] == 0
            and proof["statistics_population_passes"] == 2 and proof["statistics_context_forwards"] == len(source_pins)*2
            and proof["statistics_buffer_writes"] == 4, "Calibrated checkpoint origin or context differs")
    authorized = {f"{name}.{key}" for name, group in groups(model).items() if group == "front"
                  for key in ("running_mean", "running_var")}
    require(set(proof["replacements"]) == authorized, "Wrong restored calibration buffers")
    clone = deepcopy(model).eval()
    with torch.no_grad():
        buffers = dict(clone.named_buffers())
        for name, value in proof["replacements"].items():
            require(value.shape == buffers[name].shape and value.dtype == buffers[name].dtype and torch.isfinite(value).all()
                    and (not name.endswith("running_var") or (value >= 0).all()), "Invalid restored moment tensor")
            buffers[name].copy_(value)
    require(state_hash(clone) == proof["calibrated"], "Calibration fingerprint differs after restore")
    return clone, {"optimizer_updates": 0, "new_statistics_fitting_passes": 0}
