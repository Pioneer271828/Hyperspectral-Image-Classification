"""Read-only P4 DomainBN factorial and released-context moments."""
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from evaluation.classification_metrics import classification_metrics
from research_runtime import tensor_hash
from transfer.g3_training import _forward, state_hash


HEADS = ("fused", "graph", "cnn")
GROUPS = {"front": "backbone.CNN_denoise.", "cnn": "backbone.CNN_Branch.", "graph": "backbone.HGNN_Branch."}
ARMS = {"original": (), "front_reset": ("front",), "graph_reset": ("graph",),
        "cnn_reset": ("cnn",), "all_reset": tuple(GROUPS)}
BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def groups(model):
    result = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        matched = [key for key, prefix in GROUPS.items() if name.startswith(prefix)]
        if len(matched) != 1 or layer.running_mean is None or ".norms." not in name:
            raise ValueError("Unexpected normalization outside the registered DomainBN groups")
        result[name] = matched[0]
    if set(result.values()) != set(GROUPS):
        raise ValueError("Required P4 DomainBN groups are missing")
    return result


def fingerprint(model):
    return {name: tensor_hash(param) for name, param in model.named_parameters()}


@contextmanager
def buffer_reset(model, initial, selected_groups):
    layers = dict(model.named_modules())
    mapping = groups(model)
    before = {buffer: model.state_dict()[buffer].clone() for name, group in mapping.items()
              if group in selected_groups for key in BUFFERS for buffer in (f"{name}.{key}",)}
    with torch.no_grad():
        try:
            for name, value in before.items():
                layer, field = name.rsplit(".", 1)
                require(initial[name].shape == value.shape, "Initial BN buffer shape differs")
                getattr(layers[layer], field).copy_(initial[name])
            yield tuple(before)
        finally:
            for name, value in before.items():
                layer, field = name.rsplit(".", 1)
                getattr(layers[layer], field).copy_(value)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


class Moments:
    def __init__(self):
        self.count = 0
        self.total = None
        self.square = None
        self.by_tile = []
        self.tile = None

    def observe(self, layer, args):
        x = args[0].detach().movedim(1, -1)
        values = x.reshape(-1, x.shape[-1]).to(torch.float64)
        total = values.sum(0)
        square = values.square().sum(0)
        count = len(values)
        self.total = total if self.total is None else self.total+total
        self.square = square if self.square is None else self.square+square
        self.count += count
        self.by_tile.append({"tile": self.tile, "count": count,
                             "mean": (total/count).tolist(),
                             "population_var": (square/count-(total/count).square()).clamp_min(0).tolist()})

    def compare(self, layer):
        require(self.count > 0, "Normalization layer was not exercised by authorized training context")
        mean = self.total/self.count
        var = (self.square/self.count-mean.square()).clamp_min(0)
        saved = layer.running_mean.double(), layer.running_var.double()
        return {"observations": self.count, "context_mean": mean.tolist(), "context_var": var.tolist(),
                "buffer_mean": saved[0].tolist(), "buffer_var": saved[1].tolist(),
                "mean_rms_gap": float((mean-saved[0]).square().mean().sqrt()),
                "variance_rms_gap": float((var-saved[1]).square().mean().sqrt()),
                "buffer_batches": int(layer.num_batches_tracked), "per_block": self.by_tile}


def score(values, logits, truth):
    probs = values.double().numpy()
    raw = logits.double().numpy()
    labels = np.asarray(truth)
    pred = probs.argmax(1)+1
    true_p = probs[np.arange(len(pred)), labels-1]
    centered = raw-raw.mean(1, keepdims=True)
    per_class = [{"class_id": i, "support": int((labels == i).sum()),
                  "true_probability_mean": float(true_p[labels == i].mean()) if np.any(labels == i) else None,
                  "nll": float(-np.log(np.maximum(true_p[labels == i], 1e-8)).mean()) if np.any(labels == i) else None}
                 for i in range(1, probs.shape[1]+1)]
    return {"metrics": classification_metrics(labels, pred, range(1, probs.shape[1]+1)),
            "nll": float(-np.log(np.maximum(true_p, 1e-8)).mean()),
            "true_probability_mean": float(true_p.mean()), "per_class_probabilities": per_class,
            "entropy_mean": float(-(probs*np.log(np.maximum(probs, 1e-30))).sum(1).mean()),
            "logit_abs_max": float(np.abs(raw).max()), "centered_logit_rms": float(np.sqrt(np.mean(centered**2))),
            "probabilities": probs.tolist(), "predictions": pred.tolist(), "logits": raw.tolist()}


@torch.no_grad()
def observe(model, scene, role, views, truth, *, context=False):
    if model.training and role != "train":
        raise ValueError("Devval cannot fit or use batch-derived normalization")
    module_map = dict(model.named_modules())
    moments = {name: Moments() for name in groups(model)} if context else {}
    if context and (role != "train" or model.training):
        raise ValueError("Training-context moments are observed from fixed eval forwards only")
    norms = [module_map[name].register_forward_pre_hook(moment.observe) for name, moment in moments.items()]
    captured = {}
    features = {}
    by_head = dict(zip(HEADS, (model.heads, model.hgnn_heads, model.cnn_heads), strict=True))

    def tap(head):
        def hook(_, args, logits):
            captured[head] = logits.detach()
            features[head] = args[0].detach()
        return hook

    head_hooks = [family[scene.dataset].register_forward_hook(tap(head)) for head, family in by_head.items()]
    rows = defaultdict(list)
    raw = defaultdict(list)
    intermediate = []
    try:
        for tile, indices, view in views:
            captured.clear()
            for moment in moments.values():
                moment.tile = tile["id"]
            out = _forward(model, scene, tile, observation_view=view)
            require(set(captured) == set(HEADS), "Missing classifier logits")
            intermediate.append({"tile": tile["id"], "heads": {
                head: {"input_rms": float(x.square().mean().sqrt()), "input_sha256": tensor_hash(x),
                       "logits_sha256": tensor_hash(captured[head])} for head, x in features.items()}})
            for i, head in enumerate(HEADS):
                rows[head].append(out[i][indices].cpu())
                raw[head].append(captured[head][indices].cpu())
    finally:
        for hook in (*head_hooks, *norms):
            hook.remove()
    return {"heads": {head: score(torch.cat(v), torch.cat(raw[head]), truth) for head, v in rows.items()},
            "forwards": len(views), "target_labels": np.asarray(truth).tolist(), "intermediate": intermediate,
            "moments": {name: {**mom.compare(module_map[name]), "group": groups(model)[name]}
                        for name, mom in moments.items()}}


def factor_state(model, initial, scene, data):
    params, before = fingerprint(model), state_hash(model)
    rows = {}
    for name, affected in ARMS.items():
        with buffer_reset(model, initial, affected) as touched:
            observations = {role: observe(model, scene, role, views, labels, context=(name == "original" and role == "train"))
                            for role, (views, labels) in data.items()}
            rows[name] = {"touched_buffers": list(touched), "probability_metrics": observations}
        require(fingerprint(model) == params and state_hash(model) == before, "A factor changed the original state")
    # Native train mode yields batch statistics plus graph dropout. The disposable
    # model observes devtrain only; BN updates are counted and retained separately.
    private = deepcopy(model)
    saved = dict(private.named_buffers())
    counts = {name: int(value) for name, value in saved.items() if name.endswith("num_batches_tracked")}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41377)
        private.train()
        train_mode = observe(private, scene, "train", *data["train"])
    changed = {name: int(saved[name])-value for name, value in counts.items()}
    require(all(v == train_mode["forwards"] for v in changed.values()),
            "Disposable train-mode statistics count differs from authorized train blocks")
    require(fingerprint(private) == params and state_hash(model) == before, "Diagnostic changed original parameters or state")
    outcomes = {name: observe(model, scene, role, views, labels) for name, (role, (views, labels)) in enumerate(data.items())}
    require(all(outcomes[i]["heads"] == rows["original"]["probability_metrics"][role]["heads"]
                for i, role in enumerate(data)), "Restored model outputs differ from original")
    return {"arms": rows, "train_mode_control": {"result": train_mode, "classification": "ephemeral_train_mode_comparison",
              "role": "train", "statistics_buffer_updates": changed, "seed": 41377, "dropout": True},
            "confirmed_restoration": True, "state": before,
            "source_parameters": params, "state_calibration_applied": False,
            "context_moment_writes": 0}
