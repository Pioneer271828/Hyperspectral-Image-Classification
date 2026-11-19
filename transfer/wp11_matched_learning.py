"""Matched development learning over the original class-first support stream."""

from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from transfer.curriculum_training import CurriculumRun
from transfer.development_training import development_counts, sparse_labels
from transfer.g3_training import _forward, state_hash
from transfer.light_head_training import LightHeadRun


HEADS = ("fused", "graph", "cnn")
SEED = 64
LR = 1e-4


def validation_blocks(scene, record):
    labels = sparse_labels(record, scene.plan, "validation_support")
    for tile, role in zip(scene.plan["tiles"], record["roles"], strict=True):
        if role["role"] == "devval":
            target = labels[tile_slice(tile)].ravel()
            if (target > 0).any():
                yield tile, target


def development_predictions(scene, record, compute, heads):
    """Run actual endpoints once per V5 block, preserving the original reduction."""
    metrics = {head: {"matrix": np.zeros((scene.classes, scene.classes), dtype=np.int64),
                      "nll": 0., "hashes": [], "errors": []} for head in heads}
    count = 0
    for tile, target in validation_blocks(scene, record):
        mask = target > 0
        y = target[mask] - 1
        y0, y1, x0, x1 = tile["box"]
        global_ids = (np.arange(y0, y1)[:, None] * scene.plan["shape"][1]
                      + np.arange(x0, x1)).ravel()[mask]
        count += len(y)
        with torch.no_grad():
            outputs = compute(tile)
        for head, probabilities in zip(heads, outputs, strict=True):
            a = probabilities.detach().cpu().numpy()
            if a.shape != (len(target), scene.classes) or not np.isfinite(a).all():
                raise ValueError("Finite aligned endpoint probabilities required")
            p = a[mask]
            predicted = p.argmax(1)
            row = metrics[head]
            row["matrix"] += np.bincount(y * scene.classes + predicted,
                                        minlength=scene.classes**2).reshape(scene.classes, scene.classes)
            row["nll"] -= float(np.log(np.maximum(p[np.arange(len(y)), y], 1e-8)).sum())
            row["hashes"].append(digest(a.tolist()))
            row["errors"].extend(int(i) for i in global_ids[predicted != y])
    if count != development_counts(record)[1]:
        raise ValueError("Fixed V5 validation support differs")
    results = {}
    for head, row in metrics.items():
        matrix = row["matrix"]
        support = matrix.sum(1)
        recall = np.diag(matrix) / support
        results[head] = dict(nll=row["nll"] / count, oa=float(np.trace(matrix) / count), count=count,
                             confusion=matrix.tolist(), support=support.tolist(), recall=recall.tolist(),
                             output_hash=digest(row["hashes"]), errors=sorted(row["errors"]))
    return results


@contextmanager
def count_graphs(model):
    """Count actual calls into the learned graph builder, including evaluations."""
    builder = model.backbone.learnable_graph
    original = builder.build_graph
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    with patch.object(builder, "build_graph", new=counted):
        yield calls


def parameter_evidence(model, before):
    return {name: dict(gradient_l1=None if p.grad is None else float(p.grad.detach().abs().sum()),
                       changed=int((before[name] != p.detach()).sum())) for name, p in model.named_parameters()}


def snapshot_parameters(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


class MatchedFullRun(CurriculumRun):
    """Use the existing FP32 training/recovery contract with per-tensor evidence."""

    FORMAT = "vca-wp11-matched-full-v1"
    FAILURE_FORMAT = "vca-wp11-matched-full-failure-v1"

    def _contract_extensions(self):
        return {"consumer": "wp11-matched-fp32", "parameter_evidence": "per_named_tensor"}

    def _parameter_history(self, before):
        return {"parameter_evidence": parameter_evidence(self.model, before)}

    def _history_extension_keys(self):
        return {"parameter_evidence"}

    def _validate_history_extensions(self, row):
        evidence = row["parameter_evidence"]
        if list(evidence) != self.spec["parameters"]:
            raise ValueError("Matched per-tensor evidence differs")
        for name, item in evidence.items():
            norm, changed = item["gradient_l1"], item["changed"]
            if (set(item) != {"gradient_l1", "changed"} or type(changed) is not int or changed < 0
                    or (norm is None and (changed or name in row["gradient_parameters"]))
                    or (norm is not None and (not np.isfinite(norm) or norm < 0
                                              or name not in row["gradient_parameters"]))):
                raise ValueError("Invalid matched parameter evidence")


def construct(arm, source, scene, record, plan):
    if arm == "light":
        return LightHeadRun(source, scene, record, updates=plan.updates, seed=SEED, lr=LR, class_balance=True)
    if arm == "full":
        return MatchedFullRun(source, scene, record, plan=plan, policy=None, lr=LR, torch_seed=SEED,
                              expected_source_state=state_hash(source), expected_plan_hash=plan.plan_hash,
                              expected_policy_hash=None)
    raise ValueError("Only the two matched arms are registered")


def restore(arm, source, scene, record, plan, checkpoint, contract):
    options = dict(expected_sha256=checkpoint["sha256"], expected_contract_hash=contract)
    if arm == "light":
        return LightHeadRun.load(source, scene, record, checkpoint["path"], **options)
    if arm == "full":
        return MatchedFullRun.load(source, scene, record, checkpoint["path"], plan=plan, policy=None, **options)
    raise ValueError("Only the two matched arms are registered")


def require_exposure(run, plan):
    if len(run.history) != run.completed:
        raise ValueError("Complete event history required")
    for step, row in enumerate(run.history, 1):
        event = plan.event(step)
        if any(row[key] != event[key] for key in ("step", "tile_id", "anchor")):
            raise ValueError("Observed class-first draw differs from frozen exposure")


def advance(arm, run, plan, stop):
    """Record light-head gradients without changing its original training contract."""
    evidence = []
    start = run.completed
    light = arm == "light"
    observed_model = run.source if light else run.model
    with count_graphs(observed_model) as graphs:
        while run.completed < stop:
            before = snapshot_parameters(run.head) if light else None
            row = run.step()
            event = plan.event(run.completed)
            if any(row[key] != event[key] for key in ("step", "tile_id", "anchor")):
                raise ValueError("Class-first execution differs from frozen exposure")
            if light:
                evidence.append(dict(step=run.completed, parameters=parameter_evidence(run.head, before)))
    per_step = 0 if light else sum(1 for _ in validation_blocks(run.scene, run.record)) + 1
    if graphs["count"] != per_step * (stop - start):
        raise ValueError("Actual graph construction differs from the registered path")
    return evidence, graphs["count"]


def frozen_full(run, selected):
    result = deepcopy(run.source)
    result.load_state_dict(run.selected_state if selected else run.model.state_dict(), strict=True)
    return result.requires_grad_(False).eval()


def full_predictions(model, scene, record):
    return development_predictions(scene, record, lambda tile: _forward(model, scene, tile)[:3], HEADS)


def light_predictions(run, state):
    head = deepcopy(run.head)
    head.load_state_dict(state, strict=True)
    head.requires_grad_(False).eval()
    return development_predictions(run.scene, run.record,
                                   lambda tile: (head(run._features(tile)).softmax(-1),), ("light",))


def validation_values(row):
    return {key: row[key] for key in ("nll", "oa", "count", "confusion", "support", "recall")}
