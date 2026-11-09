"""Evaluate preregistered channel dependency groups with reversible MON probes."""
from fractions import Fraction
import statistics

import numpy as np
import torch

from evaluation.classification_metrics import class_order, classification_metrics
from model.research_config import digest
from research_runtime import tensor_hash

from .interventions import ChannelIntervention, _snapshot, isolated_probe
from .readonly import threshold
from .windows import IDENTITY


HEADS = ("fused", "graph", "cnn")
POLICY = "mon_reversible_dependency_groups_v1"


def _schedule(candidate, selection, declaration, manifest):
    calls = manifest["calls"]
    anchor_keys = [name for name, call in calls.items()
                   if (call["spec"]["path"], call["call_index"], call["spec"]["unit"]) ==
                   (selection["layer"], selection["call_index"], selection["unit"])]
    if (candidate["policy"] != "mon_sustained_train_candidates_v1" or candidate["automatic_pruning"] is not False
            or digest(selection) != candidate["plan_sha256"] or digest(candidate) != declaration["candidate_sha256"]
            or len(anchor_keys) != 1 or manifest["context"]["partition"] != "val"):
        raise ValueError("Exact train-screened candidates and a validation probe anchor required")
    context, final = manifest["context"], candidate["samples"][-1]
    if (any(context[k] != candidate["identity"][k] for k in IDENTITY if k != "condition")
            or any(context[k] != final[k] for k in ("step", "checkpoint_hash"))):
        raise ValueError("Validation probe must use the screened checkpoint and cohort")

    mapping, used_sites, used_channels = {}, set(), set()
    groups = declaration["groups"]
    for group in groups:
        name, index = group["name"], group["candidate_index"]
        if (not isinstance(name, str) or not name or name in mapping or type(index) is not int
                or index not in candidate["candidate_channel_indices"] or index in used_channels
                or index < 0 or index >= candidate["channels"]):
            raise ValueError("Unique dependency group on a screened internal channel required")
        by_call = {}
        for site in group["sites"]:
            observation, channel = site["observation"], site["channel"]
            key = (observation, channel)
            if observation not in calls or type(channel) is not int or channel < 0 or key in used_sites:
                raise ValueError("Explicit nonoverlapping dependency sites required")
            used_sites.add(key)
            by_call.setdefault(observation, []).append(channel)
        if by_call.get(anchor_keys[0]) != [index]:
            raise ValueError("Every dependency group must include its exact screened anchor")
        used_channels.add(index)
        mapping[name] = by_call
    sets = [tuple(sorted(group)) for group in declaration["combinations"]]
    if (not mapping or (len(mapping) > 1 and not sets) or len(sets) != len(set(sets))
            or any(len(set(s)) != len(s) or len(s) < 2 or not set(s) <= mapping.keys() for s in sets)
            or 1 + len(mapping) + len(sets) > 32):
        raise ValueError("Unique predefined combinations and at most 32 probe trials required")
    return mapping, [(name,) for name in mapping] + sets


def _metrics(outputs, labels, ids):
    if not isinstance(outputs, dict) or set(outputs) != set(HEADS):
        raise ValueError("Explicit fused, graph and CNN validation scores required")
    predictions, results = {}, {}
    for head in HEADS:
        value = outputs[head]
        if (not torch.is_tensor(value) or not value.is_floating_point()
                or tuple(value.shape) != (len(labels), len(ids)) or not torch.isfinite(value).all()):
            raise ValueError("Finite aligned validation scores required")
        predictions[head] = np.asarray(ids)[value.argmax(dim=1).cpu().numpy()]
        results[head] = classification_metrics(labels, predictions[head], ids)
    return results, predictions["fused"]


def _protection(baseline, current, labels, before, after, ids, small_classes, limits):
    previous, observed = baseline["confusion_matrix"], current["confusion_matrix"]
    supports = [sum(row) for row in previous]
    drops = [Fraction(previous[i][i] - observed[i][i], n) for i, n in enumerate(supports)]
    oa = sum(d * n for d, n in zip(drops, supports)) / sum(supports)
    aa = statistics.mean(drops)
    failed = [c for c, drop in zip(ids, drops) if drop > limits["recall"]]
    newly_wrong = {str(c): int(((labels == c) & (before == c) & (after != c)).sum()) for c in small_classes}
    return {"oa_drop": float(oa), "aa_drop": float(aa), "per_class_recall_drop": dict(zip(map(str, ids), map(float, drops))),
            "failed_recall_classes": failed, "new_small_class_errors": newly_wrong,
            "passes_metric_guards": oa <= limits["oa"] and aa <= limits["aa"] and not failed and not any(newly_wrong.values())}


def evaluate_groups(reference, factory, forward, inputs, probe, labels, candidate, selection, declaration,
                    *, expected_plan_sha256):
    """A no-update validation experiment, with a new isolated model per trial."""
    if (digest(declaration) != expected_plan_sha256 or declaration["probe_identity"] != probe.identity
            or declaration["reference_state"] != _snapshot(reference)["state"]):
        raise ValueError("Exact fixed intervention plan, checkpoint and probe required")
    mapping, schedule = _schedule(candidate, selection, declaration, probe.manifest)
    ids = class_order(declaration["class_ids"])
    if (not torch.is_tensor(labels) or labels.dtype not in (torch.int16, torch.int32, torch.int64)
            or labels.ndim != 1 or tensor_hash(labels) != declaration["labels_sha256"]):
        raise ValueError("Pinned one-dimensional integer validation labels required")
    truth = labels.detach().cpu().numpy().copy()
    if set(truth.tolist()) != set(ids):
        raise ValueError("Validation support is required for every declared class")
    supports = declaration["train_support"]
    small_limit = declaration["small_class_max_train_support"]
    if (set(supports) != set(map(str, ids)) or any(type(n) is not int or n <= 0 for n in supports.values())
            or type(small_limit) is not int or small_limit < 0):
        raise ValueError("Explicit positive training support per class required")
    small = [c for c in ids if supports[str(c)] <= small_limit]
    if set(declaration["max_drop"]) != {"oa", "aa", "recall"}:
        raise ValueError("Predeclared OA, AA and class recall drop limits required")
    limits = {k: Fraction(str(threshold(v))) for k, v in declaration["max_drop"].items()}
    if any(v > 1 for v in limits.values()):
        raise ValueError("Metric limits must be fractions in [0, 1]")

    baseline = isolated_probe(reference, factory, forward, inputs, probe)
    before_metrics, before = _metrics(baseline["outputs"], truth, ids)
    del baseline
    rows, single_passed = [], {}
    for names in schedule:
        by_call = {}
        for name in names:
            for observation, channels in mapping[name].items():
                by_call.setdefault(observation, []).extend(channels)
        interventions = tuple(ChannelIntervention(key, tuple(sorted(values))) for key, values in by_call.items())
        trial = isolated_probe(reference, factory, forward, inputs, probe, interventions=interventions)
        metrics, after = _metrics(trial["outputs"], truth, ids)
        guard = _protection(before_metrics["fused"], metrics["fused"], truth, before, after, ids, small, limits)
        blocked_by = [name for name in names if len(names) > 1 and not single_passed[name]]
        passed = guard["passes_metric_guards"] and not blocked_by
        if len(names) == 1:
            single_passed[names[0]] = passed
        rows.append({"groups": list(names), "metrics": metrics, "protection": guard, "failed_single_groups": blocked_by,
                     "passed": passed, "execution": trial["audit"]})
    if tensor_hash(labels) != declaration["labels_sha256"]:
        raise RuntimeError("Validation labels changed during evaluation")
    return {"policy": POLICY, "plan_sha256": expected_plan_sha256, "baseline": before_metrics,
            "trials": rows, "protected_small_classes": small, "optimizer_updates": 0,
            "query_labels_accessed": False, "automatic_pruning": False}
