"""Apply Issue249's preregistered contrasts to authenticated query metric rows."""
from fractions import Fraction
import math
from statistics import mean

from evaluation.classification_metrics import metrics_from_confusion
from g4_light_full_target import roles
from g4_light_target import CONTROLS
from g4_matched_target_analysis import FT, HEADS, METRICS, SCRATCH, _stats, _value, validate_rows
from g4_matched_target_study import MODES, SEEDS
from joint_baseline import NAMES
from model.research_config import digest


PLAN_SHA256 = "c666ca8b0c1d78471b662ef900e0c6302fb8d7de3cbd2761eed42bf6da7c1d00"


def light_rows(rows, baseline):
    expected = set(roles())
    lookup = {}
    for row in rows:
        target, seed, method, mode = key = tuple(row[k] for k in ("target", "seed", "method", "mode"))
        if type(seed) is not int or key not in expected or key in lookup or set(row["metrics"]) != set(HEADS):
            raise ValueError("Unique registered light query roles and three heads required")
        for head in HEADS:
            reference = baseline[target, seed, FT, mode]["metrics"][head]
            metric = row["metrics"][head]
            matrix = metric["confusion_matrix"]
            ids = reference["class_ids"]
            if (not isinstance(matrix, list) or len(matrix) != len(ids)
                    or any(not isinstance(r, list) or len(r) != len(ids) for r in matrix)
                    or any(type(v) is not int or v < 0 for r in matrix for v in r)
                    or [sum(r) for r in matrix] != [c["support"] for c in reference["per_class"]]):
                raise ValueError("Matched reference class supports and integer confusion counts required")
            calculated = metrics_from_confusion(matrix, ids)
            if (set(metric) != set(calculated) | {"nll"} or type(metric["nll"]) not in (int, float)
                    or not math.isfinite(metric["nll"]) or metric["nll"] < 0
                    or digest({k: v for k, v in metric.items() if k != "nll"}) != digest(calculated)):
                raise ValueError("Count-consistent light query metrics required")
        digest(row)
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError("All54 light query roles required")
    return lookup


def contrast(lookup, candidate, reference, *, primary):
    def delta(target, seed, mode, head, metric):
        a = _value(lookup[target, seed, candidate, mode], head, metric)
        b = _value(lookup[target, seed, reference, mode], head, metric)
        return [x - y for x, y in zip(a, b)] if metric == "recall" else a - b

    scene_deltas = {}
    class_failures = []
    group_deltas = []
    for target in NAMES:
        scene_deltas[target] = {
            metric: mean(delta(target, seed, mode, "fused", metric) for seed in SEEDS for mode in MODES)
            for metric in ("oa", "aa")}
        for mode in MODES:
            metrics, recalls = {}, {}
            for head in HEADS:
                metrics[head] = {metric: _stats(delta(target, seed, mode, head, metric) for seed in SEEDS)
                                 for metric in METRICS}
                recalls[head] = [mean(values) for values in zip(*[
                    delta(target, seed, mode, head, "recall") for seed in SEEDS])]
            class_failures.extend({"target": target, "mode": mode, "class_id": i, "delta_pp": float(v * 100)}
                                  for i, v in enumerate(recalls["fused"], 1) if v < Fraction(-2, 100))
            group_deltas.append({"target": target, "mode": mode, "metrics": metrics,
                                 "class_recall_delta_pp": {head: [float(v * 100) for v in values]
                                                           for head, values in recalls.items()}})
    seed_deltas = [mean(delta(t, seed, mode, "fused", "oa") for t in NAMES for mode in MODES) for seed in SEEDS]
    result = {"candidate": candidate, "reference": reference, "claim_type": "primary" if primary else "descriptive",
              "equal_mode_equal_scene_fused_oa_delta_pp": _stats(seed_deltas, 100),
              "scene_fused_deltas_pp": {t: {k: float(v * 100) for k, v in row.items()}
                                       for t, row in scene_deltas.items()},
              "scene_mode_deltas": group_deltas}
    if primary:
        gates = {"positive_macro_oa_mean": mean(seed_deltas) > 0,
                 "all_model_seeds_positive": all(d > 0 for d in seed_deltas),
                 "every_scene_aa_drop_at_most_1pp": all(r["aa"] >= Fraction(-1, 100) for r in scene_deltas.values()),
                 "honghu_oa_drop_at_most_1pp": scene_deltas["whu-honghu"]["oa"] >= Fraction(-1, 100),
                 "every_scene_mode_class_recall_drop_at_most_2pp": not class_failures}
        result.update(gates=gates, passes_all_gates=all(gates.values()), failed_class_protections=class_failures)
    return result


def analyze_rows(light, reference, *, plan_sha256):
    """Pure report arithmetic. Callers must authenticate both frozen result sets."""
    if plan_sha256 != PLAN_SHA256:
        raise ValueError("Exact Issue249 preregistered analysis plan required")
    baseline = validate_rows(reference)
    lookup = {**baseline, **light_rows(light, baseline)}
    return {
        "analysis_plan_sha256": PLAN_SHA256,
        "model_seeds": list(SEEDS),
        "support_seed_repetitions": 1,
        "significance_claim": False,
        "primary": [contrast(lookup, method, FT, primary=True) for method in CONTROLS],
        "descriptive": [contrast(lookup, method, SCRATCH, primary=False) for method in CONTROLS]
                       + [contrast(lookup, CONTROLS[1], CONTROLS[0], primary=False)],
    }
