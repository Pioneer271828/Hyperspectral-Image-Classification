"""Compare one frozen cohort from externally authenticated accuracy and cost evidence."""

from itertools import product
import statistics

from evaluation.classification_metrics import class_order, metrics_from_confusion
from evaluation.dynamic_curve import pin, timing_summary


HEADS = ("fused", "graph", "cnn")
METRICS = ("oa", "aa", "kappa", "macro_f1")
SCHEMA = "wp12-matched-cohort-v1"


def _fields(value, fields):
    if not isinstance(value, dict) or value.keys() != set(fields.split()):
        raise ValueError("Expected fields: " + fields)


def _index(rows, expected, fields):
    if not isinstance(rows, list):
        raise ValueError("Expected a list of registered cases")
    result = {}
    for row in rows:
        _fields(row, fields)
        key = row["scene"], row["seed"], row["arm"]
        if (not isinstance(key[0], str) or type(key[1]) is not int or
                not isinstance(key[2], str) or key not in expected or key in result):
            raise ValueError("Unexpected or duplicate scene/seed/arm")
        result[key] = row
    if result.keys() != expected:
        raise ValueError("Incomplete registered cohort")
    return result


def _counts(matrix, classes):
    # JSON integers only, with a total that cannot overflow the NumPy metric reducer.
    if (not isinstance(matrix, list) or len(matrix) != len(classes) or
            any(not isinstance(row, list) or len(row) != len(classes) or
                any(type(n) is not int or n < 0 for n in row) for row in matrix) or
            sum(map(sum, matrix)) > (2**63 - 1) // 2):
        raise ValueError("Invalid nonnegative confusion counts")
    if not all(map(sum, matrix)):
        raise ValueError("Every declared class must have evaluation support")
    return metrics_from_confusion(matrix, classes)


def _describe(values):
    if any(value is None for value in values):
        return {"values": values, "mean": None, "sample_sd": None}
    return {"values": values, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def _delta(left, right):
    return None if left is None or right is None else left - right


def summarize(plan, evidence):
    """Validate the declared grid and recompute descriptive metrics from its records."""
    _fields(plan, "schema protocol_sha256 scenes seeds arms baseline timing cases")
    _fields(evidence, "timing cases")
    if plan["schema"] != SCHEMA or not pin(plan["protocol_sha256"]):
        raise ValueError("Explicit evaluation protocol identity required")
    seeds, arms, scenes = (plan[k] for k in ("seeds", "arms", "scenes"))
    if (not isinstance(seeds, list) or len(seeds) < 3 or
            any(type(v) is not int or v < 0 for v in seeds) or len(set(seeds)) != len(seeds) or
            not isinstance(arms, list) or len(arms) < 2 or
            any(not isinstance(v, str) or not v for v in arms) or len(set(arms)) != len(arms) or
            plan["baseline"] not in arms or not isinstance(scenes, dict) or not scenes or
            any(not isinstance(v, str) or not v for v in scenes)):
        raise ValueError("Register scenes, three or more distinct model seeds and a baseline/control")
    for scene in scenes.values():
        _fields(scene, "class_ids workload_sha256 work_items")
        if not isinstance(scene["class_ids"], list):
            raise ValueError("Explicit class ID order required")
        class_order(scene["class_ids"])
        if not pin(scene["workload_sha256"]) or type(scene["work_items"]) is not int or not 0 < scene["work_items"] < 2**63:
            raise ValueError("Registered positive workload with identity required")
    timing = plan["timing"]
    _fields(timing, "environment_sha256 scope device warmups repeats")
    if (not pin(timing["environment_sha256"]) or timing["device"] not in ("cpu", "cuda") or
            timing["scope"] not in ("end_to_end", "resident_forward") or
            type(timing["warmups"]) is not int or timing["warmups"] < 1 or
            type(timing["repeats"]) is not int or timing["repeats"] < 4 or evidence["timing"] != timing):
        raise ValueError("Identical registered hardware, scope and timing budget required")

    expected = set(product(scenes, seeds, arms))
    registered = _index(plan["cases"], expected, "scene seed arm evaluation_sha256 checkpoint_sha256")
    measured = _index(evidence["cases"], expected, "scene seed arm accuracy cost")
    rows = {}
    baseline = plan["baseline"]
    for scene, seed, arm in product(scenes, seeds, arms):
        key = scene, seed, arm
        planned, actual = registered[key], measured[key]
        accuracy, cost = actual["accuracy"], actual["cost"]
        _fields(accuracy, "checkpoint_sha256 evaluation_sha256 heads")
        _fields(cost, "checkpoint_sha256 workload_sha256 wall_ms peak_bytes parameters package_bytes")
        checkpoint, identity = planned["checkpoint_sha256"], planned["evaluation_sha256"]
        if (not pin(checkpoint) or not pin(identity) or
                identity != registered[scene, seed, baseline]["evaluation_sha256"] or
                checkpoint != accuracy["checkpoint_sha256"] or checkpoint != cost["checkpoint_sha256"] or
                identity != accuracy["evaluation_sha256"] or cost["workload_sha256"] != scenes[scene]["workload_sha256"]):
            raise ValueError("Evaluation, checkpoint or workload identity differs")
        _fields(accuracy["heads"], " ".join(HEADS))
        heads = {head: _counts(accuracy["heads"][head], scenes[scene]["class_ids"]) for head in HEADS}
        if any(type(cost[k]) is not int or cost[k] <= 0 for k in ("peak_bytes", "parameters", "package_bytes")):
            raise ValueError("Measured peak bytes, parameters and package bytes required")
        stats = timing_summary(cost["wall_ms"], repeats=timing["repeats"], blocks=scenes[scene]["work_items"])
        stats["items_per_second"] = stats.pop("blocks_per_second")
        rows[key] = dict(scene=scene, seed=seed, arm=arm, checkpoint_sha256=checkpoint,
                         evaluation_sha256=identity, heads=heads,
                         cost={**stats, **{k: cost[k] for k in ("peak_bytes", "parameters", "package_bytes")}})

    summaries = []
    for scene, arm in product(scenes, arms):
        pairs = [(rows[scene, seed, arm], rows[scene, seed, baseline]) for seed in seeds]
        reference = [tuple(c["support"] for c in b["heads"]["fused"]["per_class"]) for _, b in pairs]
        for (row, _), support in zip(pairs, reference):
            if any(tuple(c["support"] for c in h["per_class"]) != support for h in row["heads"].values()):
                raise ValueError("Ground-truth class counts differ within paired evaluation")
        fields = ("p50_ms", "p95_ms", "items_per_second", "peak_bytes", "parameters", "package_bytes")
        summaries.append(dict(scene=scene, arm=arm, heads={
            head: {metric: {
                "absolute": _describe([a["heads"][head][metric] for a, _ in pairs]),
                "paired_delta": _describe([_delta(a["heads"][head][metric], b["heads"][head][metric])
                                          for a, b in pairs]),
            } for metric in METRICS} for head in HEADS},
            cost={field: {
                "absolute": _describe([a["cost"][field] for a, _ in pairs]),
                "paired_delta": _describe([a["cost"][field] - b["cost"][field] for a, b in pairs]),
            } for field in fields}))
    return dict(schema=SCHEMA, protocol_sha256=plan["protocol_sha256"], timing=timing,
                seeds=seeds, baseline=baseline, rows=list(rows.values()), summaries=summaries)
