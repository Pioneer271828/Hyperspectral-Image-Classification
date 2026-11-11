"""Scene-stratified endpoint and paired comparisons with explicit class IDs."""
from collections import Counter, defaultdict
from fractions import Fraction
from statistics import mean

from evaluation import allscene_probe_registry as registry, frozen_probe_summary as summary
from evaluation.frozen_probe_registry import require
from research_runtime import write_json


ENDPOINT_KEYS = ("scene", "source", "case", "seed", "method", "mode", "layer", "role", "checkpoint",
                 "case_status", "selected_step", "selected_at_budget_end", "strict_decisions",
                 "probabilities_close", "zero_recall_classes")
PAIR_KEYS = ("scene", "seed", "method", "mode", "layer", "role", "contrast", "baseline_mode",
             "baseline_layer", "strict_decisions", "declining_classes")
GROUP = ("scene", "method", "mode", "layer", "role")
PAIR_GROUP = (*GROUP, "contrast", "baseline_mode", "baseline_layer")
CLASS_VALUES = ("recall", "f1")
CLASS_DELTA = ("recall_delta", "f1_delta")
TABLES = ("metrics.csv", "class-metrics.csv", "paired-differences.csv", "paired-class-differences.csv", "summary.json")


def class_rows(row, paired=False):
    support = 10 if row["role"] == "train" else 5
    keys = PAIR_KEYS if paired else ENDPOINT_KEYS
    fields = CLASS_DELTA if paired else CLASS_VALUES
    columns = ("recall", "f1")
    for class_id in range(1, registry.CLASSES[row["scene"]]+1):
        values = {name: row[f"{column}_{class_id}"] for name, column in zip(fields, columns, strict=True)}
        if not paired:
            require(row[f"support_{class_id}"] == support, "Class support differs from the ordered release")
        yield {**{key: row[key] for key in keys}, "class_id": class_id, "support": support, **values}


def class_protection(pairs):
    """Apply the original -2 pp class cutoff as a separate diagnostic per layer."""
    groups = defaultdict(list)
    for row in pairs:
        if row["role"] == "val" and row["contrast"].startswith("residual_minus_"):
            keys = (*PAIR_GROUP, "class_id")
            groups[tuple(row[k] for k in keys)].append(row)
    decisions = []
    for group, rows in groups.items():
        scope = dict(zip((*PAIR_GROUP, "class_id"), group, strict=True))
        support = rows[0]["support"]
        delta = mean(Fraction(round(row["recall_delta"]*support), support) for row in rows)
        complete = len(rows) == len({row["seed"] for row in rows}) == 3
        decisions.append({**scope, "seed_count": len(rows), "mean_recall_delta_pp": float(100*delta),
                          "recall_protected_at_2pp": delta >= Fraction(-2, 100) if complete else None,
                          "all_strict_decisions": all(row["strict_decisions"] for row in rows)})
    return decisions


def write_outputs(out, results):
    rows = list(summary.metric_rows(results, scenes=True))
    values = (*summary.METRICS, *(f"recall_{i}" for i in range(1, 23)), *(f"f1_{i}" for i in range(1, 23)))
    pairs = list(summary.paired_rows(rows, scenes=True, values=values))
    classes = [item for row in rows for item in class_rows(row)]
    paired_classes = [item for row in pairs for item in class_rows(row, paired=True)]
    endpoint_fields, pair_fields = (*ENDPOINT_KEYS, *summary.METRICS), (*PAIR_KEYS, *summary.METRICS)
    for name, source, fields in (
            (TABLES[0], rows, endpoint_fields),
            (TABLES[1], classes, (*ENDPOINT_KEYS, "class_id", "support", *CLASS_VALUES)),
            (TABLES[2], pairs, pair_fields),
            (TABLES[3], paired_classes, (*PAIR_KEYS, "class_id", "support", *CLASS_DELTA))):
        summary.write_csv(out/name, [{k: row[k] for k in fields} for row in source], fields)
    selected = [row for row in rows if row["role"] == "val" and row["checkpoint"] == "selected"]
    scene_stats = {}
    for scene, count in registry.CLASSES.items():
        scene_cases = [r for r in results if r["role"]["target"] == scene]
        scene_selected = [r for r in selected if r["scene"] == scene]
        scene_stats[scene] = {
            "classes": count, "expected_cases": 18, "statuses": dict(Counter(r["status"] for r in scene_cases)),
            "reviewed_probes": len(scene_selected), "selected_at_last_step": sum(r["selected_at_budget_end"] for r in scene_selected),
            "zero_recall_selected_val_probes": sum(bool(r["zero_recall_classes"]) for r in scene_selected),
            "historical_probe_updates": sum(r["historical_probe_updates"] for r in scene_cases),
            "new_probe_updates": sum(r["new_probe_updates"] for r in scene_cases)}
    write_json(out/TABLES[4], {
        "scope": "Same-scene frozen probe development validation. Source, spatial and query confirmation are separate.",
        "scenes": scene_stats, "metric_rows": len(rows), "class_metric_rows": len(classes),
        "paired_rows": len(pairs), "paired_class_rows": len(paired_classes),
        "reviewed_probes": len(selected), "reviewed_selected_roles": 2*len(selected),
        "history_selections_checked": len(selected), "nonselected_event_probability_replays": 0,
        "numerical_gate_failures": [{"case": row["case"], "scene": row["role"]["target"], **failure}
            for row in results for failure in row.get("numerical_failures", [])],
        "three_seed_metrics": summary.summarize(rows, (*GROUP, "checkpoint"), summary.METRICS),
        "three_seed_class_metrics": summary.summarize(classes, (*GROUP, "checkpoint", "class_id", "support"), CLASS_VALUES),
        "three_seed_paired": summary.summarize(pairs, PAIR_GROUP, summary.METRICS),
        "three_seed_class_paired": summary.summarize(paired_classes, (*PAIR_GROUP, "class_id", "support"), CLASS_DELTA),
        "residual_val_class_protection": class_protection(paired_classes)})
    return TABLES
