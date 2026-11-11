"""Report all cases and paired layer/mode differences without selecting favorable rows."""
from collections import defaultdict
import csv
from statistics import mean, stdev

from research_runtime import write_json


METRICS = ("oa", "aa", "kappa", "macro_f1", "nll")
RECALLS = tuple(f"recall_{i}" for i in range(1, 10))
VALUES = (*METRICS, *RECALLS)


def write_csv(path, rows, empty_fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, keys, values=VALUES):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, sample in groups.items():
        stats = {}
        for metric in values:
            measured = [row[metric] for row in sample if row[metric] is not None]
            stats[metric] = {"mean": mean(measured) if measured else None,
                             "sample_sd": stdev(measured) if len(measured) > 1 else None, "seeds": len(measured)}
        decisions = [row["strict_decisions"] for row in sample if row["strict_decisions"] is not None]
        result.append({**dict(zip(keys, group, strict=True)), "seed_count": len(sample),
                       "all_strict_decisions": all(decisions) if decisions else None,
                       "estimates": stats})
    return result


def metric_rows(results, *, scenes=False):
    for row in results:
        if "layers" not in row:
            continue
        role = row["role"]
        for name, layer in row["layers"].items():
            for partition, observed in layer["roles"].items():
                for checkpoint in ("selected", "terminal"):
                    scores = observed[checkpoint]
                    classes = range(1, len(scores["support"])+1) if scenes else range(1, 10)
                    scene = {"scene": role["target"], "source": row["disposition"]} if scenes else {}
                    yield {**scene, "case": row["case"], "seed": role["seed"], "mode": role["mode"], "method": role["method"],
                           "layer": name, "role": partition, "checkpoint": checkpoint,
                           "case_status": row["status"], "selected_step": layer["selected_step"],
                           "selected_at_budget_end": layer["selected_at_budget_end"],
                           "strict_decisions": observed["cross_backend"]["strict_decisions"] if checkpoint == "selected" else None,
                           "probabilities_close": observed["cross_backend"]["probabilities_close"] if checkpoint == "selected" else None,
                           "zero_recall_classes": " ".join(map(str, scores["zero_recall_classes"])),
                           **{key: scores[key] for key in METRICS},
                           **{f"recall_{i}": v for i, v in zip(classes, scores["recall"], strict=True)},
                           **{f"support_{i}": v for i, v in enumerate(scores["support"], 1)},
                           **{f"f1_{i}": v for i, v in enumerate(scores["f1"], 1)}}


def paired_rows(rows, *, scenes=False, values=VALUES):
    lookup = {(r.get("scene"), r["seed"], r["method"], r["mode"], r["layer"], r["role"]): r
              for r in rows if r["checkpoint"] == "selected"}
    for key, current in lookup.items():
        scene, seed, method, mode, layer, role = key
        controls = []
        if layer.endswith("_cnn"):
            controls.append(("cnn_minus_graph", mode, layer.replace("_cnn", "_graph")))
        if layer.startswith("stage2_"):
            controls.append(("stage2_minus_stage1", mode, layer.replace("stage2_", "stage1_")))
        if mode == "x_residual":
            controls.extend(("residual_minus_"+base, base, layer) for base in ("capacity_control", "none"))
        for contrast, base_mode, base_layer in controls:
            base = lookup.get((scene, seed, method, base_mode, base_layer, role))
            if base is None:
                continue
            delta = {k: current[k]-base[k] if current.get(k) is not None and base.get(k) is not None else None for k in values}
            yield {**({"scene": scene} if scenes else {}), "seed": seed, "method": method, "mode": mode, "layer": layer, "role": role,
                   "contrast": contrast, "baseline_mode": base_mode, "baseline_layer": base_layer,
                   "strict_decisions": current["strict_decisions"] and base["strict_decisions"],
                   "declining_classes": " ".join(k.removeprefix("recall_") for k, value in delta.items()
                                                 if k.startswith("recall_") and value is not None and value < 0), **delta}


def write_outputs(out, results):
    rows = list(metric_rows(results))
    pairs = list(paired_rows(rows))
    write_csv(out/"metrics.csv", rows, ("case", "layer", "role", "checkpoint"))
    write_csv(out/"paired-differences.csv", pairs, ("seed", "method", "mode", "layer", "role", "contrast"))
    selected = [row for row in rows if row["checkpoint"] == "selected"]
    write_json(out/"summary.json", {
        "cases": [{k: row[k] for k in ("case", "status", "probe_updates")} for row in results],
        "reviewed_probes": len(selected)//2, "probe_updates": sum(row["probe_updates"] for row in results),
        "reviewed_selected_roles": len(selected), "history_selections_checked": len(selected)//2,
        "nonselected_event_probability_replays": 0, "metric_rows": len(rows), "paired_rows": len(pairs),
        "numerical_gate_failures": [{**fail, "case": row["case"]} for row in results
                                    for fail in row.get("numerical_failures", [])],
        "selected_at_last_step": sum(row["selected_at_budget_end"] for row in selected if row["role"] == "val"),
        "three_seed_metrics": summarize(rows, ("method", "mode", "layer", "role", "checkpoint")),
        "three_seed_paired": summarize(pairs, ("method", "mode", "layer", "role", "contrast",
                                               "baseline_mode", "baseline_layer")),
        "scope": "Descriptive fixed-column PaviaU probes on previously used selection validation."})
    return ["metrics.csv", "paired-differences.csv", "summary.json"]
