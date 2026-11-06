"""Issue184 preregistered count-based analysis; never releases query labels."""
import copy
from fractions import Fraction
import math
from pathlib import Path
import re
import statistics
import time

from evaluation.classification_metrics import metrics_from_confusion
from g4_matched_target_audit import check_files, read
from g4_matched_target_evaluation import QUERY_POLICY, new_output, paired_summary, require_frozen_boundary
from g4_matched_target_execution import read_registration, unchanged_producer
from g4_matched_target_review import POLICY as REVIEW_POLICY
from g4_matched_target_study import EXPECTED, METHODS, MODES, SEEDS, case_name, grid
from joint_baseline import NAMES
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code, write_json

POLICY = "g4_matched_full54_preregistered_analysis_v1"
PRODUCER = "331d718567b1a9726934f445665999df972392b1"
PRODUCER_SOURCE = "48b28cfeee55e4d351fbf873d35e2e0d0f22ccca12910ef8a9f147e0c9bdbad8"
REVIEW_PRODUCER = "ad60ce0a863aad1991ee6c84d547c3efd995290f"
PLAN_SHA256 = "0bd71779546fd5ee0cbaab57b7b3639a1c378bcf9d1379f8afb1a80231a92dee"
HEADS = ("fused", "graph", "cnn")
METRICS = ("oa", "aa", "kappa", "nll")
SCRATCH, FT = METHODS[:2]


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Explicit SHA256 identity required")
    return value


def _stats(values, scale=1):
    values = list(values)
    if len(values) != 3:
        raise ValueError("Exactly three paired model seeds required")
    return {"mean": float(statistics.mean(values) * scale),
            "sample_sd": float(statistics.stdev([float(v * scale) for v in values])),
            "seed_values": [float(v * scale) for v in values], "seeds": list(SEEDS)}


def validate_rows(rows):
    if not isinstance(rows, list) or len(rows) != 54:
        raise ValueError("Complete54 query rows required")
    lookup, cohorts = {}, {}
    for row in rows:
        key = tuple(row[k] for k in ("target", "seed", "method", "mode"))
        if (type(row["seed"]) is not int or key not in grid() or key in lookup
                or row["name"] != case_name(*key) or set(row["metrics"]) != set(HEADS)):
            raise ValueError("Unique named registered54 roles and three heads required")
        _hash(row["checkpoint_sha256"]); _hash(row["release_sha256"])
        ids = list(range(1, DATASETS[row["target"]][2] + 1))
        for head in HEADS:
            value = row["metrics"][head]
            matrix = value["confusion_matrix"]
            if (not isinstance(matrix, list) or len(matrix) != len(ids)
                    or any(not isinstance(r, list) or len(r) != len(ids) for r in matrix)
                    or any(type(v) is not int or v < 0 for r in matrix for v in r)
                    or not 0 < sum(v for r in matrix for v in r) <= 2 ** 53):
                raise ValueError("Bounded nonnegative integer confusion counts required")
            expected = metrics_from_confusion(matrix, ids)
            if (not expected["full_class_coverage"] or set(value) != set(expected) | {"nll"}
                    or digest({k: v for k, v in value.items() if k != "nll"}) != digest(expected)
                    or type(value["nll"]) not in (int, float) or not math.isfinite(value["nll"])
                    or value["nll"] < 0):
                raise ValueError("Complete supported count-consistent finite query metrics required")
            support = tuple(c["support"] for c in expected["per_class"])
            if row["target"] in cohorts and cohorts[row["target"]] != support:
                raise ValueError("All methods/seeds/heads must use the same target cohort")
            cohorts[row["target"]] = support
        digest(row)  # Reject NaN/Inf in preserved branch diagnostics too.
        lookup[key] = row
    if set(lookup) != set(grid()):
        raise ValueError("Complete frozen54 role grid required")
    return lookup


def _value(row, head, metric):
    value = row["metrics"][head]
    if metric == "nll":
        return value[metric]  # NLL is pinned query output, not count-recomputed.
    matrix = value["confusion_matrix"]
    supports = [sum(r) for r in matrix]
    recalls = [Fraction(r[i], supports[i]) for i, r in enumerate(matrix)]
    if metric == "recall":
        return recalls
    total = sum(supports)
    oa = Fraction(sum(matrix[i][i] for i in range(len(matrix))), total)
    if metric == "oa":
        return oa
    if metric == "aa":
        return statistics.mean(recalls)
    predicted = [sum(r[i] for r in matrix) for i in range(len(matrix))]
    chance = Fraction(sum(a * b for a, b in zip(supports, predicted)), total ** 2)
    return (oa - chance) / (1 - chance)


def contrast(lookup, a, am, b, bm, *, inferential):
    def row(target, seed, mode, method):
        return lookup[target, seed, method, mode]
    scenes, seed_macro, class_pass = [], [], True
    for target in NAMES:
        differences, recall = {}, {}
        for head in HEADS:
            differences[head] = {metric: _stats(
                _value(row(target, s, a, am), head, metric) - _value(row(target, s, b, bm), head, metric)
                for s in SEEDS) for metric in METRICS}
            recall[head] = [statistics.mean(
                _value(row(target, s, a, am), head, "recall")[i]
                - _value(row(target, s, b, bm), head, "recall")[i] for s in SEEDS)
                for i in range(DATASETS[target][2])]
        aa = statistics.mean(_value(row(target, s, a, am), "fused", "aa")
                             - _value(row(target, s, b, bm), "fused", "aa") for s in SEEDS)
        oa = statistics.mean(_value(row(target, s, a, am), "fused", "oa")
                             - _value(row(target, s, b, bm), "fused", "oa") for s in SEEDS)
        failed = [i + 1 for i, v in enumerate(recall["fused"]) if v < Fraction(-2, 100)]
        class_pass = class_pass and not failed
        scenes.append({"target": target, "three_head_metric_deltas_fraction_units": differences,
            "three_head_mean_recall_deltas_pp": {h: [float(v * 100) for v in vs] for h, vs in recall.items()},
            "fused_mean_aa_delta_pp": float(aa * 100), "fused_mean_oa_delta_pp": float(oa * 100),
            "aa_protected": aa >= Fraction(-1, 100), "class_protected": not failed,
            "failed_class_ids": failed, "honghu_oa_protected": target != "whu-honghu" or oa >= Fraction(-1, 100)})
    for seed in SEEDS:
        seed_macro.append(statistics.mean(_value(row(t, seed, a, am), "fused", "oa")
            - _value(row(t, seed, b, bm), "fused", "oa") for t in NAMES))
    gates = {"mean_equal_scene_macro_oa_positive": statistics.mean(seed_macro) > 0,
        "all_three_model_seeds_positive": all(v > 0 for v in seed_macro),
        "every_scene_mean_aa_drop_at_most_1pp": all(r["aa_protected"] for r in scenes),
        "honghu_mean_oa_drop_at_most_1pp": all(r["honghu_oa_protected"] for r in scenes),
        "every_class_mean_recall_drop_at_most_2pp": bool(class_pass)}
    return {"candidate": {"mode": a, "method": am}, "reference": {"mode": b, "method": bm},
        "equal_scene_macro_fused_oa_delta_pp": _stats(seed_macro, 100), "scenes": scenes,
        "claim_type": "preregistered_benefit" if inferential else "descriptive_only",
        **({"gates": gates, "passes_all_gates": all(gates.values())} if inferential else {})}


def analyze_rows(rows):
    lookup = validate_rows(rows)
    groups = []
    for target in NAMES:
        for method in METHODS[:2]:
            for mode in MODES:
                group = [lookup[target, s, method, mode] for s in SEEDS]
                groups.append({"target": target, "method": method, "mode": mode,
                    "heads": {h: {m: _stats(_value(r, h, m) for r in group) for m in METRICS} for h in HEADS},
                    "per_class_mean_recall": {h: [float(statistics.mean(_value(r, h, "recall")[i]
                        for r in group)) for i in range(DATASETS[target][2])] for h in HEADS}})
    primary = contrast(lookup, "x_residual", FT, "capacity_control", FT, inferential=True)
    secondary = {"residualFT_minus_noneFT": contrast(lookup, "x_residual", FT, "none", FT, inferential=True)}
    secondary.update({mode + "_FT_minus_scratch": contrast(lookup, mode, FT, mode, SCRATCH, inferential=True)
                      for mode in MODES})
    descriptive = {a + "_minus_" + b + "_scratch": contrast(lookup, a, SCRATCH, b, SCRATCH, inferential=False)
                   for a, b in (("x_residual", "capacity_control"), ("x_residual", "none"), ("capacity_control", "none"))}
    descriptive["capacityFT_minus_noneFT"] = contrast(lookup, "capacity_control", FT, "none", FT, inferential=False)
    did = {h: {m: _stats(statistics.mean(
        _value(lookup[t, s, FT, "x_residual"], h, m) - _value(lookup[t, s, FT, "capacity_control"], h, m)
        - _value(lookup[t, s, SCRATCH, "x_residual"], h, m) + _value(lookup[t, s, SCRATCH, "capacity_control"], h, m)
        for t in NAMES) for s in SEEDS) for m in METRICS} for h in HEADS}
    return {"policy": POLICY, "plan_sha256": PLAN_SHA256, "primary_name": "residualFT_minus_capacityFT",
        "primary": primary, "secondary": secondary, "descriptive": descriptive,
        "difference_in_differences_fraction_units": did, "groups": groups,
        "models": [copy.deepcopy(lookup[k]) for k in grid()], "threshold_arithmetic": "exact_rational_confusion_counts",
        "metric_units": "fractions except fields explicitly named pp", "model_seeds": list(SEEDS),
        "support_seed_repetitions": 1, "significance_claim": False, "secondary_replaces_primary": False,
        "benefit_conclusion": "primary_benefit_supported" if primary["passes_all_gates"] else "primary_benefit_not_supported"}


def report_markdown(analysis):
    lines = ["# Full54 preregistered query analysis", "", analysis["benefit_conclusion"], "",
        "Three model seeds with one support set; no significance claim. Exact count-based predicates.", "",
        "| Claim | Mean macro OA delta (pp) | Sample SD (pp) | All gates |", "|---|---:|---:|---|"]
    claims = [(analysis["primary_name"], analysis["primary"]), *analysis["secondary"].items()]
    for name, row in claims:
        stat = row["equal_scene_macro_fused_oa_delta_pp"]
        lines.append(f'| {name} | {stat["mean"]:.9f} | {stat["sample_sd"]:.9f} | {row["passes_all_gates"]} |')
    for name, row in claims:
        lines += ["", "## " + name, "", "Paired seed deltas (pp): " + str(row["equal_scene_macro_fused_oa_delta_pp"]["seed_values"]), ""]
        lines += ["- " + key + ": " + str(value) for key, value in row["gates"].items()]
        lines += ["", "| Scene | Mean OA delta (pp) | Mean AA delta (pp) | Failed recall class IDs |", "|---|---:|---:|---|"]
        lines += [f'| {r["target"]} | {r["fused_mean_oa_delta_pp"]:.9f} | {r["fused_mean_aa_delta_pp"]:.9f} | {r["failed_class_ids"]} |'
                  for r in row["scenes"]]
    lines += ["", "All54 three-head/count/class records,18 groups and descriptive contrasts are preserved in analysis.json.",
        "NLL is authenticated query output; it cannot be independently reconstructed from confusion counts.",
        "Scratch/capacity descriptive contrasts and difference-in-differences do not replace the primary claim.", ""]
    return "\n".join(lines)


def export_analysis(query_path, query_sha256, freeze_path, freeze_sha256, review_path, review_sha256, out, code):
    started = time.monotonic()
    paths = [Path(p).resolve(strict=True) for p in (query_path, freeze_path, review_path)]
    pins = {}
    def pin(path, sha):
        path = str(Path(path).resolve(strict=True)); _hash(sha)
        if path in pins and pins[path] != sha:
            raise ValueError("Authenticated pins cannot be replaced")
        pins[path] = sha
    for path, sha in zip(paths, (query_sha256, freeze_sha256, review_sha256)):
        pin(path, sha)
    check_files(pins)
    query, freeze, review = (read(p) for p in paths)
    require_frozen_boundary(freeze)
    producer = freeze["producer"]
    if producer["commit"] != PRODUCER or producer["source_hash"] != PRODUCER_SOURCE:
        raise ValueError("Original331 full54 training producer required")
    unchanged_producer(producer, code)
    for source in (freeze["code"], query["code"]):
        if source["commit"] != PRODUCER or source["source_hash"] != PRODUCER_SOURCE:
            raise ValueError("Original331 ordered freeze and query consumers required")
        unchanged_producer(producer, source)
    if (query["status"] != "passed" or query["policy"] != QUERY_POLICY or query["freeze_sha256"] != freeze_sha256
            or query["query_labels_accessed"] is not True or query["optimizer_updates"] != 0):
        raise ValueError("Completed pinned all54 ordered query required")
    root = Path(freeze["training_root"]).resolve(strict=True)
    launch_path = root / "launch.json"; pin(launch_path, freeze["launch_sha256"]); check_files(pins)
    launch = read(launch_path)
    registration = read_registration(launch["registration_path"], launch["registration_sha256"], producer)
    if (registration["experiment_issue"] != 184 or registration["analysis_plan"]["sha256"] != PLAN_SHA256
            or Path(registration["out_dir"]).resolve() != root or launch["code"] != producer):
        raise ValueError("Original Issue184 preregistration and analysis plan required")
    pin(launch["registration_path"], launch["registration_sha256"])
    pin(registration["analysis_plan"]["path"], PLAN_SHA256)
    pin(registration["engineering_acceptance"]["path"], registration["engineering_acceptance"]["sha256"])
    if (review["status"] != "passed" or review["policy"] != REVIEW_POLICY or review["producer"] != producer
            or review["code"]["commit"] != REVIEW_PRODUCER or review["query_labels_accessed"] is not False
            or review["test_scoring"] is not False or review["new_optimizer_updates"] != 0
            or review["optimizer_updates_reviewed"] != 81000
            or review["launch_sha256"] != freeze["launch_sha256"]
            or review["acceptance_sha256"] != freeze["acceptance_sha256"]
            or review["audit_sha256"] != freeze["independent_audit_sha256"]
            or len(review["cases"]) != 54 or {r["name"] for r in review["cases"]} != {case_name(*k) for k in grid()}
            or any(r["updates"] != 1500 or r["validation_events"] != 51 for r in review["cases"])):
        raise ValueError("Complete independent54 train/validation review required")
    unchanged_producer(producer, review["code"])
    expected_exports = {case_name(*k) + suffix for k in grid() for suffix in (".png", "-curve.csv")}
    if set(review["exported_files"]) != expected_exports:
        raise ValueError("All54 review curves and figures required")
    for name, sha in review["exported_files"].items():
        pin(paths[2].parent / name, sha)
    analysis = analyze_rows(query["models"])
    cost_fields = ("controller_seconds_excluding_initial_freeze", "independent_audit_seconds",
                   "sum_case_training_resume_wait_and_acceptance_seconds", "report_elapsed_seconds")
    if (any(type(review[k]) not in (int, float) or not math.isfinite(review[k]) or review[k] < 0 for k in cost_fields)
            or type(query["elapsed_seconds"]) not in (int, float) or not math.isfinite(query["elapsed_seconds"])
            or query["elapsed_seconds"] < 0):
        raise ValueError("Finite nonnegative separately scoped stage costs required")
    analysis["recorded_cost_seconds"] = {k: review[k] for k in cost_fields}
    analysis["recorded_cost_seconds"]["ordered_query"] = query["elapsed_seconds"]
    analysis["training_budgets"] = {"source": 27054, "short_target": 216, "current_target": 81000}
    analysis["cost_limitations"] = "Resource waiting is included; sums of case costs are not parallel elapsed wall time or controlled speed comparisons."
    if digest(query["paired_summary"]) != digest(paired_summary(query["models"])):
        raise ValueError("Original authenticated descriptive query summary differs")
    frozen_models = {r["name"]: r for r in freeze["models"]}
    for row in query["models"]:
        expected = frozen_models[row["name"]]
        if (row["checkpoint_sha256"] != expected["primary_sha256"]
                or any(row[k] != expected[k] for k in ("target", "seed", "method", "mode"))):
            raise ValueError("Query row/checkpoint differs from frozen54 role")
        path = paths[0].parent / (row["name"] + ".json")
        if digest(read(path)) != digest(row):
            raise ValueError("Per-model query record differs from receipt")
        pin(path, file_hash(path))
    for path, sha in freeze["protected_files"].items():
        pin(path, sha)
    out = new_output(external_directory(out), launch, tuple(p.parent for p in paths))
    if out.exists():
        raise ValueError("New analysis output required")
    check_files(pins)
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    write_json(out / "analysis.json", analysis)
    (out / "report.md").write_text(report_markdown(analysis), encoding="utf-8")
    require_frozen_boundary(freeze); check_files(pins); require_frozen_code(code)
    receipt = {"status": "passed", "policy": POLICY, "code": code, "plan_sha256": PLAN_SHA256,
        "query_sha256": query_sha256, "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
        "models_analyzed": 54, "heads_analyzed": 162, "groups": 18, "new_optimizer_updates": 0,
        "raw_labels_deserialized": False, "model_inference_executed": False, "new_query_release": False,
        "input_query_metrics_consumed": True, "files_verified_unchanged": len(pins), "protected_files": pins,
        "outputs": {p.name: file_hash(p) for p in out.iterdir()}, "elapsed_seconds": time.monotonic() - started,
        "primary_benefit_supported": analysis["primary"]["passes_all_gates"], "significance_claim": False}
    write_json(out / "receipt.json", receipt)
    return receipt
