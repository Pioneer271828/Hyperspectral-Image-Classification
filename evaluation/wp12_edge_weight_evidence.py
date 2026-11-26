"""Import a strictly paired, fixed-candidate WP04 development result."""
from collections import Counter
import csv
from math import isclose, log, sqrt
from pathlib import Path
from statistics import mean

from scripts.run_wp04_factorial import (
    EDGE_WEIGHT_CONDITIONS, EDGE_WEIGHT_CONTRASTS, EDGE_WEIGHT_GROUP, EDGE_WEIGHT_PARAMS,
    EDGE_WEIGHT_PRIMARY, EDGE_WEIGHT_SCOPE, HEADS, REAL_CATALOG_SHA, SEEDS, STEPS,
    paired_initialization, require, roster, verify,
)
from g4_support_context import read
from transfer.p4_runtime_device import CUDA_FP64_WEIGHT


def _open(pin, root):
    path = Path(pin["path"])
    require(path.is_relative_to(root) and path != root, "Evidence path outside authorized runs")
    return read(verify(pin))


def _recompute(value):
    points = value["points"]
    expected = {(c, p) for c in range(1, 10) for p in range(1, 6)}
    require(len(points) == 45 and {(p["class_id"], p["point"]) for p in points} == expected,
            "Changed class/point denominators")
    counts = Counter()
    loss = []
    for point in points:
        label, prediction = point["class_id"], point["predicted"]
        probability, margin = point["true_probability"], point["margin"]
        require(1 <= prediction <= 9 and 0 <= probability <= 1 and -1 <= margin <= 1
                and (margin >= 0 if label == prediction else margin <= 0),
                "Invalid anonymous development prediction")
        counts[label, prediction] += 1
        loss.append(-log(max(probability, 1e-8)))
    confusion = [[counts[i, j] for j in range(1, 10)] for i in range(1, 10)]
    per_class = [counts[c, c] for c in range(1, 10)]
    oa = sum(per_class) / 45
    require(confusion == value["confusion"] and oa == value["oa"] and isclose(oa, value["aa"], abs_tol=1e-12)
            and isclose(mean(loss), value["nll"], abs_tol=1e-6)
            and [row["correct"] for row in value["classes"]] == per_class
            and [row["support"] for row in value["classes"]] == [5] * 9,
            "Independent point-derived metrics differ from the audit")
    require(value["correctness"] == [
        [p["predicted"] == c for p in points if p["class_id"] == c] for c in range(1, 10)
    ], "Paired correctness differs from predictions")
    return {**value, "confusion": confusion, "oa": oa, "aa": oa}


def _wilson(correct, n=5):
    z = 1.959963984540054
    rate = correct / n
    scale = 1 + z * z / n
    center = rate + z * z / (2 * n)
    spread = z * sqrt((rate * (1 - rate) + z * z / (4 * n)) / n)
    return [max(0., (center - spread) / scale), min(1., (center + spread) / scale)]


def _per_seed(left, right):
    rows = []
    for a, b in zip(left["points"], right["points"], strict=True):
        cls, point = a["class_id"], a["point"]
        require((cls, point) == (b["class_id"], b["point"]), "Different validation point ordering")
        rows.append(dict(class_id=cls, point=point, before=b["predicted"], after=a["predicted"],
                         before_margin=b["margin"], after_margin=a["margin"],
                         before_true_p=b["true_probability"], after_true_p=a["true_probability"]))
    changes = []
    for c in range(1, 10):
        cohort = [r for r in rows if r["class_id"] == c]
        before = sum(r["before"] == c for r in cohort)
        after = sum(r["after"] == c for r in cohort)
        changes.append(dict(class_id=c, support=len(cohort), before=before, after=after,
                            lost=sum(r["before"] == c and r["after"] != c for r in cohort),
                            gained=sum(r["before"] != c and r["after"] == c for r in cohort),
                            before_wilson_95=_wilson(before), after_wilson_95=_wilson(after),
                            margin_delta=mean(r["after_margin"]-r["before_margin"] for r in cohort)))
    return dict(delta_oa=left["oa"] - right["oa"], delta_aa=left["aa"] - right["aa"],
                delta_nll=left["nll"] - right["nll"], classes=changes,
                changed_points=[r for r in rows if r["before"] != r["after"]])


def intake(root, pin):
    data = _open(pin, root)
    reg = _open(data["registration"], root)
    cases = roster(EDGE_WEIGHT_GROUP)
    require(data["scope"] == reg["scope"] == EDGE_WEIGHT_SCOPE and reg["cases"] == cases
            and reg["published_catalog"]["sha256"] == REAL_CATALOG_SHA
            and reg["role_scope"] == "published_train_dev_only"
            and data["query_access"] is reg["query_access_authorized"] is False
            and data["code"] == reg["code"] and reg["real_updates_authorized"] == data["real_updates"] == 288
            and data["research_updates"] == 144 and not any(data["lane_exit_codes"].values())
            and data["devices"] == reg["devices"] and reg["arithmetic"] == CUDA_FP64_WEIGHT
            and reg["parameters_by_arm"] == EDGE_WEIGHT_PARAMS
            and tuple(reg["primary_comparisons"]) == EDGE_WEIGHT_PRIMARY, "Different edge-weight development registration")
    _open(reg["published_catalog"], root)
    rows = [_open(p, root) for p in data["case_pins"]]
    require([r["case"] for r in rows] == cases and len(data["devices"]) == len(SEEDS),
            "Incomplete WP04 evidence roster")
    by_case = {}
    for row, case in zip(rows, cases, strict=True):
        seed, arm = case["seed"], case["arm"]
        phases = row["phases"]
        uuid = data["devices"][str(seed)]
        require(row["exact_whole_split"] and row["exact_independent_step_one"]
                and row["resume_completed_updates"] == row["audit_updates"] == 0
                and list(phases) == list(STEPS) and all(phases[k]["updates"] == n for k, n in STEPS.items())
                and all(p["gpu"]["uuid"] == p["arithmetic"]["physical_uuid"] == uuid for p in phases.values())
                and all(phases[k]["kernel_calls"]["sparse_mm"] > 0 for k in ("whole", "prefix", "resume")),
                "Unqualified arithmetic or unpaired checkpoint recovery")
        by_case[seed, arm] = {e: {h: _recompute(row["endpoints"][e][h]) for h in HEADS}
                             for e in ("selected", "last")}
    paired_initialization(rows, EDGE_WEIGHT_SCOPE)

    sizes = {"endpoints": 72, "per-class": 648, "paired": 108, "paired-per-class": 972,
             "costs": 48, "correct-point-transitions": 972}
    counts = {}
    for table in data["tables"]:
        path = Path(table["path"])
        require(path.is_relative_to(root) and path.stem not in counts, "Duplicate or outside-run table")
        with verify(table).open(encoding="utf-8", newline="") as stream:
            counts[path.stem] = sum(1 for _ in csv.DictReader(stream))
    require(counts == sizes, "Incomplete WP04 evidence tables")

    comparisons = {}
    for a, b in EDGE_WEIGHT_CONTRASTS:
        name = f"{a}-{b}"
        endpoints = {}
        for endpoint in ("selected", "last"):
            endpoints[endpoint] = {
                head: {str(seed): _per_seed(by_case[seed, a][endpoint][head], by_case[seed, b][endpoint][head])
                       for seed in SEEDS} for head in HEADS
            }
        fused = list(endpoints["selected"]["fused"].values())
        oa, aa, nll = (mean(r[f"delta_{metric}"] for r in fused) for metric in ("oa", "aa", "nll"))
        recall = {c: mean((r["classes"][c-1]["after"] - r["classes"][c-1]["before"]) / 5 for r in fused)
                  for c in range(1, 10)}
        failures = [c for c, value in recall.items() if value < -.02]
        positive = sum(r["delta_oa"] > 0 for r in fused)
        comparisons[name] = dict(primary=name in EDGE_WEIGHT_PRIMARY,
                                 criterion_passed=oa > 0 and positive >= 2 and aa >= -.005 and not failures,
                                 mean_delta=dict(oa=oa, aa=aa, nll=nll), positive_seeds=positive,
                                 mean_class_recall_delta=recall, failed_classes=failures, endpoints=endpoints)
    return dict(schema="wp12_wp04_edge_weights_development_v1", source=pin, registration=data["registration"],
                producer=data["code"], scope=EDGE_WEIGHT_SCOPE, query_access=False,
                production_accuracy_qualified=False, optimizer_updates=288, research_updates=144,
                restore_replay_updates=144, completed_reentry_updates=0, counts=counts, comparisons=comparisons,
                parameters=EDGE_WEIGHT_PARAMS, devices=data["devices"],
                observed_arms=list(EDGE_WEIGHT_CONDITIONS), validation_points_per_class=5,
                uncertainty_scope="Wilson intervals on the original five development points; seeds share point identities")
