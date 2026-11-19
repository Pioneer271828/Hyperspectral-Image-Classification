"""Matched point exposure and validated support and spatial contrasts."""

from collections import defaultdict

from evaluation.frozen_training_stages import require
from model.research_config import digest
from transfer.development_bank import _gap

MODE = "one_released_point_per_class_per_visit_v1"
FACTORS = ("support_k", "support_seed", "spatial_layout")
ROLES = {"support": "devtrain", "calibration": "devval", "evaluation": "devval"}
SEEDS = (64, 512, 2048)


def inventory(bank, roles, k):
    custody = bank.release.custody
    require(k in (1, 5, 10) and custody.get("k") == k and custody.get("support_seed") in SEEDS,
            "Matched support must bind the release's K and label draw")
    classes = defaultdict(list)
    for tile in roles["support"]:
        labels = defaultdict(list)
        for pixel, label in bank.role_row("devtrain", tile)["samples"]:
            labels[label].append(pixel)
        require(labels and all(len(spots) == k for spots in labels.values()),
                "Each support tile must release exactly K points for its assigned classes")
        for label, spots in labels.items():
            classes[label].append((tile, spots))
    require(set(classes) == set(range(1, bank.release.classes+1))
            and all(len(rows) == 1 for rows in classes.values()),
            "Each class must belong to exactly one support tile for matched exposure")
    return {label: rows[0] for label, rows in sorted(classes.items())}


def choose(points, visit):
    positions = defaultdict(list)
    for i, (_, label) in enumerate(points):
        positions[label].append(i)
    return sorted(indices[visit % len(indices)] for indices in positions.values())


def role_rows(bank, roles):
    return {role: [bank.role_row(internal, tile) for tile in roles[role]]
            for role, internal in ROLES.items()}


def validate_contrast(factor, left, right, left_group, right_group, left_entry, right_entry):
    a, ar, ai = left
    b, br, bi = right
    data_a, data_b = a.release, b.release
    require((data_a.scene, data_a.bands, data_a.shape, data_a.classes)
            == (data_b.scene, data_b.bands, data_b.shape, data_b.classes)
            and {k: v for k, v in left_group.items() if k != "release"}
                == {k: v for k, v in right_group.items() if k != "release"}
            and {k:v for k,v in left_entry.items() if k not in {"manifest","authority","k"}}
                == {k:v for k,v in right_entry.items() if k not in {"manifest","authority","k"}},
            "Support contrasts require identical model seed, backend, clock, views and device")
    ca, cb = data_a.custody, data_b.custody
    rows_a, rows_b = role_rows(a, ar), role_rows(b, br)
    require([len(rows_a[role]) for role in ROLES] == [len(rows_b[role]) for role in ROLES],
            "Support contrasts require equal role block counts")
    if factor == "spatial_layout":
        gap = max(ca["minimum_buffer"], cb["minimum_buffer"])
        require(ca["k"] == cb["k"] and ca["support_seed"] == cb["support_seed"]
                and data_a.plan_hash != data_b.plan_hash
                and len(ar["support"]) == len(br["support"])
                and all(x["parent_tile"] != y["parent_tile"] and _gap(x["parent_box"], y["parent_box"]) >= gap
                        for role_a in rows_a.values() for role_b in rows_b.values()
                        for x in role_a for y in role_b)
                and [ar["support"].index(ai[c][0]) for c in ai] ==
                    [br["support"].index(bi[c][0]) for c in bi],
                "Spatial repeats must keep support parameters and use separated actual parent blocks")
    else:
        require(data_a.plan_hash == data_b.plan_hash and data_a.statistics == data_b.statistics
                and ar == br and rows_a["calibration"] == rows_b["calibration"]
                and rows_a["evaluation"] == rows_b["evaluation"]
                and [{k:v for k,v in row.items() if k!="samples"} for row in rows_a["support"]] ==
                    [{k:v for k,v in row.items() if k!="samples"} for row in rows_b["support"]]
                and all(ai[c][0] == bi[c][0] for c in ai),
                "Support contrasts must hold actual observations, calibration, evaluation and class locations fixed")
        if factor == "support_k":
            small, large = (ai, bi) if ca["k"] < cb["k"] else (bi, ai)
            require(ca["support_seed"] == cb["support_seed"] and ca["k"] != cb["k"]
                    and all(set(small[c][1]) < set(large[c][1]) for c in small),
                    "K contrasts require strict nesting within the same label draw")
        else:
            require(ca["k"] == cb["k"] and ca["support_seed"] != cb["support_seed"]
                    and any(ai[c][1] != bi[c][1] for c in ai),
                    "Support seeds must produce distinct actual K-point draws")
    return dict(factor=factor, baseline_support=digest(ai), comparison_support=digest(bi),
                baseline_layout=data_a.plan_hash, comparison_layout=data_b.plan_hash,
                same_evaluation_points=factor != "spatial_layout",
                baseline_k=ca["k"], comparison_k=cb["k"],
                baseline_support_seed=ca["support_seed"], comparison_support_seed=cb["support_seed"],
                model_seed=left_group["seed"], steps=left_group["steps"])


def exposure(history, classes):
    """Count actual supervised points in every completed registered update."""
    return [sum(row["released"]["per_class"][i] for row in history) for i in range(classes)]
