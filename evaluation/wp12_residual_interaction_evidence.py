"""Qualify WP04 historical controls and independently tabulate residual and direct-message results."""
import csv
from pathlib import Path

import torch

from evaluation.wp12_edge_weight_evidence import _comparison, _open, _point_metrics, _qualified_phases, _recompute
from g4_support_context import context_name
from joint_training import _tree_fingerprint
from research_runtime import digest, tensor_hash
from scripts.audit_wp04_message_correspondence import scores
from scripts.run_wp04_factorial import (
    ASSIGNMENT_GROUP, ASSIGNMENT_SCOPE, HEADS, I7_GROUP, I7_SCOPE, REAL_CATALOG_SHA, REMAINING,
    SEEDS, STEPS, identity, paired_initialization, pinned, require, roster, terminal, verify,
)
from target_initialization import METHODS
from transfer.p4_runtime_device import CUDA_FP64_WEIGHT


HISTORICAL = {
    "wp04-residual-fusion-real-20261126": ("registration.json",
        "2dbd07d0aa8a2d56613da8ac64f4c4b941dd6b566bbdc9c799ebcb5584c6edd6", {"B0": "B0", "R": "R"}),
    "wp04-real-c2g-pairs-20261126": ("execution/evidence.json",
        "955c50b81d729ea0a811789eac78cde59705b446530c1e68f0ddff8ebfdc295f", {"off": "I0"}),
    "wp04-direction-capacity-20261126": ("execution/evidence.json",
        "599ba1ae59489ce93bd1f25c989cbc54c289825ce0dd7ea402b34a3b68df138b", {"add": "I1", "both": "I2"}),
    "wp04-edge-weights-real-v2-20261126": ("execution/evidence.json",
        "c4c166aa0c2b7c9ea41eaa1dde888d8e8da131feb87e117ca55b2bbbbb007159", {"Wj": "I6"}),
    "wp04-message-correspondence-real-20261126": ("execution/evidence.json",
        "e98c2d65dc6028932a3f51e1ef1d99ecdf1f673877ccff12666524cbaa39418c", {"I8": "I8"}),
}
EXPECTED_PARAMS = {"B0": 61696, "R": 69952, "I0": 61696, "I1": 94976, "I2": 120194,
                   "I6": 78592, "I8": 137090}
PAIR_COMPARISONS = {
    "R3-B0": ("R3", "B0"), "R-R3": ("R", "R3"), "R-XRegion": ("R", "XRegion"),
    "XRegion-B0": ("XRegion", "B0"), "R-B0": ("R", "B0"),
    "GridR-Grid0": ("GridR", "Grid0"), "LearnR-Learn0": ("LearnR", "Learn0"),
    "I7-I6": ("I7", "I6"), "I8-I7": ("I8", "I7"),
}
PRIMARY_PAIRS = {"R3-B0", "R-XRegion", "GridR-Grid0", "LearnR-Learn0", "I7-I6", "I8-I7"}
FACTORIALS = {
    "(LearnR-Learn0)-(GridR-Grid0)": {"LearnR": 1, "Learn0": -1, "GridR": -1, "Grid0": 1},
    "(LearnR-Learn0)-(R-B0)": {"LearnR": 1, "Learn0": -1, "R": -1, "B0": 1},
    "(I7-I6)-(I1-I0)": {"I7": 1, "I6": -1, "I1": -1, "I0": 1},
    "(I8-I6)-(I2-I0)": {"I8": 1, "I6": -1, "I2": -1, "I0": 1},
}


def published_roles(reg, root):
    catalogue = _open(reg["published_catalog"], root)
    row = next(row for row in catalogue["contexts"] if row["name"] == context_name("xuzhou", 512, 1))
    path = Path(reg["published_catalog"]["path"]).parent / row["file"]
    require(reg["published_catalog"]["sha256"] == REAL_CATALOG_SHA and path.is_relative_to(root),
            "Only the established published observation catalogue is valid")
    pin = {"path": str(path), "sha256": row["sha256"]}
    overlay = torch.load(verify(pin), map_location="cpu", weights_only=True)
    keys = ("train_indices", "train_labels", "val_indices", "val_labels")
    require(all(name in overlay for name in keys) and overlay["train_labels"].numel() == 9
            and overlay["val_labels"].numel() == 45
            and not (set(overlay["train_indices"].tolist()) & set(overlay["val_indices"].tolist()))
            and torch.equal(torch.bincount(overlay["val_labels"]), torch.full((9,), 5)),
            "The original development role denominators or separation changed")
    return pin, {key: tensor_hash(overlay[key]) for key in keys}, overlay["val_labels"]


def contract(result, seed):
    options = result["settings"]
    cfg = options["config"]
    require(result["status"] == "completed" and result["optimizer_steps"] == 12
            and result["test_scoring"] is False and options["target"] == "xuzhou"
            and options["method"] == METHODS[0] and options["model_seed"] == seed
            and options["head_seed"] == seed + 1000 and options["support_seed"] == 512
            and options["support_per_class"] == 1 and options["validation_per_class"] == 5,
            "Supervisor, role, seed or terminal budget differs")
    return {"training": cfg["training"], "loss": cfg["loss"], "data": cfg["data"],
            "augmentation": cfg["augmentation"], "scheduler": options["scheduler"],
            "schedule_sha256": result["schedule_sha256"], "validation_policy": options["validation_policy"]}


def _source_case(campaign, evidence, seed, arm):
    path = campaign / "execution" / str(seed) / f"{seed}-{arm}" / "audit.json"
    pin = pinned(path)
    if "case_pins" in evidence:
        require(pin in evidence["case_pins"] or pin in evidence.get("reused_case_pins", []),
                "Historical row is not pinned by its authoritative evidence")
    else:
        require({"seed": seed, "arm": arm, "mode": "none" if arm == "B0" else "x_residual", "fusion": "fixed"}
                in evidence["cases"], "Historical residual row is absent from its original registration")
    return pin


def historical(root, publication_pin):
    role_pin, roles, truth = published_roles({"published_catalog": publication_pin}, root)
    evidence_pins, cases = [], []
    for folder, (name, sha, mapping) in HISTORICAL.items():
        campaign = root / folder
        authority = {"path": str(campaign / name), "sha256": sha}
        source = _open(authority, root)
        reg = _open(source["registration"], root) if "registration" in source else source
        require(reg["published_catalog"] == publication_pin and reg["phase_updates"] == STEPS
                and reg["arithmetic"] == CUDA_FP64_WEIGHT and reg["query_access_authorized"] is False,
                "Historical catalogue, CUDA arithmetic or role permission differs")
        evidence_pins.append(authority)
        for seed in SEEDS:
            for arm, canonical in mapping.items():
                receipt = _source_case(campaign, source, seed, arm)
                row = _open(receipt, root)
                uuid = reg["devices"][str(seed)]
                _qualified_phases(row, uuid)
                require(row["case"]["seed"] == seed and row["case"]["arm"] == arm
                        and row["init"]["parameters"] == EXPECTED_PARAMS[canonical]
                        and all(row["inputs"][k] == v for k, v in roles.items())
                        and row["steps"] == {"selected": 12, "last": 12},
                        "Historical paired inputs, capacity or endpoints differ")
                tree = Path(receipt["path"]).parent / "split"
                result_pin = pinned(tree / "result.json")
                result = _open(result_pin, root)
                require(identity(result["start_code"]) == reg["code"] and result["selected_step"] == 12,
                        "Historical checkpoint producer or first-min endpoint differs")
                settings = contract(result, seed)
                output_pin = {"path": str(tree / "selected-probabilities.pt"),
                              "sha256": result["artifacts"]["selected-probabilities.pt"]}
                values = torch.load(verify(output_pin), map_location="cpu", weights_only=True)
                progress = terminal(tree)
                whole = terminal(tree.parent / "whole")
                require(progress["best_step"] == 12
                        and min(progress["history"], key=lambda r: r["val_loss"])["step"] == 12
                        and all(_tree_fingerprint(whole[k]) == _tree_fingerprint(progress[k])
                                for k in ("model", "optimizer", "scheduler", "history", "trace", "rng",
                                          "best_model", "best_probabilities")),
                        "Original historical selection or exact terminal model equality differs")
                endpoints = {}
                for head in HEADS:
                    value = values["xuzhou"]["val"][head]
                    materialized = scores(value, truth)
                    metrics = _point_metrics(materialized["points"])
                    refs = [row["endpoints"][e][head] for e in ("selected", "last")]
                    require(all(ref["probability_sha256"] == materialized["probability_sha256"]
                                and abs(ref["oa"] - metrics["oa"]) < 1e-12
                                and abs(ref["aa"] - metrics["aa"]) < 1e-12
                                and abs(ref["nll"] - metrics["nll"]) < 1e-6
                                and [c["correct"] for c in ref["classes"]] == metrics["correct_by_class"]
                                for ref in refs), "Historical selected/last point view differs")
                    endpoints[head] = {**metrics, "probability_sha256": materialized["probability_sha256"]}
                cases.append({"seed": seed, "arm": canonical, "origin": receipt, "result": result_pin,
                              "probabilities": output_pin, "producer": reg["code"], "contract": settings,
                              "input": row["inputs"], "shared": row["init"]["shared"], "initial": row["init"],
                              "device": uuid, "phases": row["phases"], "endpoints": {"selected": endpoints, "last": endpoints}})
    require(len(cases) == len(EXPECTED_PARAMS) * len(SEEDS), "Incomplete independent historical roles")
    return {"schema": "wp04_frozen_remaining_baselines_v1", "evidence": evidence_pins, "release": publication_pin,
            "role_overlay": role_pin, "supervision": roles, "query_access": False, "updates": 0, "cases": cases}


def _new_campaign(root, pin, group):
    data = _open(pin, root)
    reg = _open(data["registration"], root)
    scope, parameters, graph = REMAINING[group]
    expected = roster(group)
    research = len(expected) * STEPS["whole"]
    require(data["scope"] == reg["scope"] == scope and reg["group"] == group and reg["graph"] == graph
            and reg["cases"] == expected and reg["parameters_by_arm"] == parameters
            and data["research_updates"] == research and data["real_updates"] == reg["real_updates_authorized"] == 2 * research
            and data["code"] == reg["code"] and data["devices"] == reg["devices"]
            and data["query_access"] is reg["query_access_authorized"] is False
            and not any(data["lane_exit_codes"].values()) and reg["published_catalog"]["sha256"] == REAL_CATALOG_SHA,
            "Inadmissible new residual or direct-edge evidence")
    raw = [_open(case_pin, root) for case_pin in data["case_pins"]]
    require([row["case"] for row in raw] == expected, "Wrong new seed/arm roster")
    paired_initialization(raw, scope)
    counts = {}
    for table in data["tables"]:
        path = verify(table)
        require(path.is_relative_to(root) and path.stem not in counts, "Duplicate or unauthorized table")
        with path.open(encoding="utf8", newline="") as stream:
            counts[path.stem] = sum(1 for _ in csv.DictReader(stream))
    n = len(expected)
    sizes = {"endpoints": n * 6, "per-class": n * 54, "costs": n * 4}
    if group == ASSIGNMENT_GROUP:
        sizes.update({"paired": 54, "paired-per-class": 486, "correct-point-transitions": 324})
    require(counts == sizes, "New evidence tables do not cover the registered matrix")
    results = []
    for row, case_pin in zip(raw, data["case_pins"], strict=True):
        seed = row["case"]["seed"]
        uuid = data["devices"][str(seed)]
        _qualified_phases(row, uuid)
        result_path = Path(case_pin["path"]).parent / "split/result.json"
        result_pin = pinned(result_path)
        result = _open(result_pin, root)
        require(identity(result["start_code"]) == data["code"] and result["selected_step"] == row["steps"]["selected"],
                "New endpoint or producer differs")
        points = {e: {h: _recompute(row["endpoints"][e][h]) for h in HEADS} for e in ("selected", "last")}
        results.append({"seed": seed, "arm": row["case"]["arm"], "input": row["inputs"], "device": uuid,
                        "shared": row["init"]["shared"], "initial": row["init"], "origin": case_pin,
                        "contract": contract(result, seed), "endpoints": points, "phases": row["phases"],
                        "graph": row["diagnostics"], "components": row["learned_components"],
                        "steps": row["steps"]})
    return reg, data, counts, results


def _factorial(cases, weights):
    result = {}
    for end in ("selected", "last"):
        result[end] = {}
        for head in HEADS:
            observations = {}
            for seed in SEEDS:
                rows = [(cases[seed, arm]["endpoints"][end][head], weight) for arm, weight in weights.items()]
                class_delta = [sum(weight * sum(p["predicted"] == cls for p in row["points"] if p["class_id"] == cls)
                                   for row, weight in rows) for cls in range(1, 10)]
                observations[str(seed)] = {"delta_oa": sum(class_delta) / 45, "delta_aa": sum(class_delta) / 45,
                                           "delta_nll": sum(w * x["nll"] for x, w in rows),
                                           "correct_delta_by_class": class_delta}
            result[end][head] = observations
    return result


def intake(root, residual_pin, joint_pin, history_pin):
    rreg, residual, rcounts, rrows = _new_campaign(root, residual_pin, ASSIGNMENT_GROUP)
    ireg, joint, icounts, irows = _new_campaign(root, joint_pin, I7_GROUP)
    require(rreg["published_catalog"] == ireg["published_catalog"], "Different published supervision")
    saved = _open(history_pin, root)
    checked = historical(root, rreg["published_catalog"])
    require(saved == checked, "Historical selected controls failed independent zero-update verification")
    rows = rrows + irows + checked["cases"]
    cases = {(row["seed"], row["arm"]): row for row in rows}
    require(len(cases) == len(rows), "A new arm duplicated a frozen case")
    for seed in SEEDS:
        cohort = [row for row in rows if row["seed"] == seed]
        require(len({digest(row["input"]) for row in cohort}) == len({row["shared"] for row in cohort})
                == len({digest(row["contract"]) for row in cohort}) == 1,
                "Paired seeds, labels, shared initialization or optimizer/validation contract changed")
        require(all(cases[seed, a]["device"] == rreg["devices"][str(seed)] for a in ("B0", "R", "R3", "XRegion"))
                and all(cases[seed, a]["device"] == ireg["devices"][str(seed)] for a in ("I6", "I7", "I8"))
                and cases[seed, "R"]["initial"]["residual"] == cases[seed, "XRegion"]["initial"]["residual"]
                and all(cases[seed, "I7"]["graph"][e][k]
                        == _open(cases[seed, a]["origin"], root)["diagnostics"][e][k]
                        for a in ("I6", "I8") for e in ("initial", "selected", "last")
                        for k in ("support_sha256", "prior_sha256")),
                "Same-card residual/I7 pairing, initial projection or fixed edge prior differs")
    endpoints = {key: row["endpoints"] for key, row in cases.items()}
    pairs = {}
    for name, (a, b) in PAIR_COMPARISONS.items():
        pair = _comparison(endpoints, a, b, primary=name in PRIMARY_PAIRS)
        # Resolve near-zero means from integer paired counts.
        selected = pair["endpoints"]["selected"]["fused"]
        net = sum(r["after"] - r["before"] for item in selected.values() for r in item["classes"])
        pair["mean_delta"]["oa"] = pair["mean_delta"]["aa"] = net / (45 * len(SEEDS))
        pair["criterion_passed"] = net > 0 and pair["positive_seeds"] >= 2 and not pair["failed_classes"]
        pairs[name] = pair
    factorials = {}
    for name, design in FACTORIALS.items():
        factorials[name] = {"effects": _factorial(cases, design),
            "devices": {str(seed): {arm: cases[seed, arm]["device"] for arm in design} for seed in SEEDS}}
    return {"schema": "wp12_wp04_residual_assignment_and_direct_joint_v1",
            "qualification": "paired_released_development_12_steps", "query_access": False,
            "production_accuracy_qualified": False, "history": history_pin, "sources": [residual_pin, joint_pin],
            "new_cases": len(rrows + irows), "reused_cases": len(checked["cases"]),
            "research_updates": residual["research_updates"] + joint["research_updates"],
            "real_updates": residual["real_updates"] + joint["real_updates"], "historical_new_updates": 0,
            "table_counts": {ASSIGNMENT_SCOPE: rcounts, I7_SCOPE: icounts},
            "pairs": pairs, "factorials": factorials,
            "cases": [{k: v for k, v in row.items() if k not in ("contract", "initial")} for row in rows]}


def write_paper_tables(summary, out):
    comparisons = [(name, row["endpoints"], row["primary"], row["criterion_passed"])
                   for name, row in summary["pairs"].items()]
    comparisons.extend((name, row["effects"], False, None) for name, row in summary["factorials"].items())
    points, classes = [], []
    for name, endpoints, primary, passed in comparisons:
        for endpoint, heads in endpoints.items():
            for head, seeds in heads.items():
                for seed, effect in seeds.items():
                    index = {"comparison": name, "endpoint": endpoint, "head": head, "seed": seed}
                    points.append({**index, "oa_difference_pp": 100 * effect["delta_oa"],
                                   "aa_difference_pp": 100 * effect["delta_aa"],
                                   "nll_difference": effect["delta_nll"], "primary": primary,
                                   "passed_primary_gate": passed if primary else None})
                    if "classes" in effect:
                        class_rows = [(x["class_id"], x["after"] - x["before"], x["lost"], x["gained"])
                                      for x in effect["classes"]]
                    else:
                        class_rows = [(i, n, None, None) for i, n in enumerate(effect["correct_delta_by_class"], 1)]
                    for label, delta, lost, gained in class_rows:
                        classes.append({**index, "class_id": label, "support_per_arm": 5,
                                        "correct_count_difference": delta, "lost": lost, "gained": gained})
    for filename, rows in (("paper-comparisons.csv", points), ("paper-classes.csv", classes)):
        with (out / filename).open("x", encoding="utf8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
