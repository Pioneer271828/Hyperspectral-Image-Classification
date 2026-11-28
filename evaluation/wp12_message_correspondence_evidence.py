"""Independently admit trained and frozen WP04 correspondence comparisons."""
import csv
from pathlib import Path

from evaluation.wp12_edge_weight_evidence import (
    _comparison, _open, _per_seed, _point_metrics, _qualified_phases, _recompute,
    intake as edge_weight_intake,
)
from scripts.audit_wp04_message_correspondence import PRIOR_JOINT, SHUFFLE_BASE
from scripts.run_wp04_factorial import (
    HEADS, MESSAGE_GROUP, MESSAGE_PARAMS, MESSAGE_SCOPE, REAL_CATALOG_SHA, SEEDS, STEPS,
    paired_initialization, require, roster, verify,
)
from transfer.p4_runtime_device import CUDA_FP64_WEIGHT


def intake(root, pin, frozen_pins, adapter):
    data = _open(pin, root)
    reg = _open(data["registration"], root)
    expected = roster(MESSAGE_GROUP)
    updates = len(expected) * sum(STEPS.values())
    require(reg["scope"] == data["scope"] == MESSAGE_SCOPE and reg["cases"] == expected
            and reg["group"] == MESSAGE_GROUP and reg["arithmetic"] == CUDA_FP64_WEIGHT
            and reg["published_catalog"]["sha256"] == REAL_CATALOG_SHA
            and reg["role_scope"] == "published_train_dev_only"
            and data["query_access"] is reg["query_access_authorized"] is False
            and data["code"] == reg["code"] and reg["real_updates_authorized"] == data["real_updates"] == updates
            and data["research_updates"] == len(expected) * STEPS["whole"]
            and not any(data["lane_exit_codes"].values()) and data["devices"] == reg["devices"]
            and reg["parameters_by_arm"] == MESSAGE_PARAMS
            and reg["primary_comparisons"] == ["I8-I9-message", "Wj-Wg65"], "Inadmissible correspondence registration")
    _open(reg["published_catalog"], root)
    rows = [_open(item, root) for item in data["case_pins"]]
    require([row["case"] for row in rows] == expected, "Wrong correspondence cases")
    paired_initialization(rows, MESSAGE_SCOPE)
    costs = {}
    by_case = {}
    by_key = {(row["case"]["seed"], row["case"]["arm"]): row for row in rows}
    for (seed, arm), row in by_key.items():
        _qualified_phases(row, data["devices"][str(seed)])
        operators = row["diagnostics"]["selected"]["edge_path"]
        costs.setdefault(str(seed), {})[arm] = {"stage2": operators, "phases": row["phases"]}
        by_case[seed, arm] = {endpoint: {head: _recompute(row["endpoints"][endpoint][head]) for head in HEADS}
                              for endpoint in ("selected", "last")}
    counts = {}
    for table in data["tables"]:
        path = Path(table["path"])
        require(path.is_relative_to(root) and path.stem not in counts, "Duplicate or outside-run table")
        with verify(table).open(encoding="utf-8", newline="") as stream:
            counts[path.stem] = sum(1 for _ in csv.DictReader(stream))
    require(counts == {"endpoints": 54, "per-class": 486, "paired": 18, "paired-per-class": 162,
                       "costs": 36, "correct-point-transitions": 162}, "Wrong correspondence CSV row count")

    frozen = [_open(item, root) for item in frozen_pins]
    require(tuple(row["seed"] for row in frozen) == SEEDS, "Missing or unordered frozen seed")
    rebound = frozen[0]["rebound_context"]
    require(_open(rebound, root)["code"]["commit"] == adapter, "Wrong replay context producer")
    prior_pin = frozen[0]["prior_edge"]
    require(prior_pin["sha256"] == PRIOR_JOINT, "Unexpected historical Wj source")
    edge_weight_intake(root, prior_pin)
    prior = _open(prior_pin, root)
    prior_reg = _open(prior["registration"], root)
    require(all(reg[key] == prior_reg[key] for key in ("published_catalog", "target", "support_seed", "per_class",
                                                       "method", "phase_updates", "selection", "role_scope",
                                                       "devices", "arithmetic")), "Historical scope differs")
    changes = {}
    for receipt in frozen:
        seed = receipt["seed"]
        uuid = data["devices"][str(seed)]
        i8 = by_key[seed, "I8"]
        wide = by_key[seed, "Wg65"]
        selected = i8["endpoints"]["selected"]
        support = i8["diagnostics"]["selected"]
        require(receipt["schema"] == "wp04_frozen_message_and_prior_joint_v1"
                and receipt["source"] == pin and receipt["producer"] == adapter and receipt["prior_edge"] == prior_pin
                and receipt["rebound_context"] == rebound
                and receipt["updates"] == 0 and receipt["query_access"] is False and receipt["device"] == uuid
                and receipt["shuffle_seed"] == SHUFFLE_BASE + seed and receipt["regions"] == support["nodes"]
                and 0 <= receipt["fixed_points"] < receipt["regions"]
                and receipt["selected_step"] == i8["steps"]["selected"]
                and all(receipt["original"][h][k] == selected[h][k] for h in HEADS for k in ("probability_sha256", "points")),
                "Frozen treatment lacks its original paired endpoint")
        compatibility = receipt["adapter_compatibility"]
        require(set(compatibility) == set(MESSAGE_PARAMS)
                and all(compatibility[arm][e][h][k] == by_key[seed, arm]["endpoints"][e][h][k]
                        for arm in MESSAGE_PARAMS for e in ("selected", "last")
                        for h in HEADS for k in ("probability_sha256", "points")),
                "The standalone adapter cannot replay the original trained endpoints exactly")
        summary = {}
        for treatment in ("shuffled", "global_mean"):
            observed = receipt[treatment]
            require(observed["support_sha256"] == support["support_sha256"] and observed["calls"] > 0,
                    "Frozen treatment changed candidate support")
            summary[treatment] = {h: _per_seed(_point_metrics(observed["heads"][h]["points"]), selected[h])
                                  for h in HEADS}
        changes[str(seed)] = summary

        old = receipt["prior_joint"]
        require(old["case"] in prior["case_pins"], "Historical case is absent from the authoritative archive")
        archived = _open(old["case"], root)
        require(archived["case"] == {"seed": seed, "arm": "Wj", "condition": "joint"}
                and old["input_sha256"] == archived["inputs"] == wide["inputs"]
                and old["parameters"] == archived["init"]["parameters"] == wide["init"]["parameters"]
                and archived["init"]["shared"] == wide["init"]["shared"]
                and all(archived["diagnostics"][e][k] == wide["diagnostics"][e][k]
                        for e in ("initial", "selected", "last") for k in ("support_sha256", "prior_sha256"))
                and all(old["endpoints"][e][h][k] == archived["endpoints"][e][h][k]
                        for e in ("selected", "last") for h in HEADS for k in ("probability_sha256", "points")),
                "The archived Wj and expanded Wg differ in paired identity or historical replay")
        _qualified_phases(archived, uuid)
        by_case[seed, "Wj"] = {e: {h: _recompute(archived["endpoints"][e][h]) for h in HEADS}
                               for e in ("selected", "last")}
        cost_fields = ("feature_channels", "embedding_channels", "projection_dense_macs", "edge_dot_macs")
        costs[str(seed)]["Wj"] = {"stage2": {field: old[field] for field in cost_fields}, "phases": archived["phases"]}

    comparisons = {f"{a}-{b}": _comparison(by_case, a, b, primary=True)
                   for a, b in (("I8", "I9-message"), ("Wj", "Wg65"))}
    return {"schema": "wp12_wp04_message_correspondence_v1", "source": pin, "registration": data["registration"],
            "producer": data["code"], "frozen_replay": frozen_pins, "rebound_context": rebound, "prior_joint_source": prior_pin,
            "scope": MESSAGE_SCOPE, "query_access": False, "production_accuracy_qualified": False,
            "optimizer_updates": updates, "research_updates": data["research_updates"],
            "completed_reentry_updates": 0, "frozen_updates": 0, "reused_joint_cases": len(SEEDS),
            "counts": counts, "exact_adapter_head_replays": len(expected) * 2 * len(HEADS),
            "comparisons": comparisons, "phase_and_operation_costs": costs,
            "frozen_message_comparisons": changes, "validation_points_per_class": 5}
