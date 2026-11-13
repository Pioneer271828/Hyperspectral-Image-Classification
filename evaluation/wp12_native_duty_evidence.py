"""Accept separate engineering and read-only WP06/WP07 records."""
from pathlib import Path

from research_runtime import hash_file
from evaluation.wp12_p4_released_evidence import read, require


DUTIES = "wp07_native_released_duties_v1"
ARMS = {"off", "fixed", "mean", "content", "metadata", "joint", "cnn_content", "graph_content", "constant_joint"}
SEEDS = (64, 512, 2048)
SCENES = {"xuzhou": 9, "whu-honghu": 22}


def case_ids():
    choices = [(1, "off"), *((f, arm) for f in (2, 4) for arm in ARMS)]
    return {f"{scene}/{seed}/{factor}/{arm}" for scene in SCENES for seed in SEEDS for factor, arm in choices}


def duties(prepared, audit, root, original):
    plan = read(prepared, root)
    preflight = read(plan["preflight"], root)
    require(plan["schema"] == "wp07_native_released_comparison_matrix_v1"
            and plan["design"] == preflight["design"] == DUTIES and plan["synthetic"] is False
            and plan["optimization_updates"] == 0 and plan["optimizer_updates"] == 11058
            and plan["authorization"] == "separate_release_required" and plan["query_authorized"] is False
            and preflight["origin"] == original and preflight["status"] == "preflight_only"
            and preflight["optimization_updates"] == 0 and preflight["raw_gt_loaded"] is preflight["query_accessed"] is False
            and len(plan["cases"]) == 114 and {c["id"] for c in plan["cases"]} == case_ids(),
            "Expected the separate, complete, zero-update real duties matrix")
    for scene, item in preflight["scenes"].items():
        row = read(item, root)
        inventory = preflight["inventory"][scene]
        require(row["scene"] == scene and row["draft"]["authorized_for_training"] is False
                and inventory["train_support_per_class"] == [5]*SCENES[scene]
                and inventory["point_exposures_per_case"] == [10]*SCENES[scene]
                and set(inventory["parameters"]) == ARMS and inventory["draws"] == 57
                and inventory["parameters"]["joint"] == inventory["parameters"]["constant_joint"],
                "Released duties do not preserve train support and same capacity")
    report = read(audit, root)
    matrix = read(report["matrix"], root)
    scope = read(report["release"], root)
    require(report["schema"] == "wp07_native_released_comparison_audit_v1"
            and report["status"] == "passed" and report["synthetic"] is matrix["synthetic"] is scope["synthetic"] is True
            and report["optimizer_updates"] == matrix["optimizer_updates"] == scope["optimizer_updates"] == 1368
            and report["audit_optimizer_updates"] == 0 and report["query_authorized"] is scope["query_authorized"] is False
            and matrix["design"] == DUTIES and scope["matrix"] == report["matrix"]
            and report["producer"] == matrix["controller"]["commit"] == plan["controller"]["commit"]
            and len(report["cases"]) == 114 and {c["case"] for c in report["cases"]} == case_ids(),
            "Expected qualified synthetic duties audit from the registered worker")
    for row in report["cases"]:
        classes = SCENES[row["case"].split("/")[0]]
        require(row["optimizer_updates"] == 12 and row["completed_resume_updates"] == 0
                and len(row["released_train_exposures"]) == classes and set(row["head_metrics"]) == {"fused", "cnn", "graph"}
                and row["selected_step"] in (0, 6, 12)
                and all(len(metrics["per_class"]) == classes for by_band in row["head_metrics"].values()
                        for metrics in by_band.values()), "Missing duty support, schedule or head measures")
        for item in row["stages"]:
            stage = read(item, root)
            path = Path(stage["checkpoint"]["path"]).resolve(strict=True)
            require(root in path.parents and hash_file(path) == stage["checkpoint"]["sha256"],
                    "Synthetic duty checkpoint bytes changed")
    read_path = Path(report["selected_devval_table"]["path"]).resolve(strict=True)
    require(root in read_path.parents and hash_file(read_path) == report["selected_devval_table"]["sha256"],
            "Synthetic duty metrics bytes changed")
    return {"real": {"scope": "prepare_only", "matrix": prepared, "worker": plan["controller"],
                     "planned_updates": 11058, "actual_updates": 0, "case_count": 114, "cost": None, "gain": None,
                     "paviaU": preflight["paviaU"], "inventory": preflight["inventory"]},
            "synthetic": {"scope": "engineering", "audit": audit, "worker": matrix["controller"],
                          "optimizer_updates": 1368, "case_count": 114, "completed_resume_updates": 0,
                          "audit_optimizer_updates": 0, "selected_metrics": report["selected_devval_table"],
                          "audit_wall_seconds": report["elapsed_seconds"]}}


def qualification(item, root, scientific_worker):
    row = read(item, root)
    wait, resumed, done, fail = (row[k] for k in ("resource_wait", "continuation", "completed", "failure"))
    require(row["schema"] == "wp07_native_dual_slot_cpu_qualification_v1" and row["status"] == "passed"
            and row["worker"] == scientific_worker and row["synthetic"] is True and row["physical_gpus"] == 0
            and row["query_accessed"] is row["raw_gt_loaded"] is False
            and row["completed_resume_updates"] == 0 and row["real_scientific_cost"] is None
            and (wait["status"], resumed["status"], done["status"], fail["status"]) ==
                ("waiting", "completed", "completed", "failed")
            and wait["new_updates"] == resumed["new_updates"] == 12 and done["new_updates"] == 0
            and row["actual_optimizer_updates"] == 48 + fail["new_updates"]
            and len(row["paired_states"]) == 2
            and all(v["model_adam_rng_history_selected_exact"] is True for v in row["paired_states"].values()),
            "Wrong qualification scope or subprocess evidence")
    for entry in row["paired_states"].values():
        for key in ("subprocess_completed", "direct_completed"):
            require(read(entry[key], root)["updates"] == 0, "Subprocess recovery consumed new updates")
    return {"scope": "cpu_subprocess_engineering", "receipt": item, "worker": row["worker"],
            "controller": row["controller"], "physical_gpus_qualified": 0,
            "optimizer_updates": row["actual_optimizer_updates"], "completed_resume_updates": 0, "real_cost": None}


def intervention(item, root, audit, original):
    row = read(item, root)
    endpoints = row["synth_trained"]
    require(row["schema"] == "wp06_released_graph_representation_evidence_v1" and row["status"] == "passed"
            and row["frozen_worker_audit"] == audit and row["real_preflight"] == original
            and row["real_optimizer_updates"] == row["completed_resume_updates"] == 0
            and row["real_effect_measured"] is row["raw_gt_loaded"] is row["query_accessed"] is False
            and row["real_scientific_cost"] is None and row["synth_optimizer_updates"] == 48
            and len(endpoints) == 12 and {v["case"] for v in endpoints} ==
                {case for case in case_ids() if case.endswith("/joint")}
            and {r["classes"] for r in row["synth_recovery"]} == {9, 22},
            "Wrong frozen, synthetic recovery or real-observability intervention scope")
    for endpoint in endpoints:
        require(endpoint["advance_updates"] == 0 and len({v["event"]["band"] for v in endpoint["views"]}) == 2
                and all(v["frozen_state_unchanged"] is True for v in endpoint["views"]),
                "Intervention changed a frozen endpoint or lost a released band")
    donors = row["synth_donor_endpoints"]
    frozen = read(row["synthetic_donor_source"], root)
    require(frozen["schema"] == "wp07_native_released_synthetic_acceptance_v1"
            and frozen["scope"] == "synthetic_engineering" and frozen["status"] == "passed"
            and frozen["raw_gt_loaded"] is frozen["query_accessed"] is False,
            "Donor source must be synthetic")
    require(len(donors) == 6 and {entry["case"] for entry in donors} ==
            {f"{classes}-{seed}-f2-joint" for classes in SCENES.values() for seed in SEEDS},
            "Missing cross-seed donor endpoints")
    for endpoint in donors:
        case = frozen["cases"][endpoint["case"]]
        require(endpoint["optimizer_updates"] == 0 and endpoint["checkpoint"] == case["stages"][-1]["checkpoint"]
                and len({view["event"]["band"] for view in endpoint["views"]}) == 2
                and all(view["frozen_state_unchanged"] and view["arms"]["donor"]["status"] == "measured"
                        for view in endpoint["views"]), "Donor evidence changed the frozen model or scope")
    for group in row["synth_recovery"]:
        require(set(group["arms"]) == {"baseline", "zero"} and all(a["exact_full_split_complete"] and
                [s["updates"] for s in a["stages"]] == [6, 2, 4, 0] for a in group["arms"].values()),
                "Fixed intervention recovery violated the optimizer budget")
    coverage = {}
    for scene, classes in SCENES.items():
        real = row["real_coverage"][scene]
        require(real["optimization_updates"] == 0 and real["frozen_endpoint_status"] ==
                "compatible_real_trained_endpoint_unavailable", "Coverage cannot claim fitted real performance")
        coverage[scene] = {}
        for role, group in real["roles"].items():
            coverage[scene][role] = group["totals"]
            require(all(sum(col.values()) == 5*classes for col in group["totals"].values()),
                    "Real boundary summary lost released support")
    return {"receipt": item, "producer": row["producer"], "source_hash": row["source_hash"],
            "real": {"scope": "observability_only", "updates": 0, "accuracy": None, "cost": None, "coverage": coverage},
            "synthetic": {"scope": "frozen_fixture_and_bounded_recovery", "endpoints": len(endpoints),
                          "updates": row["synth_optimizer_updates"], "completed_updates": 0,
                          "frozen_donor_endpoints": len(donors)}}


def p4_fit(item, root, resource, release, p4):
    row = read(item, root)
    require(row["schema"] == "wp05_p4_fit_sufficiency_v1" and row["resource"] == resource and row["release"] == release
            and row["new_optimizer_updates"] == 0 and row["raw_gt_loaded"] is row["query_accessed"] is False
            and row["planned_cases"] == 20 and row["complete_cases"] == len(row["cases"])
            and 0 < row["complete_cases"] <= p4["reviewed_cases"] and row["actual_scope_updates"] <= p4["optimizer_updates"]
            and row["validation_steps"] == [30, 60] and all(case["actual_optimizer_updates_on_one_path"] == 60
                and set(case["selected"]) == {"fused", "cnn", "graph"} and len(case["class_to_column"]) == 22
                and len(case["events"]) == 2 for case in row["cases"]), "P4 diagnostic exceeds committed real evidence")
    return {"scope": "selected_checkpoint_diagnostic", "receipt": item, "producer": row["producer"],
            "observed_cases": row["complete_cases"], "committed_updates_at_diagnosis": row["actual_scope_updates"],
            "new_updates": 0, "full_run_effect": None}
