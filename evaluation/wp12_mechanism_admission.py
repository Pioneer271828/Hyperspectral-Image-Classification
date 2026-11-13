"""Scope-separated WP12 ingestion of crash safety and trained mechanism evidence."""
from pathlib import Path

from evaluation.wp12_p4_released_evidence import OLD, read, require
from research_runtime import hash_file
from transfer import registered_development as transport


SCHEMA = "wp12_mechanism_admission_v1"


def scoped_pin(pin, root):
    path = Path(pin["path"])
    require(path.resolve() == path and root in path.parents and hash_file(path) == pin["sha256"],
            "Pinned mechanism evidence changed or escaped the runs folder")
    return path


def attest(row, name):
    require(row["schema"] == name and row.get("raw_gt_loaded", False) is False
            and row.get("query_accessed", False) is False, "Unexpected evidence schema or data role")


def crash(item, root, registered):
    row = read(item, root)
    attest(row, "wp07_native_dual_slot_crash_qualification_v2")
    require(row["status"] == "passed" and row["worker"] == registered["science_worker"]
            and row["controller"] == registered["crash_controller"]
            and row["synthetic"] is True and row["physical_gpus"] == 0 and row["completed_resume_updates"] == 0
            and row["concurrent_exit"] == 75 and row["failure"]["status"] == "failed"
            and row["completed"]["new_updates"] == 0 and row["real_scientific_cost"] is None
            and row["failure"]["first_failure"] is not None, "Scope cannot promote CPU engineering or omit first failure")
    attempts = (row["resource_wait"], row["continuation"], row["completed"], row["failure"])
    counted = 0
    for record in attempts:
        attest(record, "wp07_native_released_dual_attempt_v2")
        attempt = read(record["attempt"], root)
        route = read(record["routing"], root)
        require(record["controller"] == row["controller"] and record["worker"] == row["worker"]
                and route["controller"] == row["controller"] and attempt["identity"]["routing"] == record["routing"]
                and route["synthetic"] is True and record["synthetic"] is True
                and record["physical_uuids"] == [], "Foreign attempt or physical slot evidence")
        cases = [stage for slot in record["slots"].values() for stage in slot["stages"]]
        accounted = 0
        receipt_count = 0
        for pin in record["operation_acks"]:
            ack = read(pin, root)
            lanc = read(ack["launch"], root)
            dispatch = read(lanc["dispatch"], root)
            require(dispatch["attempt"] == record["attempt"] and dispatch["physical_uuid"] is None
                    and ack["returncode"] in (0, 75), "Attempt includes a failed or foreign acknowledgement")
            if ack["returncode"] == 0:
                result = read(ack["receipt"], root)
                require(result["pid"] == lanc["pid"] and result["case"] == dispatch["case"],
                        "Missing committed subprocess receipt")
                phase = read(result["result"], root)
                if dispatch["phase"] != "audit":
                    scoped_pin(phase["checkpoint"], root)
                accounted += result["new_updates"]
                receipt_count += 1
            else:
                require(ack["receipt"] is None, "WAITING cannot claim a scientific stage")
        require(receipt_count == len(cases) and accounted == record["new_updates"] == sum(s["new_updates"] for s in cases),
                "Incomplete acknowledged attempt")
        counted += accounted
    if registered.get("real_calibration") is not None:
        require(row.get("alternate_journal_refused") is True,
                "The real mechanism supplement requires the final single-owner controller qualification")
    for key in ("coordinator_failure_window", "unacknowledged_success"):
        test = row[key]
        gate = read(test["gate"], root)
        attest(gate, "wp07_dual_first_failure_v2")
        require(test["new_updates_on_reentry"] == 0 and test["unchanged_science"] is True
                and gate["identity"]["controller"] == row["controller"] and "acknowledged" in gate["reason"],
                "Unknown coordinator exit must have a preserved refusal")
    gate = read(row["failure"]["first_failure"], root)
    require(gate["schema"] == "wp07_dual_first_failure_v2" and gate["returncode"] < 0
            and isinstance(gate["dispatch"], dict) and row["actual_optimizer_updates"] == counted+24,
            "Drained failure is unrelated to its original dispatch")
    read(gate["dispatch"], root)
    return {"scope": "cpu_crash_qualification", "receipt": item, "controller": row["controller"],
            "actual_committed_cpu_updates": row["actual_optimizer_updates"] + row["orphan_committed_optimizer_updates"],
            "physical_gpus_qualified": 0, "real_cost": None, "real_gain": None}


def learning(item, root, registered):
    row = read(item, root)
    attest(row, "wp06_synthetic_learned_channel_v1")
    expected = {(seed, arm) for seed in (64, 512, 2048) for arm in ("parallel", "bottleneck", "local", "shuffle")}
    require(row["scope"] == "synthetic_learning" and row["status"] == "measured"
            and row["producer"] == registered["learned_worker"]
            and row["real_updates"] == row["confirmation_optimizer_updates"] == 0
            and row["training_optimizer_updates"] == 2304
            and all(row[v] is None for v in ("real_hsi_gain", "cross_sensor_gain", "oversquashing_gain"))
            and len(row["cases"]) == 12 and {(x["seed"], x["arm"]) for x in row["cases"]} == expected,
            "Expected a sealed, bounded synthetic learning campaign")
    freeze = read(row["freeze"], root)
    claim = read(row["confirmation_claim"], root)
    reg = read(row["registration"], root)
    require(claim == {"registration": row["registration"], "freeze": row["freeze"]}
            and freeze.get("schema") == "wp06_learned_channel_frozen_v1"
            and freeze["registration"] == row["registration"] and freeze["confirmation_consumed"] is False
            and freeze["optimizer_updates"] == reg["design"]["budget"] == 2304
            and reg["producer"] == row["producer"] and reg["design"]["scope"] == "synthetic_learning"
            and set(reg["design"]["role_seeds"]) == {"train", "devval", "confirmation"}
            and reg["design"]["classes"] == [0, 1], "Confirmation lacked a frozen matched roster")
    if row["review_repair"] is not None:
        repair = row["review_repair"]
        require(repair["consumer"] == registered["learned_review"],
                "Only the declared numerical auditor repair may read the original confirmed states")
        log = scoped_pin(repair["failed_confirmation"], root)
        repair_pin = transport._pin(Path(row["freeze"]["path"]).parent / "confirmation.repair.json")
        require("Trained signal Jacobian differs" in log.read_text()
                and read(repair_pin, root) == {**repair, "claim": row["confirmation_claim"], "optimizer_updates": 0},
                "Missing unpublished confirmation failure or immutable repair receipt")
    for result, case in zip(row["cases"], freeze["cases"], strict=True):
        require((result["seed"], result["arm"], result["selected_step"], result["source"]) ==
                (case["seed"], case["arm"], case["selected_step"], case["selected_logits_source"])
                and case["complete_exact"] is True and set(case["stages"]) == {"full", "prefix", "resume", "completed"}
                and set(result["confirmation"]) == {"reachable", "capacity", "beyond", "disconnected"},
                "Frozen seed, selection or route differs")
        for phase, updates in (("full", 96), ("prefix", 17), ("resume", 79), ("completed", 0)):
            stage = read(case["stages"][phase], root)
            require(stage["updates"] == updates and read(stage["intent"], root)["phase"] == phase,
                    "Missing or improperly scoped checkpoint stage")
            scoped_pin(stage["checkpoint"], root)
    return {"scope": "synthetic_learning", "receipt": item, "optimizer_updates": 2304,
            "real_hsi_gain": None, "cross_sensor_gain": None, "oversquashing_gain": None,
            "positive_control": row["positive_control"], "negative_control": row["negative_control"],
            "confirmation_optimizer_updates": 0, "cases": len(row["cases"])}


def bn(item, root, registered):
    row = read(item, root)
    attest(row, "wp05_p4_bounded_domainbn_factor_v1")
    require(row["scope"] == "frozen_real_devtrain_devval_diagnostic" and row["status"] == "measured"
            and row["producer"] == registered["bn_diagnostic"] and row["real_gain"] is None
            and row["validation_statistics_fit"] is False and row["confirmed_metrics"] is None
            and row["training_updates"] == row["persistent_state_updates"] == row["statistics_calibration_updates"] == 0,
            "Real fitting diagnostic cannot promote calibration or validation status")
    prior = read(row["prior_bounded_diagnostic"], root)
    require(prior["resource"] == row["resource"] and prior["release"] == row["release"]
            and registered["p4_resource"] == row["resource"] and len(row["cases"]) == 2,
            "Frozen real diagnostic differs from currently approved resource")
    per_private_readonly = 0
    for case in row["cases"]:
        scoped_pin(case["source"], root)
        require(set(case["states"]) == {"initial", "selected", "last"} and case["updates_in_original_path"] == 60,
                "Missing frozen P4 initial, selected or last endpoint")
        for state in case["states"].values():
            private = state["train_mode_control"]
            require(state["confirmed_restoration"] is True and state["state_calibration_applied"] is False
                    and state["context_moment_writes"] == 0 and private["role"] == "train"
                    and private["classification"] == "ephemeral_train_mode_comparison"
                    and len(private["statistics_buffer_updates"]) == 5
                    and all(x == private["result"]["forwards"] for x in private["statistics_buffer_updates"].values()),
                    "Ephemeral train-mode normalization changes cannot be labeled read-only statistics")
            per_private_readonly += sum(private["statistics_buffer_updates"].values())
    return {"scope": "bounded_readonly_development_comparison", "receipt": item,
            "persistent_state_updates": 0, "private_statistics_batch_updates": per_private_readonly,
            "persistent_calibration_updates": 0, "real_gain": None, "confirmation": None}


def adapter(item, root, registered):
    row = read(item, root)
    attest(row, "wp06_native_trained_region_analysis_v1")
    plan = read(row["registration"], root)
    require(row["producer"] == plan["producer"] == registered["region_adapter"]
            and row["status"] == "measured" and row["real_gain"] is None
            and row["real_optimizer_updates"] == row["synthetic_optimizer_updates"] == 0
            and row["scope"] == plan["scope"] == "frozen_synthetic_with_real_prepare_only"
            and len(row["cases"]) == len(plan["cases"]) == 14
            and set(row["real"]) == {"xuzhou", "whu-honghu", "paviaU"},
            "Invalid prepared real admission or frozen synthetic adapter scope")
    for name in ("xuzhou", "whu-honghu"):
        real = row["real"][name]
        require(real["checkpoint_status"] == "compatible_real_selected_checkpoint_unavailable"
                and real["optimization_updates"] == real["spectral_array_reads"] == 0 and real["metrics"] is None,
                "Real observation admission was promoted to measured accuracy")
    views = 0
    for result, source in zip(row["cases"], plan["cases"], strict=True):
        scoped_pin(result["checkpoint"], root)
        require(all(result[k] == v for k, v in source.items()) and result["views"], "Analysis roster or source changed")
        for view in result["views"]:
            require(view["role"] == "devval" and view["state_and_outputs_restored"] is True
                    and len(view["stage_support"]) == 2 and set(view["ranges"]) in ({"single"}, {"wide", "narrow"})
                    and set(view["plans"]) == {"core", "edge", "least"},
                    "Analysis lacked region provenance, support or restored state")
            views += 1
    return {"scope": "frozen_synthetic_region_and_prepare_only", "receipt": item,
            "views": views, "real_optimizer_updates": 0, "real_gain": None}


def real_calibration(item, root, registered):
    row = read(item, root)
    attest(row, "wp05_p4_real_front_population_v1")
    plan = read(row["registration"], root)
    require(row["scope"] == plan["scope"] == "real_frozen_p4_development_only"
            and row["status"] == "measured" and row["producer"] == plan["producer"] == registered["real_calibration"]
            and plan["inputs"]["resource"] == registered["p4_resource"]
            and row["optimizer_updates"] == row["original_state_updates"] == row["restore_refits"] == 0
            and row["validation_statistics_fit"] is row["confirmation_accessed"] is row["production_default_changed"] is False
            and row["statistics_forwards"] == plan["statistics_forward_limit"] == 528
            and row["evaluation_forwards"] == plan["evaluation_forward_limit"] == 768
            and row["private_buffer_writes"] == plan["statistics_buffer_write_limit"] == 48
            and row["restore_evaluation_forwards"] == plan["restore_evaluation_forward_limit"] == 12
            and len(row["cases"]) == len(plan["cases"]) == 4
            and plan["selection"] == "original_selection_frozen",
            "Real development calibration must declare all private fitted writes and evaluation scope")
    passed = []
    for case, authority in zip(row["cases"], plan["cases"], strict=True):
        require(all(case[k] == authority[k] for k in ("case", "checkpoint", "selected_step"))
                and set(case["states"]) == {"initial", "selected", "last"} and len(authority["contexts"]) == 22,
                "Calibration changed the independently audited case or context")
        scoped_pin(case["checkpoint"], root)
        for state in case["states"].values():
            proof = state["proof"]
            require(proof["role"] == "devtrain" and proof["statistics_population_passes"] == 2
                    and proof["statistics_context_forwards"] == 44 and proof["statistics_buffer_writes"] == 4
                    and state["restore"] == {"evaluation_forwards": 1, "optimizer_updates": 0,
                                             "new_statistics_fitting_passes": 0},
                    "Calibration state changes or recovery costs were misreported")
            scoped_pin(state["buffers"], root)
        passed.extend(head["passes"] for head in case["states"]["selected"]["val"]["protection"].values())
    require(bool(passed) and all(passed) is row["selected_all_heads_protected"],
            "Calibration protection must reflect all original selected heads")
    return {"scope": row["scope"], "receipt": item, "private_statistic_buffer_writes": row["private_buffer_writes"],
            "statistics_forwards": row["statistics_forwards"], "evaluation_forwards": row["evaluation_forwards"],
            "real_optimizer_updates": 0, "selected_all_heads_protected": row["selected_all_heads_protected"],
            "production_default_changed": False, "independent_confirmation": None}


def legacy_region(item, root, registered):
    row = read(item, root)
    attest(row, "wp07_real_xuzhou_legacy_v5_region_intervention_v1")
    compatible = read(row["compatibility"], root)
    require(row["producer"] == registered["legacy_region"] and row["status"] == "measured"
            and row["backend"] == registered["legacy_worker"] and row["real_gain"] is None
            and row["scope"] == "legacy_real_xuzhou_v5_development_only"
            and row["original_protocol"] == compatible["original_protocol"] == "wp07_xuzhou_multiscale_v1"
            and row["optimizer_updates"] == row["bn_updates"] == row["calibration_updates"] == 0
            and row["independent_confirmation"] is row["new_native_protocol"] is False
            and compatible["status"] == "exact_historical_predictions" and compatible["source"] == row["source"]
            and row["original_forwards"] == compatible["forward_calls"] == 80 and len(row["records"]) == 16,
            "Legacy V5 mechanism output needs exact original predictions and separate protocol")
    old = read(row["source"]["audit"], root)
    expected = {(arm, endpoint) for arm in old["results"] for endpoint in ("selected", "last")}
    require({(value["arm"], value["endpoint"]) for value in row["records"]} == expected,
            "Legacy V5 endpoint census changed")
    forwards = 0
    for value in row["records"]:
        binding = compatible["endpoints"][value["arm"]][value["endpoint"]]
        witness = old["results"][value["arm"]]["endpoints"][value["endpoint"]]
        require(binding["audit_probability_hashes_equal"] and value["step"] == binding["step"] == witness["step"]
                and value["state_sha256"] == binding["state"] and value["original_state_restored"]
                and len(value["tiles"]) == 5 and all(set(t["observed"]) in
                    ({"self", "zero", "replace"}, {"self", "zero"}) for t in value["tiles"])
                and all(value["heads"]["baseline"][head]["metrics"]["confusion_matrix"] ==
                        witness["metrics"][head]["confusion_matrix"] for head in ("fused", "cnn", "graph")),
                "Original audited V5 endpoint differs")
        forwards += sum(1+len(tile["observed"]) for tile in value["tiles"])
    require(forwards == row["intervention_and_restoration_forwards"] <= 320, "Wrong bounded intervention census")
    return {"scope": row["scope"], "receipt": item, "endpoints": len(row["records"]),
            "operation_and_restore_forwards": row["intervention_and_restoration_forwards"],
            "optimizer_updates": 0, "new_native_matrix": None, "independent_confirmation": None, "real_gain": None}


def admit(out, root, receipts, approved, producer):
    registered = read(approved, root)
    base = {"crash", "learning", "bn", "region"}
    extension = {"calibration", "legacy-region"}
    require(set(receipts) in (base, base | extension), "Expected paired real development supplements")
    extra = set(receipts) == base | extension
    keys = ("science_worker", "crash_controller", "learned_worker", "learned_review", "bn_diagnostic", "region_adapter")
    if extra:
        keys += ("real_calibration", "legacy_region", "legacy_worker")
    for key in keys:
        identity = registered[key]
        require(set(identity) == {"commit", "source_hash"} and len(identity["commit"]) == 40
                and len(identity["source_hash"]) == 64, "Exact code and tree identities required")
    require(registered["schema"] == SCHEMA
            and registered["receipts"] == receipts and registered["query_authorized"] is False,
            "Explicit scope and pinned sources required")
    entries = {"dual_slots": crash(receipts["crash"], root, registered),
               "learning": learning(receipts["learning"], root, registered),
               "normalization": bn(receipts["bn"], root, registered),
               "region": adapter(receipts["region"], root, registered)}
    if extra:
        entries["front_population_calibration"] = real_calibration(receipts["calibration"], root, registered)
        entries["legacy_v5_region_analysis"] = legacy_region(receipts["legacy-region"], root, registered)
    retained = {}
    for folder, tables in OLD.items():
        for table, sha in tables.items():
            item = {"path": str(root / folder / table), "sha256": sha}
            scoped_pin(item, root)
            retained[table] = item
    out.mkdir()
    for name, item in retained.items():
        dest = out / name
        dest.write_bytes(scoped_pin(item, root).read_bytes())
        require(hash_file(dest) == item["sha256"], "Preserved WP12 table changed during export")
    return transport._json_new(out / "supplement.json", {"schema": SCHEMA, "status": "passed",
        "producer": producer, "authorities": approved, "modules": entries,
        "preserved": retained, "query_accessed": False, "real_confirmation_accessed": False})
