"""Plan, admit and independently inspect a fresh WP03 light-adaptation campaign."""
from pathlib import Path

from g4_light_adapter_memory import RECOMPUTE
import g4_light_engineering_registration as engineering
from g4_light_engineering_review import HISTORICAL
import g4_light_probe_pair as probe
from g4_light_target import ADAPTER_HEADS
from g4_matched_target_evaluation import new_output
from g4_matched_target_study import METHODS, select_case
from g4_support_context import check_files
from g4_support_target import source_pins
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json

PLAN = "wp03_new_full54_plan_v1"
REGISTRATION = "wp03_new_full54_registration_v1"
APPROVAL = "wp03_new_full54_release_v1"
REAL = "real_wp03_full54"
FIXTURE = "synthetic_wp03_admission"
BUDGET = {
    "cases": 54, "per_case_updates": 1500, "optimizer_updates": 81000,
    "support_seed": 64, "support_per_class": 10, "validation_per_class": 5,
    "validation_events": 51, "selection": "first_min_fused_validation_nll",
    "query_access": False,
}
READY_SHA = "50cc39499b0255df6ce602263d964ee7eb74bc7e3cb830f808fb41e9b0c6f469"


def contained(paths, candidate):
    return any(candidate == path or candidate in path.parents or path in candidate.parents for path in paths)


def historical(registration, inventory):
    if (registration["sha256"] != probe.REGISTRATION_SHA256 or inventory["sha256"] != HISTORICAL):
        raise ValueError("Exact frozen Issue249 registration and inventory required")
    original, prior = (engineering.pinned(pin) for pin in (registration, inventory))
    if (original["experiment_issue"] != 249 or original["producer_commit"] != probe.PRODUCER_COMMIT
            or prior["producer_commit"] != probe.PRODUCER_COMMIT or prior["registration_sha256"] != registration["sha256"]
            or prior["campaign_status"] != "failed" or prior["resume_authorized"] is not False
            or prior["global_acceptance"] is not False
            or prior["levels"] != {"coordinator_receipt_revalidated": 35, "external_receipt_revalidated": 1,
                                   "not_started": 17, "failed_case": 1}):
        raise ValueError("Original failed batch custody or accepted counts differ")
    return original, prior


def cohort(request, code):
    original, prior = historical(request["original_registration"], request["inventory"])
    legacy_root = Path(original["out_dir"]).resolve(strict=True)
    if request["scope"] == FIXTURE:
        if set(request) != {"scope", "original_registration", "inventory", "fixture", "output", "devices", "active"}:
            raise ValueError("Exact engineering admission request required")
        launch, inputs = engineering.read_fixture(request["fixture"], code)
        binding = request["fixture"]
        phase = engineering.PHASE
    elif request["scope"] == REAL:
        if set(request) != {"scope", "original_registration", "inventory", "reference_producer", "output", "devices",
                            "issue", "readiness", "diagnosis", "qualifications", "forecast"}:
            raise ValueError("Exact real admission request required")
        binding = original["reference_launch"]
        launch = engineering.pinned(binding)
        snapshot = code_snapshot(Path(request["reference_producer"]).resolve(strict=True))
        if (snapshot["dirty"] or snapshot["commit"] != probe.PRODUCER_COMMIT
                or launch["code"]["dirty"] or digest(launch["code"]["files"]) != launch["code"]["source_hash"]
                or any(snapshot["files"].get(k) != sha for k,sha in launch["code"]["files"].items())):
            raise ValueError("Frozen reference must be authenticated against the clean original producer")
        phase = "full_campaign"
        inputs = {}
    else:
        raise ValueError("Known admission scope required")
    return launch, phase, binding, inputs, legacy_root, prior


def check_review(doc_pin, code, *, scope, device=None, replay=True):
    row = engineering.pinned(doc_pin)
    if (row.get("policy") != probe.PROBE_REVIEW or row.get("scope") != scope
            or row.get("arithmetic") != probe.ARITHMETIC or row.get("status") != "within_existing_tolerance"
            or (device is not None and row.get("device") != device)):
        raise ValueError("Qualified same-device arithmetic and successful independent diagnostic required")
    request = engineering.pinned(row["request"])
    if request["producer"] != engineering.code_identity(code):
        raise ValueError("Diagnostic evidence binds another executable source")
    if replay and row != probe.review_pair(row["request"], row["samples"]):
        raise ValueError("Diagnostic receipt differs from its independent numerical replay")
    if replay and scope == probe.REAL:
        if request != probe.probe_intent(request["registration"], row["device"], code,
                                        migration=request["migration"], authority=request["authority"]):
            raise ValueError("Real role diagnosis binds another executable source or approval")
    return row


def production_gaps(request, code, devices, *, replay):
    gaps = []
    if request["issue"] is None:
        gaps.append("fresh_experiment_issue")
    elif type(request["issue"]) is not int or request["issue"] <= 249:
        raise ValueError("A fresh non-Issue249 experiment identity is required")
    if request["readiness"]["sha256"] != READY_SHA:
        raise ValueError("Pinned verified WP03 engineering readiness required")
    ready = engineering.pinned(request["readiness"])
    if (ready["reuse_as_accepted_new_strategy"] != 0 or ready["reuse_as_historical_evidence"] != 36
            or ready["historical_inventory"] != request["inventory"]):
        raise ValueError("Historical evidence must stay outside the new acceptance set")
    diagnosis = request["diagnosis"]
    if diagnosis is None:
        gaps.append("same_role_real_zero_update_independent_diagnosis")
    else:
        report = check_review(diagnosis, code, scope=probe.REAL, replay=replay)
        if engineering.pinned(report["request"])["registration"] != request["original_registration"]:
            raise ValueError("Diagnosis used a foreign Issue249 registration")
    qualifications = request["qualifications"]
    if not isinstance(qualifications, dict) or not set(qualifications) <= {d["uuid"] for d in devices}:
        raise ValueError("Physical slot qualifications must bind the proposed cards")
    for d in devices:
        if d["uuid"] not in qualifications:
            gaps.append("device_arithmetic:" + d["uuid"])
        else:
            check_review(qualifications[d["uuid"]], code, scope=probe.FIXTURE, device=d, replay=replay)
    if request["forecast"] is None:
        gaps.append("frozen_runtime_budget")
    else:
        estimate = engineering.pinned(request["forecast"])
        if (estimate.get("policy") != "wp03_full54_duration_forecast_v1"
                or estimate.get("inventory") != request["inventory"] or estimate.get("budget") != BUDGET):
            raise ValueError("Forecast must use the same cohort and the fixed 81,000 updates")
    return gaps


def prepare(request_pin, code, *, verify_files, replay_diagnostics=True):
    from g4_light_full_target import roles

    request = engineering.pinned(request_pin)
    launch, phase, binding, inputs, legacy, prior = cohort(request, code)
    synthetic = request["scope"] == FIXTURE
    devices = request["devices"]
    slots = engineering.verify_devices(devices)
    if synthetic != all(isinstance(slot, str) for slot in slots):
        raise ValueError("Engineering CPU slots and real physical cards cannot be mixed")
    out = new_output(request["output"], launch, (legacy,))
    matrix = roles()
    select_case(launch, launch["code"], matrix[0][0], matrix[0][1], METHODS[1], matrix[0][3], phase=phase)
    references = {(c["target"], c["seed"], c["mode"]): c for c in launch["cases"] if c["method"] == METHODS[1]}
    active = {tuple(role) for role in request["active"]} if synthetic else set(matrix)
    if not active or not active <= set(matrix) or synthetic and len(active) != len(request["active"]):
        raise ValueError("Distinct active engineering cases must belong to the full 54-role roster")
    rows = []
    for index, key in enumerate(matrix):
        scene, seed, method, mode = key
        case = references[scene, seed, mode]
        context = case["target_context"]
        if (context["support_seed"] != BUDGET["support_seed"] or context["support_per_class"] != 10
                or context["validation_per_class"] != 5 or context["query_labels_accessed"] is not False):
            raise ValueError("K10/V5 support, validation and access differ")
        if verify_files:
            inputs.update(context["protected_files"])
            inputs.update(source_pins(case["source_spec"]))
        rows.append({"role": list(key), "slot": slots[index % len(slots)], "scheduled": key in active,
                     "registered_updates": BUDGET["per_case_updates"],
                     "strategy": RECOMPUTE if method == ADAPTER_HEADS else None,
                     "reference_sha256": digest(case), "initial_sha256": case["initial_state_sha256"],
                     "input_sha256": digest(context["protected_files"]), "source_sha256": digest(case["source_spec"]),
                     "graph_sha256": digest(case["graph_binding"]), "training_sha256": digest(case["settings"])})
    if {row["slot"] for row in rows if row["scheduled"]} != set(slots):
        raise ValueError("Every registered slot needs a scheduled case")
    if contained({Path(p).resolve() for p in inputs} | {Path(request_pin["path"]), Path(binding["path"])}, out):
        raise ValueError("Execution overlaps a protected input")
    if verify_files:
        check_files(inputs)
    missing = [] if synthetic else production_gaps(request, code, devices, replay=replay_diagnostics)
    return {"policy": PLAN, "scope": request["scope"], "request": request_pin, "producer": engineering.code_identity(code),
            "reference": binding, "output": str(out), "arithmetic": None if synthetic else probe.ARITHMETIC,
            "budget": BUDGET, "devices": devices, "roles": rows, "missing": missing,
            "historical": {"inventory": request["inventory"], "levels": prior["levels"],
                           "retained_original_updates": prior["historically_accepted_updates_revalidated"],
                           "counted_in_new_acceptance": 0}, "query_authorized": False}


def release_for(plan_pin, plan, *, decision):
    return {"policy": APPROVAL, "decision": decision, "plan": plan_pin, "producer": plan["producer"],
            "scope": plan["scope"], "output": plan["output"], "devices": plan["devices"], "budget": BUDGET,
            "query_authorized": False, "real_training_authorized": decision == "approved" and plan["scope"] == REAL}


def activate(plan_pin, authority, code, *, replay=True):
    plan = engineering.pinned(plan_pin)
    if plan != prepare(plan["request"], code, verify_files=False, replay_diagnostics=replay):
        raise ValueError("Registration plan no longer reproduces")
    if plan["missing"]:
        raise ValueError("Missing admission evidence: " + ", ".join(plan["missing"]))
    if engineering.pinned(authority) != release_for(plan_pin, plan, decision="approved"):
        raise ValueError("Separate approved plan release is required")
    return {"policy": REGISTRATION, "plan": plan_pin, "authority": authority, "producer": engineering.code_identity(code)}


def open_registration(registration, code, *, replay=False):
    if registration != activate(registration["plan"], registration["authority"], code, replay=replay):
        raise ValueError("Invalid full54 execution admission")
    plan = engineering.pinned(registration["plan"])
    synthetic = plan["scope"] == FIXTURE
    request = engineering.pinned(plan["request"])
    launch, _, _, _, _, _ = cohort(request, code)
    pins = (registration["plan"], registration["authority"], plan["request"], plan["reference"],
            request["original_registration"], request["inventory"])
    if not synthetic:
        pins += (request["diagnosis"], request["forecast"], request["readiness"], *request["qualifications"].values())
    return plan, launch, {p["path"]: p["sha256"] for p in pins}


def launch_view(registration, plan):
    synthetic = plan["scope"] == FIXTURE
    return {**registration, "out_dir": plan["output"], "roles": plan["roles"], "devices": plan["devices"],
            "gpu_ids": [d["slot"] for d in plan["devices"]], "planned_optimizer_updates": BUDGET["optimizer_updates"],
            "scheduled_optimizer_updates": sum(row["registered_updates"] for row in plan["roles"] if row["scheduled"]),
            "adapter_backward_policy": RECOMPUTE, "synthetic": synthetic}


def check_worker_environment(device):
    import torch

    if probe.arithmetic() != probe.ARITHMETIC or torch.cuda.device_count() != 1:
        raise ValueError("Worker CUDA arithmetic or visible physical device differs")
    probe.hardware_gate(device, 12288)


def independent_campaign(registration_pin, code):
    from g4_light_full_campaign import POLICY, case_directory
    from g4_light_full_target import roles

    registration = engineering.pinned(registration_pin)
    plan, _, pins = open_registration(registration, code, replay=True)
    root = Path(plan["output"]).resolve(strict=True)
    synthetic = plan["scope"] == FIXTURE
    status = engineering.pinned(engineering.pin(root / "status.json"))
    acceptance = engineering.pinned(engineering.pin(root / ("engineering-acceptance.json" if synthetic
                                                           else "training-acceptance.json")))
    grid = [tuple(row["role"]) for row in plan["roles"] if row["scheduled"]]
    names = [case_directory(root, key).name for key in grid]
    if (status["status"] != "completed" or status["errors"] or acceptance["policy"] != POLICY
            or status["completed_cases"] != len(grid) or acceptance["status"] != "passed" or acceptance["code"] != code
            or acceptance["registration_sha256"] != registration_pin["sha256"]
            or acceptance["admission_plan"] != registration["plan"] or acceptance["cases"] != names
            or acceptance["optimizer_updates"] != len(grid) * BUDGET["per_case_updates"]
            or acceptance["query_labels_accessed"] is not False or (not synthetic and grid != roles())
            or any(acceptance["protected_files"].get(k) != sha for k,sha in pins.items())):
        raise ValueError("Complete current registration, protected files and exact case budget required")
    check_files(acceptance["protected_files"])
    if set(acceptance["audit_receipts"]) != {str(root / "audits" / (name + ".json")) for name in names}:
        raise ValueError("Audit receipts do not cover the exact new cohort")
    audits = []
    for key, name in zip(grid, names, strict=True):
        audit_path = str(root / "audits" / (name + ".json"))
        item = {"path": audit_path, "sha256": acceptance["audit_receipts"][audit_path]}
        row = engineering.pinned(item)
        role = next(r for r in plan["roles"] if tuple(r["role"]) == key)
        if (row["status"] != "passed" or row["name"] != name or row["ledger"] != role
                or row["registration_sha256"] != registration_pin["sha256"] or row["updates"] != 1500
                or row["validation_events"] != 51 or row["exact_reload_repeats"] != 3
                or row["new_optimizer_updates"] != 0 or row["query_labels_accessed"] is not False
                or any(acceptance["protected_files"].get(k) != sha for k,sha in row["artifacts"].items())):
            raise ValueError("A new case audit differs from its own registered role or artifacts")
        audits.append(item)
    require_frozen_code(code)
    return {"policy": "wp03_independent_campaign_review_v1", "scope": plan["scope"], "registration": registration_pin,
            "new_cases": len(grid), "new_updates": acceptance["optimizer_updates"], "audits": audits,
            "historical_separate": plan["historical"], "query_authorized": False}
