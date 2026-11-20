"""Independent read-only review and engineering proofs for a completed light queue."""
from pathlib import Path

import g4_light_engineering_registration as eng
from g4_light_full_campaign import POLICY as QUEUE, case_directory
from g4_light_full_target import UPDATES, device_for, read_registration
from g4_light_trainval_review import derive_case
from g4_support_context import check_files, read
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code, write_json

REVIEW = "g4_light_synthetic_batch_review_v1"
WORKER = "g4_light_synthetic_worker_proof_v1"
CAMPAIGN = "g4_light_synthetic_campaign_proof_v1"
HISTORICAL = "37b398e982f4e9069147ccada2bf55da71b544810c03ecd24620e03ed304ce51"


def scheduled(registration):
    return [row for row in registration["roles"] if row["scheduled"]]


def verify_queue(registration_pin, code):
    registration, inputs = read_registration(registration_pin["path"],registration_pin["sha256"],code)
    if registration["policy"] != eng.REGISTRATION:
        raise ValueError("Independent engineering review requires a synthetic registration")
    root = external_directory(registration["out_dir"])
    status_pin = eng.pin(root / "status.json")
    status = eng.pinned(status_pin)
    if status["status"] != "completed":
        raise ValueError("Failed or incomplete light campaign cannot enter engineering review")
    receipt_pin = eng.pin(root / "engineering-acceptance.json")
    receipt = eng.pinned(receipt_pin)
    roster = scheduled(registration)
    folders = [case_directory(root,tuple(row["role"])) for row in roster]
    names = [path.name for path in folders]
    audit_paths = {str(root / "audits" / (name+".json")) for name in names}
    producer = eng.code_identity(code)
    common = {"policy":QUEUE,"scope":eng.SCOPE,"producer":producer,"registration_sha256":registration_pin["sha256"],
              "gpu_ids":registration["gpu_ids"],"planned_cases":54,"planned_optimizer_updates":81000,
              "scheduled_cases":len(roster),"scheduled_optimizer_updates":len(roster)*UPDATES,"real_training_authorized":False}
    if (any(doc.get(key)!=value for doc in (status,receipt) for key,value in common.items())
            or status["completed_cases"]!=len(roster) or status["errors"]
            or receipt["status"]!="passed" or receipt["code"]!=code or receipt["cases"]!=names
            or receipt["optimizer_updates"]!=len(roster)*UPDATES or set(receipt["audit_receipts"])!=audit_paths
            or receipt["query_labels_accessed"] is not False
            or any(receipt["protected_files"].get(path)!=value for path,value in inputs.items())):
        raise ValueError("Incomplete or mixed light campaign receipt")
    pins = {**receipt["protected_files"],**receipt["audit_receipts"],
            status_pin["path"]:status_pin["sha256"],receipt_pin["path"]:receipt_pin["sha256"]}
    check_files(pins)
    rows = []
    for binding,folder in zip(roster,folders,strict=True):
        name = folder.name
        audit_path = root / "audits" / (name+".json")
        audit = read(audit_path)
        if (audit["status"]!="passed" or audit["name"]!=name or audit["role"]!=binding["role"]
                or audit["ledger"]!=binding or audit["scope"]!=eng.SCOPE or audit["producer"]!=producer
                or audit["registration_sha256"]!=registration_pin["sha256"] or audit["physical_gpu"]!=binding["slot"]
                or audit["device"]!=device_for(binding["slot"]) or audit.get(eng.MEMORY)!=binding["strategy"]
                or audit["updates"]!=UPDATES or audit["validation_events"]!=51 or audit["exact_reload_repeats"]!=3
                or audit["new_optimizer_updates"]!=0 or audit["query_labels_accessed"] is not False
                or any(pins.get(p)!=h for group in ("artifacts","protected_inputs") for p,h in audit[group].items())):
            raise ValueError("Wrong method, producer, slot or policy in case audit")
        case = {n:read(folder / (n+".json")) for n in ("result","history","trace","adaptation-cost")}
        result = case["result"]
        primary = folder / "primary.pt"
        if (audit["result_sha256"]!=pins[str(folder/"result.json")] or audit["primary_sha256"]!=pins[str(primary)]
                or audit["sidecar_sha256"]!=pins[str(primary)+".json"] or result["identity"]["prepared"]["scheduled_role"]!=binding
                or result["settings"]["registration_sha256"]!=registration_pin["sha256"] or result["optimizer_steps"]!=UPDATES
                or result["settings"]["reference_case_sha256"]!=binding["reference_sha256"]):
            raise ValueError("Case result differs from the registered ledger")
        reviewed = derive_case({"role":binding["role"],"selected_step":audit["selected_step"],"name":name},
                              result,audit,case["history"],case["trace"],case["adaptation-cost"])
        rows.append({"name":name,"binding":binding,"audit":eng.pin(audit_path),
                     "checkpoint":eng.pin(primary),"metadata":eng.pin(str(primary)+".json"),
                     "class_recalls":audit["class_recalls"],"head_metrics":audit["head_metrics"],
                     "exact_reload_repeats":audit["exact_reload_repeats"],"trainval":reviewed})
    check_files(pins)
    require_frozen_code(code)
    return {"policy":REVIEW,"scope":eng.SCOPE,"producer":producer,"registration":registration_pin,
            "fixture":registration["fixture"],"roles":registration["roles"],"devices":registration["devices"],
            "queue_status":status_pin,"queue_acceptance":receipt_pin,"rows":rows,
            "input_files":inputs,"planned_cases":54,"planned_updates":81000,"executed_cases":len(rows),
            "executed_updates":UPDATES*len(rows),"unexecuted_roles":54-len(rows),
            "real_updates":0,"query_access":False,"real_training_authorized":False,
            "accuracy_scope":"synthetic_support_and_validation_only","cost_scope":"adaptation_phases_only",
            "elapsed_seconds":receipt["elapsed_seconds_this_coordinator"]}


def inspect(review, code):
    data = eng.pinned(review)
    if data["policy"]!=REVIEW or data["producer"]!=eng.code_identity(code):
        raise ValueError("Wrong-code or legacy light batch evidence")
    if data!=verify_queue(data["registration"],code):
        raise ValueError("Light review differs from its independently replayed batch")
    return data


def evidence_proofs(review_pins, code, out):
    reports = [inspect(pin,code) for pin in review_pins]
    if not reports or len({r["registration"]["sha256"] for r in reports})!=len(reports):
        raise ValueError("Distinct verified synthetic runs required")
    required = {"source_pretrained_head_only","source_pretrained_adapter_heads"}
    for report in reports:
        if {r["binding"]["role"][2] for r in report["rows"]}!=required:
            raise ValueError("Both light methods must be executed and accepted")
    if any(r["fixture"]!=reports[0]["fixture"] for r in reports):
        raise ValueError("Proof mixes different synthetic source or starting states")
    bindings = [{k:report[k] for k in ("registration","fixture","devices","executed_cases","executed_updates")}
                for report in reports]
    common = {"scope":eng.SCOPE,"producer":eng.code_identity(code),"reviews":review_pins,
              "bindings":bindings,eng.MEMORY:eng.RECOMPUTE,"labels":{"query":False,"real":False},
              "real_training_authorized":False}
    worker = {**common,"policy":WORKER,"selection":"first_min_fused_validation_nll",
              "events_per_case":51,"updates_per_executed_case":UPDATES,"exact_selected_reloads":3}
    out.mkdir()
    write_json(out/"worker-proof.json",worker)
    campaign = {**common,"policy":CAMPAIGN,"worker_proof":eng.pin(out/"worker-proof.json"),
                "planned_cases":54,"planned_updates":81000,"real_updates":0,
                "cpu_verified":any(all("uuid" not in d for d in r["devices"]) for r in reports),
                "physical_multigpu_verified":any(sum("uuid" in d for d in r["devices"])>=2 for r in reports)}
    write_json(out/"campaign-proof.json",campaign)


def readiness(worker_pin,campaign_pin, inventory_pin, code):
    worker,campaign = eng.pinned(worker_pin),eng.pinned(campaign_pin)
    identity=eng.code_identity(code)
    if (worker["policy"]!=WORKER or campaign["policy"]!=CAMPAIGN or campaign["worker_proof"]!=worker_pin
            or any(row["producer"]!=identity or row["scope"]!=eng.SCOPE or row[eng.MEMORY]!=eng.RECOMPUTE
                   or row["real_training_authorized"] is not False for row in (worker,campaign))
            or any(worker[k]!=campaign[k] for k in ("reviews","bindings","labels"))
            or worker["labels"]!={"query":False,"real":False} or worker["events_per_case"]!=51
            or worker["updates_per_executed_case"]!=UPDATES or worker["exact_selected_reloads"]!=3
            or campaign["planned_cases"]!=54 or campaign["planned_updates"]!=81000 or campaign["real_updates"]!=0):
        raise ValueError("Old, mixed or foreign worker/campaign proof")
    expected_bindings=[]
    for pin in campaign["reviews"]:
        report=inspect(pin,code)
        expected_bindings.append({k:report[k] for k in ("registration","fixture","devices","executed_cases","executed_updates")})
    if expected_bindings!=campaign["bindings"]:
        raise ValueError("Proof counts or inputs differ from verified reviews")
    inventory = eng.pinned(inventory_pin)
    if (inventory_pin["sha256"]!=HISTORICAL or inventory["levels"]!={"coordinator_receipt_revalidated":35,
            "external_receipt_revalidated":1,"failed_case":1,"not_started":17}
            or inventory["resume_authorized"] is not False):
        raise ValueError("Original failed-role inventory required")
    return {"policy":"g4_light_real_registration_readiness_v1","producer":identity,"worker":worker_pin,
            "campaign":campaign_pin,"historical_inventory":inventory_pin,"historical_levels":inventory["levels"],
            "reuse_as_historical_evidence":36,"reuse_as_accepted_new_strategy":0,
            "new_registration_required":True,"eligibility":{"real_training_authorized":False,
                "query_authorized":False,"historical_gpu_probe_required":True},
            "remaining":["same_role_real_zero_update_numeric_probe","new_registered_case_assignments",
                         "independent_campaign_admission","all54_acceptance_before_query"],
            "query_access":False,"real_updates":0}
