"""Reconstruct paired P4 acceptance from pinned stage and case records."""
from pathlib import Path
import json

from research_runtime import hash_file, write_json
from transfer import p4_coverage_batch as batch, p4_full_budget_campaign as campaign
from transfer import p4_independent_queue as queue, p4_independent_registration as scope
from transfer import p4_runtime_device as device


SCHEMA = "wp05_p4_independent_slot_acceptance_v1"


def verify_attempt(resource, item, out, case, slot, stage, prefix, attempt):
    name, operation, start, until = stage
    core = {"resource_sha256": item["sha256"], "case": case["id"], "slot": slot, "path": name,
            "operation": operation, "start": start, "until": until}
    gpu = case["physical_uuid"]
    lease = out / f".{slot}.lock" if gpu is None else campaign.slot_lock(out, gpu)
    scope.require(set(attempt) == set(core) | {"utc", "inventory", "cuda_visible_devices", "lease", "log"}
                  and all(attempt[key] == value for key, value in core.items())
                  and attempt["cuda_visible_devices"] == (gpu or "") and attempt["lease"] == str(lease),
                  "Stage receipt has an unregistered dispatch identity or lease")
    log = Path(attempt["log"])
    scope.require(log.parent == out / "logs" and log.resolve() == log
                  and log.name == f"{case['id']}-{prefix.name}-" + log.name.rsplit("-", 1)[-1],
                  "Stage log is outside its registered identity")
    if gpu is None:
        scope.require(attempt["inventory"] is None, "CPU dispatch cannot claim a physical snapshot")
    else:
        inventory = attempt["inventory"]
        scope.require(set(inventory) == {"physical_uuid", "free_mib", "utilization", "driver"}
                      and inventory["physical_uuid"] == gpu and device.eligible(inventory),
                      "Missing eligible per-operation physical admission")
    # A coordinator can die after worker commit and before the last stdout
    # flush. The worker ledger, rather than stdout, is the commit authority.
    return scope.pin(log) if log.is_file() else None


def reconstruct(item, resource, science):
    out = Path(resource["execution"])
    sha = resource["science"]["sha256"]
    owner_file = out / "resource-owner.json"
    worker_owner = campaign.owner(out, resource["science"]["path"], sha)
    scope.require(not (out / "failure.json").exists()
                  and campaign.read(owner_file) == {"resource": item, "worker_owner": worker_owner}
                  and campaign.read(out / "owner.json") == worker_owner,
                  "No valid nonfailed execution to accept")
    pins = {item["path"]: item["sha256"], resource["science"]["path"]: sha,
            **science["protected_files"], str(owner_file): hash_file(owner_file),
            str(out / "owner.json"): hash_file(out / "owner.json")}
    for recovery in resource["physical_recoveries"].values():
        scope.read(recovery)
        pins[recovery["path"]] = recovery["sha256"]
    cases = {case["id"]: case for case in science["cases"]}
    rows, slots, audits = [], {}, {}
    for slot, spec in resource["slots"].items():
        total = 0
        for case_id in spec["cases"]:
            case = cases[case_id]
            folder = campaign.bound_folder(out, case)
            row = campaign.accepted(folder, science, case, sha)
            scope.require(row["numeric"]["physical_uuid"] == case["physical_uuid"]
                          and row["class_count" if science["purpose"] == scope.RECOVERY else "case"]
                          == (resource["classes"] if science["purpose"] == scope.RECOVERY else case),
                          "Accepted class count or physical UUID differs")
            pins.update(row["artifact_pins"])
            path = folder / "audit.json"
            audits[str(path)] = hash_file(path)
            for index, stage in enumerate(queue.stages(science)):
                prefix = out / "stages" / case_id / f"{index:02}"
                intent = prefix.with_suffix(".intent.json")
                receipt = prefix.with_suffix(".receipt.json")
                dispatch = campaign.read(intent)
                committed = queue.evidence(out, case, science, sha, stage)
                scope.require(committed is not None and campaign.read(receipt) == {
                    "attempt": scope.pin(intent), "evidence": committed}, "Stage receipt differs from its ledger")
                log = verify_attempt(resource, item, out, case, slot, stage, prefix, dispatch)
                attempts = sorted(prefix.parent.glob(prefix.name + ".*.attempt.json"))
                scope.require(bool(attempts) and campaign.read(attempts[-1]) == dispatch,
                              "Committed stage is missing its recorded dispatch")
                if log is not None:
                    pins[log["path"]] = log["sha256"]
                for attempt in attempts:
                    prior_log = verify_attempt(resource, item, out, case, slot, stage, prefix, campaign.read(attempt))
                    if prior_log is not None:
                        pins[prior_log["path"]] = prior_log["sha256"]
                for evidence in (*attempts, intent, receipt):
                    pins[str(evidence)] = hash_file(evidence)
            total += row["optimizer_updates"]
            rows.append(row)
        scope.require(total == spec["updates"], "Slot update count differs from its frozen allocation")
        slots[slot] = {"cases": len(spec["cases"]), "optimizer_updates": total}
    pins.update(audits)
    total = sum(row["optimizer_updates"] for row in rows)
    scope.require(len(rows) == len(cases) and total == science["budget"]["planned_optimizer_updates"],
                  "Only the full registered matrix qualifies")
    result = {"schema": SCHEMA, "status": "passed", "resource_scope": item, "science_scope": resource["science"],
              "controller": resource["controller"], "worker": scope.identity(resource["worker"]),
              "synthetic": science["synthetic"], "cases": len(rows), "optimizer_updates": total,
              "audit_optimizer_updates": 0, "completed_resume_updates": 0, "slots": slots,
              "raw_gt_loaded": False, "query_accessed": False, "case_audits": audits, "protected_files": pins}
    if resource["mode"] == scope.REAL:
        result["validation_diagnostics"] = campaign.admission_validation(rows)
    batch.check_pins(pins)
    return result


def publish(item, code):
    resource = scope.read(item)
    science = scope.validate(resource, code=code, live_worker=True)
    scope.require(resource["mode"] != scope.REAL, "Real independent scope is preparation-only")
    out = Path(resource["execution"])
    with device.lease(out / ".coordinator.lock"):
        result = reconstruct(item, resource, science)
        path = out / "independent-acceptance.json"
        if path.exists():
            scope.require(campaign.read(path) == result, "Published independent receipt differs")
        else:
            write_json(path, result)
        return scope.pin(path)


def predecessor(resource_pin, receipt_pin, approved_controller):
    """Validate a separately reviewed release's real P4 predecessor.

    The calling release carries the controller identity and the receipt hash.
    Deterministic reconstruction rejects incomplete, synthetic or edited evidence.
    This does not create a fitting release.
    """
    resource = scope.read(resource_pin)
    science = scope.validate(resource, live_worker=True)
    scope.require(resource["mode"] == scope.REAL and resource["controller"] == approved_controller,
                  "Expected separately approved real independent P4 producer")
    result = scope.read(receipt_pin)
    out = Path(resource["execution"])
    scope.require(Path(receipt_pin["path"]) == out / "independent-acceptance.json" and result["schema"] == SCHEMA
                  and result["synthetic"] is False and result["cases"] == 20 and result["optimizer_updates"] == 2400,
                  "Full 20-case and 2400-update independent predecessor required")
    with device.lease(out / ".coordinator.lock"):
        scope.require(result == reconstruct(resource_pin, resource, science),
                      "Independent predecessor differs from its full stage and case evidence")
    return {**result["protected_files"], receipt_pin["path"]: receipt_pin["sha256"]}
