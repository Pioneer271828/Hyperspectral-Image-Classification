"""Reconstruct paired P4 acceptance from pinned stage and case records."""
from pathlib import Path
import re

from research_runtime import hash_file, write_json
from transfer import p4_coverage_batch as batch, p4_full_budget_campaign as campaign
from transfer import p4_independent_queue as queue, p4_independent_registration as scope
from transfer import p4_runtime_device as device


SCHEMA = "wp05_p4_independent_slot_acceptance_v1"


def verify_attempt(item, out, case, slot, stage, index, serial, attempt):
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
                  and log.name == f"{case['id']}-{index:02d}-{serial:04d}.log",
                  "Stage log is outside its registered identity")
    if gpu is None:
        scope.require(attempt["inventory"] is None, "CPU dispatch cannot claim a physical snapshot")
    else:
        inventory = attempt["inventory"]
        scope.require(set(inventory) == {"physical_uuid", "free_mib", "utilization", "driver"}
                      and inventory["physical_uuid"] == gpu and device.eligible(inventory)
                      and type(inventory["free_mib"]) is int and type(inventory["utilization"]) is int
                      and 0 <= inventory["utilization"] <= 100,
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
        proof = scope.read(recovery)
        pins[recovery["path"]] = recovery["sha256"]
        for case in proof["cases"]:
            pins[case["scope"]] = case["scope_sha256"]
            pins[case["acceptance"]] = case["acceptance_sha256"]
            saved = scope.read({"path": case["acceptance"], "sha256": case["acceptance_sha256"]})
            for path, checksum in saved["protected_files"].items():
                scope.require(pins.setdefault(path, checksum) == checksum, "Recovery and science pins disagree")
    cases = {case["id"]: case for case in science["cases"]}
    rows, slots, audits = [], {}, {}
    for slot, spec in resource["slots"].items():
        total = 0
        for case_id in spec["cases"]:
            case = cases[case_id]
            folder = campaign.bound_folder(out, case)
            row = campaign.accepted(folder, science, case, sha)
            gpu = case["physical_uuid"]
            heads = row["fit"]["selected"]["val"]["heads"]
            scope.require(row["numeric"]["physical_uuid"] == gpu
                          and (science["purpose"] != scope.RECOVERY or row["class_count"] == resource["classes"])
                          and row["raw_gt_loaded"] is row["query_accessed"] is False
                          and row["nonselected_event_probability_replays"] == 0
                          and set(heads) == {"fused", "graph", "cnn"}
                          and all(len(head["confusion"]) == resource["classes"]
                                  and [sum(values) for values in head["confusion"]] == [5] * resource["classes"]
                                  for head in heads.values()),
                          "Accepted class count, validation supports or physical UUID differs")
            if gpu is not None:
                qualification = science["qualifications"][gpu]
                qualified = scope.read({"path": qualification, "sha256": science["protected_files"][qualification]})
                scope.require(row["numeric"] == qualified["numeric"], "Accepted arithmetic differs from qualification")
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
                attempts = sorted(prefix.parent.glob(prefix.name + ".*.attempt.json"))
                scope.require(bool(attempts) and campaign.read(attempts[-1]) == dispatch,
                              "Committed stage is missing its recorded dispatch")
                for serial, attempt in enumerate(attempts, 1):
                    scope.require(attempt == prefix.parent / f"{index:02d}.{serial:04d}.attempt.json"
                                  and attempt.resolve() == attempt, "Unexpected or missing dispatch attempt")
                    log = verify_attempt(item, out, case, slot, stage, index, serial, campaign.read(attempt))
                    if log is not None:
                        pins[log["path"]] = log["sha256"]
                for proof in (*attempts, intent, receipt):
                    pins[str(proof)] = hash_file(proof)
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
    scope.require(resource["mode"] == scope.REAL and resource["controller"] == approved_controller
                  and set(approved_controller) == {"commit", "source_hash"}
                  and re.fullmatch("[a-f0-9]{40}", approved_controller["commit"])
                  and re.fullmatch("[a-f0-9]{64}", approved_controller["source_hash"]),
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
