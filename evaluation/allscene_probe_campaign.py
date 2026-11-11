"""Execute only new scene probes and audit the complete frozen three-scene roster."""
import json
from pathlib import Path
import time

from evaluation import allscene_probe_registry as registry, allscene_probe_summary as summary
from evaluation import frozen_probe_audit as audit, frozen_probe_campaign as campaign, frozen_probe_registry as reg
from g4_support_context import check_files
from research_runtime import hash_file, require_frozen_code, write_json


def allocation(scope_path, scope_sha, out, code):
    return {"scope_sha256": scope_sha, "consumer_commit": code["commit"],
            "execution": str(out/"execution.json"), "registration": str(scope_path)}


def ledger_rows(cases, reused):
    for case, _, _ in cases:
        name, disposition = case["name"], case["disposition"]
        yield {"case": name, "scene": case["target"], "disposition": disposition,
               "status": {"historical": "reused", "new": "not_run", "missing_capture": "missing_capture"}[disposition],
               "historical_probe_updates": reused[name]["probe_updates"] if name in reused else 0, "new_probe_updates": 0}


def run(scope_path, scope_sha, out, code, *, authority=registry, run_case=campaign.run_case):
    started = time.monotonic()
    scope_path = Path(scope_path).resolve(strict=True)
    scope, launch, cases, pins, reused, elapsed = authority.load(scope_path, scope_sha, code)
    out = authority.output(out, launch, scope, scope_path.parent)
    reg.require(not out.exists(), "Fresh external execution required")
    owner = scope_path.with_name("execution-owner.json")
    with owner.open("x", encoding="utf-8") as stream:
        json.dump(allocation(scope_path, scope_sha, out, code), stream, indent=2)
        stream.write("\n")
    owner_sha = hash_file(owner)
    out.mkdir(exist_ok=False)
    rows = list(ledger_rows(cases, reused))
    state = {"policy": scope["policy"], "scope_sha256": scope_sha, "commit": code["commit"], "cases": rows,
             "status": "running", "history": scope["history"], "input_authentication_seconds": elapsed,
             "allocation_sha256": owner_sha, "backbone_updates": 0, "bn_updates": 0, "query_labels_accessed": False}
    status_file = out/"execution.json"
    write_json(status_file, state)
    for row, (case, bound, local) in zip(rows, cases, strict=True):
        if row["status"] != "not_run":
            continue
        row["status"] = "running"

        def record_update(receipt):
            row["new_probe_updates"] += receipt["probe_updates"]
            write_json(status_file, state)

        try:
            sha = run_case(campaign.folder_at(out, case["name"]), bound, case, local,
                                    registry.case_scope(scope, case), scope_sha, record_update)
            row.update(status="executed", execution_sha256=sha)
        except Exception as error:
            row.update(status="execution_failed", error=repr(error))
        write_json(status_file, state)
    state["status"] = ("executed" if all(row["status"] in ("executed", "reused") for row in rows)
                       else "incomplete_or_failed")
    state["historical_probe_updates"] = sum(row["historical_probe_updates"] for row in rows)
    state["new_probe_updates"] = sum(row["new_probe_updates"] for row in rows)
    state["total_probe_updates"] = state["historical_probe_updates"] + state["new_probe_updates"]
    write_json(status_file, state)
    check_files(registry.merge_pins(pins, {str(owner): owner_sha}))
    require_frozen_code(code)
    state.update(protected_files_unchanged=True, wall_seconds=time.monotonic()-started)
    write_json(status_file, state)
    return status_file


def validate_run(run_path, run_sha, scope_path, scope_sha, scope, cases, reused, code):
    run = reg.pinned_json(run_path, run_sha)
    owner = scope_path.with_name("execution-owner.json")
    reg.require(reg.pinned_json(owner, run["allocation_sha256"]) == allocation(scope_path, scope_sha, run_path.parent, code),
                "Execution differs from the once-only registered allocation")
    reg.require(run_path.name == "execution.json" and run["policy"] == scope["policy"]
                and run["commit"] == code["commit"] and run["scope_sha256"] == scope_sha
                and run["status"] in ("executed", "incomplete_or_failed")
                and run["history"] == scope["history"] and run["protected_files_unchanged"] is True
                and run["backbone_updates"] == run["bn_updates"] == 0 and run["query_labels_accessed"] is False,
                "All-scene execution source or boundary differs")
    rows = run["cases"]
    updates = len(reg.LAYERS)*scope["paths"]["full"][0]*2
    for actual, expected in zip(rows, ledger_rows(cases, reused), strict=True):
        status, amount = actual["status"], actual["new_probe_updates"]
        keys = ("case", "scene", "disposition", "historical_probe_updates")
        reg.require(all(actual[key] == expected[key] for key in keys), "Execution ledger role or historical budget differs")
        if expected["disposition"] != "new":
            reg.require(actual == expected, "Historical or missing-capture role must have zero new updates")
        else:
            reg.require(type(amount) is int and (status == "executed" and amount == updates or
                        status == "execution_failed" and 0 <= amount <= updates), "New role update budget differs")
    new = sum(row["new_probe_updates"] for row in rows)
    historical = sum(row["historical_probe_updates"] for row in rows)
    complete = all(row["status"] in ("executed", "reused") for row in rows)
    reg.require((run["status"] == "executed") == complete
        and (new, historical, new+historical) == (run["new_probe_updates"], run["historical_probe_updates"],
                                                 run["total_probe_updates"])
        and historical == scope["reused_probe_updates"] and new <= scope["planned_new_probe_updates"],
        "Actual execution update total differs")
    return run, {str(owner): run["allocation_sha256"], str(run_path): run_sha}


def review_case(row, case, bound, local, scope, scope_sha, run_path):
    folder = campaign.folder_at(run_path.parent, case["name"])
    pack, state, report, probabilities, counts, files = campaign.checked_execution(folder, row["execution_sha256"],
        scope, scope_sha, case, local)
    reg.require(row["new_probe_updates"] == sum(counts.values()) and counts["completed"] == 0,
                "Actual completed update count differs")
    feature_check = audit.audit_pack(pack, bound, scope, case, local)
    metrics = audit.audit_probes(pack, scope, state, report, probabilities)
    result = {**row, **metrics, **feature_check, "role": case, "updates_by_path": counts,
              "status": "failed_numerical_gate" if metrics["numerical_failures"] else "accepted",
              "exact_continuous_split_and_completed": True, "audit_probe_updates": 0,
              "raw_gt_deserialized": False, "query_labels_accessed": False}
    return result, {str(folder/"execution.json"): row["execution_sha256"], **files}


def review(scope_path, scope_sha, run_path, run_sha, out, code):
    started = time.monotonic()
    scope_path, run_path = (Path(path).resolve(strict=True) for path in (scope_path, run_path))
    scope, launch, cases, inputs, reused, elapsed = registry.load(scope_path, scope_sha, code)
    run, artifacts = validate_run(run_path, run_sha, scope_path, scope_sha, scope, cases, reused, code)
    out = registry.output(out, launch, scope, scope_path.parent, run_path.parent)
    out.mkdir(exist_ok=False)
    results = []
    for row, (case, bound, local) in zip(run["cases"], cases, strict=True):
        if row["status"] == "reused":
            result = {**reused[case["name"]], **row, "role": case, "status": "accepted"}
        elif row["status"] == "executed":
            try:
                result, files = review_case(row, case, bound, local, registry.case_scope(scope, case), scope_sha, run_path)
                artifacts = registry.merge_pins(artifacts, files)
            except Exception as error:
                result = {**row, "role": case, "status": "failed_review", "error": repr(error)}
        else:
            result = {**row, "role": case}
        write_json(campaign.folder_at(out, case["name"]+".json"), result)
        results.append(result)
    tables = summary.write_outputs(out, results)
    reviewed = [row for row in results if "layers" in row]
    passed_cases = sum(row["status"] == "accepted" for row in results)
    failed_probes = sum(len({failure["layer"] for failure in row["numerical_failures"]}) for row in reviewed)
    complete = (passed_cases == len(results) == len(reviewed) == 54 and failed_probes == 0
                and run["status"] == "executed" and run["new_probe_updates"] == scope["planned_new_probe_updates"]
                and run["total_probe_updates"] == scope["expected_probe_updates"])
    all_pins = registry.merge_pins(inputs, artifacts)
    check_files(all_pins)
    require_frozen_code(code)
    result = {"policy": registry.POLICY, "status": "passed" if complete else "incomplete_or_failed",
        "commit": code["commit"], "synthetic": scope["synthetic"], "scope_sha256": scope_sha, "execution_sha256": run_sha,
        "history": scope["history"], "cases": [{k: row[k] for k in ("case", "scene", "disposition", "status",
                  "new_probe_updates", "historical_probe_updates")} for row in results],
        "accepted_cases": passed_cases, "reviewed_probes": len(reg.LAYERS)*len(reviewed),
        "accepted_probes": len(reg.LAYERS)*len(reviewed)-failed_probes, "failed_probes": failed_probes,
        "historical_probe_updates": run["historical_probe_updates"], "new_probe_updates": run["new_probe_updates"],
        "total_probe_updates": run["total_probe_updates"], "strict_decisions": complete,
        "audit_probe_updates": 0, "completed_resume_updates": 0, "backbone_updates": 0, "bn_updates": 0,
        "raw_gt_deserialized": False, "query_labels_accessed": False,
        "input_authentication_seconds": elapsed, "wall_seconds": time.monotonic()-started,
        "protected_files": all_pins, "protected_files_unchanged": True,
        "outputs": campaign.pins_for(out, (*tables, *(row["case"]+".json" for row in results)))}
    path = out/"receipt.json"
    write_json(path, result)
    if complete:
        write_json(out/"acceptance.json", {"status": "passed", "policy": registry.POLICY, "scope_sha256": scope_sha,
            "independent_receipt_sha256": hash_file(path), "cases": 54, "probes": 216,
            "historical_probe_updates": run["historical_probe_updates"], "new_probe_updates": run["new_probe_updates"],
            "total_probe_updates": run["total_probe_updates"], "synthetic": scope["synthetic"]})
    return path
