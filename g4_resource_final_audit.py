"""Read-only supplement for Issue184's registered51+3 resource handoff.

Consumes the original full54 independent audit; never imports model/producer code.
Events prove recorded stage order, not an independent historical concurrency trace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PRODUCER = "331d718567b1a9726934f445665999df972392b1"
CONTROLLER = "5db71e82ad3fc36f3cef7046a22c1162be756097"
POLICY = "g4_issue184_resource_final_independent_v1"
STAGES = ("first2", "continue", "readonly", "audit")
RESULTS = {"first2": {"operation": "interrupted", "updates_this_call": 2},
           "continue": {"operation": "completed", "updates_this_call": 1498},
           "readonly": {"operation": "verified_completed", "updates_this_call": 0}}
WAIT = {"operation": "resource_wait", "updates_this_call": 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value):
    raise ValueError("Nonfinite JSON: " + value)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def keyed(rows):
    result = {row["name"]: row for row in rows}
    require(len(result) == len(rows), "Duplicate case")
    return result


def validate_events(jobs, gpus, events):
    require(len(jobs) == 3 and set(jobs.values()) == {"first2"}, "Registered three fresh jobs required")
    require(len(gpus) == 3 and len(set(gpus)) == 3, "Three distinct registered cards required")
    cursor = dict.fromkeys(jobs, 0)
    waits = updates = 0
    for event in events:
        name, stage = event["case"], event["stage"]
        require(name in cursor and cursor[name] < 4, "Unknown or already completed job")
        require(event["exit_code"] == 0 and event["gpu"] in gpus, "Failed stage or unregistered card")
        require(stage == STAGES[cursor[name]], "Missing, reordered or duplicate stage")
        result = event["result"]
        if result == WAIT:
            waits += 1
            continue
        if stage == "audit":
            require(set(result) == {"audit", "sha256"}, "Unexpected audit stage result")
        else:
            require(result == RESULTS[stage], "Exact2+1498+readonly0 budget required")
            updates += result["updates_this_call"]
        cursor[name] += 1
    require(set(cursor.values()) == {4} and updates == 4500, "Incomplete stage coverage/budget")
    return {"successful_stages": 12, "resource_wait_events": waits, "recorded_updates": updates}


def validate_rows(launch, acceptance, progress, audit, amendment):
    names = keyed(launch["cases"])
    rows = keyed(progress)
    accepted = keyed(acceptance["models"])
    audited = keyed(audit["cases"])
    require(len(names) == 54 and set(names) == set(rows) == set(accepted) == set(audited), "All54 identity required")
    require(rows == accepted, "Acceptance/progress mismatch")
    old = keyed(amendment["completed_rows"])
    require(len(old) == 51 and set(old).isdisjoint(amendment["jobs"]), "Original51 partition")
    require(set(old) | set(amendment["jobs"]) == set(rows), "Exact51+3 partition")
    require(all(rows[name] == row for name, row in old.items()), "Preserved51 progress changed")
    for name, row in rows.items():
        require(row["status"] == "passed" and row["updates"] == 1500 and row["validation_events"] == 51, "Case acceptance/budget")
        require(row["head_metrics"] == 6 and row["exact_reload_repeats"] == 3, "Metrics/reload counts")
        require(row["completed_resume_updates"] == 0 and row["query_labels_accessed"] is False, "Readonly/query contract")
        require(all(row[k] == value for k, value in audited[name].items()), "Independent model audit differs")
    totals = {k: sum(row[k] for row in rows.values()) for k in
              ("updates", "head_metrics", "class_recalls", "exact_reload_repeats")}
    require(totals == dict(updates=81000, head_metrics=324, class_recalls=4320, exact_reload_repeats=162), "Full54 totals")
    return totals


def checked_json(path, expected):
    require(sha(path) == expected, "Pinned JSON hash differs: " + str(path))
    return read(path)


def verify_files(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, "Protected file changed: " + str(path))


def merge_pins(destination, incoming):
    for path, expected in incoming.items():
        require(path not in destination or destination[path] == expected, "Conflicting protected file pins")
        destination[path] = expected


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_code(root, snapshot, expected):
    root = Path(root).resolve(strict=True)
    require(snapshot["commit"] == expected and snapshot["dirty"] is False, "Registered clean code required")
    require(git(root, "rev-parse", "HEAD") == expected and not git(root, "status", "--porcelain"), "Actual code differs/dirty")
    for relative, expected_sha in snapshot["files"].items():
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), "Code file outside checkout")
        require(sha(path) == expected_sha, "Frozen code file changed")
    return {"commit": expected, "branch": git(root, "branch", "--show-current"),
            "files": len(snapshot["files"])}


def audit_resource(args):
    started = time.monotonic()
    require(sys.platform == "linux" and __debug__, "Authorized Linux and enabled checks required")
    here = Path(__file__).resolve().parent
    require(git(here, "rev-parse", "HEAD") == args.expected_commit and not git(here, "status", "--porcelain"), "Clean expected consumer required")
    run = Path(args.run).resolve(strict=True)
    resources = Path(args.amendment).resolve(strict=True).parent
    out = Path(args.out).resolve()
    require(out.parent == run.parent and not out.exists() and out not in (run, resources), "Fresh sibling output required")
    launch_path, acceptance_path = run / "launch.json", run / "training-acceptance.json"
    launch = checked_json(launch_path, args.launch_sha256)
    acceptance = checked_json(acceptance_path, args.acceptance_sha256)
    audit = checked_json(args.audit, args.audit_sha256)
    amendment = checked_json(args.amendment, args.amendment_sha256)
    execution = acceptance["resource_execution"]
    require(Path(execution["amendment"]).resolve() == Path(args.amendment).resolve() and execution["sha256"] == args.amendment_sha256, "Acceptance resource binding")
    receipt_path = Path(execution["receipt"]).resolve(strict=True)
    require(receipt_path == resources / "receipt.json", "Resource receipt path")
    receipt = checked_json(receipt_path, execution["receipt_sha256"])
    require(audit["status"] == "passed" and audit["policy"] == "g4_matched_full54_independent_audit_v2", "Full54 independent audit required")
    require(audit["code"]["commit"] == audit["producer"]["commit"] == PRODUCER, "Original331 audit required")
    require(audit["launch_sha256"] == args.launch_sha256 and audit["acceptance_sha256"] == args.acceptance_sha256, "Independent audit input binding")
    require(audit["new_optimizer_updates"] == 0 and audit["query_labels_accessed"] is False and audit["test_scoring"] is False, "Independent audit access contract")
    require(acceptance["status"] == "passed" and acceptance["code"]["commit"] == PRODUCER and acceptance["optimizer_updates"] == 81000, "Producer acceptance/budget")
    require(acceptance["query_labels_accessed"] is False and acceptance["test_scoring"] is False, "Producer query contract")
    require(read(run / "status.json")["status"] == "completed", "Training not completed")
    require(amendment["policy"] == receipt["policy"] == "g4_registered_resource_handoff_v1", "Registered resource policy")
    require(Path(amendment["run"]).resolve() == run and amendment["gpus"] == [1, 3, 7], "Registered run/card pool")
    require(amendment["optimizer_updates_total"] == 81000 and amendment["remaining_updates"] == receipt["remaining_updates"] == 4500, "Handoff allocation budget")
    require(amendment["query_labels_accessed"] is False and receipt["query_labels_accessed"] is False, "Resource query contract")
    require(receipt["status"] == "passed" and receipt["amendment_sha256"] == args.amendment_sha256, "Resource receipt binding")
    require(not amendment["preserved_partial_files"], "This supplement covers registered51+3 fresh jobs")
    require(len(amendment["preserved_completed_files"]) == receipt["old_completed_files_verified_unchanged"] == 5763, "Preserved file count")
    require(amendment["launch"] == {"path": str(launch_path), "sha256": args.launch_sha256}, "Launch binding")
    require(acceptance["registration_sha256"] == amendment["registration"]["sha256"], "Registration binding")
    progress = read(run / "progress.json")
    totals = validate_rows(launch, acceptance, progress, audit, amendment)
    events_path = resources / "events.json"
    events = read(events_path)
    require(events == receipt["events"], "Final event ledger differs")
    stages = validate_events(amendment["jobs"], amendment["gpus"], events)
    require(len(events) == 13 and stages["resource_wait_events"] == 1, "Actual13 events including one resource wait required")
    for event in events:
        log = Path(event["log"]).resolve(strict=True)
        require(log.is_relative_to(resources / "logs"), "Event log path")
        require(sha(log) == event["log_sha256"], "Stage log changed")
        if "audit" in event["result"]:
            result = event["result"]
            path = Path(result["audit"]).resolve(strict=True)
            require(path == resources / "events" / (event["case"] + "-audit.json"), "Per-case audit path")
            row = checked_json(path, result["sha256"])
            final_row = keyed(progress)[event["case"]]
            require(all(final_row[k] == v for k, v in row.items()), "Recorded per-case audit differs")
    protected_path = run / "protected-files.json"
    protected = checked_json(protected_path, acceptance["protected_manifest_sha256"])
    require(len(protected) == 7930, "Original7930 protected inputs required")
    require(audit["protected_manifest_sha256"] == acceptance["protected_manifest_sha256"], "Independent protection binding")
    pins = dict(protected)
    merge_pins(pins, amendment["preserved_completed_files"])
    merge_pins(pins, amendment["pins"])
    merge_pins(pins, {str(p): sha(p) for folder in (run, resources) for p in folder.rglob("*") if p.is_file()})
    merge_pins(pins, {str(Path(args.audit).resolve()): args.audit_sha256,
                      str(Path(args.amendment).resolve()): args.amendment_sha256})
    verify_files(pins)
    producer = verify_code(amendment["frozen_checkout"], amendment["producer"], PRODUCER)
    controller = verify_code(amendment["controller_checkout"], amendment["controller_code"], CONTROLLER)
    verify_files(pins)
    verify_code(amendment["frozen_checkout"], amendment["producer"], PRODUCER)
    verify_code(amendment["controller_checkout"], amendment["controller_code"], CONTROLLER)
    require(git(here, "rev-parse", "HEAD") == args.expected_commit and not git(here, "status", "--porcelain"), "Consumer changed")
    result = {"policy": POLICY, "status": "passed", "consumer": {"commit": args.expected_commit, "branch": git(here, "branch", "--show-current"), "script_sha256": sha(__file__)},
              "producer": producer, "controller": controller,
              "launch_sha256": args.launch_sha256, "acceptance_sha256": args.acceptance_sha256,
              "full_audit_sha256": args.audit_sha256, "amendment_sha256": args.amendment_sha256,
              "resource_receipt_sha256": execution["receipt_sha256"], "events_sha256": sha(events_path),
              "totals": totals, "models": 54, "preserved_rows": 51, "preserved_completed_files": 5763,
              "protected_inputs": 7930, "files_checked_before_and_after": len(pins), "stages": stages,
              "new_optimizer_updates": 0, "query_labels_accessed": False, "test_scoring": False,
              "elapsed_seconds": time.monotonic() - started,
              "cost_scopes": {"acceptance_seconds": acceptance["elapsed_seconds"], "acceptance_scope": acceptance["elapsed_scope"],
                              "resource_controller_seconds": receipt["elapsed_seconds"]},
              "limits": ["Recorded stage ordering and zero-update waits only; no independent full historical concurrency timeline.",
                         "No model/probability recomputation in this supplement; original331 full audit provides selected-state verification."]}
    out.mkdir()
    (out / "protected-consumer-inputs.json").write_text(json.dumps(pins, indent=2) + "\n")
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(out / "receipt.json"), "sha256": sha(out / "receipt.json"), **result}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "run", "launch-sha256", "acceptance-sha256", "audit", "audit-sha256", "amendment", "amendment-sha256", "out"):
        parser.add_argument("--" + key, required=True)
    audit_resource(parser.parse_args())


if __name__ == "__main__":
    main()

