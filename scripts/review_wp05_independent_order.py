"""Re-run a bounded 22-class case through the original dispatcher for scientific parity."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_runtime import code_snapshot, external_directory, write_json
from transfer import p4_independent_registration as scope


def compare(resource_pin, out):
    resource = scope.read(resource_pin)
    science = scope.validate(resource, live_worker=True)
    scope.require(resource["mode"] == scope.CPU and resource["classes"] == 22,
                  "Expected the bounded CPU 22-class scheduling fixture")
    root = resource["worker"]["root"]
    candidate = Path(resource["execution"])
    case_id = resource["slots"]["cpu-b"]["cases"][0]
    case = next(case for case in science["cases"] if case["id"] == case_id)
    case_audit = candidate / "cases" / case_id / "audit.json"
    baseline = out / "cases" / case_id
    script = """
import sys
from pathlib import Path

from research_runtime import code_snapshot, write_json
from transfer import p4_full_budget_campaign as c, p4_full_budget_registration as r, p4_runtime_device as d

path, sha, destination, case_id = sys.argv[1:]
path, out = Path(path), Path(destination)
code = code_snapshot()
scope, _ = r.load(path, sha, code)
out = c.batch.sibling(out, path)
case, = (case for case in scope["cases"] if case["id"] == case_id)
out.mkdir()
(out / "cases").mkdir()
(out / "logs").mkdir()
write_json(out / "owner.json", c.owner(out, path, sha))
with d.lease(out / ".coordinator.lock") as coordinator, d.lease(c.slot_lock(out, None)) as cpu:
    for name, stops in scope["paths"].items():
        for stop in stops:
            c.invoke_operation(path, sha, out, code, case, "advance", stop, (coordinator, cpu), c.invoke, path_name=name)
        c.invoke_operation(path, sha, out, code, case, "completed", None, (coordinator, cpu), c.invoke, path_name=name)
    c.invoke_operation(path, sha, out, code, case, "audit", None, (coordinator, cpu), c.invoke)
    c.accepted(c.bound_folder(out, case), scope, case, sha)
"""
    subprocess.run([sys.executable, "-c", script, resource["science"]["path"], resource["science"]["sha256"],
                    str(out), case_id], cwd=root, env=scope.environment(root), check=True)
    baseline_pin, queue_pin = scope.pin(baseline / "audit.json"), scope.pin(case_audit)
    left, right = scope.read(baseline_pin), scope.read(queue_pin)
    fields = ["exact_path_fingerprints", "selected_step", "fit", "exposure", "populations", "coverage_participation"]
    scope.require(all(left[key] == right[key] for key in fields) and left["optimizer_updates"] == 120,
                  "Changing queue order changed scientific output")
    result = {"status": "passed", "case": case, "resource": resource_pin, "original_worker_launch": baseline_pin,
              "independent_queue": queue_pin, "identical_fields": fields, "baseline_updates": 120,
              "audit_updates": 0, "raw_gt_loaded": False, "query_accessed": False}
    write_json(out / "schedule-parity.json", result)
    print(json.dumps(scope.pin(out / "schedule-parity.json")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    scope.require(code["commit"] == args.expected_commit and not code["dirty"], "Review code identity differs")
    compare(scope.pin(args.resource), external_directory(args.out))


if __name__ == "__main__":
    main()
