"""Real source-only three-role engineering check: 3*(2+6) updates, no query."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from g4_none_source_launch import prepare_launch, select_bound_case
from loadData.dataset_registry import file_hash
from research.reproduction.m4g4b.none_source_evidence import audit_case
from research_runtime import code_snapshot, require_frozen_code, write_json
from source_data_context import load_source_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "reference-launch", "reference-launch-sha256", "out-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protected-root", action="append", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean remote expected commit required")
    roots = [Path(p).resolve(strict=True) for p in args.protected_root]
    pins = {str(p): file_hash(p) for root in roots for p in root.rglob("*") if p.is_file()}
    if not pins:
        raise ValueError("Existing frozen source cohorts must be protected")
    started = time.monotonic()
    launch = prepare_launch(args.reference_launch, args.reference_launch_sha256, args.out_dir, code, full=False)
    root = Path(launch["out_dir"])
    lp = root / "launch.json"
    sha = file_hash(lp)
    pins[str(lp)] = sha
    write_json(root / "protected-files.json", pins)
    rows = []
    for case in launch["cases"]:
        select_bound_case(launch, code, case["held_out"], case["seed"], full=False)
        command = [sys.executable, "scripts/train_none_source_ordered_short.py", "--expected-commit", args.expected_commit,
                   "--launch-plan", str(lp), "--launch-sha256", sha, "--held-out", case["held_out"], "--seed", str(case["seed"])]
        operations = []
        for suffix, extra in (("first2", ["--stop-after", "2"]), ("continue", ["--resume"]), ("readonly", ["--resume"])):
            folder = Path(case["out"])
            before = {str(p): file_hash(p) for p in folder.rglob("*") if p.is_file()} if suffix == "readonly" else None
            with (root / "logs" / (case["name"] + "-" + suffix + ".log")).open("x") as log:
                subprocess.run(command + extra, check=True, stdout=log, stderr=subprocess.STDOUT)
            value = json.loads((root / "logs" / (case["name"] + "-" + suffix + ".log")).read_text().splitlines()[-1])
            if (value["operation"], value["updates_this_call"]) != {"first2": ("interrupted", 2), "continue": ("completed", 6),
                                                                   "readonly": ("verified_completed", 0)}[suffix]:
                raise ValueError("Engineering interruption/resume budget differs")
            if before is not None and before != {str(p): file_hash(p) for p in folder.rglob("*") if p.is_file()}:
                raise ValueError("Read-only source recovery changed files")
            operations.append(value)
        context = load_source_context(launch["prepared"], launch["data_root"], held_out=case["held_out"],
            consumer=code, manifest_sha256=launch["prepared_manifest_sha256"])
        pins.update(context.record["protected_files"])
        row = audit_case(launch, case, context, sha)
        row["operations"] = operations
        rows.append(row)
        write_json(root / "progress.json", rows)
        print("NONE_ORDERED_SHORT_CASE", row, flush=True)
        del context
    require_frozen_code(code)
    if any(file_hash(p) != h for p, h in pins.items()):
        raise ValueError("Protected historical source/input/launch changed")
    if sum(r["metrics"] for r in rows) != 36 or sum(r["recalls"] for r in rows) != 480:
        raise ValueError("Complete independent short source metric coverage differs")
    write_json(root / "receipt.json", {"status": "passed", "code": code, "launch_sha256": sha,
        "optimizer_updates": 24, "cases": rows, "protected_files_and_inputs": len(pins),
        "query_labels_accessed": False, "test_scoring": False, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
