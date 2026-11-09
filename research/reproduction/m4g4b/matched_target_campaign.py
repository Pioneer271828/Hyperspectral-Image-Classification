"""Registered single-GPU full54 queue; no retries by score and no query reader."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from g4_matched_target_audit import ACCEPTANCE_POLICY, audit_case, check_files, files
from g4_matched_target_execution import freeze_campaign, read_registration
from g4_matched_target_study import select_case
from loadData.dataset_registry import file_hash
from research.reproduction.m4g4b.none_source_campaign import wait_gpu
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


def run_queue(launch, registration, registration_path, registration_sha256, code):
    root = external_directory(launch["out_dir"])
    lp, rows, started = root / "launch.json", [], time.monotonic()
    sha = file_hash(lp)
    protected = json.loads((root / "protected-files.json").read_text())
    try:
        for case in launch["cases"]:
            select_case(launch, code, case["target"], case["seed"], case["method"], case["mode"], phase="full_campaign")
            folder, begin = Path(case["out"]), time.monotonic()
            command = [sys.executable, "scripts/train_g4_matched_target_full.py", "--expected-commit", code["commit"],
                "--registration", str(registration_path), "--registration-sha256", registration_sha256,
                "--launch-plan", str(lp), "--launch-sha256", sha, "--target", case["target"], "--seed", str(case["seed"]),
                "--method", case["method"], "--mode", case["mode"]]
            for suffix, extra in (("first2", ["--stop-after", "2"]), ("continue", ["--resume"]), ("readonly", ["--resume"])):
                wait_gpu(registration["gpu"], root)
                write_json(root / "status.json", {"status": "running", "case": case["name"], "operation": suffix,
                    "completed_cases": len(rows), "planned_optimizer_updates": 81000})
                before = files(folder) if suffix == "readonly" else None
                path = root / "logs" / (case["name"] + "-" + suffix + ".log")
                with path.open("x") as log:
                    subprocess.run(command + extra, check=True, stdout=log, stderr=subprocess.STDOUT)
                operation = json.loads(path.read_text().splitlines()[-1])
                expected = {"first2": ("interrupted", 2), "continue": ("completed", 1498), "readonly": ("verified_completed", 0)}[suffix]
                if (operation["operation"], operation["updates_this_call"]) != expected:
                    raise ValueError("Full target interrupted/resumed/completed budget differs")
                if before is not None and before != files(folder):
                    raise ValueError("Completed full target recovery mutated files")
            wait_gpu(registration["gpu"], root)
            row = audit_case(case, code, sha, registration_sha256)
            row.update(completed_resume_updates=0, elapsed_seconds=time.monotonic() - begin)
            rows.append(row)
            write_json(root / "progress.json", rows)
            print("MATCHED_FULL_TARGET_CASE", case["name"], row["selected_step"], row["elapsed_seconds"], flush=True)
            torch.cuda.empty_cache()
        require_frozen_code(code)
        check_files(protected)
        if (len(rows) != 54 or sum(r["updates"] for r in rows) != 81000
                or sum(r["head_metrics"] for r in rows) != 324 or sum(r["class_recalls"] for r in rows) != 4320
                or sum(r["exact_reload_repeats"] for r in rows) != 162):
            raise ValueError("Complete full54 independent acceptance coverage required")
        write_json(root / "training-acceptance.json", {"policy": ACCEPTANCE_POLICY, "status": "passed", "code": code,
            "launch_sha256": sha, "registration_sha256": registration_sha256, "optimizer_updates": 81000,
            "models": rows, "protected_manifest_sha256": file_hash(root / "protected-files.json"),
            "protected_files_and_inputs": len(protected), "query_labels_accessed": False, "test_scoring": False,
            "elapsed_seconds": time.monotonic() - started, "independent_audit_required": True})
        write_json(root / "status.json", {"status": "completed", "completed_cases": 54, "optimizer_updates": 81000,
            "independent_audit_required": True})
    except BaseException as error:
        write_json(root / "status.json", {"status": "failed", "completed_cases": len(rows),
            "error": type(error).__name__ + ": " + str(error), "retry_policy": "preserve_and_diagnose"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "registration", "registration-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected registered remote commit required")
    registration = read_registration(args.registration, args.registration_sha256, code)
    launch = freeze_campaign(args.registration, args.registration_sha256, code)
    run_queue(launch, registration, Path(args.registration).resolve(), args.registration_sha256, code)


if __name__ == "__main__":
    main()
