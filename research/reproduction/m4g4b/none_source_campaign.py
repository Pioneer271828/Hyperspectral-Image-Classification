"""Single-GPU nine-source control campaign after accepted real short engineering."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from g4_none_source_launch import prepare_launch, select_bound_case
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research.reproduction.m4g4b.none_source_evidence import audit_case
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from source_data_context import load_source_context


def accepted_short(root, receipt_sha256, consumer):
    root = Path(root).resolve(strict=True)
    rp, lp = root / "receipt.json", root / "launch.json"
    if file_hash(rp) != receipt_sha256:
        raise ValueError("Pinned real short acceptance changed")
    receipt, launch = json.loads(rp.read_text()), json.loads(lp.read_text())
    producer = launch["code"]
    if (receipt["status"] != "passed" or receipt["code"] != producer or producer["dirty"]
            or producer["source_hash"] != digest(producer["files"]) or consumer["dirty"]
            or any(consumer["files"].get(k) != sha for k, sha in producer["files"].items())
            or receipt["launch_sha256"] != file_hash(lp) or receipt["optimizer_updates"] != 24
            or receipt["query_labels_accessed"] is not False or receipt["test_scoring"] is not False
            or len(receipt["cases"]) != 3 or sum(c["metrics"] for c in receipt["cases"]) != 36
            or sum(c["recalls"] for c in receipt["cases"]) != 480
            or Path(launch["out_dir"]).resolve() != root):
        raise ValueError("Real short coverage or additive campaign implementation differs")
    rows = {r["name"]: r for r in receipt["cases"]}
    if len(rows) != 3:
        raise ValueError("Distinct real short roles required")
    for case in launch["cases"]:
        select_bound_case(launch, producer, case["held_out"], case["seed"], full=False)
        row = rows[case["name"]]
        folder = Path(case["out"])
        if (row["updates"] != 8 or row["exact_reload_repeats"] != 3 or row["metrics"] != 12
                or row["recalls"] != 6 * sum(DATASETS[n][2] for n in case["settings"]["source_tasks"])
                or file_hash(folder / "result.json") != row["result_sha256"]):
            raise ValueError("Accepted short case changed")
        result = json.loads((folder / "result.json").read_text())
        if (result["status"] != "completed" or result["optimizer_steps"] != 8
                or result["start_code"] != producer or result["settings"] != case["settings"]):
            raise ValueError("Accepted short result identity differs")
        for name, sha in result["artifacts"].items():
            if Path(name).name != name or file_hash(folder / name) != sha:
                raise ValueError("Accepted short artifact changed")
    return receipt


def wait_gpu(index, root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(index):
        raise ValueError("Explicit single physical GPU queue required")
    while True:
        memory, util = [int(x.strip()) for x in subprocess.check_output(
            ["nvidia-smi", "-i", str(index), "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).strip().split(",")]
        if memory >= 12288 and util <= 60:
            return
        write_json(root / "status.json", {"status": "waiting_for_resources", "gpu": index,
                   "free_mib": memory, "utilization": util})
        time.sleep(30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "reference-launch", "reference-launch-sha256", "short-run",
                 "short-receipt-sha256", "out-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--protected-root", action="append", required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean remote expected commit required")
    accepted_short(args.short_run, args.short_receipt_sha256, code)
    roots = [Path(x).resolve(strict=True) for x in [args.short_run, *args.protected_root]]
    out = external_directory(args.out_dir)
    if any(out == r or r in out.parents or out in r.parents for r in roots):
        raise ValueError("Fresh independent campaign output required")
    protected = {str(p): file_hash(p) for r in roots for p in r.rglob("*") if p.is_file()}
    launch = prepare_launch(args.reference_launch, args.reference_launch_sha256, out, code, full=True)
    lp, started = out / "launch.json", time.monotonic()
    sha = file_hash(lp)
    protected[str(lp)] = sha
    for case in launch["cases"]:
        protected.update(case["source_context"]["protected_files"])
    write_json(out / "protected-files.json", protected)
    write_json(out / "registration.json", {"code": code, "launch_sha256": sha,
        "short_receipt_sha256": args.short_receipt_sha256, "short_run": args.short_run,
        "expected_optimizer_updates": 9018, "gpu": args.gpu, "query_labels_accessed": False})
    rows = []
    try:
        for case in launch["cases"]:
            select_bound_case(launch, code, case["held_out"], case["seed"], full=True)
            command = [sys.executable, "scripts/train_none_source_ordered_full.py", "--expected-commit", args.expected_commit,
                "--launch-plan", str(lp), "--launch-sha256", sha, "--held-out", case["held_out"], "--seed", str(case["seed"])]
            start = time.monotonic()
            folder = Path(case["out"])
            for suffix, extra in (("first2", ["--stop-after", "2"]), ("continue", ["--resume"]), ("readonly", ["--resume"])):
                wait_gpu(args.gpu, out)
                write_json(out / "status.json", {"status": "running", "case": case["name"], "operation": suffix,
                           "completed_cases": len(rows), "expected_cases": 9, "launch_sha256": sha})
                before = {str(p): file_hash(p) for p in folder.rglob("*") if p.is_file()} if suffix == "readonly" else None
                log_path = out / "logs" / (case["name"] + "-" + suffix + ".log")
                with log_path.open("x") as log:
                    subprocess.run(command + extra, check=True, stdout=log, stderr=subprocess.STDOUT)
                value = json.loads(log_path.read_text().splitlines()[-1])
                if (value["operation"], value["updates_this_call"]) != {"first2": ("interrupted", 2),
                        "continue": ("completed", 1000), "readonly": ("verified_completed", 0)}[suffix]:
                    raise ValueError("Full source execution/resume budget differs")
                if before is not None and before != {str(p): file_hash(p) for p in folder.rglob("*") if p.is_file()}:
                    raise ValueError("Completed source recovery changed files")
            context = load_source_context(launch["prepared"], launch["data_root"], held_out=case["held_out"],
                consumer=code, manifest_sha256=launch["prepared_manifest_sha256"])
            row = audit_case(launch, case, context, sha)
            row.update(elapsed_seconds=time.monotonic() - start, completed_resume_updates=0,
                       primary_sha256=file_hash(folder / "primary.pt"), sidecar_sha256=file_hash(folder / "primary.pt.json"))
            rows.append(row)
            write_json(out / "progress.json", rows)
            print("NONE_ORDERED_FULL_CASE", {k: row[k] for k in ("name", "updates", "selected_step", "elapsed_seconds")}, flush=True)
            del context
            torch.cuda.empty_cache()
        require_frozen_code(code)
        if any(file_hash(p) != h for p, h in protected.items()):
            raise ValueError("Protected historical sources/short/input/launch changed")
        write_json(out / "training-acceptance.json", {"status": "passed", "code": code, "launch_sha256": sha,
            "optimizer_updates": 9018, "cases": rows, "protected_files_and_inputs": len(protected),
            "query_labels_accessed": False, "test_scoring": False, "elapsed_seconds": time.monotonic() - started})
        write_json(out / "status.json", {"status": "completed", "completed_cases": 9, "optimizer_updates": 9018,
                   "independent_audit_required": True})
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "completed_cases": len(rows),
                   "error": type(error).__name__ + ": " + str(error), "retry_policy": "preserve_and_diagnose"})
        raise


if __name__ == "__main__":
    main()
