"""Register the frozen54 resident benchmark and resume it on one physical GPU."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp12_cost_preparation import POLICY as PREPARATION
from evaluation.wp12_cost_primitives import fields, read_pinned
from evaluation.wp12_evidence import COST_POLICY, accuracy_records, join_hardware
from evaluation.wp12_resident_cost import JOB_POLICY, MEMORY, admit_gpu, assemble, pinned_path, validate_job
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from scripts.measure_wp12_historical_cost import preserved_runtime


POLICY = "wp12-frozen54-resident-campaign-v1"
WORKER_COMMIT = "ad107501595f9b9cb625086e8457a852d7a60789"
BASELINE = "target_scratch_full/none"
WARMUPS, REPEATS, CASE_TIMEOUT = 3, 12, 900
CASE_FIELDS = "scene seed arm evaluation_sha256 checkpoint_sha256".split()


def read_spec(spec):
    fields(spec, "path sha256")
    if not Path(spec["path"]).is_absolute():
        raise ValueError("Absolute input artifact paths required")
    return read_pinned(spec["path"], spec["sha256"])


def source_accuracy(inputs):
    fields(inputs, "analysis receipt freeze")
    freeze = read_spec(inputs["freeze"])
    accuracy = accuracy_records(read_spec(inputs["analysis"]), read_spec(inputs["receipt"]), freeze,
                                analysis_sha=inputs["analysis"]["sha256"], freeze_sha=inputs["freeze"]["sha256"])
    return accuracy, freeze


def register(inputs, out):
    fields(inputs, "preparation accuracy")
    prepared = read_spec(inputs["preparation"])
    accuracy, freeze = source_accuracy(inputs["accuracy"])
    if (prepared["policy"] != PREPARATION or prepared["status"] != "prepared"
            or prepared["freeze_sha256"] != inputs["accuracy"]["freeze"]["sha256"]
            or prepared["analysis_sha256"] != inputs["accuracy"]["analysis"]["sha256"]
            or prepared["launch_sha256"] != freeze["launch_sha256"]
            or prepared["accuracy_protocol_sha256"] != accuracy["protocol_sha256"]):
        raise ValueError("Preparation and frozen accuracy must belong to the same cohort")
    key = lambda row: (row["scene"], row["seed"], row["arm"])
    lookup = {key(r): r for r in prepared["cases"]}
    if len(lookup) != len(prepared["cases"]) or lookup.keys() != {key(r) for r in accuracy["cases"]}:
        raise ValueError("Exactly the 54 frozen preparation cases required")
    if prepared["workloads"].keys() != {r["scene"] for r in accuracy["cases"]}:
        raise ValueError("Exactly the frozen scene workloads required")
    workloads = {}
    for scene, workload in prepared["workloads"].items():
        pinned_path(workload["file"])
        workloads[scene] = {k: workload[k] for k in ("file", "tensor_sha256", "work_items")}
    models = {(r["target"], r["seed"], f'{r["method"]}/{r["mode"]}'): r for r in freeze["models"]}
    for row in accuracy["cases"]:
        actual, model = lookup[key(row)], models[key(row)]
        folder = Path(freeze["training_root"]) / "cases" / model["name"]
        if ({k: actual[k] for k in CASE_FIELDS} != {k: row[k] for k in CASE_FIELDS}
                or actual["checkpoint"] != {"path": str(folder / "primary.pt"), "sha256": model["primary_sha256"]}
                or actual["sidecar"] != {"path": str(folder / "primary.pt.json"), "sha256": model["sidecar_sha256"]}):
            raise ValueError("Prepared case and checkpoint/sidecar must match the freeze")
        pinned_path(actual["checkpoint"])
        pinned_path(actual["sidecar"])
    # Rotate the six arms across the nine scene/seed blocks to vary serial position.
    arms = accuracy["arms"]
    blocks = sorted({key(row)[:2] for row in accuracy["cases"]})
    ordered = [lookup[(*block, arm)] for i, block in enumerate(blocks)
               for arm in arms[i % len(arms):] + arms[:i % len(arms)]]
    registration = dict(policy=POLICY, inputs=inputs, protocol_sha256=accuracy["protocol_sha256"],
                        baseline=BASELINE, worker_commit=WORKER_COMMIT, warmups=WARMUPS,
                        repeats=REPEATS, case_timeout_seconds=CASE_TIMEOUT, scope="resident_forward",
                        arithmetic="ordered", device="cuda", cases=ordered, workloads=workloads)
    out.mkdir()
    write_json(out / "registration.json", registration)
    return registration


def cost_plan(registration, environment):
    timing = {k: registration[k] for k in ("scope", "device", "warmups", "repeats")}
    timing["environment_sha256"] = digest(environment)
    return dict(policy=COST_POLICY, protocol_sha256=registration["protocol_sha256"], timing=timing,
                cases=[{k: row[k] for k in CASE_FIELDS} for row in registration["cases"]],
                workloads={s: dict(workload_sha256=w["tensor_sha256"], work_items=w["work_items"])
                           for s, w in registration["workloads"].items()})


def checked_result(result, job, job_sha, worker_commit, identity):
    if (result.get("job") != job or result.get("job_sha256") != job_sha
            or result.get("producer_commit") != worker_commit or result.get("fresh_process") is not True
            or any(result.get(k) != v for k, v in identity.items())):
        raise ValueError("Measurement differs from the registered job or preserved worker")
    return result


def publish(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing campaign artifact differs: {path.name}")
    else:
        write_json(path, value)


def execute(registration, out, accuracy, environment, identity, invoke, *, max_cases, finish):
    plan = cost_plan(registration, environment)
    publish(out / "cost-plan.json", plan)
    plan_sha = hash_file(out / "cost-plan.json")
    results, manifest, started = [], [], 0
    for i, (case, row) in enumerate(zip(plan["cases"], registration["cases"]), 1):
        job = dict(policy=JOB_POLICY, case=case, timing=plan["timing"], arithmetic=registration["arithmetic"],
                   checkpoint=row["checkpoint"], sidecar=row["sidecar"], workload=registration["workloads"][case["scene"]])
        validate_job(job)
        job_path, accepted = out / f"job-{i:02}.json", out / f"accepted-{i:02}.json"
        publish(job_path, job)
        job_sha = hash_file(job_path)
        if accepted.exists():
            spec = json.loads(accepted.read_text(encoding="utf-8"))
            result = read_spec(spec)
        else:
            if max_cases is not None and started >= max_cases:
                return len(results)
            attempt = Path(tempfile.mkdtemp(prefix=f"attempt-{i:02}-", dir=out))
            path = attempt / "result.json"
            invoke(job_path, job_sha, path)
            spec = dict(path=str(path), sha256=hash_file(path))
            result = read_spec(spec)
            started += 1
        checked_result(result, job, job_sha, registration["worker_commit"], identity)
        assemble(dict(plan, cases=[case]), [result], plan_sha256=plan_sha)
        publish(accepted, spec)
        results.append(result)
        manifest.append(spec)
    receipt = assemble(plan, results, plan_sha256=plan_sha)
    report = join_hardware(accuracy, plan, receipt, cost_plan_sha=plan_sha, baseline=registration["baseline"])
    finish()
    publish(out / "measurements.json", manifest)
    publish(out / "hardware-receipt.json", receipt)
    publish(out / "combined-report.json", dict(status="combined", measurement_manifest_sha256=hash_file(out / "measurements.json"),
                                              hardware_receipt_sha256=hash_file(out / "hardware-receipt.json"),
                                              accuracy_protocol_sha256=accuracy["protocol_sha256"],
                                              preregistered_benefit_conclusion=accuracy["preregistered_benefit_conclusion"],
                                              combined=report))
    return len(results)


def run_campaign(args, out, code):
    registration_path = out / "registration.json"
    registration = read_pinned(registration_path, args.registration_sha256)
    if (registration["policy"] != POLICY or registration["worker_commit"] != WORKER_COMMIT
            or registration["device"] != "cuda" or len(registration["cases"]) != 54):
        raise ValueError("Registered historical CUDA cohort required")
    worker = Path(args.worker_root).resolve(strict=True)
    if out == worker or worker in out.parents:
        raise ValueError("Campaign outputs must be outside the worker checkout")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu_uuid):
        raise ValueError("Explicit physical GPU UUID required")
    worker_code = code_snapshot(worker)
    if worker_code["commit"] != WORKER_COMMIT:
        raise ValueError("Historical worker commit differs from registration")
    accuracy, freeze = source_accuracy(registration["inputs"]["accuracy"])
    identity = preserved_runtime(freeze, worker_code)
    freeze_spec = registration["inputs"]["accuracy"]["freeze"]
    command = [sys.executable, "scripts/measure_wp12_historical_cost.py", "--expected-commit", WORKER_COMMIT,
               "--freeze", freeze_spec["path"], "--freeze-sha256", freeze_spec["sha256"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")

    def invoke(path, operation):
        admit_gpu({"gpu": {"uuid": args.gpu_uuid}})
        with (path.parent / "worker.log").open("xb") as log:
            subprocess.run(command + ["--out", str(path)] + operation, cwd=worker, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=registration["case_timeout_seconds"])

    with (out / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment_path = out / "environment.json"
        if not environment_path.exists():
            path = Path(tempfile.mkdtemp(prefix="bind-", dir=out)) / "environment.json"
            invoke(path, ["environment", "--device", "cuda", "--arithmetic", "ordered"])
            environment = json.loads(path.read_text(encoding="utf-8"))
        else:
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
        if (environment["device"] != "cuda" or environment["gpu"]["uuid"] != args.gpu_uuid
                or environment["arithmetic"] != "ordered" or environment["memory"] != MEMORY["cuda"]):
            raise ValueError("Campaign must stay on its bound physical GPU and ordered environment")
        publish(environment_path, environment)

        def finish():
            require_frozen_code(code)
            if hash_file(registration_path) != args.registration_sha256 or code_snapshot(worker) != worker_code:
                raise ValueError("Registration or worker changed during execution")
            source_accuracy(registration["inputs"]["accuracy"])

        count = execute(registration, out, accuracy, environment, identity,
                        lambda j, sha, dest: invoke(dest, ["measure", "--job", str(j), "--job-sha256", sha]),
                        max_cases=args.max_cases, finish=finish)
        print(f"Accepted {count}/54 frozen cost cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", required=True, type=Path)
    sub = parser.add_subparsers(dest="operation", required=True)
    plan = sub.add_parser("register")
    plan.add_argument("--inputs", required=True)
    plan.add_argument("--inputs-sha256", required=True)
    run = sub.add_parser("run")
    for name in ("registration-sha256", "worker-root", "gpu-uuid"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux campaign checkout required")
    out = external_directory(args.out)
    if args.operation == "register":
        if out.exists():
            raise ValueError("A new external campaign directory is required")
        register(read_pinned(args.inputs, args.inputs_sha256), out)
        print(hash_file(out / "registration.json"))
    else:
        if args.max_cases is not None and not 1 <= args.max_cases <= 54:
            parser.error("--max-cases must be in [1, 54]")
        run_campaign(args, out, code)


if __name__ == "__main__":
    main()
