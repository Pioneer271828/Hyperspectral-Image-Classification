"""Durable registered 2–3 GPU queues and independent acceptance for light adaptation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time

from g4_light_full_audit import audit_case
import g4_light_engineering_registration as engineering
from g4_light_engineering_devices import LEASE_FD, check_child, device_admitted, open_lease
from g4_light_full_target import MEMORY_FIELD, UPDATES, claim_output, device_for, read_registration, reference_launch, require_gpu, roles
from g4_light_target import CONTROLS
from g4_matched_target_audit import check_files, files, read
from g4_matched_target_evaluation import new_output
from g4_matched_target_execution import unchanged_producer
from g4_matched_target_study import METHODS, MODES, SEEDS, case_name, select_case
from joint_baseline import NAMES
from loadData.dataset_registry import file_hash
from research_runtime import ROOT, code_snapshot, require_frozen_code, write_json


POLICY = "g4_light_full_multigpu_acceptance_v1"
RESOURCE_ERROR = re.compile(r"RuntimeError: GPU \d+ is below the resource gate \(\d+ MiB free, \d+% utilized\)")


def case_directory(root, key):
    target, seed, method, mode = key
    name = case_name(target, seed, METHODS[1], mode).removesuffix(METHODS[1]) + method
    folder = root / "cases" / name
    if folder.resolve() != folder:
        raise ValueError("Campaign case path cannot traverse a symlink")
    return folder


def next_stage(folder):
    if not folder.exists():
        return "first2", ["--stop-after", "2"], ("interrupted", 2)
    progress = read(folder / "progress.json")
    cursor = progress["cursor"]
    if type(cursor) is not int or not 0 < cursor <= UPDATES or progress["state_file"] != f"state-{cursor:06d}.pt":
        raise ValueError("Existing case has no trustworthy recovery cursor")
    if file_hash(folder / progress["state_file"]) != progress["state_sha256"]:
        raise ValueError("Recovery checkpoint changed")
    if (folder / "result.json").is_file():
        if cursor != UPDATES or progress["status"] != "completed":
            raise ValueError("Completed case has an inconsistent cursor")
        return "readonly", ["--resume"], ("verified_completed", 0)
    if progress["status"] not in ("running", "interrupted", "completed"):
        raise ValueError("Unrecognized light worker progress")
    return "resume", ["--resume"], ("completed", UPDATES - cursor)


def eligible(gpu):
    if device_for(gpu) == "cpu":
        return True
    free, util = map(int, subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip().split(","))
    return free >= 12288 and util <= 60


def invoke(root, folder, gpu, command, stage, lease, *, shared=None):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if device_for(gpu) == "cpu" else str(gpu), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    descriptors = (lease.fileno(),)
    if shared is not None:
        env[LEASE_FD] = str(shared.fileno())
        descriptors += (shared.fileno(),)
    with tempfile.NamedTemporaryFile(dir=root / "logs", prefix=f"{folder.name}-{stage}-",
                                    suffix=".log", delete=False) as log:
        # The dispatch lease stays held by a live child after a hard coordinator exit.
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 pass_fds=descriptors)
    lines = Path(log.name).read_text().splitlines()
    if process.returncode:
        if lines and RESOURCE_ERROR.fullmatch(lines[-1]):
            return None
        raise RuntimeError(f"{stage} failed for {folder.name}; evidence {log.name}")
    return json.loads(lines[-1])


def run_role(root, key, gpu, base, stop, publish, lease, *, engineering_role=None, shared=None, uuid=None):
    folder = case_directory(root, key)
    role = ["--target", key[0], "--seed", str(key[1]), "--method", key[2], "--mode", key[3]]
    worker = [sys.executable, "scripts/train_g4_light_target_full.py", *base, *role]
    audit = [sys.executable, "-m", "g4_light_full_campaign", "audit", *base, *role]
    options = {"shared":shared} if shared is not None else {}
    wait_seconds = 30 if engineering_role is None else 5
    waiting_since = time.monotonic()

    def admitted():
        return device_admitted(gpu,uuid) if uuid else eligible(gpu)

    def wait(stage):
        if engineering_role is not None and time.monotonic()-waiting_since > 300:
            raise RuntimeError(f"Synthetic resource wait expired for {folder.name} {stage}")
        publish(gpu, "waiting_for_resources", folder.name, stage)
        stop.wait(wait_seconds)

    while not stop.is_set():
        stage, flags, expected = next_stage(folder)
        if not admitted():
            wait(stage)
            continue
        if stop.is_set():
            return None
        if publish(gpu, "running", folder.name, stage) is False:
            return None
        before = files(folder) if stage == "readonly" else None
        outcome = invoke(root, folder, gpu, worker + flags, stage, lease, **options)
        if outcome is None:
            wait(stage)
            continue
        waiting_since = time.monotonic()
        if (outcome["operation"], outcome["updates_this_call"]) != expected:
            raise ValueError("Worker operation or recovery budget differs")
        if stage == "readonly":
            if files(folder) != before:
                raise ValueError("Completed light target recovery changed files")
            break
    else:
        return None
    while not stop.is_set():
        if not admitted():
            wait("audit")
            continue
        if publish(gpu, "running", folder.name, "audit") is False:
            return None
        row = invoke(root, folder, gpu, audit, "audit", lease, **options)
        if row is None:
            wait("audit")
            continue
        if (row["name"] != folder.name or row["status"] != "passed" or row["role"] != list(key)
                or row["physical_gpu"] != gpu or row["updates"] != UPDATES or row["validation_events"] != 51
                or row["exact_reload_repeats"] != 3 or row["new_optimizer_updates"] != 0
                or row["query_labels_accessed"] is not False):
            raise ValueError("Independent light case acceptance differs")
        if engineering_role is not None and (row.get("ledger") != engineering_role
                or row.get("scope") != engineering.SCOPE or row.get(MEMORY_FIELD) != engineering_role["strategy"]):
            raise ValueError("Independent synthetic role or adapter policy differs")
        write_json(root / "audits" / (folder.name + ".json"), row)
        return row
    return None


def run_queues(root, gpus, base, code, registration_sha256, pins, acceptance_sha256, *, resume, synthetic=None):
    claim_output(root, registration_sha256)
    with (root / ".coordinator.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = root / "status.json"
        if resume != status_path.is_file():
            raise ValueError("Resume requires existing campaign status")
        header = {"policy": POLICY, "registration_sha256": registration_sha256,
                  "campaign_engineering_sha256": acceptance_sha256, "gpu_ids": gpus}
        grid = roles()
        scheduled = {}
        uuid_by_slot = {}
        if synthetic is not None:
            scheduled = {tuple(row["role"]):row for row in synthetic["roles"] if row["scheduled"]}
            grid = list(scheduled)
            uuid_by_slot = {row["slot"]:row.get("uuid") for row in synthetic["devices"]}
            header.update(scope=engineering.SCOPE, producer=engineering.code_identity(code),
                          planned_cases=len(synthetic["roles"]), planned_optimizer_updates=synthetic["planned_optimizer_updates"],
                          scheduled_cases=len(grid), scheduled_optimizer_updates=synthetic["scheduled_optimizer_updates"],
                          real_training_authorized=False)
        accepted_path = root / ("engineering-acceptance.json" if synthetic is not None else "training-acceptance.json")
        if resume:
            previous = read(status_path)
            if any(previous[k] != value for k, value in header.items()) or previous["status"] == "failed":
                raise ValueError("Campaign identity differs or a failure requires diagnosis")
            if accepted_path.is_file():
                accepted = read(accepted_path)
                expected_names = [case_directory(root, key).name for key in grid]
                if (any(accepted[k] != value for k, value in header.items()) or accepted["status"] != "passed"
                        or accepted["code"] != code or accepted["cases"] != expected_names
                        or set(accepted["audit_receipts"]) != {str(root / "audits" / (name + ".json")) for name in expected_names}
                        or accepted["optimizer_updates"] != len(grid) * UPDATES
                        or accepted["query_labels_accessed"] is not False):
                    raise ValueError("Published acceptance identity differs")
                check_files(accepted["protected_files"])
                check_files(accepted["audit_receipts"])
                if previous["status"] != "completed":
                    write_json(status_path, {**previous, "status": "completed"})
                return
            if previous["status"] == "completed":
                raise ValueError("Completed campaign lacks acceptance")
        for name in ("logs", "audits"):
            (root / name).mkdir(exist_ok=True)
        started = time.monotonic()
        slots, completed, errors = {}, {}, {}
        mutex, stop = threading.Lock(), threading.Event()

        def write_status(status):
            write_json(status_path, {"planned_cases":len(grid), "planned_optimizer_updates":len(grid)*UPDATES,
                **header, "status": status, "slots": slots, "completed_cases": len(completed), "errors": errors})

        def publish(gpu, status, name=None, stage=None):
            with mutex:
                if status == "running" and stop.is_set():
                    return False
                slots[str(gpu)] = {"status": status, "case": name, "stage": stage}
                waiting = slots and all(s["status"] in ("waiting_for_resources", "finished") for s in slots.values())
                write_status("draining_after_failure" if errors else "waiting_for_resources" if waiting else "running")

        def queue(index, gpu):
            try:
                with ExitStack() as stack:
                    dispatch = stack.enter_context((root / f".dispatch-{gpu}.lock").open("a"))
                    uuid = uuid_by_slot.get(gpu)
                    shared = stack.enter_context(open_lease(uuid)) if uuid else None
                    started_waiting = time.monotonic()
                    while not stop.is_set():
                        try:
                            fcntl.flock(dispatch, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            if shared is not None:
                                fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            if synthetic is not None and time.monotonic()-started_waiting > 300:
                                raise RuntimeError("Synthetic worker lease wait expired")
                            publish(gpu, "waiting_for_worker")
                            stop.wait(5 if synthetic is not None else 30)
                        else:
                            break
                    owned = [key for key in grid if scheduled[key]["slot"] == gpu] if scheduled else grid[index::len(gpus)]
                    for key in owned:
                        if stop.is_set():
                            break
                        options = {"engineering_role":scheduled[key], "shared":shared, "uuid":uuid} if scheduled else {}
                        row = run_role(root, key, gpu, base, stop, publish, dispatch, **options)
                        if row is not None:
                            with mutex:
                                completed[key] = row
                                write_status("draining_after_failure" if errors else "running")
                publish(gpu, "finished")
            except Exception as error:
                stop.set()
                with mutex:
                    errors[str(gpu)] = type(error).__name__ + ": " + str(error)
                    slots[str(gpu)] = {**slots.get(str(gpu), {}), "status": "failed"}
                    write_status("draining_after_failure")

        write_status("running")
        try:
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = [pool.submit(queue, i, gpu) for i, gpu in enumerate(gpus)]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    stop.set()
                    raise
            if errors or len(completed) != len(grid):
                raise RuntimeError("Incomplete or failed light campaign; see status and preserved logs")
            require_frozen_code(code)
            all_pins = dict(pins)
            for row in completed.values():
                for group in ("protected_inputs", "artifacts"):
                    for path, sha in row[group].items():
                        if all_pins.setdefault(path, sha) != sha:
                            raise ValueError("Conflicting full campaign artifact pin")
            check_files(all_pins)
            audit_pins = {str(root / "audits" / (row["name"] + ".json")):
                          file_hash(root / "audits" / (row["name"] + ".json")) for row in completed.values()}
            write_json(accepted_path, {**header, "status": "passed", "code": code,
                "cases": [completed[key]["name"] for key in grid], "audit_receipts": audit_pins,
                "protected_files": all_pins, "optimizer_updates": len(grid) * UPDATES,
                "query_labels_accessed": False, "elapsed_seconds_this_coordinator": time.monotonic() - started})
            write_status("completed")
        except BaseException as error:
            errors.setdefault("coordinator", type(error).__name__ + ": " + str(error))
            write_status("failed" if isinstance(error, Exception) else "interrupted")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run, audit = sub.add_parser("run"), sub.add_parser("audit")
    for entry in (run, audit):
        for field in ("expected-commit", "registration", "registration-sha256"):
            entry.add_argument("--" + field, required=True)
    run.add_argument("--campaign-engineering")
    run.add_argument("--campaign-engineering-sha256")
    run.add_argument("--resume", action="store_true")
    audit.add_argument("--target", choices=NAMES, required=True)
    audit.add_argument("--seed", type=int, choices=SEEDS, required=True)
    audit.add_argument("--method", choices=CONTROLS, required=True)
    audit.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean registered Linux commit required")
    registration, pins = read_registration(args.registration, args.registration_sha256, code)
    synthetic = registration["policy"] == engineering.REGISTRATION
    launch = reference_launch(registration)
    unchanged_producer(launch["code"], code)
    select_case(launch, launch["code"], NAMES[0], 64, METHODS[1], "none",
                phase=engineering.PHASE if synthetic else "full_campaign")
    root = new_output(registration["out_dir"], launch)
    gpus = registration["gpu_ids"]
    if args.action == "audit":
        key = (args.target, args.seed, args.method, args.mode)
        gpu = gpus[roles().index(key) % len(gpus)]
        with (root / f".gpu-{gpu}.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if synthetic:
                check_child(registration,gpu)
            require_gpu(gpu)
            result = audit_case(args.registration, args.registration_sha256, code, key, device=device_for(gpu))
        print(json.dumps(result, allow_nan=False), flush=True)
        return
    base = ["--expected-commit", code["commit"], "--registration", str(Path(args.registration).resolve()),
            "--registration-sha256", args.registration_sha256]
    if synthetic:
        if args.campaign_engineering or args.campaign_engineering_sha256:
            raise ValueError("Legacy campaign proof cannot be attached to synthetic execution")
        run_queues(root, gpus, base, code, args.registration_sha256, pins, None,
                   resume=args.resume, synthetic=registration)
        return
    if not args.campaign_engineering or not args.campaign_engineering_sha256:
        raise ValueError("Pinned campaign engineering proof is required")
    proof = Path(args.campaign_engineering).resolve(strict=True)
    if root in proof.parents or file_hash(proof) != args.campaign_engineering_sha256:
        raise ValueError("Pinned campaign engineering proof required outside the output")
    accepted = read(proof)
    if (accepted["issue"] != 247 or accepted["status"] != "engineering_passed"
            or any(accepted[k] is not True for k in ("queue_recovery_passed", "cpu_audit_passed", "gpu_audit_passed"))
            or accepted["real_training_updates"] != 0 or accepted["query_labels_accessed"] is not False):
        raise ValueError("Accepted campaign and independent audit required")
    if accepted.get(MEMORY_FIELD) != registration.get(MEMORY_FIELD):
        raise ValueError("Campaign acceptance does not cover the registered adapter policy")
    unchanged_producer(accepted["code"], code)
    pins[str(proof)] = args.campaign_engineering_sha256
    if not args.resume and not all(eligible(gpu) for gpu in gpus):
        raise RuntimeError("All registered cards must meet the resource gate at campaign launch")
    run_queues(root, gpus, base, code, args.registration_sha256, pins,
               args.campaign_engineering_sha256, resume=args.resume)


if __name__ == "__main__":
    main()
