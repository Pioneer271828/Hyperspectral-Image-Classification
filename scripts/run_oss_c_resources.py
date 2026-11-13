"""Register and execute the original OSS-C cells with whole-seed GPU queues."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock, Thread
from time import monotonic

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.research_config import digest
from research_runtime import code_snapshot, hash_file, require_frozen_code
from scripts.run_oss_c_pair import ARMS, SEEDS, campaign, release, run_case
from transfer.container_runtime import training_container_kind


OLD_COMMIT = "a36a389d1ff55a77f921cf7e829f42f5abd1beaa"
OLD_SOURCE = "cfde802fc22458c36e6a42698ccbfa1d55724f096f2c4f9fa4999d8a2e07cc44"
OLD_SHA = "8186b1fa5f822247a4f2fa5dd39f1825f45d9d553d0962c6f9b075dea8fef57e"
OLD_PATH = Path("oss-c-review01/real-registration04/registration.json")
RELATIVE_ROOT = Path("oss-c-resource-short01")
MAIN_CHAT = "01a1aa2e-9fe8-7782-a6e9-327ef5f16c57"
AUTHORIZATION = "OSS-C-R：自然空闲设备登记、按种子配对的多卡调度、108真实短测及完整独立审计"
DEVICE = "cuda:0"
MODEL = "NVIDIA GeForce RTX 4090"
MIN_FREE = 12288
MAX_UTIL = 60
KEYS = tuple(f"{arm}-{seed}" for seed in SEEDS for arm in ARMS)
GPU_PATTERN = re.compile(r"GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_source(source):
    unchanged = {name: sha for name, sha in source["files"].items()
                 if name != "scripts/run_oss_c_resources.py"}
    require(digest(unchanged) == OLD_SOURCE, "Frozen production files differ from the original producer")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def create(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def inventory(uuid=None):
    args = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"]
    if uuid is not None:
        args[1:1] = ["-i", uuid]
    records = subprocess.check_output(args, text=True)
    rows = {}
    for index, gpu, name, total, free, used, util in csv.reader(records.splitlines(), skipinitialspace=True):
        require(GPU_PATTERN.fullmatch(gpu) and gpu not in rows, "Unexpected GPU inventory")
        rows[gpu] = dict(index=int(index), model=name, total_mib=int(total), free_mib=int(free),
                         device_used_mib=int(used), utilization=int(util))
    return rows


def qualifies(row):
    return row["model"] == MODEL and row["free_mib"] >= MIN_FREE and row["utilization"] <= MAX_UTIL


@contextmanager
def lease(runs, uuid):
    require(GPU_PATTERN.fullmatch(uuid), "Invalid GPU UUID")
    with (runs / ".p4_gpu_leases" / f"{uuid}.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def old_absent(runs):
    require(not os.path.lexists((runs / OLD_PATH).parent / "cases")
            and not os.path.lexists(runs / "oss-c-real-execution01"),
            "Previous registration already has execution evidence")
    commands = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    require(not any("scripts/run_oss_c_pair.py" in line and " run " in line
                    and OLD_COMMIT in line for line in commands), "Previous worker is active")


def preserved_campaign(runs, source, loaded=None):
    old_path = runs / OLD_PATH
    require(not old_path.is_symlink() and hash_file(old_path) == OLD_SHA, "Previous registration bytes differ")
    old = read(old_path)
    previous = {"commit": OLD_COMMIT, "source_hash": OLD_SOURCE}
    require(old == campaign(runs, previous, loaded=loaded), "Previous experiment contract differs")
    # All model settings, initial states, schedules, inputs and gates come directly from the pinned prior record.
    unchanged = {key: value for key, value in old.items()
                 if key not in ("schema", "source_commit", "source_hash", "gpu_uuid", "gpu_policy")}
    return dict(unchanged, schema="oss-c-seed-queues-v1", source_commit=source["commit"],
                source_hash=source["source_hash"], supersedes=dict(path=str(OLD_PATH), sha256=OLD_SHA,
                    relationship="supersedes-before-first-update", previous_updates=0),
                unchanged_contract_sha256=digest(unchanged),
                gpu_policy=dict(model=MODEL, min_free_mib=MIN_FREE, max_utilization=MAX_UTIL,
                                device_assignment="same model seed on the same UUID"))


def seed_mapping(uuids):
    require(1 <= len(uuids) <= 3 and len(set(uuids)) == len(uuids)
            and all(GPU_PATTERN.fullmatch(uuid) for uuid in uuids), "Use one to three distinct GPU UUIDs")
    return {key: uuids[SEEDS.index(int(key.rsplit("-", 1)[1])) % len(uuids)] for key in KEYS}


def registration_path(runs):
    return runs / RELATIVE_ROOT / "registration.json"


def verify_registration(runs, source, expected_sha, loaded=None):
    path = registration_path(runs)
    require(not path.parent.is_symlink() and not path.is_symlink() and hash_file(path) == expected_sha,
            "New registration file or digest differs")
    registered = read(path)
    devices = registered["devices"]
    require(set(registered["case_gpu"]) == set(KEYS)
            and registered["case_gpu"] == seed_mapping(tuple(devices))
            and all(qualifies(row) for row in devices.values())
            and len({row["total_mib"] for row in devices.values()}) == 1, "Frozen GPU queues differ")
    require(registered == dict(preserved_campaign(runs, source, loaded), devices=devices,
                               case_gpu=registered["case_gpu"]), "Frozen registration or source differs")
    old_absent(runs)
    return registered


def register(runs, source, uuids):
    case_gpu = seed_mapping(uuids)
    old_absent(runs)
    rows = inventory()
    devices = {}
    for uuid in uuids:
        require(uuid in rows and qualifies(rows[uuid]), "Registration requires individually eligible RTX 4090 devices")
        with lease(runs, uuid):
            devices[uuid] = rows[uuid]
    require(len({row["total_mib"] for row in devices.values()}) == 1, "Registered device capacities differ")
    registered = dict(preserved_campaign(runs, source), devices=devices, case_gpu=case_gpu)
    path = registration_path(runs)
    path.parent.mkdir(exist_ok=False)
    create(path, registered)
    return dict(registration_sha256=hash_file(path), campaign_sha256=digest(registered), case_gpu=case_gpu,
                updates=108, actual_updates=0)


def launch_record(registered, sha, key):
    return dict(schema="oss-c-seed-queue-launch-v1", campaign_sha256=digest(registered),
                registration_sha256=sha, source_commit=registered["source_commit"], case=key, updates=12,
                gpu_uuid=registered["case_gpu"][key], scope="train_val_only", budget_notice_sent=True)


def case_path(runs, key):
    return registration_path(runs).parent / "cases" / key


def snapshot(directory):
    entries = sorted(directory.iterdir())
    require(all(path.is_file() and not path.is_symlink() for path in entries), "Unexpected completed case file")
    return {path.name: hash_file(path) for path in entries}


def run_worker(runs, source, sha, key, launch_sha, resume):
    loaded = release(runs)
    registered = verify_registration(runs, source, sha, loaded)
    execution = registration_path(runs).parent / "execution"
    claim_path = execution / "claims" / f"{key}.json"
    receipt = execution / "receipts" / f"{key}.json"
    expected = launch_record(registered, sha, key)
    require(not execution.is_symlink() and not receipt.is_symlink() and hash_file(receipt) == launch_sha
            and read(receipt) == expected, "Wrong or missing independent launch receipt")
    uuid = expected["gpu_uuid"]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "Worker visibility or deterministic workspace differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    directory = case_path(runs, key)
    require(not directory.parent.is_symlink() and not directory.is_symlink(), "Invalid frozen case path")
    with lease(runs, uuid):
        row = inventory(uuid).get(uuid)
        require(row is not None and qualifies(row)
                and row["total_mib"] == registered["devices"][uuid]["total_mib"], "Worker GPU admission failed")
        if resume:
            require(read(claim_path) == expected and (directory / "result.json").is_file(),
                    "Only a completed case can reenter this batch")
        else:
            require(not os.path.lexists(directory), "Case already exists")
            create(claim_path, expected)
        result = run_case(registered, key, loaded[0], source, directory, resume=resume, device=DEVICE)
    return {**{name: result[name] for name in ("operation", "updates_this_call")}, "admission": row}


def worker_command(source, runs, sha, key, launch_sha, resume):
    command = [sys.executable, str(Path(__file__).resolve()), "--expected-commit", source["commit"],
               "--runs", str(runs), "run", "--registration-sha256", sha, "--case", key,
               "--launch-sha256", launch_sha]
    return command + (["--resume"] if resume else [])


def sample_gpu(uuid, path, stop):
    with path.open("x", encoding="utf-8") as stream:
        while True:
            try:
                row = inventory(uuid)[uuid]
                stream.write(json.dumps(dict(time=utc(), **row)) + "\n")
                stream.flush()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                stream.write(json.dumps(dict(time=utc(), sampling_error=str(error))) + "\n")
                stream.flush()
            if stop.wait(1):
                return


def dispatch(runs, source, sha):
    registered = verify_registration(runs, source, sha)
    root = registration_path(runs).parent
    require(not os.path.lexists(root / "cases"), "Cases already exist for this registration")
    execution = root / "execution"
    execution.mkdir(exist_ok=False)
    receipts = execution / "receipts"
    receipts.mkdir()
    (execution / "claims").mkdir()
    hashes = {}
    for key in KEYS:
        path = receipts / f"{key}.json"
        create(path, launch_record(registered, sha, key))
        path.chmod(0o444)
        hashes[key] = hash_file(path)
    create(execution / "authorization.json", dict(source_chat=MAIN_CHAT, notice=AUTHORIZATION,
            registration_sha256=sha, campaign_sha256=digest(registered), source_commit=source["commit"],
            source_hash=source["source_hash"], case_gpu=registered["case_gpu"], budget=108, receipts=hashes))
    failure = Event()
    event_lock = Lock()

    def emit(uuid, phase, key, action, **values):
        with (execution / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time=utc(), uuid=uuid, phase=phase, case=key, action=action, **values)) + "\n")

    def queue(uuid):
        keys = [key for key in KEYS if registered["case_gpu"][key] == uuid]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for phase in ("fresh", "completed"):
            for key in keys:
                with event_lock:
                    if failure.is_set():
                        return
                    emit(uuid, phase, key, "start")
                started = monotonic()
                stop = Event()
                sampler = None
                code = 1
                directory = case_path(runs, key)
                try:
                    if phase == "fresh":
                        sampler = Thread(target=sample_gpu, args=(uuid, execution / f"{key}.gpu.jsonl", stop))
                        sampler.start()
                    else:
                        create(execution / f"{key}.before.json", snapshot(directory))
                    path = execution / f"{key}.{phase}.log"
                    with path.open("x", encoding="utf-8") as log:
                        child = subprocess.run(worker_command(source, runs, sha, key, hashes[key], phase == "completed"),
                                               stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
                    require(child.returncode == 0, f"Worker exit {child.returncode}")
                    operation = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
                    require((operation["operation"], operation["updates_this_call"]) ==
                            (("completed", 12) if phase == "fresh" else ("verified_completed", 0)),
                            "Worker update receipt differs")
                    if phase == "completed":
                        after = snapshot(directory)
                        create(execution / f"{key}.after.json", after)
                        require(read(execution / f"{key}.before.json") == after, "Completed resume modified files")
                    code = 0
                except Exception as error:
                    with event_lock:
                        failure.set()
                        emit(uuid, phase, key, "failure", error=str(error))
                finally:
                    stop.set()
                    if sampler is not None:
                        sampler.join()
                    with event_lock:
                        emit(uuid, phase, key, "exit", code=code, seconds=monotonic() - started)

    with ThreadPoolExecutor(max_workers=len(registered["devices"])) as pool:
        list(pool.map(queue, registered["devices"]))
    create(execution / "controller.json", dict(status="failed" if failure.is_set() else "completed", time=utc()))
    require(not failure.is_set(), "At least one queue failed. Consult the preserved events and case artifacts.")
    return dict(status="completed", execution=str(execution))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    new = sub.add_parser("register")
    new.add_argument("--gpu", action="append", required=True)
    controller = sub.add_parser("dispatch")
    worker = sub.add_parser("run")
    for mode in (controller, worker):
        mode.add_argument("--registration-sha256", required=True)
    worker.add_argument("--case", choices=KEYS, required=True)
    worker.add_argument("--launch-sha256", required=True)
    worker.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    source = code_snapshot()
    require(training_container_kind() == "lxc" and source["commit"] == args.expected_commit,
            "Authorized LXC and exact runtime commit required")
    require_frozen_code(source)
    verify_source(source)
    runs = args.runs.resolve(strict=True)
    require(runs.name == "runs", "Authorized runs directory required")
    if args.mode == "register":
        result = register(runs, source, args.gpu)
    elif args.mode == "dispatch":
        result = dispatch(runs, source, args.registration_sha256)
    else:
        result = run_worker(runs, source, args.registration_sha256, args.case, args.launch_sha256, args.resume)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
