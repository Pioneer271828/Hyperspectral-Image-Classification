"""Register, execute, freeze and independently review a native long coverage cohort."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.frozen_training_stages import file_pin, json_pin, require
from transfer.native_long_run import REFERENCE
from transfer.p4_runtime_device import UUID, WORKSPACE


def commit(path):
    value = subprocess.check_output(["git","status","--porcelain"],cwd=path,text=True)
    require(not value, "One-click flow requires a clean producer and consumer")
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=path,text=True).strip()


def invoke(root, python, script, arguments, log, gpu=None):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE":"1", "OMP_NUM_THREADS":"1",
           "OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":gpu or "",
           "CUBLAS_WORKSPACE_CONFIG":WORKSPACE}
    command = [python,"-B",script,*map(str,arguments)]
    if gpu:
        command += ["--gpu",gpu]
    for _ in range(30):
        attempt = log.with_name(f"{log.name}-{uuid4().hex[:12]}.log")
        with attempt.open("x") as stream:
            result = subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        text = attempt.read_text()
        if result.returncode == 0:
            return json.loads(text.splitlines()[-1])
        if "ResourceUnavailable" not in text:
            raise RuntimeError(f"{script} failed. See {attempt}")
        sleep(5)
    raise RuntimeError(f"Resource admission remained unavailable. See {attempt}")


def signal_reference(producer, python, common, campaign, log):
    """Drain at the first durable training checkpoint, then use normal resume."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE":"1", "CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"1",
           "OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    with log.open("x") as stream:
        process = subprocess.Popen([python,"-B","scripts/run_native_released_points.py",*map(str,common),
                    "--mode","coverage-long-run","--case",REFERENCE],cwd=producer,env=env,stdout=stream,stderr=subprocess.STDOUT)
        pulse = campaign/"cases"/REFERENCE/"clock-0150.pt"
        while process.poll() is None and not pulse.exists():
            sleep(.1)
        if process.poll() is None:
            process.terminate()
        require(process.wait() == 0, f"Orderly drain did not exit normally. See {log}")
    drained = json.loads(log.read_text().splitlines()[-1])
    require(drained["status"] == "drained" and drained["signal"] == "SIGTERM",
            "Expected a committed partial trajectory after the process signal")


def recorded_device(case_folder, default):
    reg = case_folder/"registration.json"
    return json.loads(reg.read_bytes())["numeric"]["physical_uuid"] if reg.is_file() else default


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs","campaign","review"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--producer-checkout",type=Path,default=ROOT)
    parser.add_argument("--gpu",action="append",default=[])
    parser.add_argument("--review-only",action="store_true")
    parser.add_argument("--historical-snapshot",nargs=2,metavar=("PATH","SHA256"))
    args = parser.parse_args(argv)
    root = args.runs.resolve(strict=True)
    campaign, review = args.campaign.resolve(), args.review.resolve()
    producer = args.producer_checkout.resolve(strict=True)
    prod_sha, consumer_sha = commit(producer), commit(ROOT)
    require(root in campaign.parents and root in review.parents and
            campaign not in review.parents and review not in campaign.parents and campaign != review,
            "Use two isolated external output directories")
    require(len(args.gpu) <= 3 and len(set(args.gpu)) == len(args.gpu)
            and all(UUID.fullmatch(item) for item in args.gpu), "Specify up to three distinct full physical UUIDs")
    review.mkdir(exist_ok=True)
    logs = review/"commands"
    logs.mkdir(exist_ok=True)
    py = sys.executable
    common = ["--expected-commit",prod_sha,"--runs",root,"--out",campaign]

    def train(*extra,gpu=None):
        key = extra[-1] if extra[-2:-1] == ("--case",) else extra[1]
        return invoke(producer,py,"scripts/run_native_released_points.py",
                      [*common,*extra],logs/key,gpu)

    plan_path = campaign/"campaign.json"
    if not args.review_only and not plan_path.exists():
        train("--mode","coverage-long-register")
    plan = json_pin(root,file_pin(plan_path))
    require(plan["producer"] == prod_sha, "Requested producer checkout is foreign to the campaign")
    if not args.review_only:
        require(plan["scope"] == "synthetic_only", "The one-click training flow admits synthetic releases only")
        cpu = campaign/"cases"/REFERENCE
        if not (cpu/"registration.json").exists():
            signal_reference(producer,py,common,campaign,logs/f"ordered-drain-{uuid4().hex}.log")
        expected = cycle(args.gpu or [None])
        groups = {}
        paired = {}
        for name, profile in plan["cases"].items():
            key = (profile["scene"],profile["model_seed"])
            paired.setdefault(key,[]).append(name)
        for names in paired.values():
            existing = [recorded_device(campaign/"cases"/name,None) for name in names
                        if (campaign/"cases"/name/"registration.json").is_file()]
            default = existing[0] if existing else None if REFERENCE in names else next(expected)
            for name in names:
                folder = campaign/"cases"/name
                if not (folder/"receipt.json").is_file():
                    gpu = recorded_device(folder,default)
                    groups.setdefault(gpu,[]).append(name)

        def group(item):
            gpu, names = item
            for name in names:
                train("--mode","coverage-long-run","--case",name,gpu=gpu)

        with ThreadPoolExecutor(max_workers=max(1,len(groups))) as pool:
            list(pool.map(group,groups.items()))
        if not (campaign/"reference"/REFERENCE/"receipt.json").is_file():
            train("--mode","coverage-long-reference","--case",REFERENCE)
        reentry = train("--mode","coverage-long-run","--case",REFERENCE)
        require(reentry["updates_this_invocation"] == 0, "Completed trajectory made more updates")
        train("--mode","coverage-long-freeze")
    frozen = file_pin(campaign/"freeze.json")
    source = json_pin(root,frozen)
    receipts = review/"cases"
    receipts.mkdir(exist_ok=True)
    scope = ["--expected-commit",consumer_sha,"--runs",root,"--campaign",campaign,
             "--freeze",frozen["path"],frozen["sha256"]]

    def audit(item):
        name, original = item
        row = json_pin(root,original)
        gpu = row["device"]["physical_uuid"]
        dest = receipts/f"{name}.json"
        if not dest.is_file():
            invoke(ROOT,py,"scripts/ingest_wp12_evidence.py",
                   ["--long-stage-case",*scope,"--case",name,"--out",dest],logs/f"audit-{name}",gpu)

    grouped = {}
    for name, original in source["cases"].items():
        row = json_pin(root,original)
        grouped.setdefault(row["device"]["physical_uuid"],[]).append((name,original))

    def audit_group(entries):
        for item in entries:
            audit(item)

    with ThreadPoolExecutor(max_workers=max(1,len(grouped))) as pool:
        list(pool.map(audit_group,grouped.values()))
    out = review/"wp12.json"
    if out.is_file():
        value = json_pin(root,file_pin(out))
        require(value["freeze"] == frozen and value["consumer"] == consumer_sha and value["status"] == "passed",
                "Existing WP12 publication belongs to a different frozen campaign")
    else:
        history = ["--historical-snapshot",*args.historical_snapshot] if args.historical_snapshot else []
        invoke(ROOT,py,"scripts/ingest_wp12_evidence.py",
               ["--long-stage-publish",*scope,"--receipts",receipts,"--out",out,*history],logs/"publish")
    print(json.dumps(file_pin(out)),flush=True)


if __name__ == "__main__":
    main()
