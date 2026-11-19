"""One checked-out entry for the fixed synthetic matrix and independent review."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from evaluation.frozen_training_stages import file_pin, json_pin, require
from scripts.run_native_long_coverage import commit, invoke, recorded_device
from transfer.p4_runtime_device import UUID


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("runs","campaign","review"):
        parser.add_argument("--"+item,type=Path,required=True)
    parser.add_argument("--gpu",action="append",default=[])
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--producer-checkout",type=Path,default=ROOT)
    parser.add_argument("--review-only",action="store_true")
    parser.add_argument("--v1",nargs=2,metavar=("PATH","SHA256"))
    args = parser.parse_args(argv)
    root,producer = args.runs.resolve(strict=True), args.producer_checkout.resolve(strict=True)
    campaign,review = args.campaign.resolve(),args.review.resolve()
    shell_sha, audit_sha = commit(producer), commit(ROOT)
    require(root in campaign.parents and root in review.parents and campaign != review
            and campaign not in review.parents and review not in campaign.parents,
            "Use separate synthetic optimization and review directories")
    require(1 <= args.workers <= 16 and len(set(args.gpu)) == len(args.gpu) <= 3
            and all(UUID.fullmatch(value) for value in args.gpu), "Expected bounded worker count and physical UUIDs")
    review.mkdir(exist_ok=True)
    logs = review/"commands"
    logs.mkdir(exist_ok=True)
    producer_cli = "scripts/run_native_released_points.py"
    audit_cli = "scripts/ingest_wp12_evidence.py"
    common = ["--expected-commit",shell_sha,"--runs",root,"--out",campaign]

    def run_train(mode, name=None, gpu=None):
        details = ["--case",name] if name else []
        return invoke(producer,sys.executable,producer_cli,[*common,"--mode",mode,*details],logs/(name or mode),gpu)

    plan_path = campaign/"plan.json"
    if not args.review_only and not plan_path.is_file():
        run_train("positive-control-register")
    plan = json_pin(root,file_pin(plan_path))
    require(plan["producer"] == shell_sha and plan["scope"] == "synthetic_only","Foreign training roster")
    groups = {}
    for name,profile in plan["cases"].items():
        groups.setdefault((profile["family"],profile["seed"]),[]).append(name)
    if not args.review_only:
        jobs = {}
        for index, names in enumerate(groups.values()):
            bindings = {recorded_device(campaign/"cases"/name,None) for name in names
                        if (campaign/"cases"/name/"registration.json").exists()}
            require(len(bindings) <= 1,"Existing triplet is split between physical devices")
            gpu = next(iter(bindings)) if bindings else (args.gpu[index] if index < len(args.gpu) else None)
            for name in names:
                if not (campaign/"cases"/name/"receipt.json").is_file():
                    jobs.setdefault(gpu,[]).append(name)

        def optimize(name_gpu):
            name,gpu = name_gpu
            return run_train("positive-control-run",name,gpu)

        operate(jobs,optimize,args.workers)
        run_train("positive-control-freeze")

    freeze_pin = file_pin(campaign/"freeze.json")
    frozen = json_pin(root,freeze_pin)
    receipts = review/"cases"
    receipts.mkdir(exist_ok=True)
    results = review/"wp12.json"
    scope = ["--expected-commit",audit_sha,"--runs",root,"--review",review,"--campaign",campaign,
             "--freeze",freeze_pin["path"],freeze_pin["sha256"]]

    def audit(name_gpu):
        name,gpu = name_gpu
        out = receipts/f"{name}.json"
        if not out.is_file():
            invoke(ROOT,sys.executable,audit_cli,["--positive-control-case",*scope,"--case",name,"--out",out],
                   logs/f"audit-{name}",gpu)

    jobs = {}
    for name,pin in frozen["cases"].items():
        record = json_pin(root,pin)
        spec = json_pin(root,record["spec"])
        if not (receipts/f"{name}.json").is_file():
            jobs.setdefault(spec["numeric"]["physical_uuid"],[]).append(name)
    operate(jobs,audit,args.workers)
    if results.is_file():
        value = json_pin(root,file_pin(results))
        require(value["frozen_source"] == freeze_pin and value["consumer"] == audit_sha
                and value["status"] == "passed","Prior publication differs")
    else:
        old = ["--v1",*args.v1] if args.v1 else []
        invoke(ROOT,sys.executable,audit_cli,["--positive-control-publish",*scope,"--out",results,*old],logs/"aggregate")
    print(json.dumps(file_pin(results)),flush=True)


def operate(groups, job, workers):
    # Serial work per physical device. CPU jobs can use a bounded group.
    cpu = groups.get(None,[])
    gpu = [(device,names) for device,names in groups.items() if device is not None]
    def device_work(group):
        device,names = group
        for name in names:
            job((name,device))
    with ThreadPoolExecutor(max_workers=max(1,min(len(cpu),workers)+len(gpu))) as pool:
        queued = [pool.submit(device_work, group) for group in gpu]
        queued.extend(pool.submit(job,(name,None)) for name in cpu)
        for item in queued:
            item.result()


if __name__ == "__main__":
    main()
