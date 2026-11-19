"""Register, replay and publish frozen development stage comparisons."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import subprocess
import sys

import torch

from evaluation import frozen_stage_archives as archives, frozen_training_stages as stages
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer.registered_development import _json_new


def pin(pair):
    return dict(zip(("path", "sha256"), pair, strict=True))


def replay_cli(code, root, out, gpu, release, *, plan_pin):
    if sys.platform != "linux" or not __debug__:
        raise ValueError("Frozen comparisons require the verified Linux runtime")
    if release is not None:
        if gpu is not None or plan_pin is not None:
            raise ValueError("Registration accepts a release only")
        document = stages.stage_plan(root, pin(release))
        require_frozen_code(code)
        print(json.dumps(_json_new(out, document)), flush=True)
        return
    if plan_pin is None:
        raise ValueError("Replay requires the exact registered plan pin")
    plan = stages.json_pin(root, pin(plan_pin))
    if gpu is not None:
        from evaluation.wp11_deployment_cost import gpu_scope

        lease = gpu_scope(gpu)
    else:
        lease = nullcontext((None, None, None))
    with lease as (admission, numeric, _):
        result = stages.run(root, plan, torch.device("cpu" if gpu is None else "cuda:0"))
        result.update(code=code["commit"], admission=admission, numeric=numeric)
    require_frozen_code(code)
    print(json.dumps(_json_new(out, result)), flush=True)


def checked_checkout(args):
    code = code_snapshot()
    require_frozen_code(code)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    if sys.platform != "linux" or not __debug__ or code["commit"] != args.expected_commit or root not in out.parents:
        raise ValueError("Clean expected LXC checkout and external runs output required")
    torch.set_num_threads(1)
    return code, root, out


def audited_execution_commit(code, previous):
    if previous is None or previous == code["commit"]:
        return code["commit"]
    if not re.fullmatch("[0-9a-f]{40}", previous):
        raise ValueError("Full frozen execution commit required")
    subprocess.run(["git", "merge-base", "--is-ancestor", previous, code["commit"]], check=True)
    changed = subprocess.check_output(["git", "diff", "--name-only", previous, code["commit"]], text=True)
    safe = {"evaluation/frozen_stage_archives.py", "evaluation/frozen_stage_cli.py",
            "evaluation/frozen_stage_g4_replay.py", "scripts/ingest_wp12_evidence.py"}
    if any(path not in safe and not path.startswith(("tests/", "docs/", "research/reports/"))
           for path in changed.splitlines()):
        raise ValueError("Frozen execution producer changed the production runtime")
    return previous


def historical_cli(argv):
    from evaluation.frozen_stage_g4_replay import replay
    from transfer.p4_runtime_device import gate
    from transfer.shared_gpu_lease import physical_lease

    parser = argparse.ArgumentParser(description="Replay one frozen whole-scene scratch representative per scene")
    parser.add_argument("--historical-stage-run", action="store_true", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    for key in ("launch", "independent", "freeze"):
        parser.add_argument("--" + key, nargs=2, required=True)
    args = parser.parse_args(argv)
    code, root, out = checked_checkout(args)
    archive_pins = {key: pin(getattr(args, key)) for key in ("launch", "independent", "freeze")}
    plan = archives.admission(root, archive_pins)
    if (out/"receipt.json").exists():
        current = stages.json_pin(root, stages.file_pin(out/"receipt.json"))
        if current["producer"] != code["commit"] or current["plan_sha256"] != stages.digest(plan):
            raise ValueError("Completed historical replay belongs to a different source or plan")
        print(json.dumps(stages.file_pin(out/"receipt.json")), flush=True)
        return
    with physical_lease(args.gpu):
        admission = gate(args.gpu)
        result = replay(root, plan, out, torch.device("cuda:0"))
        result.update(producer=code["commit"], admission=admission, physical_uuid=args.gpu)
    require_frozen_code(code)
    print(json.dumps(_json_new(out/"receipt.json", result)), flush=True)


def publish_cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-stages", action="store_true", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution-commit")
    for key in ("plan", "cpu", "gpu", "launch", "independent", "freeze", "honghu-release",
                "honghu-review", "honghu-status", "honghu-alternate", "synthetic-queue", "historical-replay"):
        parser.add_argument("--" + key, nargs=2, metavar=("PATH", "SHA256"), required=key not in ("gpu", "honghu-alternate", "synthetic-queue", "historical-replay"))
    args = parser.parse_args(argv)
    code, root, out = checked_checkout(args)
    execution = audited_execution_commit(code, args.execution_commit)
    if out.exists():
        raise ValueError("New WP12 output required")
    wp11 = stages.json_pin(root, pin(args.plan))
    if wp11 != stages.stage_plan(root, wp11["release"]):
        raise ValueError("Frozen stage plan changed")
    cpu = stages.json_pin(root, pin(args.cpu))
    if cpu["device"] != "cpu" or cpu["code"] != execution:
        raise ValueError("Expected the accepted CPU stage receipt")
    wp11_summary = stages.summarize(wp11, cpu)
    if args.gpu:
        gpu = stages.json_pin(root, pin(args.gpu))
        if gpu["device"] != "cuda:0" or gpu["code"] != execution or not gpu["admission"]:
            raise ValueError("Expected strict admitted CUDA stage receipt")
        stages.summarize(wp11, gpu)
        device_check = stages.compare_devices(cpu, gpu)
    else:
        device_check = None
    archive_pins = {key: pin(getattr(args, key)) for key in ("launch", "independent", "freeze")}
    historical = archives.admission(root, archive_pins)
    case_review = archives.honghu(root, pin(args.honghu_release), pin(args.honghu_review), pin(args.honghu_status),
                                  pin(args.honghu_alternate) if args.honghu_alternate else None)
    long_archive = archives.archive(root, historical)
    replayed = stages.json_pin(root, pin(args.historical_replay)) if args.historical_replay else None
    if replayed is not None:
        require = stages.require
        require(replayed["plan_sha256"] == long_archive["plan_sha256"] and replayed["producer"] == code["commit"]
                and len(replayed["cases"]) == 3 and replayed["optimizer_updates"] == 0
                and replayed["raw_gt_loaded"] is replayed["query_accessed"] is False,
                "Historical representative replay is foreign to the frozen archive")
    synthetic = None
    if args.synthetic_queue:
        from evaluation.frozen_stage_synthetic import publish

        synthetic = publish(root, pin(args.synthetic_queue), {**code, "commit": execution})
    result = dict(policy=stages.POLICY, producer=code["commit"], original_archive=long_archive, synthetic=synthetic,
                  native_summary=wp11_summary, device_check=device_check, execution_commit=execution,
                  historical_replay=replayed,
                  native_plan=pin(args.plan), native_cpu=pin(args.cpu), native_gpu=pin(args.gpu) if args.gpu else None,
                  honghu=case_review, optimizer_updates=0, raw_gt_loaded=False, query_accessed=False)
    require_frozen_code(code)
    print(json.dumps(_json_new(out, result)), flush=True)
