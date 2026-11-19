"""Register, replay and publish frozen development stage comparisons."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
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


def publish_cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-stages", action="store_true", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for key in ("plan", "cpu", "gpu", "launch", "independent", "freeze", "honghu-release",
                "honghu-review", "honghu-status", "honghu-alternate", "synthetic-queue"):
        parser.add_argument("--" + key, nargs=2, metavar=("PATH", "SHA256"), required=key not in ("gpu", "honghu-alternate", "synthetic-queue"))
    args = parser.parse_args(argv)
    code = code_snapshot()
    require_frozen_code(code)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    if sys.platform != "linux" or not __debug__ or code["commit"] != args.expected_commit or root not in out.parents:
        raise ValueError("Clean expected LXC checkout and external runs output required")
    if out.exists():
        raise ValueError("New WP12 output required")
    torch.set_num_threads(1)
    wp11 = stages.json_pin(root, pin(args.plan))
    if wp11 != stages.stage_plan(root, wp11["release"]):
        raise ValueError("Frozen stage plan changed")
    cpu = stages.json_pin(root, pin(args.cpu))
    if cpu["device"] != "cpu" or cpu["code"] != code["commit"]:
        raise ValueError("Expected the accepted CPU stage receipt")
    wp11_summary = stages.summarize(wp11, cpu)
    if args.gpu:
        gpu = stages.json_pin(root, pin(args.gpu))
        if gpu["device"] != "cuda:0" or gpu["code"] != code["commit"] or not gpu["admission"]:
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
    synthetic = None
    if args.synthetic_queue:
        from evaluation.frozen_stage_synthetic import publish

        synthetic = publish(root, pin(args.synthetic_queue), code)
    result = dict(policy=stages.POLICY, producer=code["commit"], original_archive=long_archive, synthetic=synthetic,
                  native_summary=wp11_summary, device_check=device_check,
                  native_plan=pin(args.plan), native_cpu=pin(args.cpu), native_gpu=pin(args.gpu) if args.gpu else None,
                  honghu=case_review, optimizer_updates=0, raw_gt_loaded=False, query_accessed=False)
    require_frozen_code(code)
    print(json.dumps(_json_new(out, result)), flush=True)
