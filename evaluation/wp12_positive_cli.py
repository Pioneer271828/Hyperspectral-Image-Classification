"""Invoke synthetic confirmation from WP12 without an optimizer entry point."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

from evaluation.frozen_training_stages import json_pin, require
from evaluation.wp12_long_cli import production
from evaluation.wp12_positive_control import audit_case, publish
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer.native_scale_device import arithmetic
from transfer.p4_runtime_device import CPU, CUDA_FP64_WEIGHT
from transfer.shared_gpu_lease import physical_lease


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--positive-control-case",action="store_true")
    modes.add_argument("--positive-control-publish",action="store_true")
    for name in ("runs","campaign","review","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--freeze",nargs=2,required=True,metavar=("PATH","SHA256"))
    parser.add_argument("--v1",nargs=2,metavar=("PATH","SHA256"))
    parser.add_argument("--case")
    parser.add_argument("--gpu")
    args = parser.parse_args(argv)
    code = code_snapshot()
    require_frozen_code(code)
    root = args.runs.resolve(strict=True)
    place,review,out = (external_directory(p) for p in (args.campaign,args.review,args.out))
    require(code["commit"] == args.expected_commit and all(root in p.parents for p in (place,review,out))
            and place != review and place not in review.parents and review not in place.parents
            and review in out.parents and out.parent.is_dir(),"Expected isolated registered prediction scope")
    pin = dict(zip(("path","sha256"),args.freeze,strict=True))
    freeze = json_pin(root,pin)
    plan = json_pin(root,freeze["plan"])
    producer = production(code,plan["producer"],plan["source_hash"])
    if args.positive_control_case:
        require(args.case and not args.v1 and out == review/"cases"/f"{args.case}.json",
                "Specify one frozen case under the audit case directory")
        lease = nullcontext() if args.gpu is None else physical_lease(args.gpu)
        with lease,arithmetic(CPU if args.gpu is None else CUDA_FP64_WEIGHT,args.gpu) as execution:
            result = audit_case(root,place,review,out,code,producer,pin,args.case,execution)
    else:
        require(not args.case and not args.gpu and out == review/"wp12.json", "Publish with the sealed case directory only")
        old = dict(zip(("path","sha256"),args.v1,strict=True)) if args.v1 else None
        result = publish(root,place,review,out,code,producer,pin,old)
    print(json.dumps(result),flush=True)
