"""Publish and independently score the preregistered native paired head experiments."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from evaluation import wp12_paired_head_capture as frozen, wp12_paired_head_intake as custody
from evaluation import wp12_paired_head_review as consumer
from evaluation.frozen_training_stages import file_pin, require
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.check_wp12_production import prior_runtime
from scripts.run_native_released_points import arithmetic
from transfer import native_paired_head_campaign as campaign


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("capture","trace","inventory","review","real-preflight"))
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--producer")
    parser.add_argument("--program",type=Path)
    parser.add_argument("--case")
    parser.add_argument("--gpu")
    parser.add_argument("--cpu",type=Path)
    parser.add_argument("--cuda",type=Path)
    parser.add_argument("--inventory",type=Path)
    parser.add_argument("--trace",type=Path)
    parser.add_argument("--request",nargs=2,metavar=("PATH","SHA256"))
    parser.add_argument("--approval",nargs=2,metavar=("PATH","SHA256"))
    args=parser.parse_args()
    root=args.runs.resolve(strict=True)
    out=external_directory(args.out)
    require(root in out.parents,"Paired evidence must remain under authorized runs")
    code=code_snapshot()
    require(code["commit"]==args.expected_commit,"Expected the frozen clean checkout")
    require_frozen_code(code)
    if args.gpu:
        require(args.action in ("capture","trace") and args.gpu==os.environ.get("CUDA_VISIBLE_DEVICES"),
                "One physical GPU must be isolated for measured inference")
    if args.action=="inventory":
        result=custody.inspect(root)
    elif args.action=="real-preflight":
        require(args.request and args.approval,"Real preflight is read-only and requires separate holder pins")
        result=custody.preflight(root,dict(zip(("path","sha256"),args.request,strict=True)),
                           dict(zip(("path","sha256"),args.approval,strict=True)),code["commit"])
    elif args.action=="trace":
        with arithmetic(root,args.gpu) as execution:
            result=frozen.trace(root,execution)
    else:
        require(args.producer and args.program is not None and root in args.program.resolve(strict=True).parents,
                "Provide a registered paired program and runtime producer")
        prior_runtime(args.producer,code["commit"])
        if args.action=="capture":
            require(args.case in campaign.design() and (not args.gpu or args.case in campaign.REFERENCE.values()),
                    "GPU inference requires a registered representative paired case")
            with arithmetic(root,args.gpu) as execution:
                result=frozen.capture(root,args.program,out,args.case,args.producer,code,execution)
            print(json.dumps(result),flush=True)
            return
        require(args.cpu and args.cuda and args.inventory and args.trace,
                "Provide pinned scope, source trace and independent CPU/GPU receipts")
        out.mkdir(parents=True,exist_ok=False)
        result=consumer.review(root,args.program,args.cpu,args.cuda,out,args.producer,code["commit"],
                               file_pin(args.inventory.resolve(strict=True)),file_pin(args.trace.resolve(strict=True)))
        out=out/"paired-review.json"
    require_frozen_code(code)
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,result)
    print(json.dumps(dict(schema=result["schema"],status="passed")),flush=True)


if __name__=="__main__":
    main()
