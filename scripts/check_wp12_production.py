"""Freeze and independently audit retained-observation production and diagnostic evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import wp12_locked_support as locked, wp12_native_production as production
from evaluation.frozen_training_stages import require
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.run_native_released_points import arithmetic


def prior_runtime(producer, current):
    if producer==current:
        return
    require(len(producer)==40 and all(c in "0123456789abcdef" for c in producer),
            "Use the complete runtime producer commit")
    subprocess.run(["git","merge-base","--is-ancestor",producer,current],check=True)
    changes = subprocess.check_output(["git","diff","--diff-filter=DMRT","--name-only",
                                       producer,current,"--","*.py","configs"],text=True)
    require(not changes.strip(), "A changed production runtime requires a separate acceptance run")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("locked","capture","audit","real-preflight"))
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--producer")
    parser.add_argument("--program",type=Path)
    parser.add_argument("--locked-cases",type=Path)
    parser.add_argument("--cpu",type=Path)
    parser.add_argument("--cuda",type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--model-seed",type=int)
    parser.add_argument("--request",nargs=2,metavar=("PATH","SHA256"))
    parser.add_argument("--authority",nargs=2,metavar=("PATH","SHA256"))
    args=parser.parse_args()
    root=args.runs.resolve(strict=True)
    out=external_directory(args.out)
    require(root in out.parents, "Deliverables must stay under the authorized runs")
    code=code_snapshot()
    require(code["commit"]==args.expected_commit, "Checkout SHA differs")
    require_frozen_code(code)
    if args.action=="real-preflight":
        from transfer.native_production_intake import real_preflight

        require(args.request and args.authority and not args.gpu, "Read-only real preflight requires separate holder pins")
        result=real_preflight(root,dict(zip(("path","sha256"),args.request,strict=True)),
                              dict(zip(("path","sha256"),args.authority,strict=True)),code["commit"])
    else:
        require(args.producer and not args.request and not args.authority, "Specify the prior production source commit")
        prior_runtime(args.producer,code["commit"])
        if args.action=="locked":
            require(args.locked_cases is not None and not args.gpu, "Provide the frozen locked-readout case directory")
            result=locked.assemble(root,args.locked_cases,args.producer)
        elif args.action=="audit":
            require(args.program is not None and args.cpu is not None and not args.gpu,
                    "Provide the registered program and independent CPU receipts")
            result=production.audit(root,args.program,args.cpu,args.cuda,code,args.producer)
        else:
            require(args.program is not None, "Provide the registered production program")
            require((args.gpu is None and args.model_seed is None)
                    or (args.gpu==os.environ.get("CUDA_VISIBLE_DEVICES") and args.model_seed==64),
                    "GPU requires a registered physical device and representative paired seed")
            with arithmetic(root,args.gpu) as execution:
                production.write_capture(root,args.program,out,code,args.producer,execution,only=args.model_seed)
            return
    require_frozen_code(code)
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,result)
    print(json.dumps(dict(schema=result["schema"],status="written")),flush=True)


if __name__=="__main__":
    main()
