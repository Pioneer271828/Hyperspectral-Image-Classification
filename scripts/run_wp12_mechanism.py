"""Preregister and execute retained-observation WP05/WP06 mechanism verification."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation import wp12_mechanism_real as real
from evaluation.frozen_training_stages import file_pin, json_pin, require
from evaluation.wp11_deployment_cost import gpu_scope
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


def register(root,out,code):
    document = dict(schema="wp12_mechanism_registration_v1", producer=code["commit"],
        localizations=dict(cases=list(real.CASES),views=list(real.VIEWS),sites=list(real.SITES),
            scope="released_development_offline_only",rivals=real.RIVALS,
            control=["identity","aligned","reverse","random"],pool_binding="exact_Q",
            edge_cases=list(real.EDGE_CASES)),
        real_updates=0,query_accessed=False,raw_gt_loaded=False)
    out.mkdir(parents=True,exist_ok=True)
    path=out/"registration.json"
    if path.exists():
        require(json_pin(root,file_pin(path)) == json.loads(json.dumps(document)), "Offline intervention registry differs")
    else:
        write_json(path,document)
    return file_pin(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("register","real"))
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--case",choices=real.CASES)
    parser.add_argument("--gpu")
    args=parser.parse_args()
    root=args.runs.resolve(strict=True)
    out=external_directory(args.out)
    require(root in out.parents,"Mechanism results must stay in authorized runs")
    code=code_snapshot()
    require(code["commit"] == args.expected_commit, "Expected exact source commit")
    require_frozen_code(code)
    scope=register(root,out,code)
    if args.action=="register":
        require(args.case is args.gpu is None, "Registration makes neither predictions nor updates")
        print(json.dumps(scope),flush=True)
        return
    require(args.case is not None and (args.gpu is None or args.gpu==os.environ.get("CUDA_VISIBLE_DEVICES")),
            "Use a registered mechanism case and isolated device")
    device=torch.device("cuda" if args.gpu else "cpu")
    folder=out/("cuda" if args.gpu else "cpu")
    folder.mkdir(exist_ok=True)
    destination=folder/(args.case+".json")
    if destination.exists():
        old=json_pin(root,file_pin(destination))
        require(old["producer"]==code["commit"] and old["case"]==args.case
                and old["registration"]==scope["sha256"],"Completed mechanism receipt differs")
        return
    with (gpu_scope(args.gpu) if args.gpu else nullcontext((None,None,None))) as (row,numeric,_):
        result=real.run_case(root,args.case,code["commit"],device)
    result.update(registration=scope["sha256"],device=str(device),admission=row,numeric=numeric)
    require_frozen_code(code)
    write_json(destination,result)
    print(f"accepted {args.case} {device}",flush=True)


if __name__=="__main__":
    main()
