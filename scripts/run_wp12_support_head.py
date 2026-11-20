"""Fit, provision, deploy and independently review the frozen support-only control."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import wp12_support_head as wp12, wp12_support_head_cost as cost
from evaluation.frozen_training_stages import file_pin, require
from inference import support_readout
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("fit","prepare","deploy","review"))
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--case",choices=wp12.PINS,required=True)
    for key in ("artifact","plan","cost"):
        parser.add_argument("--"+key,nargs=2,metavar=("PATH","SHA256"))
    parser.add_argument("--gpu",help="Single authorized physical CUDA UUID for the deployment worker")
    parser.add_argument("--selection",help="Pinned released support subset for prepare, defaults to all")
    args = parser.parse_args()
    required = {"fit":set(),"prepare":{"artifact"},"deploy":{"plan"},"review":{"artifact","cost"}}
    received = {key for key in ("artifact","plan","cost") if getattr(args,key) is not None}
    require(required[args.operation]==received and (args.gpu is None or args.operation=="deploy")
            and (args.selection is None or args.operation=="prepare"),
            "The operation requires its specific frozen input receipts")
    code = code_snapshot()
    require_frozen_code(code)
    require(sys.platform=="linux" and code["commit"]==args.expected_commit,"Expected clean LXC checkout")
    root = args.runs.resolve(strict=True)
    out = external_directory(args.out)
    require(root in out.parents and not out.exists() and out.parent.is_dir(),
            "Expected fresh output under the authorized runs root")
    pins = {key:dict(zip(("path","sha256"),getattr(args,key),strict=True)) for key in received}
    if args.operation=="deploy":
        plan = cost.open_plan(root,pins["plan"],code["commit"])
        require(plan["source"]["fit_receipt"]==wp12.PINS[args.case],"Deployment plan binds a foreign case")
        scope = nullcontext()
        if args.gpu is not None:
            from evaluation.wp11_deployment_cost import gpu_scope

            require(os.environ.get("CUDA_VISIBLE_DEVICES")==args.gpu,"Expose exactly the planned physical device")
            scope = gpu_scope(args.gpu)
        out.mkdir()
        with scope:
            result = cost.deploy(root,pins["plan"],out,code,"cpu" if args.gpu is None else "cuda:0")
    elif args.operation=="fit":
        out.mkdir()
        result = wp12.fit_case(root,args.case,out)
    else:
        _,source = wp12.frozen_case(root,args.case)
        if args.operation=="prepare":
            artifact = support_readout.load(root,pins["artifact"],source)
            selection = args.selection or "all"
            require(selection in artifact["selections"],"Unknown released support subset")
            out.mkdir()
            result = cost.prepare_plan(root,source,pins["artifact"],out,code["commit"],selection)
        else:
            deployed = cost.evidence(root,pins["cost"],code["commit"],source,pins["artifact"])
            review = wp12.review_case(root,args.case,pins["artifact"],deployed)
            write_json(out,dict(schema=wp12.SCHEMA,producer=code["commit"],case=args.case,development_only=True,
                raw_gt_loaded=False,query_accessed=False,backbone_optimizer_updates=0,cost=pins["cost"],results=review))
            result = file_pin(out)
    require_frozen_code(code)
    print(result,flush=True)


if __name__=="__main__":
    cli()
