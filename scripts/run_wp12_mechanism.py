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
from evaluation import wp12_mechanism_factorial as factorial
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


def frozen_factorial(root,out,code,args):
    from scripts.run_native_released_points import arithmetic
    from model.research_config import digest

    out.mkdir(parents=True,exist_ok=True)
    code_identity=code["commit"]
    scope=factorial.registration(root)
    require(args.gpu is None or args.gpu==os.environ.get("CUDA_VISIBLE_DEVICES"),"Expose only the registered GPU UUID")
    mode="cuda" if args.gpu else "cpu"
    suffix=args.case if args.action=="factorial" else "coverage"
    require(suffix is not None and suffix and Path(suffix).name==suffix and
            (args.action=="factorial") == bool(args.case),"A factorial case or bounded coverage proof is required")
    destination=out/(f"{mode}-{suffix}.json")
    if destination.exists():
        previous=json_pin(root,file_pin(destination))
        require(previous["execution_commit"]==code_identity and previous["manifest"]==digest(scope),
                "Existing frozen factorial receipt differs")
        return
    manifest=out/"factorial-registration.json"
    if manifest.exists():
        require(json_pin(root,file_pin(manifest))==json.loads(json.dumps(scope)),"Factorial registration differs")
    else:
        write_json(manifest,scope)
    with arithmetic(root,args.gpu) as execution:
        result=(factorial.run(root,args.case,execution) if args.action=="factorial" else factorial.coverage(root,execution,out))
    require_frozen_code(code)
    write_json(destination,result|dict(execution_commit=code_identity,manifest=digest(scope)))
    print(f"accepted {suffix} {mode}",flush=True)


def edge_campaign(root,out,code,args):
    from scripts.run_native_released_points import arithmetic
    from transfer import native_mechanism_campaign as campaign

    if args.action=="edge-register":
        require(args.case is args.gpu is args.capture is args.producer is None,
                "Only synthetic geometry is opened during registration")
        print(json.dumps(campaign.register(root,out,code)),flush=True)
        return
    require(args.case in campaign.GROUPS and (args.gpu is None or args.gpu==os.environ.get("CUDA_VISIBLE_DEVICES")),
            "Use an exact synthetic edge case and isolated device")
    with arithmetic(root,args.gpu) as execution:
        if args.action=="edge-run":
            require(args.capture is args.producer is None,"Edge optimization uses only its original registered source")
            print(json.dumps(campaign.fit(root,out,code,args.case,execution)),flush=True)
            return
        require(args.capture is not None,"Confirm into a separate authorized directory")
        report=external_directory(args.capture)
        require(root in report.parents and out!=report, "Independent points must be saved outside training")
        report.mkdir(parents=True,exist_ok=True)
        destination=report/(args.case+".json")
        if destination.exists():
            old=json_pin(root,file_pin(destination))
            require(old["consumer"]==code["commit"] and old["producer"]==(args.producer or code["commit"])
                    and old["device"]==execution.numeric,"An existing independent edge receipt differs")
            return
        result=campaign.confirm(root,out,args.producer or code["commit"],code,args.case,execution)
        write_json(destination,result)
        print(f"accepted {args.case} {execution.device}",flush=True)


def shared_edges(root, out, code, args):
    from evaluation import wp12_shared_edge_interaction as experiment

    require(args.producer is args.capture is None, "Use the original frozen localizations")
    rule = experiment.registration(code["commit"])
    path = out / "shared-edge-registration.json"
    if args.action == "shared-edge-register":
        require(args.case is args.gpu is args.interaction_reference is None and not out.exists(),
                "The preregistration requires a fresh directory without inference")
        out.mkdir(parents=True)
        write_json(path, rule)
        print(json.dumps(file_pin(path)), flush=True)
        return
    require(args.case in experiment.CASES and json_pin(root, file_pin(path)) == rule
            and (args.gpu is None or args.gpu == os.environ.get("CUDA_VISIBLE_DEVICES"))
            and bool(args.interaction_reference) == bool(args.gpu),
            "Use the exact preregistration and a pinned CPU mask for GPU replay")
    reference = None
    if args.interaction_reference:
        reference = json_pin(root, dict(zip(("path", "sha256"), args.interaction_reference, strict=True)))
        experiment.checked_receipt(reference, code["commit"])
    mode = "cuda" if args.gpu else "cpu"
    destination = out / f"shared-edge-{mode}-{args.case}.json"
    require(not destination.exists(), "Use a fresh receipt; retain the original shared-edge observations")
    torch.set_num_threads(1)
    with (gpu_scope(args.gpu) if args.gpu else nullcontext((None, None, None))) as (admission, numeric, _):
        value = experiment.run(root, args.case, code["commit"], torch.device(mode), reference=reference)
    value.update(admission=admission, numeric=numeric, reference=args.interaction_reference)
    require_frozen_code(code)
    write_json(destination, value)
    print(json.dumps(file_pin(destination)), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("register","real","factorial","coverage","edge-register","edge-run","edge-confirm",
                                         "shared-edge-register","shared-edge-run"))
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--case")
    parser.add_argument("--gpu")
    parser.add_argument("--producer")
    parser.add_argument("--capture",type=Path)
    parser.add_argument("--interaction-reference",nargs=2,metavar=("PATH","SHA256"))
    args=parser.parse_args()
    root=args.runs.resolve(strict=True)
    out=external_directory(args.out)
    require(root in out.parents,"Mechanism results must stay in authorized runs")
    code=code_snapshot()
    require(code["commit"] == args.expected_commit, "Expected exact source commit")
    require_frozen_code(code)
    if args.action in ("shared-edge-register", "shared-edge-run"):
        return shared_edges(root, out, code, args)
    require(args.interaction_reference is None, "Only the shared-edge replay accepts a fixed CPU mask")
    if args.action.startswith("edge-"):
        return edge_campaign(root,out,code,args)
    if args.action in ("factorial","coverage"):
        return frozen_factorial(root,out,code,args)
    scope=register(root,out,code)
    if args.action=="register":
        require(args.case is args.gpu is None, "Registration makes neither predictions nor updates")
        print(json.dumps(scope),flush=True)
        return
    require(args.case in real.CASES and (args.gpu is None or args.gpu==os.environ.get("CUDA_VISIBLE_DEVICES")),
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
