"""Register and execute an exact, CPU-only WP11 model manifest."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.wp11_execution import MODES, build, execute
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer.container_runtime import training_container_kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "run"):
        stage = commands.add_parser(name)
        stage.add_argument("--input", required=True)
        stage.add_argument("--input-sha256", required=True)
        stage.add_argument("--out", required=True)
        if name == "run":
            stage.add_argument("--mode", choices=MODES, required=True)
            stage.add_argument("--save-predictions", action="store_true")
    deployment_commands(commands)
    args = parser.parse_args()
    code = code_snapshot()
    if args.command.startswith("deployment-"):
        deployment(args,code)
        return
    torch.set_num_threads(1)
    if (not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit
            or torch.cuda.is_available()):
        raise ValueError("Clean expected Linux CPU checkout required")
    pin = dict(path=args.input, sha256=args.input_sha256)
    out = external_directory(args.out)
    if args.command == "register":
        build(pin, code, out)
    else:
        execute(pin, code, args.mode, out, export=args.save_predictions)


def deployment_commands(commands):
    from evaluation.wp11_deployment_cost import CASES, SCOPES
    for name in ("register","measure","worker","synthetic","audit"):
        stage=commands.add_parser("deployment-"+name)
        stage.add_argument("--runs",required=True,type=Path)
        stage.add_argument("--out",required=True,type=Path)
        if name in ("measure","synthetic"):
            stage.add_argument("--gpu")
        if name in ("measure","worker","audit"):
            stage.add_argument("--plan-sha",required=True)
        if name=="measure":
            stage.add_argument("--limit",type=int)
        if name=="worker":
            stage.add_argument("--case",choices=CASES,required=True)
            stage.add_argument("--scope",choices=(SCOPES[0],SCOPES[-1]),required=True)
            stage.add_argument("--device",choices=("cpu","cuda:0"),required=True)
            stage.add_argument("--sync-folder",type=Path)
        if name=="audit":
            stage.add_argument("--synthetic",nargs="+",type=Path,required=True)


def deployment(args,code):
    import json
    from evaluation import wp11_deployment_cost as cost
    from evaluation.wp12_deployment_report import publish
    from evaluation.wp11_deployment_synthetic import exercise

    torch.set_num_threads(1)
    if not training_container_kind() or not __debug__ or code["dirty"] or code["commit"]!=args.expected_commit:
        raise ValueError("Clean expected Linux deployment checkout required")
    root,out=external_directory(args.runs),external_directory(args.out)
    if not root.is_dir() or root not in out.parents:
        raise ValueError("Deployment workload must be within the authorized runs")
    action=args.command.removeprefix("deployment-")
    if action=="register":
        result=cost.register(root,out,code)
    elif action=="measure":
        if args.limit is not None and args.limit<=0:
            raise ValueError("Measurement stage limit must be positive")
        result=cost.run(root,out,code,args.gpu,plan_sha=args.plan_sha,limit=args.limit)
    elif action=="worker":
        if args.sync_folder is not None and out not in args.sync_folder.resolve().parents:
            raise ValueError("Worker barrier must be within its registered deployment run")
        with cost.inherited_arithmetic(args.device):
            result=cost.internal(root,out,code,args.case,args.scope,args.device,plan_sha=args.plan_sha,
                                 sync_folder=args.sync_folder)
    elif action=="synthetic":
        if args.gpu is None:
            result=exercise(root,out,code)
        else:
            with cost.gpu_scope(args.gpu) as (admit,_,__):
                result=exercise(root,out,code,"cuda:0",physical_admission=admit)
    else:
        result=publish(root,out,code,args.plan_sha,args.synthetic)
    require_frozen_code(code)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
