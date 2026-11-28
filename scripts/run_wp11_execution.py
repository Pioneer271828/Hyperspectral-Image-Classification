"""Register WP11 manifests, measure deployment or use frozen WP10/WP11 capsules."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import wp11_wp03_static as wp03
from inference.wp11_execution import MODES, build, execute
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from transfer.container_runtime import training_container_kind
from transfer.registered_wp11_learning import read


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "run"):
        stage = commands.add_parser(name)
        stage.add_argument("--input", required=True)
        stage.add_argument("--input-sha256", required=True)
        stage.add_argument("--out", required=True)
        if name == "run":
            stage.add_argument("--mode", choices=(*MODES, wp03.MODE), required=True)
            stage.add_argument("--save-predictions", action="store_true")
    deployment_commands(commands)
    frozen_commands(commands)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if args.command.startswith("deployment-"):
        deployment(args,code)
        return
    torch.set_num_threads(1)
    gpu = getattr(args, "gpu", None)
    if (not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit
            or (gpu is None and torch.cuda.is_available())):
        raise ValueError("Clean expected Linux checkout and explicit CPU/CUDA scope required")
    out = external_directory(args.out)
    if args.command in ("deploy", "export-precision", "deploy-precision"):
        frozen_deployment(args, code, out)
        return
    pin = dict(path=args.input, sha256=args.input_sha256)
    policy = read(pin).get("policy")
    if args.command == "register" and policy == wp03.REQUEST:
        wp03.register(pin, code, out)
        return
    if args.command == "run" and policy == wp03.MANIFEST:
        wp03.run(pin, code, args.mode, out, export=args.save_predictions)
        return
    if args.command == "register":
        build(pin, code, out)
    else:
        execute(pin, code, args.mode, out, export=args.save_predictions)


def frozen_commands(commands):
    deploy = commands.add_parser("deploy", help="Deploy the frozen WP11 development capsule")
    deploy.add_argument("--mode", choices=("full", "light"), default="full")
    export = commands.add_parser("export-precision", help="Copy a protected original WP10 precision release")
    export.add_argument("--source-plan", type=Path, required=True)
    export.add_argument("--input-sha256", required=True)
    export.add_argument("--out", required=True)
    precision = commands.add_parser("deploy-precision", help="Deploy the archived WP10 precision under its guards")
    precision.add_argument("--arm", help="Archived requested precision or seed-specific rank arm")
    precision.add_argument("--torchao-source", type=Path)
    for stage in (deploy, precision):
        stage.add_argument("--inputs", type=Path, required=True)
        stage.add_argument("--input-sha256", required=True)
        stage.add_argument("--partition", choices=("native", "grid", "slic"), default="native")
        stage.add_argument("--gpu", metavar="PHYSICAL_UUID")
        stage.add_argument("--out", required=True)


def frozen_deployment(args, code, out):
    if args.command == "export-precision":
        from inference.wp10_precision_deployment import export_capsule

        export_capsule(args.source_plan, args.input_sha256, out)
        return
    if args.command == "deploy-precision":
        from inference.wp10_precision_deployment import execute as deploy_precision

        deploy_precision(args.inputs, args.input_sha256, out, code, requested=args.arm,
                         partition=args.partition, gpu=args.gpu, provider=args.torchao_source)
        return

    from inference.wp11_frozen_deployment import deployment_arithmetic, open_runtime, place_requests, read_inputs, sha
    from transfer.p4_runtime_device import snapshot

    capsule = args.inputs.resolve()
    if out.is_relative_to(capsule):
        raise ValueError("Deployment outputs must be outside the frozen input capsule")
    if out.exists():
        raise FileExistsError(out)
    inputs, requests = read_inputs(capsule, args.input_sha256)
    gpu = args.gpu
    before = snapshot(gpu) if gpu is not None else None
    kernels = Counter()
    with deployment_arithmetic(gpu, kernel_calls=kernels) as numeric:
        if gpu is not None:
            torch.cuda.reset_peak_memory_stats()
            requests = place_requests(requests, numeric["device"])
        runtime = open_runtime(capsule, inputs, args.mode, partition=args.partition, device=numeric["device"])
        result = runtime.run(requests, args.mode)
        predictions_cpu = [tuple(head.cpu() for head in block) for block in result.pop("predictions")]
        peaks = (dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
                 if gpu is not None else None)
    after = snapshot(gpu) if gpu is not None else None
    require_frozen_code(code)
    out.mkdir()
    predictions = out / "predictions.pt"
    with predictions.open("xb") as stream:
        torch.save(predictions_cpu, stream)
    write_json(out / "receipt.json", dict(result, partition=args.partition, mode=args.mode,
                                         commit=code["commit"], states=dict(runtime.identity),
                                         inputs_sha256=args.input_sha256, predictions_sha256=sha(predictions),
                                         arithmetic=numeric, cuda_peak_bytes=peaks, graph_kernels=dict(kernels),
                                         device_before=before, device_after=after,
                                         optimizer_updates=0, query_reads=0))


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
