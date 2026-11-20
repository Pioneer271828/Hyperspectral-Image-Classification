"""Register and execute verified WP10 or WP11 frozen deployment."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compression.low_rank_recovery_cost import cost
from inference import wp10_low_rank_execution as lr
from inference import wp11_execution as wp11
from research_runtime import code_snapshot, external_directory
from transfer import matched_route_admission as admission
from transfer.container_runtime import training_container_kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "run", "startup", "cost", "warm-register", "warm-cost",
                 "warm-reference", "onednn-register", "onednn-cost"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--input", nargs=2, required=True, metavar=("PATH", "SHA256"))
        if name != "startup":
            cmd.add_argument("--out", required=True)
        if name in ("run", "startup"):
            cmd.add_argument("--mode", required=True)
        if name == "run":
            cmd.add_argument("--save-predictions", action="store_true")
        if name in ("warm-register", "warm-cost", "onednn-register"):
            cmd.add_argument("--reference", nargs=2, required=name != "warm-cost", metavar=("PATH", "SHA256"))
        if name == "warm-cost":
            cmd.add_argument("--gpu", metavar="PHYSICAL_UUID")
        if name == "onednn-cost":
            cmd.add_argument("--torchao-source", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    torch.set_num_threads(1)
    gpu = getattr(args, "gpu", None)
    if (not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit
            or (gpu is None and torch.cuda.is_available())):
        raise ValueError("Clean isolated authorized checkout at the expected commit required")
    if args.command == "warm-cost" and (gpu is None) != (args.reference is None):
        parser.error("A physical CUDA UUID and its bitwise CPU reference must be supplied together")
    pin = dict(zip(("path", "sha256"), args.input, strict=True))
    kind = admission.read(pin).get("kind")
    if args.command == "register":
        out = external_directory(args.out)
        if kind == lr.KIND:
            result = lr.freeze(pin, code, out)
        elif kind is None:
            wp11.build(pin, code, out)
            result = dict(status="registered", manifest=str(out / "manifest.json"))
        else:
            raise ValueError("Unknown frozen CPU package kind")
    elif args.command == "run":
        out = external_directory(args.out)
        if kind == lr.KIND:
            result = lr.execute(pin, code, args.mode, out, export=args.save_predictions)
        elif kind is None:
            wp11.execute(pin, code, args.mode, out, export=args.save_predictions)
            result = dict(status="passed", evidence=str(out / "execution.json"))
        else:
            raise ValueError("Unknown frozen CPU package kind")
    elif args.command == "onednn-register":
        from inference.wp10_onednn_deployment import preparation

        observation = dict(zip(("path", "sha256"), args.reference, strict=True))
        result = preparation(pin, observation, code, args.out)
    elif args.command == "onednn-cost":
        from inference.wp10_onednn_deployment import measure

        result = measure(pin, code, args.out, args.torchao_source)
    elif kind != lr.KIND:
        raise ValueError("Startup and cost require a registered low-rank manifest")
    elif args.command == "startup":
        result = lr.startup(pin, code, args.mode)
    elif args.command == "warm-register":
        from inference.wp10_deployment_cost import prepare

        reference = dict(zip(("path", "sha256"), args.reference, strict=True))
        result = prepare(pin, reference, code, args.out)
    elif args.command == "warm-cost":
        if gpu is None:
            from inference.wp10_deployment_cost import measure

            result = measure(pin, code, args.out)
        else:
            from inference.wp10_cross_device import cuda_measure

            reference = dict(zip(("path", "sha256"), args.reference, strict=True))
            result = cuda_measure(pin, reference, code, args.out, gpu)
    elif args.command == "warm-reference":
        from inference.wp10_cross_device import rank_reference

        result = rank_reference(pin, code, args.out)
    else:
        result = cost(pin, code, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
