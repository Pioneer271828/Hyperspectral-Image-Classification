"""Register and execute a verified WP10 or WP11 frozen CPU deployment."""

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
    for name in ("register", "run", "startup", "cost"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--input", nargs=2, required=True, metavar=("PATH", "SHA256"))
        if name != "startup":
            cmd.add_argument("--out", required=True)
        if name in ("run", "startup"):
            cmd.add_argument("--mode", required=True)
        if name == "run":
            cmd.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    code = code_snapshot()
    torch.set_num_threads(1)
    if (not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit
            or torch.cuda.is_available()):
        raise ValueError("Clean authorized CPU checkout at the expected commit required")
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
    elif kind != lr.KIND:
        raise ValueError("Startup and cost require a registered low-rank manifest")
    elif args.command == "startup":
        result = lr.startup(pin, code, args.mode)
    else:
        result = cost(pin, code, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
