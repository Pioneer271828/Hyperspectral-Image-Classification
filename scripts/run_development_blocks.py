"""Publish accepted spatial blocks and run the existing CPU zero-update consumers."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.wp12_development_admission import supplement
from research_runtime import code_snapshot, require_frozen_code
from transfer.container_runtime import training_container_kind
from transfer.development_block_probe import probe, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    admission = commands.add_parser("probe")
    admission.add_argument("--out", type=Path, required=True)
    admission.add_argument("--scenes", nargs="+")
    admission.add_argument("--resume", action="store_true")
    repeated = commands.add_parser("replay")
    repeated.add_argument("--admission", nargs=2, required=True, metavar=("PATH", "SHA256"))
    repeated.add_argument("--scene", required=True)
    status = commands.add_parser("supplement")
    status.add_argument("--out", type=Path, required=True)
    status.add_argument("--admission", nargs=2, required=True, metavar=("PATH", "SHA256"))
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error("Authorized Linux container with assertions required")
    torch.set_num_threads(1)
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        parser.error("Exact frozen source SHA required")
    require_frozen_code(code)
    if args.command == "probe":
        options = {} if args.scenes is None else {"names": args.scenes}
        result = probe(args.runs, args.out, code["commit"], resume=args.resume, **options)
    else:
        pin = dict(zip(("path", "sha256"), args.admission, strict=True))
        if args.command == "replay":
            result = replay(args.runs, pin, code["commit"], args.scene)
        else:
            result = supplement(args.runs, pin, args.out)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
