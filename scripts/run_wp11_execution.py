"""Register and execute an exact, CPU-only WP11 model manifest."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.wp11_execution import MODES, build, execute
from research_runtime import code_snapshot, external_directory
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
    args = parser.parse_args()
    code = code_snapshot()
    torch.set_num_threads(1)
    if (not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit
            or torch.cuda.is_available()):
        raise ValueError("Clean expected Linux CPU checkout required")
    pin = dict(path=args.input, sha256=args.input_sha256)
    out = external_directory(args.out)
    if args.command == "register":
        build(pin, code, out)
    else:
        execute(pin, code, args.mode, out)


if __name__ == "__main__":
    main()
