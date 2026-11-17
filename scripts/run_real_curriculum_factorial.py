"""Authenticate controls, register or execute the three new real factorial trajectories."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transfer.curriculum_cli import _code
from transfer.curriculum_factorial_controls import authenticate
from transfer.curriculum_factorial_real import complete, execute, register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    def pinned(command, name, required=True):
        command.add_argument("--"+name, nargs=2, metavar=("PATH", "SHA256"), required=required)

    controls, registration, stage, completion = (commands.add_parser(key) for key in ("controls", "register", "stage", "complete"))
    for cmd in (controls, registration):
        pinned(cmd, "request")
        cmd.add_argument("--out", required=True)
    pinned(registration, "controls")
    for cmd in (stage, completion):
        pinned(cmd, "registration")
        pinned(cmd, "notice")
    stage.add_argument("--case", required=True)
    stage.add_argument("--stage", required=True, choices=("full", "prefix", "resume", "completed"))
    pinned(stage, "prior", required=False)

    args = parser.parse_args()
    code = _code(args.expected_commit)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Frozen CPU only")

    def pin(key):
        value = getattr(args, key)
        return None if value is None else dict(zip(("path", "sha256"), value, strict=True))

    if args.command == "controls":
        result = authenticate(pin("request"), args.out, code)
    elif args.command == "register":
        result = register(pin("request"), pin("controls"), args.out, code)
    elif args.command == "stage":
        result = execute(pin("registration"), code, pin("notice"), args.case, args.stage, pin("prior"))
    else:
        result = complete(pin("registration"), code, pin("notice"))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
