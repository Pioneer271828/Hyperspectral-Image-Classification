"""Register, execute, complete or independently audit the registered CUR-O arms."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transfer.curriculum_cli import _code
from transfer.curriculum_observation_real import complete, execute, register
from transfer.curriculum_observation_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(command, name):
        command.add_argument("--" + name, nargs=2, metavar=("PATH", "SHA256"), required=True)

    plan = commands.add_parser("register")
    add(plan, "admission")
    plan.add_argument("--out", required=True)
    stage, end, check = (commands.add_parser(x) for x in ("stage", "complete", "audit"))
    for command in (stage, end, check):
        add(command, "registration")
    for command in (stage, end):
        add(command, "notice")
    stage.add_argument("--case", required=True)
    stage.add_argument("--stage", required=True, choices=("full", "prefix", "resume", "completed"))
    stage.add_argument("--prior", nargs=2)
    add(check, "completion")
    args = parser.parse_args()
    code = _code(args.expected_commit)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Frozen CPU only")

    def pin(name):
        value = getattr(args, name)
        return None if value is None else dict(zip(("path", "sha256"), value, strict=True))

    if args.command == "register":
        result = register(pin("admission"), args.out, code)
    elif args.command == "stage":
        result = execute(pin("registration"), code, pin("notice"), args.case, args.stage, pin("prior"))
    elif args.command == "complete":
        result = complete(pin("registration"), code, pin("notice"))
    else:
        result = audit(pin("registration"), pin("completion"), code)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
