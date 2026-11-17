"""Read-only audits for the registered real factorial and its teacher diagnosis."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transfer.curriculum_cli import _code
from transfer.curriculum_factorial_real_audit import audit
from transfer.curriculum_joint_diagnosis import freeze, review_joint, teacher_diagnosis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, names in {
        "audit": ("registration", "completion", "campaign"),
        "freeze-teacher": ("request",),
        "diagnose-teacher": ("manifest",),
        "joint": ("request", "teacher", "audit"),
    }.items():
        sub = commands.add_parser(command)
        for name in names:
            sub.add_argument("--"+name, nargs=2, metavar=("PATH", "SHA256"), required=True)
        sub.add_argument("--out", required=True)
    args = parser.parse_args()
    code = _code(args.expected_commit)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Frozen CPU only")

    def pin(name):
        return dict(zip(("path", "sha256"), getattr(args, name), strict=True))

    if args.command == "audit":
        result = audit(pin("registration"), pin("completion"), pin("campaign"), args.out, code)
    elif args.command == "freeze-teacher":
        result = freeze(pin("request"), args.out, code)
    elif args.command == "diagnose-teacher":
        result = teacher_diagnosis(pin("manifest"), args.out, code)
    else:
        result = review_joint(pin("request"), pin("teacher"), pin("audit"), args.out, code)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
