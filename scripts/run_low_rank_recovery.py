"""Run the frozen WP10 low-rank campaign in the authorized CPU environment."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compression.low_rank_recovery_audit import audit
from compression.registered_low_rank_recovery import complete, execute, register
from research_runtime import code_snapshot
from transfer.container_runtime import training_container_kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    def argument(command, name):
        command.add_argument("--" + name, nargs=2, metavar=("PATH", "SHA256"), required=True)

    plan = commands.add_parser("register")
    argument(plan, "request")
    plan.add_argument("--out", required=True)
    run, end, replay = (commands.add_parser(v) for v in ("stage", "complete", "audit"))
    for command in (run, end, replay):
        argument(command, "registration")
    for command in (run, end):
        argument(command, "notice")
    run.add_argument("--case", required=True)
    run.add_argument("--stage", choices=("full", "prefix", "resume", "completed"), required=True)
    run.add_argument("--prior", nargs=2)
    argument(replay, "completion")
    args = parser.parse_args()
    code = code_snapshot()
    torch.set_num_threads(1)
    if (not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit
            or torch.cuda.is_available()):
        raise ValueError("Expected exact clean authorized CPU container checkout")

    def pin(name):
        value = getattr(args, name)
        return dict(zip(("path", "sha256"), value, strict=True)) if value else None

    if args.command == "register":
        result = register(pin("request"), args.out, code)
    elif args.command == "stage":
        result = execute(pin("registration"), code, pin("notice"), args.case, args.stage, pin("prior"))
    elif args.command == "complete":
        result = complete(pin("registration"), code, pin("notice"))
    else:
        result = audit(pin("registration"), pin("completion"), code)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
