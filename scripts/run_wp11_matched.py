"""Run the registered WP11 matched light/full learning and freeze audit."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp11_learning_audit import audit
from research_runtime import code_snapshot, external_directory
from transfer.container_runtime import training_container_kind
from transfer.registered_wp11_learning import ARMS, execute, register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    commands = parser.add_subparsers(dest="operation", required=True)

    def add_pin(command, name):
        command.add_argument("--" + name, required=True)
        command.add_argument("--" + name + "-sha256", required=True)

    plan = commands.add_parser("register")
    add_pin(plan, "request")
    plan.add_argument("--out", required=True)
    run = commands.add_parser("run")
    freeze = commands.add_parser("audit")
    for command in (run, freeze):
        add_pin(command, "registration")
        add_pin(command, "notice")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--stage", choices=("continuous", "prefix", "resume", "completed"), required=True)
    freeze.add_argument("--out", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux container checkout required")
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only matched campaign requires CUDA hidden")

    def pin(name):
        return dict(path=getattr(args, name), sha256=getattr(args, name + "_sha256"))

    if args.operation == "register":
        register(pin("request"), external_directory(args.out), code)
    elif args.operation == "run":
        execute(pin("registration"), pin("notice"), code, args.arm, args.stage)
    else:
        audit(pin("registration"), pin("notice"), code, external_directory(args.out))


if __name__ == "__main__":
    main()
