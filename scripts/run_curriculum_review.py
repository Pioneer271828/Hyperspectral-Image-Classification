"""Review frozen support trajectories or register a zero-update observation plan."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transfer.curriculum_cli import _code
from transfer.curriculum_observation_admission import execute_synthetic, register_observation
from transfer.curriculum_support_review import review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    obs = sub.add_parser("register-observation")
    mechanism = sub.add_parser("review-support")
    for cmd in (obs, mechanism):
        cmd.add_argument("--sample-registration", required=True, nargs=2)
        cmd.add_argument("--out", required=True)
    mechanism.add_argument("--sample-audit", required=True, nargs=2)
    synth = sub.add_parser("synthetic-stage")
    synth.add_argument("--registration", required=True, nargs=2)
    synth.add_argument("--case", required=True)
    synth.add_argument("--stage", choices=("full", "prefix", "resume", "completed"), required=True)
    synth.add_argument("--prior", nargs=2)
    args = parser.parse_args()
    code = _code(args.expected_commit)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Frozen CPU only")

    def pin(name):
        value = getattr(args, name)
        return None if value is None else dict(zip(("path", "sha256"), value, strict=True))

    if args.command == "register-observation":
        result = register_observation(pin("sample_registration"), args.out, code)
    elif args.command == "review-support":
        result = review(pin("sample_registration"), pin("sample_audit"), args.out, code)
    else:
        result = execute_synthetic(pin("registration"), code, args.case, args.stage, pin("prior"))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
