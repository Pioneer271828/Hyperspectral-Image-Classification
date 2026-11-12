"""Run registered regional support interventions and fixed-support recovery."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_runtime import code_snapshot, require_frozen_code
from transfer import controlled_support_campaign as campaign
from transfer.container_runtime import training_container_kind


def pin(args):
    path, checksum = args
    return dict(path=str(Path(path).resolve(strict=True)), sha256=checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    registration = actions.add_parser("register")
    registration.add_argument("--root", required=True)
    for name in campaign.OLD:
        registration.add_argument("--" + name, nargs=2, required=True)
    for name in ("stage", "audit", "intervene", "cost", "supplement"):
        action = actions.add_parser(name)
        action.add_argument("--registration", nargs=2, required=True)
        if name == "stage":
            action.add_argument("--case", required=True)
            action.add_argument("--arm", required=True)
            action.add_argument("--phase", choices=("full", "prefix", "resume", "completed"), required=True)
            action.add_argument("--prior", nargs=2)
        if name == "supplement":
            action.add_argument("--previous", required=True)
        if name in ("audit", "intervene", "cost", "supplement"):
            action.add_argument("--output", required=True)
            action.add_argument("--producer-commit")
        if name in ("cost", "supplement"):
            action.add_argument("--intervention", nargs=2, required=True)
        if name == "supplement":
            action.add_argument("--recovery-audit", nargs=2, required=True)
            action.add_argument("--cost-evidence", nargs=2, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if (not training_container_kind() or not __debug__ or code["commit"] != args.expected_commit
            or torch.cuda.is_available()):
        parser.error("Clean exact commit in authorized CPU LXC with assertions required")
    torch.set_num_threads(1)
    require_frozen_code(code)
    if args.action == "register":
        result = campaign.register(args.root, {name: pin(getattr(args, name)) for name in campaign.OLD}, code)
    elif args.action == "stage":
        result = campaign.stage(pin(args.registration), code, args.case, args.arm, args.phase,
                                pin(args.prior) if args.prior else None)
    else:
        from transfer import controlled_support_evidence as evidence
        options = dict(producer_commit=args.producer_commit)
        registration = pin(args.registration)
        if args.action == "supplement":
            result = evidence.supplement(registration, code, args.output, args.previous, pin(args.intervention),
                                         pin(args.recovery_audit), pin(args.cost_evidence), **options)
        elif args.action == "cost":
            from inference.controlled_support_cost import cost
            result = cost(registration, code, args.output, pin(args.intervention), **options)
        else:
            result = getattr(evidence, args.action)(registration, code, args.output, **options)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
