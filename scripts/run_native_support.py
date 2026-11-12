"""Register, preflight, execute and audit bounded WP07/WP08 CPU experiments."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from research_runtime import code_snapshot, require_frozen_code
from transfer.container_runtime import training_container_kind
from transfer import native_support_campaign as campaign


def pin(value):
    path, checksum = value
    return dict(path=str(Path(path).resolve(strict=True)), sha256=checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    registration = actions.add_parser("register")
    registration.add_argument("--root", required=True)
    for name in ("preflight", "stage", "freeze", "audit", "cost"):
        action = actions.add_parser(name)
        action.add_argument("--registration", nargs=2, required=True, metavar=("PATH", "SHA256"))
        if name in ("audit", "cost"):
            action.add_argument("--freeze", nargs=2, required=True, metavar=("PATH", "SHA256"))
        if name == "stage":
            action.add_argument("--case", required=True)
            action.add_argument("--arm", required=True)
            action.add_argument("--phase", choices=("full", "prefix", "resume", "completed"), required=True)
            action.add_argument("--prior", nargs=2, metavar=("PATH", "SHA256"))
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error("Authorized Linux container with assertions required")
    torch.set_num_threads(1)
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        parser.error("Exact frozen experiment SHA required")
    require_frozen_code(code)
    if args.action == "register":
        result = campaign.register(args.root, code)
    elif args.action == "preflight":
        result = campaign.preflight(pin(args.registration), code)
    elif args.action == "stage":
        result = campaign.execute(pin(args.registration), code, args.case, args.arm, args.phase,
                                  None if args.prior is None else pin(args.prior))
    else:
        from transfer import native_support_audit as audit
        if args.action == "freeze":
            result = audit.freeze(pin(args.registration), code)
        elif args.action == "audit":
            result = audit.audit(pin(args.registration), pin(args.freeze), code)
        else:
            from inference.native_support_cost import measure
            result = measure(pin(args.registration), pin(args.freeze), code)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
