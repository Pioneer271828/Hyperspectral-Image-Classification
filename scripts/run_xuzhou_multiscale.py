"""Register, run or audit the bounded WP07 Xuzhou development experiment."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from research_runtime import code_snapshot, require_frozen_code
from transfer.container_runtime import training_container_kind
from transfer import xuzhou_multiscale as campaign, xuzhou_multiscale_audit as audit


def pin(value):
    path, checksum = value
    return dict(path=str(Path(path).resolve(strict=True)), sha256=checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    register = actions.add_parser("register")
    register.add_argument("--root", required=True)
    register.add_argument("--issue", type=int, required=True)
    register.add_argument("--prior", nargs=2, required=True, metavar=("PATH", "SHA256"))
    register.add_argument("--donor-audit", nargs=2, required=True, metavar=("PATH", "SHA256"))
    for name, extra in (("run", "--notice"), ("audit", "--complete")):
        action = actions.add_parser(name)
        action.add_argument("--registration", nargs=2, required=True, metavar=("PATH", "SHA256"))
        action.add_argument(extra, nargs=2, required=True, metavar=("PATH", "SHA256"))
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error("Authorized Linux container with assertions required")
    torch.set_num_threads(1)
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        parser.error("Expected exact WP07 code")
    require_frozen_code(code)
    if args.action == "register":
        result = campaign.register(args.root, pin(args.prior), pin(args.donor_audit), code=code, issue=args.issue)
    elif args.action == "run":
        result = campaign.execute(pin(args.registration), code, notice=pin(args.notice))
    else:
        result = audit.audit(pin(args.registration), pin(args.complete), code)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
