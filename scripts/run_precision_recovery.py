"""Run the registered WP10 precision recovery and CPU deployment checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compression import precision_recovery_audit, precision_recovery_cost, registered_precision_recovery as campaign
from research_runtime import code_snapshot, require_frozen_code
from scripts.run_xuzhou_multiscale import pin
from transfer.container_runtime import training_container_kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    actions = parser.add_subparsers(dest='action', required=True)
    register = actions.add_parser('register')
    register.add_argument('--request', nargs=2, required=True, metavar=('PATH', 'SHA256'))
    register.add_argument('--out', type=Path, required=True)
    for name in ('stage', 'complete', 'audit', 'cost', 'startup'):
        action = actions.add_parser(name)
        action.add_argument('--registration', nargs=2, required=True, metavar=('PATH', 'SHA256'))
        if name in ('cost', 'startup'):
            action.add_argument('--audit', nargs=2, required=True, metavar=('PATH', 'SHA256'))
        if name == 'audit':
            action.add_argument('--complete', nargs=2, required=True, metavar=('PATH', 'SHA256'))
        if name == 'stage':
            action.add_argument('--notice', nargs=2, required=True, metavar=('PATH', 'SHA256'))
            action.add_argument('--prior', nargs=2, metavar=('PATH', 'SHA256'))
            action.add_argument('--stage', choices=('full', 'prefix', 'resume', 'completed'), required=True)
        if name in ('stage', 'startup'):
            action.add_argument('--arm', required=True)
    args = parser.parse_args()
    if training_container_kind() != 'lxc' or not __debug__:
        parser.error('Authorized Linux container with assertions required')
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        parser.error('Expected exact WP10 code')
    require_frozen_code(code)
    if args.action == 'register':
        result = campaign.register(pin(args.request), args.out, code)
    else:
        registration = pin(args.registration)
        if args.action == 'stage':
            result = campaign.execute(registration, code, pin(args.notice), args.arm, args.stage,
                                      None if args.prior is None else pin(args.prior))
        elif args.action == 'complete':
            result = campaign.complete(registration, code)
        elif args.action == 'audit':
            result = precision_recovery_audit.audit(registration, pin(args.complete), code)
        elif args.action == 'cost':
            result = precision_recovery_cost.cost(registration, pin(args.audit), code)
        else:
            result = precision_recovery_cost.startup(registration, pin(args.audit), args.arm, code)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
