"""Register or execute a fixed development-only CPU curriculum; no query access."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transfer.curriculum_cli import register_curriculum, execute_curriculum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    register = sub.add_parser('register', help='Freeze external hashes without optimization')
    for name in ('source', 'metadata', 'development', 'exposure', 'sample-policy', 'observation-policy'):
        required = name not in ('sample-policy', 'observation-policy')
        register.add_argument('--'+name, type=Path, required=required)
        register.add_argument('--'+name+'-sha256', required=required)
    register.add_argument('--lr', type=float, required=True)
    register.add_argument('--torch-seed', type=int, required=True)
    register.add_argument('--out', type=Path, required=True)
    run = sub.add_parser('run', help='Advance to an absolute step or restore a completed run with zero updates')
    run.add_argument('--registration', type=Path, required=True)
    run.add_argument('--registration-sha256', required=True)
    run.add_argument('--to-step', type=int, required=True)
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--resume', type=Path)
    run.add_argument('--resume-sha256')
    args = parser.parse_args()

    def pin(name):
        path, sha = getattr(args, name), getattr(args, name+'_sha256')
        if (path is None) != (sha is None):
            parser.error(name+' path and external hash must be paired')
        return None if path is None else dict(path=str(path.resolve()), sha256=sha)

    torch.set_num_threads(1)
    if args.command == 'register':
        inputs = {name: pin(name) for name in ('source', 'metadata', 'development', 'exposure', 'sample_policy', 'observation_policy')}
        result = register_curriculum(inputs, out=args.out, lr=args.lr, torch_seed=args.torch_seed,
                                     expected_commit=args.expected_commit)
    else:
        status = execute_curriculum(pin('registration'), out=args.out, to_step=args.to_step,
                                   expected_commit=args.expected_commit, resume=pin('resume'))
        result = {key: status[key] for key in ('status', 'completed_updates', 'new_updates', 'checkpoint', 'query')}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
