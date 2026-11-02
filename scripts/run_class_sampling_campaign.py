"""Frozen G3-E4 queue: remote checks, then four paired class-first training runs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_runtime import ROOT, code_snapshot, external_directory, require_frozen_code, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--hsi-root', type=Path, default=Path('/home/train/hsi-pioneer271828'))
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if (sys.platform != 'linux' or not Path('/.dockerenv').exists() or code['dirty']
            or code['commit'] != args.expected_commit or sys.flags.optimize):
        parser.error('Require container, assertions and clean exact SHA')
    root = external_directory(args.out_dir)
    inputs = [args.hsi_root / p for p in ('datasets/current', 'runs/g3-development-20261102-01',
              'runs/transfer-protocol-20261031-02', 'runs/g3-budget-repeat-20261102-01',
              'runs/g3-replication-20261101-01')]
    for item in inputs:
        item = item.resolve(strict=True)
        if root == item or item in root.parents or root in item.parents:
            parser.error('Output overlaps frozen inputs')
    root.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'code': code, 'seeds': [2046, 2056],
             'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
             'checks': [], 'runs': [], 'planned_new_target_updates': 20000}
    started = time.monotonic()

    def save():
        write_json(root / 'campaign.json', state)

    def run(command, name, records):
        row = {'name': name, 'command': command, 'status': 'running'}
        records.append(row)
        save()
        before = time.monotonic()
        with (root / f'{name}.log').open('xb') as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'G3_DEVICE': 'cuda'})
        row.update(returncode=completed.returncode, wall_seconds=time.monotonic() - before)
        if completed.returncode:
            row['status'] = 'failed'
            raise RuntimeError(f'{name} failed; outputs retained, no automatic retry')
        row['status'] = 'passed'
        save()

    save()
    try:
        run(['git', 'diff', '--check'], 'diff-check', state['checks'])
        for pattern in ('test_tile_sampling.py', 'test_development_budget_controls.py', 'test_development_training.py'):
            run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v'],
                pattern[:-3], state['checks'])
        run([sys.executable, 'scripts/run_development_budget.py', '--help'], 'cli-help', state['checks'])
        for seed in state['seeds']:
            command = [sys.executable, '-u', 'scripts/run_development_budget.py',
                       '--expected-commit', args.expected_commit, '--seed', str(seed), '--class-first',
                       '--baseline-run', str(args.hsi_root / f'runs/g3-budget-repeat-20261102-01/seed-{seed}'),
                       '--development-root', str(inputs[1]), '--protocol-root', str(inputs[2]),
                       '--data-root', str(inputs[0]),
                       '--source-run', str(args.hsi_root / f'runs/g3-replication-20261101-01/run-s{seed}'),
                       '--out-dir', str(root / f'seed-{seed}')]
            run(command, f'seed-{seed}', state['runs'])
            manifest = json.loads((root / f'seed-{seed}/manifest.json').read_text())
            assert manifest['status'] == 'passed' and manifest['optimizer_steps'] == 10000
        require_frozen_code(code)
        state.update(status='passed', completed_new_target_updates=20000)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(ended_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
        save()


if __name__ == '__main__':
    main()
