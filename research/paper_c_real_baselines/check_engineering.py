"""Run synthetic C8 checks in the authorized LXC, then record repository status."""
import argparse
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter
import unittest

from c8_bridge import ROOT, clean, fresh, write, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = fresh(args.expected_commit, args.out)
    virtualization = subprocess.check_output(['systemd-detect-virt'], text=True).strip()
    if virtualization != 'lxc':
        raise RuntimeError('Current user authorizes LXC execution only')
    out.mkdir()
    began = perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    opened, denied = [], []
    temp_root = Path(tempfile.gettempdir()).resolve()
    def audit(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path.suffix.lower() in ('.mat', '.pt', '.pth', '.ckpt') or (path.suffix.lower() in ('.npy', '.npz') and not path.is_relative_to(temp_root)):
            denied.append(path.suffix)
            raise PermissionError('Synthetic engineering run forbids real datasets and weights')
        if path.suffix.lower() in ('.npy', '.npz'):
            opened.append(path.suffix)
    sys.addaudithook(audit)
    folder = Path(__file__).resolve().parent
    with (out/'tests.log').open('x') as stream:
        suite = unittest.defaultTestLoader.discover(str(folder), pattern='test_*.py')
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    with (out/'repository.log').open('x') as stream:
        repository = subprocess.run([sys.executable, str(ROOT/'scripts/check_repository.py'), '--expected-commit', args.expected_commit], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    packages = {name: importlib.import_module(name).__version__ for name in ('numpy', 'scipy', 'sklearn')}
    references = {name: digest(importlib.import_module(name).__file__) for name in ('sklearn.linear_model._linear_loss', 'sklearn._loss.loss', 'sklearn.neighbors._nearest_centroid')}
    clean(args.expected_commit)
    receipt = dict(source=args.expected_commit, branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(), hostname=platform.node(), virtualization=virtualization, python=platform.python_version(), packages=packages, installed_reference_module_sha256=references, started_UTC=started, ended_UTC=datetime.now(timezone.utc).isoformat(), wall_seconds=perf_counter()-began, tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped), synthetic_tests_passed=result.wasSuccessful(), repository_check_exit=repository.returncode, repository_acceptance_passed=repository.returncode == 0, synthetic_array_opens=len(opened), denied_real_array_or_weight_opens=denied, real_dataset_reads=0, real_fitting_updates=0, query_reads=0, training_state='paused_by_current_user_instruction', scope='Synthetic engineering only; generated complete package replay/export/copy restore is not real classification', files=[dict(name=path.name, bytes=path.stat().st_size, sha256=digest(path)) for path in sorted(out.glob('*.log'))])
    write(out/'receipt.json', receipt)
    print(__import__('json').dumps(receipt))
    if not result.wasSuccessful() or denied:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
