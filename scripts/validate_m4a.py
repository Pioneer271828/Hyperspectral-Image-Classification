"""Remote synthetic acceptance for standalone M4-A; no dataset or long training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = '3cdcfc9c09b37842847d516940e523ae901ce5d5'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_files():
    paths = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().rstrip('\0').split('\0')
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
            if p.endswith('.py') or (p.startswith('configs/') and p.endswith('.json'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    require(not out.is_relative_to(ROOT) and not ROOT.is_relative_to(out), 'output overlaps checkout')
    out.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'expected_commit': args.expected_commit,
              'scope': 'M4-A standalone synthetic engineering; no real-data accuracy evidence',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'checks': []}
    try:
        require(sys.platform == 'linux' and __debug__, 'use Linux training container with assertions enabled')
        container = subprocess.run(['systemd-detect-virt', '--container'], capture_output=True, text=True)
        require(Path('/.dockerenv').exists() or container.returncode == 0, 'training container not confirmed')
        require(re.fullmatch('[0-9a-f]{40}', args.expected_commit) is not None, 'expected full SHA required')
        require(git('rev-parse', 'HEAD') == args.expected_commit, 'unexpected checkout SHA')
        require(not git('status', '--porcelain'), 'dirty checkout')
        require(git('rev-parse', '--is-shallow-repository') == 'false', 'full history required')
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', BASE, 'HEAD'], check=True)
        changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', '-z', BASE, 'HEAD']).decode().rstrip('\0')
        allowed = {'model/region_residual_decomposition.py', 'tests/test_region_residual_decomposition.py',
                   'scripts/validate_m4a.py', 'README.md', 'docs/architecture/module-map.json',
                   'docs/guides/region-residual-m4.md'}
        require(all(p in allowed or p.startswith('research/') for p in changed.split('\0') if p),
                'changes outside independent M4-A scope')
        before = source_files()
        import torch
        require(torch.cuda.is_available(), 'CUDA acceptance required')
        record.update(commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
                      checkout=str(ROOT), base=BASE, source_files=before,
                      environment={'python': sys.version, 'torch': torch.__version__,
                                   'cuda': torch.version.cuda, 'container': container.stdout.strip(),
                                   'gpu': torch.cuda.get_device_name(0),
                                   'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
        json.loads((ROOT / 'docs/architecture/module-map.json').read_text())
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   M4_DEVICE='cpu', G3_DEVICE='cpu', GRAPH_CACHE_DEVICE='cpu', DYNAMIC_HEAD_DEVICE='cpu')
        checks = [
            ('focused-cpu', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_region_residual_decomposition.py', '-v'], {}),
            ('full-cpu', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'], {}),
            ('focused-cuda', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_region_residual_decomposition.py', '-v'], {'M4_DEVICE': 'cuda'}),
            ('smoke-cpu', ['smoke_test.py', '--device', 'cpu'], {}),
            ('smoke-cuda', ['smoke_test.py', '--device', 'cuda'], {}),
        ]
        for name, command, overrides in checks:
            start = time.monotonic()
            item = {'name': name, 'command': [sys.executable, *command],
                    'environment': {**{key: env[key] for key in ('M4_DEVICE', 'G3_DEVICE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, **overrides}}
            record['checks'].append(item)
            with (out / (name + '.log')).open('w') as log:
                result = subprocess.run(item['command'], cwd=ROOT, env={**env, **overrides},
                                        stdout=log, stderr=subprocess.STDOUT, timeout=240)
            item.update(returncode=result.returncode, seconds=time.monotonic() - start)
            require(result.returncode == 0, name + ' failed; see preserved log')
        subprocess.run(['git', '-C', str(ROOT), 'diff', '--check', BASE, 'HEAD'], check=True)
        require(git('rev-parse', 'HEAD') == args.expected_commit and not git('status', '--porcelain'),
                'checkout changed during acceptance')
        require(source_files() == before, 'source fingerprints changed')
        record.update(status='passed', dirty_after=False)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        (out / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in record.items() if key != 'source_files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
