"""Remote-only E0a engineering validation; requires accepted matching M8 sources."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M8_PATHS = ['model/', 'train_vca_multidataset.py', 'train_spatial.py',
            'tests/test_fusion_m8.py', 'scripts/validate_m8.py',
            'configs/experiments/m8_spatial_default.json', 'configs/experiments/m8_spatial_aux_off.json',
            'configs/experiments/m8_spatial_alpha035.json']


def git(*options):
    return subprocess.check_output(['git', '-C', str(ROOT), *options], text=True).strip()


def fixture_cli(out, commit):
    import numpy as np
    fixture = out / 'fixture'
    fixture.mkdir()
    labels = np.array([[1, 1, 1, 2, 2, 2], [1, 1, 0, 2, 2, 2]])
    fused, cnn, graph = labels.copy(), labels.copy(), labels.copy()
    fused[0, 0] = 2; cnn[0, 1] = 2; graph[1, 5] = 1
    np.save(fixture / 'labels.npy', labels)
    np.save(fixture / 'mask.npy', np.ones(labels.shape, dtype=bool))
    np.savez_compressed(fixture / 'arrays.npz', prediction=fused, cnn_prediction=cnn, graph_prediction=graph)
    (fixture / 'checkpoint.pt').write_bytes(b'synthetic provenance fixture, not a trained checkpoint')
    command = [sys.executable, 'export_research_diagnostics.py', '--arrays', str(fixture / 'arrays.npz'),
               '--labels', str(fixture / 'labels.npy'), '--mask', str(fixture / 'mask.npy'),
               '--checkpoint', str(fixture / 'checkpoint.pt'), '--class-ids', '1,2,3',
               '--prediction-commit', commit, '--expected-commit', commit, '--dataset', 'synthetic_fixture',
               '--protocol-id', 'fixed_fixture', '--partition', 'test', '--out-dir', str(out / 'fixture-report')]
    with (out / 'fixture-cli.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
    if result.returncode:
        raise RuntimeError('Frozen fixture CLI failed')
    report = json.loads((out / 'fixture-report' / 'research_diagnostics.json').read_text())
    metrics = report['classification']['prediction']
    if metrics['count'] != 11 or abs(metrics['oa'] - 10 / 11) > 1e-12 or metrics['missing_class_ids'] != [3]:
        raise AssertionError('Frozen fixture metrics differ from hand calculation')
    return {'command': command, 'status': 'passed', 'count': metrics['count'], 'oa': metrics['oa'],
            'missing_class_ids': metrics['missing_class_ids'], 'synthetic_only': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--m8-manifest', type=Path, required=True, help='Successful scripts/validate_m8.py manifest')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists():
        parser.error('Run in the training container using ssh rmvision')
    if not re.fullmatch(r'[0-9a-f]{40}', args.expected_commit) or git('rev-parse', 'HEAD') != args.expected_commit:
        parser.error('Full expected SHA differs from HEAD')
    if git('status', '--porcelain', '--untracked-files=all') or sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        parser.error('Require clean checkout and assertions enabled')
    out = args.out_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('Write validation artifacts outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'module': 'E0a', 'commit': args.expected_commit, 'branch': git('branch', '--show-current'),
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'checks': [],
                'real_data_validation': 'not_performed_by_this_fixture_acceptance'}
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        upstream = json.loads(args.m8_manifest.read_text())
        if upstream.get('module') != 'M8' or upstream.get('status') != 'passed':
            raise ValueError('Complete M8 acceptance first; a pending report is insufficient')
        upstream_sha = upstream.get('commit', '')
        if not re.fullmatch(r'[0-9a-f]{40}', upstream_sha):
            raise ValueError('M8 manifest lacks a full tested SHA')
        comparison = subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', upstream_sha, 'HEAD', '--', *M8_PATHS],
                                    capture_output=True, text=True)
        if comparison.returncode:
            raise RuntimeError('Validated M8 source differs or commit unavailable; synchronize accepted upstream first')
        manifest['accepted_m8_commit'] = upstream_sha
        import torch
        from research_runtime import code_snapshot
        torch.set_num_threads(1)
        snapshot = code_snapshot()
        manifest.update(source_hash=snapshot['source_hash'], python=sys.version, torch=torch.__version__)
        if not torch.cuda.is_available():
            raise RuntimeError('Configured remote acceptance requires CUDA smoke')
        manifest['gpu'] = torch.cuda.get_device_name(0)
        for name, arguments in (
            ('unit', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']),
            ('smoke_cpu', ['smoke_test.py', '--device', 'cpu']), ('smoke_cuda', ['smoke_test.py', '--device', 'cuda']),
        ):
            command = [sys.executable, *arguments]
            with (out / (name + '.log')).open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
            manifest['checks'].append({'name': name, 'command': command, 'exit_code': result.returncode})
            save()
            if result.returncode:
                raise RuntimeError(name + ' failed; preserve this attempt and use a new output directory')
        manifest['fixture_export'] = fixture_cli(out, args.expected_commit)
        final = code_snapshot()
        if final['dirty'] or final['commit'] != snapshot['commit'] or final['source_hash'] != snapshot['source_hash']:
            raise RuntimeError('Source changed during validation')
        manifest['status'] = 'passed'
    except Exception:
        manifest['status'] = 'failed'; manifest['error'] = traceback.format_exc(); raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__': main()
