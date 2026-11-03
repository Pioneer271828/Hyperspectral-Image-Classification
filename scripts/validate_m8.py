"""Frozen M8 acceptance with named-loss logging and GPU checkpoint predictions."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = 'b2809302a44d1ff454004361a42d34001235616d'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def real_checks(data_root, out, snapshot):
    import numpy as np
    import torch
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS, inspect_dataset
    from loadData.data_read import readWavelengths
    from loadData.spatial_tiles import observe_tile
    from train_spatial import SpatialConfig, execute, restore_spatial_checkpoint
    record, labels = inspect_dataset(data_root, 'paviaU')
    _, key, _, _ = DATASETS['paviaU']
    cube = loadmat(data_root / record['image_path'], variable_names=[key])[key]
    wave, _ = readWavelengths('paviaU', cube.shape[2])
    loader = lambda: (cube, labels, record, wave)
    results = []
    for variant in ('default', 'aux_off', 'alpha035'):
        config_path = ROOT / 'configs' / 'experiments' / ('m8_spatial_' + variant + '.json')
        config = SpatialConfig(json.loads(config_path.read_text()))
        completed = execute(config, out / 'training', snapshot, loader)
        if completed['status'] != 'completed': raise AssertionError('Training did not complete')
        folder = out / 'training' / completed['run_id'] / 'attempts' / completed['attempt']
        metrics = json.loads((folder / 'metrics.json').read_text())
        if metrics['optimizer_steps'] != 2: raise AssertionError('Wrong training budget')
        history = metrics['loss_component_history']
        if len(history) != 1: raise AssertionError('Missing epoch loss record')
        terms = history[0]['mean_per_step']
        if len(terms) != 13 or not all(np.isfinite(v) for v in terms.values()):
            raise AssertionError('Invalid loss decomposition')
        weighted_sum = sum(v for k, v in terms.items() if k.startswith('weighted_'))
        if not np.isclose(weighted_sum, terms['total'], rtol=1e-6, atol=1e-6):
            raise AssertionError('Weighted terms do not sum to total')
        if variant == 'aux_off' and (terms['weighted_graph_supervised'] != 0 or terms['weighted_cnn_supervised'] != 0):
            raise AssertionError('Disabled auxiliary supervision still contributes')
        model, plan, stats = restore_spatial_checkpoint(folder / 'model.pt', folder / 'protocol.json', folder / 'statistics.json', 'cuda')
        if model.backbone.alph != config.research.model.alph: raise AssertionError('Alpha not restored')
        tile = next(t for t in plan['tiles'] if t['partition'] == 'val')
        obs = observe_tile(cube, plan, stats, tile)
        x = torch.tensor(np.concatenate((obs.spectral, obs.coordinates), axis=-1), device='cuda')
        wavelength = torch.as_tensor(wave, device='cuda')
        with torch.no_grad():
            probability = model('paviaU', x, wavelengths=wavelength, return_graph_info=True)
            logits = model('paviaU', x, wavelengths=wavelength, return_graph_info=True, return_logits=True)
            differences = []
            for raw, expected in zip(logits[:3], probability[:3]):
                actual = raw.softmax(-1)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                differences.append(float((actual - expected).abs().max()))
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('Successful resume must skip optimizer')):
            resumed = execute(config, out / 'training', snapshot, loader, resume=True)
        if resumed['status'] != 'skipped': raise AssertionError('Resume did not skip')
        results.append({'variant': variant, 'config': str(config_path.relative_to(ROOT)),
                        'run_id': completed['run_id'], 'attempt': completed['attempt'], 'completed': completed['status'],
                        'resumed': resumed['status'], 'optimizer_steps': metrics['optimizer_steps'],
                        'alph': model.backbone.alph, 'selection': metrics['selection'],
                        'loss_means': terms, 'weighted_sum_error': abs(weighted_sum - terms['total']),
                        'logits_softmax_max_diff': differences, 'output_shapes': [list(p.shape) for p in probability[:3]],
                        'plan_hash': metrics['plan_hash'], 'statistics_hash': metrics['statistics_hash'],
                        'test_missing_classes': metrics['coverage']['missing_classes']['test']})
        del model
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists(): parser.error('Use ssh rmvision in the training container')
    if len(args.expected_commit) != 40 or git('rev-parse', 'HEAD') != args.expected_commit: parser.error('Expected full SHA differs from HEAD')
    if git('status', '--porcelain', '--untracked-files=all') or sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        parser.error('Require clean checkout and assertions enabled')
    out = args.out_dir.resolve()
    if out == ROOT or ROOT in out.parents: parser.error('Write outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from research_runtime import code_snapshot
    torch.set_num_threads(1)
    snapshot = code_snapshot()
    manifest = {'module': 'M8', 'baseline_commit': BASE, 'commit': snapshot['commit'], 'branch': snapshot['branch'],
                'source_hash': snapshot['source_hash'], 'python': sys.version, 'torch': torch.__version__, 'device': 'cuda',
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'checks': []}
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('M8 acceptance requires CUDA')
        manifest['gpu'] = torch.cuda.get_device_name(0)
        for name, command in (
            ('unit', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']),
            ('smoke_cpu', ['smoke_test.py', '--device', 'cpu']), ('smoke_cuda', ['smoke_test.py', '--device', 'cuda']),
        ):
            command = [sys.executable, *command]
            with (out / (name + '.log')).open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
            manifest['checks'].append({'name': name, 'command': command, 'exit_code': result.returncode})
            save()
            if result.returncode: raise RuntimeError(name + ' failed; preserve logs and retry in a new directory')
        manifest['spatial_training'] = real_checks(args.data_root.resolve(strict=True), out, snapshot)
        torch.cuda.synchronize()
        manifest['training_process_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        manifest['training_process_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        final = code_snapshot()
        if final['commit'] != snapshot['commit'] or final['dirty'] or final['source_hash'] != snapshot['source_hash']:
            raise RuntimeError('Code changed during validation')
        manifest['status'] = 'passed'
    except Exception:
        manifest['status'] = 'failed'; manifest['error'] = traceback.format_exc(); raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__': main()
