"""M5 frozen acceptance: regression, CUDA smoke, and two-step spatial runs."""
import argparse
import contextlib
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
BASE = 'b9ef4c98eea3a1df6f4ac4deebc2767fe94050cd'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def real_training(data_root, out, snapshot):
    import numpy as np
    import torch
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS, inspect_dataset
    from loadData.data_read import readWavelengths
    from train_spatial import SpatialConfig, execute, restore_spatial_checkpoint
    from model.VCASAHGCN import VCASAHGCNBackbone
    record, labels = inspect_dataset(data_root, 'paviaU')
    _, key, _, _ = DATASETS['paviaU']
    cube = loadmat(data_root / record['image_path'], variable_names=[key])[key]
    wave, _ = readWavelengths('paviaU', cube.shape[2])
    loader = lambda: (cube, labels, record, wave)
    results = []
    for variant in ('plain', 'fixed_q', 'joint'):
        config_path = ROOT / 'configs' / 'experiments' / ('m5_spatial_' + variant + '.json')
        config = SpatialConfig(json.loads(config_path.read_text()))
        trace = {name: {'calls': 0, 'shapes': set(), 'gradient_l1': []}
                 for name in ('cnn_stage1', 'graph_stage1', 'cnn_stage2', 'graph_stage2')}
        with contextlib.ExitStack() as stack:
            for name, item in trace.items():
                original = getattr(VCASAHGCNBackbone, name)
                def wrapped(self, *args, _original=original, _item=item, **kwargs):
                    output = _original(self, *args, **kwargs)
                    _item['calls'] += 1
                    _item['shapes'].add(tuple(output.shape))
                    if output.requires_grad:
                        output.register_hook(lambda gradient, target=_item: target['gradient_l1'].append(float(gradient.abs().sum())))
                    return output
                stack.enter_context(patch.object(VCASAHGCNBackbone, name, wrapped))
            completed = execute(config, out / 'training', snapshot, loader)
        if completed['status'] != 'completed':
            raise AssertionError('Training did not complete')
        folder = out / 'training' / completed['run_id'] / 'attempts' / completed['attempt']
        metrics = json.loads((folder / 'metrics.json').read_text())
        if metrics['optimizer_steps'] != 2:
            raise AssertionError('Incorrect training budget')
        if len({value['calls'] for value in trace.values()}) != 1:
            raise AssertionError('Stages executed unequal numbers of times')
        for name, item in trace.items():
            if len(item['gradient_l1']) != 2 or not all(np.isfinite(v) and v > 0 for v in item['gradient_l1']):
                raise AssertionError('Stage did not receive two finite nonzero gradients: ' + name)
            item['shapes'] = sorted(item['shapes'])
        restored, plan, stats = restore_spatial_checkpoint(folder / 'model.pt', folder / 'protocol.json', folder / 'statistics.json', 'cuda')
        if restored.backbone.graph_assignment_mode != config.research.model.graph_assignment_mode:
            raise AssertionError('Restored wrong graph control')
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('Successful resume must not optimize')):
            resumed = execute(config, out / 'training', snapshot, loader, resume=True)
        if resumed['status'] != 'skipped':
            raise AssertionError('Successful resume did not skip')
        results.append({'variant': variant, 'config': str(config_path.relative_to(ROOT)),
                        'run_id': completed['run_id'], 'attempt': completed['attempt'],
                        'completed': completed['status'], 'resumed': resumed['status'],
                        'optimizer_steps': metrics['optimizer_steps'], 'selection': metrics['selection'],
                        'plan_hash': metrics['plan_hash'], 'statistics_hash': metrics['statistics_hash'],
                        'test_missing_classes': metrics['coverage']['missing_classes']['test'],
                        'graph_assignment_mode': restored.backbone.graph_assignment_mode,
                        'graph_adjacency_mode': restored.backbone.graph_adjacency_mode,
                        'stage_trace': trace})
        del restored
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists():
        parser.error('Use ssh rmvision in the training container')
    if len(args.expected_commit) != 40 or git('rev-parse', 'HEAD') != args.expected_commit:
        parser.error('Full expected SHA must equal HEAD')
    if git('status', '--porcelain', '--untracked-files=all') or sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        parser.error('Require clean checkout and assertions enabled')
    out = args.out_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('Write artifacts outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from research_runtime import code_snapshot
    torch.set_num_threads(1)
    snapshot = code_snapshot()
    manifest = {'module': 'M5', 'baseline_commit': BASE, 'commit': snapshot['commit'],
                'branch': snapshot['branch'], 'source_hash': snapshot['source_hash'],
                'python': sys.version, 'torch': torch.__version__, 'device': 'cuda',
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'checks': []}
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('M5 acceptance requires CUDA')
        manifest['gpu'] = torch.cuda.get_device_name(0)
        for name, command in (
            ('unit', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']),
            ('smoke_cpu', ['smoke_test.py', '--device', 'cpu']),
            ('smoke_cuda', ['smoke_test.py', '--device', 'cuda']),
        ):
            command = [sys.executable, *command]
            with (out / (name + '.log')).open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
            manifest['checks'].append({'name': name, 'command': command, 'exit_code': result.returncode})
            save()
            if result.returncode:
                raise RuntimeError(name + ' failed; retain logs and retry in a new directory')
        manifest['spatial_training'] = real_training(args.data_root.resolve(strict=True), out, snapshot)
        torch.cuda.synchronize()
        manifest['training_process_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        manifest['training_process_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        final = code_snapshot()
        if final['commit'] != snapshot['commit'] or final['dirty'] or final['source_hash'] != snapshot['source_hash']:
            raise RuntimeError('Code changed during validation')
        manifest['status'] = 'passed'
    except Exception:
        manifest['status'] = 'failed'
        manifest['error'] = traceback.format_exc()
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
