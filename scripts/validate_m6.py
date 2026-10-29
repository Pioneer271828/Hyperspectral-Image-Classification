"""Remote M6 acceptance: frozen source, direction traces and six GPU smoke runs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = '637e21b23481150bc36fb8006c659cc0dc83d021'
VARIANTS = ('none', 'add', 'gated', 'cnn_to_graph', 'graph_to_cnn', 'global')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


class InteractionTrace:
    """Temporary validation hooks; no training code, model state or loss changes."""
    def __init__(self):
        self.calls = Counter()
        self.directions = {}
        self.gradients = {}
        self.registered = set()
        self.handles = []
        self.gates = {}
        self.train_update_norms = {}

    def gradient_hook(self, name):
        def record(gradient):
            import torch
            item = self.gradients.setdefault(name, {'count': 0, 'finite': True, 'nonzero_count': 0})
            item['count'] += 1
            item['finite'] &= bool(torch.isfinite(gradient).all())
            item['nonzero_count'] += int(bool((gradient != 0).any()))
        return record

    def __enter__(self):
        import contextlib
        import torch
        from model.cross_branch_interaction import BidirectionalSelectiveInteraction, SelectiveMessage
        from model.region_pixel_alignment import RegionPixelAlignment
        self.stack = contextlib.ExitStack()
        aligned = BidirectionalSelectiveInteraction.forward_aligned
        message = SelectiveMessage.forward
        trace = self
        def exchange(layer, pixels, regions, assignment, alignment, **kwargs):
            trace.calls['exchange_train' if layer.training else 'exchange_eval'] += 1
            for direction in ('graph_to_cnn', 'cnn_to_graph'):
                module = getattr(layer, direction)
                if module is not None:
                    trace.directions[id(module)] = direction
            return aligned(layer, pixels, regions, assignment, alignment, **kwargs)
        def send(layer, target, source):
            direction = trace.directions[id(layer)]
            trace.calls[direction] += 1
            if layer.training and torch.is_grad_enabled() and id(layer) not in trace.registered:
                trace.registered.add(id(layer))
                for name, parameter in layer.named_parameters():
                    trace.handles.append(parameter.register_hook(trace.gradient_hook(direction + '.' + name)))
            result, gate = message(layer, target, source)
            with torch.no_grad():
                if not torch.isfinite(result).all() or not torch.isfinite(gate).all():
                    raise AssertionError('Nonfinite message or gate')
                current = trace.gates.setdefault(direction, {'min': 1., 'max': 0., 'sum': 0., 'count': 0})
                current['min'] = min(current['min'], float(gate.min()))
                current['max'] = max(current['max'], float(gate.max()))
                current['sum'] += float(gate.sum()); current['count'] += gate.numel()
                if layer.training:
                    trace.train_update_norms.setdefault(direction, []).append(float((result - target).norm()))
            return result, gate
        self.stack.enter_context(patch.object(BidirectionalSelectiveInteraction, 'forward_aligned', exchange))
        self.stack.enter_context(patch.object(SelectiveMessage, 'forward', send))
        for name in ('to_pixels', 'to_regions'):
            original = getattr(RegionPixelAlignment, name)
            def align(layer, *args, _name=name, _original=original):
                trace.calls[_name] += 1
                return _original(layer, *args)
            self.stack.enter_context(patch.object(RegionPixelAlignment, name, align))
        return self

    def __exit__(self, *args):
        for handle in self.handles: handle.remove()
        return self.stack.__exit__(*args)

    def check(self, config, plan):
        active = config.research.interaction.interaction_mode != 'none'
        direction = config.research.interaction.interaction_direction
        calls = 2 + sum(t['partition'] in ('val', 'test') for t in plan['tiles'])
        expected = {'exchange_train': 2 if active else 0, 'exchange_eval': calls - 2 if active else 0}
        for name, alignment in (('graph_to_cnn', 'to_pixels'), ('cnn_to_graph', 'to_regions')):
            enabled = active and (direction == 'both' or direction == name)
            expected[name] = expected[alignment] = calls if enabled else 0
            if enabled:
                if len(self.train_update_norms[name]) != 2 or min(self.train_update_norms[name]) <= 0:
                    raise AssertionError('Enabled direction did not update features twice')
                gate = self.gates[name]
                if not 0 <= gate['min'] <= gate['max'] <= 1:
                    raise AssertionError('Gate outside [0,1]')
                required = [name + '.projection.weight']
                if config.research.interaction.interaction_mode == 'gated':
                    required += [name + '.gate.0.weight', name + '.gate.2.weight']
                for parameter in required:
                    grad = self.gradients[parameter]
                    if grad['count'] != 2 or grad['nonzero_count'] != 2 or not grad['finite']:
                        raise AssertionError('Missing finite nonzero message gradient: ' + parameter)
        if any(self.calls[key] != value for key, value in expected.items()):
            raise AssertionError('Direction/alignment call trace differs: ' + str(self.calls))
        if any(not value['finite'] or value['count'] != 2 for value in self.gradients.values()):
            raise AssertionError('Invalid interaction parameter gradient')
        return {'calls': dict(self.calls), 'expected_calls': expected, 'parameter_gradients': self.gradients,
                'train_update_norms': self.train_update_norms,
                'gates': {key: {'min': value['min'], 'max': value['max'], 'mean': value['sum']/value['count'],
                                'count': value['count']} for key, value in self.gates.items()}}


def frozen_report(model, plan, stats, folder, cube, labels, wave, scale, out, commit):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from loadData.spatial_protocol import forward_partition
    from loadData.spatial_tiles import observe_tile, tile_slice
    from research_runtime import tensor_hash
    from export_research_diagnostics import export_frozen_predictions, file_hash
    dataset = model.training_config['spatial_config']['dataset']
    classes = model.heads[dataset].out_features
    before = {key: tensor_hash(value) for key, value in model.state_dict().items()}
    names = ('model.pt', 'model.pt.json', 'protocol.json', 'statistics.json', 'metrics.json')
    artifacts = {name: file_hash(folder / name) for name in names}
    recorded = json.loads((folder / 'manifest.json').read_text())['artifacts']
    if any(value != recorded[name] for name, value in artifacts.items()):
        raise AssertionError('Training artifacts differ from completed C0 attempt')
    arrays = {key: np.zeros(labels.shape, dtype=np.int64) for key in ('prediction', 'graph_prediction', 'cnn_prediction')}
    mask = np.zeros(labels.shape, dtype=bool)
    count = 0
    with torch.no_grad():
        for tile in plan['tiles']:
            if tile['partition'] != 'test': continue
            obs = observe_tile(cube, plan, stats, tile)
            outputs = forward_partition(model, dataset, obs, wave, scale)
            ys, xs = tile_slice(tile)
            if mask[ys, xs].any(): raise AssertionError('Overlapping test tiles')
            mask[ys, xs] = True; count += 1
            for key, prediction in zip(arrays, outputs[:3]):
                arrays[key][ys, xs] = prediction.argmax(-1).reshape(obs.spectral.shape[:2]).cpu().numpy() + 1
    if before != {key: tensor_hash(value) for key, value in model.state_dict().items()}:
        raise AssertionError('Frozen evaluation changed model state')
    frozen = out / 'frozen'; frozen.mkdir(parents=True)
    np.savez_compressed(frozen / 'arrays.npz', **arrays)
    np.save(frozen / 'labels.npy', labels); np.save(frozen / 'mask.npy', mask)
    report = export_frozen_predictions(frozen / 'arrays.npz', frozen / 'labels.npy', frozen / 'mask.npy', folder / 'model.pt',
        list(range(1, classes + 1)), out / 'report', prediction_commit=commit, exporter_commit=commit,
        dataset=dataset, protocol_id=plan['plan_hash'], partition='test')
    actual = report['classification']['prediction']
    expected = json.loads((folder / 'metrics.json').read_text())['test']
    if actual['confusion_matrix'] != expected['confusion'] or actual['count'] != expected['count']:
        raise AssertionError('Frozen confusion differs from training adapter')
    differences = {a: abs(actual[a] - expected[b]) for a, b in (('oa', 'oa'), ('aa', 'aa_supported'), ('kappa', 'kappa'))}
    valid = mask & (labels > 0)
    f1 = float(f1_score(labels[valid], arrays['prediction'][valid], labels=actual['macro_class_ids'], average='macro', zero_division=0))
    if max(differences.values()) > 1e-12 or abs(actual['macro_f1'] - f1) > 1e-12:
        raise AssertionError('Frozen metrics differ')
    if artifacts != {name: file_hash(folder / name) for name in names}:
        raise AssertionError('Frozen evaluation changed artifacts')
    return {'status': 'passed', 'test_tile_count': count, 'count': actual['count'],
            'missing_class_ids': actual['missing_class_ids'], 'confusion_matches_training': True,
            'metric_differences': differences, 'macro_f1_sklearn_difference': abs(actual['macro_f1'] - f1),
            'parameters_and_buffers_unchanged': True,
            'fused_spatial_counts': {w: {'boundary': v['boundary']['count'], 'interior': v['interior']['count']}
                                     for w, v in report['spatial']['prediction']['widths'].items()},
            'branch_strata': report['branches']['strata']}


def real_checks(data_root, out, snapshot, manifest, save):
    import numpy as np
    import torch
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS, inspect_dataset
    from loadData.data_read import readWavelengths
    from train_spatial import SpatialConfig, execute, restore_spatial_checkpoint
    from export_research_diagnostics import file_hash
    record, labels = inspect_dataset(data_root, 'paviaU')
    _, key, _, _ = DATASETS['paviaU']
    cube = loadmat(data_root / record['image_path'], variable_names=[key])[key]
    wave, _ = readWavelengths('paviaU', cube.shape[2])
    loader = lambda: (cube, labels, record, wave)
    reference_identity = None
    for variant in VARIANTS:
        config_path = ROOT / 'configs/experiments' / ('m6_spatial_' + variant + '.json')
        config = SpatialConfig(json.loads(config_path.read_text()))
        if config.budget['optimizer_steps'] != 2 or config.research.interaction.graph_update_mode != 'none':
            raise AssertionError('M6 engineering budget or edge control changed')
        with InteractionTrace() as trace:
            completed = execute(config, out / 'training', snapshot, loader)
        if completed['status'] != 'completed': raise AssertionError('Training did not complete')
        folder = out / 'training' / completed['run_id'] / 'attempts' / completed['attempt']
        metrics = json.loads((folder / 'metrics.json').read_text())
        if metrics['optimizer_steps'] != 2: raise AssertionError('Wrong actual training budget')
        model, plan, stats = restore_spatial_checkpoint(folder / 'model.pt', folder / 'protocol.json', folder / 'statistics.json', 'cuda')
        if model.backbone.interaction_config != config.research.interaction:
            raise AssertionError('Interaction configuration not restored')
        identity = (metrics['plan_hash'], metrics['statistics_hash'])
        if reference_identity is None: reference_identity = identity
        if identity != reference_identity: raise AssertionError('Variants use different spatial protocol or statistics')
        observed = trace.check(config, plan)
        diagnostics = frozen_report(model, plan, stats, folder, cube, labels, wave,
            config.research.data.pavia_superpixel_scale, out / 'diagnostics' / variant, snapshot['commit'])
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('Successful resume must skip optimizer')):
            resumed = execute(config, out / 'training', snapshot, loader, resume=True)
        if resumed['status'] != 'skipped': raise AssertionError('Resume did not skip')
        manifest['spatial_training'].append({'variant': variant, 'config': str(config_path.relative_to(ROOT)),
            'run_id': completed['run_id'], 'attempt': completed['attempt'], 'completed': completed['status'],
            'resumed': resumed['status'], 'optimizer_steps': metrics['optimizer_steps'], 'selection': metrics['selection'],
            'interaction': config.to_dict()['interaction'], 'trace': observed, 'frozen_diagnostics': diagnostics,
            'plan_hash': identity[0], 'statistics_hash': identity[1], 'test_missing_classes': metrics['coverage']['missing_classes']['test']})
        save(); del model
    for kind in ('image', 'label'):
        if file_hash(data_root / record[kind + '_path']) != record[kind + '_sha256']:
            raise AssertionError('Raw input changed: ' + kind)
    manifest['raw_data_unchanged'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists(): parser.error('Use ssh rmvision in the training container')
    if not re.fullmatch(r'[0-9a-f]{40}', args.expected_commit) or git('rev-parse', 'HEAD') != args.expected_commit:
        parser.error('Expected full SHA differs from HEAD')
    if git('status', '--porcelain', '--untracked-files=all') or sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        parser.error('Require clean checkout and assertions enabled')
    out = args.out_dir.resolve()
    if out == ROOT or ROOT in out.parents: parser.error('Write outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'module': 'M6', 'baseline_commit': BASE, 'commit': args.expected_commit,
                'branch': git('branch', '--show-current'), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'running', 'checks': [], 'spatial_training': []}
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        import torch
        from research_runtime import code_snapshot
        torch.set_num_threads(1)
        snapshot = code_snapshot()
        manifest.update(source_hash=snapshot['source_hash'], python=sys.version, torch=torch.__version__)
        if not torch.cuda.is_available(): raise RuntimeError('M6 acceptance requires CUDA')
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
            if result.returncode: raise RuntimeError(name + ' failed; preserve attempt and use a new directory')
        real_checks(args.data_root.resolve(strict=True), out, snapshot, manifest, save)
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
