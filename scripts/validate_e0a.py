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


def real_spatial_check(m8_manifest_path, data_root, out, commit):
    """Freeze the accepted M8 default checkpoint's entire test partition; no fitting."""
    import numpy as np
    import torch
    from scipy.io import loadmat
    from sklearn.metrics import f1_score
    from unittest.mock import patch
    from loadData.dataset_registry import DATASETS, inspect_dataset
    from loadData.data_read import readWavelengths
    from loadData.spatial_tiles import observe_tile, tile_slice
    from loadData.spatial_protocol import forward_partition
    from train_spatial import SpatialConfig, restore_spatial_checkpoint
    from research_runtime import tensor_hash
    from evaluation.region_diagnostics import graph_diagnostics
    from export_research_diagnostics import export_frozen_predictions, file_hash

    upstream = json.loads(m8_manifest_path.read_text())
    run = next(value for value in upstream['spatial_training'] if value['variant'] == 'default')
    training_root = (m8_manifest_path.resolve().parent / 'training').resolve(strict=True)
    folder = (training_root / run['run_id'] / 'attempts' / run['attempt']).resolve(strict=True)
    if training_root not in folder.parents:
        raise ValueError('Checkpoint attempt escapes M8 training root')
    attempt = json.loads((folder / 'manifest.json').read_text())
    if attempt['status'] != 'completed':
        raise ValueError('M8 checkpoint attempt is not complete')
    names = ('model.pt', 'model.pt.json', 'protocol.json', 'statistics.json', 'metrics.json')
    for name in names:
        if file_hash(folder / name) != attempt['artifacts'][name]:
            raise ValueError('Accepted checkpoint artifact changed: ' + name)
    model, plan, stats = restore_spatial_checkpoint(folder / 'model.pt', folder / 'protocol.json', folder / 'statistics.json', 'cuda')
    dataset = model.training_config['spatial_config']['dataset']
    raw, labels = inspect_dataset(data_root, dataset)
    if raw != attempt['prepared_data']['raw']:
        raise ValueError('Raw dataset differs from M8 accepted data')
    classes = model.heads[dataset].out_features
    _, key, expected_classes, _ = DATASETS[dataset]
    if classes != expected_classes:
        raise ValueError('Checkpoint class count differs from dataset registry')
    cube = loadmat(data_root / raw['image_path'], variable_names=[key])[key]
    wave, _ = readWavelengths(dataset, cube.shape[2])
    config = model.training_config['spatial_config']
    resolved = SpatialConfig(config).research
    scale = resolved.data.pavia_superpixel_scale if dataset == 'paviaU' else resolved.data.superpixel_scale
    expected = json.loads((folder / 'metrics.json').read_text())['test']
    predictions = {name: np.zeros(labels.shape, dtype=np.int64) for name in ('prediction', 'graph_prediction', 'cnn_prediction')}
    selected = np.zeros(labels.shape, dtype=bool)
    states_before = {name: tensor_hash(value) for name, value in model.state_dict().items()}
    capture = {}
    original = model.backbone.learnable_graph.build_graph
    def capture_first(*args, **kwargs):
        state = original(*args, **kwargs)
        if not capture:
            capture.update(node_count=state.region_count, edge_indices=state.A0.indices().detach().cpu().numpy().copy(),
                           weights=state.A0.values().detach().cpu().numpy().copy())
        return state
    test_tiles = [tile for tile in plan['tiles'] if tile['partition'] == 'test']
    with torch.no_grad(), patch.object(model.backbone.learnable_graph, 'build_graph', side_effect=capture_first):
        for tile in test_tiles:
            obs = observe_tile(cube, plan, stats, tile)
            outputs = forward_partition(model, dataset, obs, wave, scale)
            local_shape = obs.spectral.shape[:2]
            ys, xs = tile_slice(tile)
            if selected[ys, xs].any():
                raise AssertionError('Test blocks overlap')
            for name, value in zip(('prediction', 'graph_prediction', 'cnn_prediction'), outputs[:3]):
                predictions[name][ys, xs] = value.argmax(-1).reshape(local_shape).cpu().numpy() + 1
            selected[ys, xs] = True
    if states_before != {name: tensor_hash(value) for name, value in model.state_dict().items()}:
        raise AssertionError('Frozen evaluation changed parameters or buffers')
    root = out / 'real-frozen'
    root.mkdir()
    np.savez_compressed(root / 'arrays.npz', **predictions)
    np.save(root / 'labels.npy', labels)
    np.save(root / 'mask.npy', selected)
    report = export_frozen_predictions(root / 'arrays.npz', root / 'labels.npy', root / 'mask.npy', folder / 'model.pt',
        list(range(1, classes + 1)), out / 'real-report', prediction_commit=commit, exporter_commit=commit,
        dataset=dataset, protocol_id=plan['plan_hash'], partition='test')
    actual = report['classification']['prediction']
    if actual['confusion_matrix'] != expected['confusion'] or actual['count'] != expected['count']:
        raise AssertionError('Frozen prediction confusion differs from accepted M8 metrics')
    differences = {}
    for key, other in (('oa', 'oa'), ('aa', 'aa_supported'), ('kappa', 'kappa')):
        if actual[key] is None or expected[other] is None:
            if actual[key] is not None or expected[other] is not None:
                raise AssertionError('Metric availability differs: ' + key)
            differences[key] = 0.0
        else:
            differences[key] = abs(actual[key] - expected[other])
    if any(value > 1e-12 for value in differences.values()):
        raise AssertionError('E0a metrics differ from accepted M8 metrics')
    valid = selected & (labels > 0)
    supported = actual['macro_class_ids']
    sklearn_f1 = float(f1_score(labels[valid], predictions['prediction'][valid], labels=supported, average='macro', zero_division=0))
    if abs(actual['macro_f1'] - sklearn_f1) > 1e-12:
        raise AssertionError('Supported-class macro F1 differs from sklearn')
    strata = report['branches']['strata']
    if sum(value['count'] for value in strata.values()) != actual['count']:
        raise AssertionError('Branch strata do not partition the evaluated pixels')
    for value in report['spatial']['prediction']['widths'].values():
        if value['boundary']['count'] + value['interior']['count'] != actual['count']:
            raise AssertionError('Boundary/interior split differs from evaluation mask')
    graph = graph_diagnostics(capture['node_count'], capture['edge_indices'], capture['weights'])
    np.savez_compressed(root / 'first-test-tile-graph.npz', edge_indices=capture['edge_indices'], weights=capture['weights'])
    for name in names:
        if file_hash(folder / name) != attempt['artifacts'][name]:
            raise ValueError('Accepted artifact changed during inference: ' + name)
    for kind in ('image', 'label'):
        if file_hash(data_root / raw[kind + '_path']) != raw[kind + '_sha256']:
            raise ValueError('Raw dataset changed during inference: ' + kind)
    return {'status': 'passed', 'dataset': dataset, 'checkpoint': str(folder / 'model.pt'), 'optimizer_steps': 0,
            'test_tile_count': len(test_tiles), 'count': actual['count'], 'missing_class_ids': actual['missing_class_ids'],
            'macro_class_ids': supported, 'aa_full': actual['aa_full'], 'confusion_matches_m8': True,
            'metric_differences': differences, 'macro_f1_sklearn_difference': abs(actual['macro_f1'] - sklearn_f1),
            'parameters_and_buffers_unchanged': True, 'plan_hash': plan['plan_hash'], 'statistics_hash': stats['statistics_hash'],
            'fused_spatial_counts': {w: {'boundary': v['boundary']['count'], 'interior': v['interior']['count']}
                                     for w, v in report['spatial']['prediction']['widths'].items()},
            'branch_strata': strata, 'first_tile_graph': {'scope': 'first_test_tile_only', 'tile': test_tiles[0], **graph}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--m8-manifest', type=Path, required=True, help='Successful scripts/validate_m8.py manifest')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--real-data-root', type=Path, help='Optional read-only dataset root for accepted M8 test predictions')
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
        if args.real_data_root is not None:
            manifest['real_data_validation'] = real_spatial_check(args.m8_manifest, args.real_data_root.resolve(strict=True), out, args.expected_commit)
        final = code_snapshot()
        if final['dirty'] or final['commit'] != snapshot['commit'] or final['source_hash'] != snapshot['source_hash']:
            raise RuntimeError('Source changed during validation')
        manifest['status'] = 'passed'
    except Exception:
        manifest['status'] = 'failed'; manifest['error'] = traceback.format_exc(); raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__': main()
