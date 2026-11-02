"""G3-E4 on a new container: rebuild source weights and paired same-environment controls."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.container_runtime import training_container_kind
from train_cross_sensor import read_protocol, load_scene
from transfer.g3_training import train_source, _forward
from transfer.development_training import adapt, prepare_scene, sparse_labels, validate
from loadData.spatial_tiles import tile_slice
from loadData.transfer_protocol import freeze_task
from model.checkpoint_io import load_model_checkpoint

DEVELOPMENT_HASH = '1cf6c1394c1f5e6e82c0bf9c1d6c08004b6b477edefcaa73b73fdfe22ecfc984'
METHODS = ('target_scratch_full', 'source_pretrained_full_finetune')


def review_training(result):
    trace = result['trace']
    assert result['status'] == 'completed' and result['steps'] == 5000
    assert [x['step'] for x in trace] == list(range(1, 5001))
    assert [x['step'] for x in result['validation']] == list(range(100, 5001, 100))
    losses = np.array([x['loss']['total'] for x in trace])
    assert np.isfinite(losses).all()
    selected = min(result['validation'], key=lambda x: x['nll'])
    assert selected['step'] == result['selected_step']
    metadata = result['sampler']
    counts = dict(zip(metadata['tile_ids'], metadata['tile_class_counts']))
    sampled = np.array([counts[x['tile_id']] for x in trace])
    return {'selected': selected, 'final_validation': result['validation'][-1],
            'first_100_train_loss': float(losses[:100].mean()),
            'last_100_train_loss': float(losses[-100:].mean()),
            'class_participation_updates': (sampled > 0).sum(0).tolist(),
            'class_label_exposures': sampled.sum(0).tolist(),
            'anchor_counts': [sum(x.get('anchor_class') == c for x in trace) for c in metadata['classes']],
            'backbone_changed': result['initial_backbone_hash'] != result['final_backbone_hash'],
            'heads_changed': result['initial_head_hash'] != result['final_head_hash'],
            'sampler': metadata}


def validation_branches(scene, record, folder):
    """Read three heads at the already selected checkpoint; never select another step."""
    model = load_model_checkpoint(folder / 'model.pt', 'cuda')
    labels = sparse_labels(record, scene.plan, 'validation_support')
    matrices = [np.zeros((9, 9), dtype=np.int64) for _ in range(3)]
    losses = np.zeros(3); targets = []; predictions = []
    with torch.no_grad():
        for role in record['roles']:
            if role['role'] != 'devval':
                continue
            local = labels[tile_slice(role)].ravel(); valid = local > 0
            if not valid.any():
                continue
            target = local[valid] - 1
            outputs = _forward(model, scene, scene.plan['tiles'][role['id']])
            predicted = []
            for i in range(3):
                probabilities = outputs[i].detach().cpu().numpy()[valid]
                prediction = probabilities.argmax(1); predicted.append(prediction)
                matrices[i] += np.bincount(target * 9 + prediction, minlength=81).reshape(9, 9)
                losses[i] -= np.log(np.maximum(probabilities[np.arange(len(target)), target], 1e-8)).sum()
            targets.extend(target.tolist()); predictions.extend(np.stack(predicted, axis=1).tolist())
    assert len(targets) == 45
    report = {}
    for i, name in enumerate(('fusion', 'graph', 'cnn')):
        assert matrices[i].sum(axis=1).tolist() == [5] * 9
        recall = np.diag(matrices[i]) / 5
        report[name] = {'nll': float(losses[i] / 45), 'oa': float(np.trace(matrices[i]) / 45),
                        'aa': float(recall.mean()), 'recall': recall.tolist(), 'confusion': matrices[i].tolist()}
    targets = np.array(targets); predictions = np.array(predictions)
    report['per_class_both_branches_wrong'] = [
        float(((predictions[targets == c, 1] != c) & (predictions[targets == c, 2] != c)).mean()) for c in range(9)]
    report['scope'] = 'auxiliary readouts, not independently retrained CNN/Graph ablations'
    write_json(folder / 'validation-branches.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    for name in ('protocol-root', 'development-root', 'data-root', 'out-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(); code = code_snapshot(); kind = training_container_kind()
    if (not kind or sys.flags.optimize or code['dirty'] or code['commit'] != args.expected_commit
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        p.error('Require verified container, assertions, clean exact SHA and exactly one visible CUDA GPU')
    out = external_directory(args.out_dir)
    for path in (args.protocol_root, args.development_root, args.data_root):
        path = path.resolve(strict=True)
        if out == path or out in path.parents or path in out.parents:
            p.error('Output overlaps inputs')
    task, plans, support, frozen = read_protocol(args.protocol_root)
    record_path = args.development_root / 'development.json'
    record = json.loads(record_path.read_text()); frozen[str(record_path)] = hash_file(record_path)
    development_manifest = json.loads((args.development_root / 'manifest.json').read_text())
    frozen[str(args.development_root / 'manifest.json')] = hash_file(args.development_root / 'manifest.json')
    assert development_manifest['status'] == 'passed' and development_manifest['parent_task_hash'] == task['task_hash']
    validate(record, plans['target']); assert record['validation_hash'] == DEVELOPMENT_HASH
    for role in ('source', 'target'):
        for field in ('image', 'label'):
            identity = task[role + '_identity']
            path = (args.data_root / identity[field + '_path']).resolve(strict=True)
            assert path.is_relative_to(args.data_root.resolve(strict=True))
            assert hash_file(path) == identity[field + '_sha256']
            frozen[str(path)] = identity[field + '_sha256']
    base = json.loads((Path(__file__).resolve().parents[1] / 'configs/experiments/screen_m6_none.json').read_text())
    out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    state = {'status': 'running', 'code': code, 'container': kind, 'pid': os.getpid(),
             'started_at': datetime.now(timezone.utc).isoformat(), 'seeds': [2046, 2056],
             'development_hash': DEVELOPMENT_HASH, 'input_hashes': frozen, 'test_evaluated': False,
             'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                             'numpy': np.__version__, 'cuda': torch.version.cuda,
                             'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu': torch.cuda.get_device_name(0)},
             'plan': {'source_updates_per_seed': 10000, 'target_updates_per_run': 5000,
                      'new_source_updates': 20000, 'new_target_updates': 40000,
                      'val_every': 100, 'selection': 'minimum fusion validation NLL, earliest tie',
                      'backbone_lr': 5e-5, 'head_lr': 5e-4,
                      'baseline': 'uniform retrained in this environment; historical controls not pooled'},
             'runs': [], 'sources': [], 'pairs': []}
    started = time.monotonic()

    def save(): write_json(out / 'campaign.json', state)
    def progress(event): write_json(out / 'progress.json', event)
    save()
    try:
        for seed in state['seeds']:
            require_frozen_code(code)
            seed_root = out / f'seed-{seed}'; seed_root.mkdir()
            config = copy.deepcopy(task['config'])
            config.update(model_seed=seed, target_head_seed=seed + 2,
                          source_sampler_seed=seed + 3, target_sampler_seed=seed + 4)
            seed_task = freeze_task(config, task['source_identity'], task['target_identity'],
                                    plans['source'], plans['target'], support, task['sensor_evidence'])
            write_json(seed_root / 'frozen-task.json', seed_task)
            before = time.monotonic()
            source_scene, source_labels = load_scene(args.data_root, task['source_identity'], plans['source'], with_labels=True)
            source, source_result = train_source(source_scene, source_labels, base, seed_root / 'source',
                steps=10000, val_every=100, model_seed=seed, sampler_seed=seed + 3,
                device='cuda', task_hash=seed_task['task_hash'], progress=progress)
            del source_scene, source_labels
            state['sources'].append({'seed': seed, 'wall_seconds': time.monotonic() - before,
                                     'selected_step': source_result['selected_step'],
                                     'checkpoint_sha256': source_result['checkpoint_sha256'],
                                     'best_validation': min(source_result['history'], key=lambda x: x['val']['nll']),
                                     'last_validation': source_result['history'][-1], 'test_evaluated': False})
            save()
            raw_target, _ = load_scene(args.data_root, task['target_identity'], plans['target'], with_labels=False)
            target = prepare_scene(raw_target.cube, raw_target.bands, plans['target'], record)
            del raw_target
            controls = {}
            for mode in ('uniform', 'class_first'):
                (seed_root / mode).mkdir()
                for method in METHODS:
                    require_frozen_code(code); before = time.monotonic()
                    folder = seed_root / mode / f'{method}-5000'
                    result = adapt(target, record, base, folder, method=method, source=source,
                                   steps=5000, val_every=100, device='cuda', seed=seed, progress=progress, sampling=mode)
                    review = review_training(result)
                    assert hash_file(folder / 'model.pt') == result['checkpoint_sha256']
                    assert hash_file(folder / 'model.pt.json') == result['sidecar_sha256']
                    branches = validation_branches(target, record, folder)
                    review['branches'] = branches
                    write_json(folder / 'training-review.json', review)
                    if mode == 'uniform':
                        controls[method] = (result, review)
                    else:
                        baseline, baseline_review = controls[method]
                        for key in ('initial_backbone_hash', 'initial_head_hash', 'development_hash', 'resolved_config'):
                            assert result[key] == baseline[key]
                        assert result['sampler']['tile_class_counts'] == baseline['sampler']['tile_class_counts']
                        assert [x['tile_id'] for x in result['trace']] != [x['tile_id'] for x in baseline['trace']]
                        state['pairs'].append({'seed': seed, 'method': method,
                            'baseline': baseline_review['selected'], 'candidate': review['selected'],
                            'nll_delta': review['selected']['nll'] - baseline_review['selected']['nll'],
                            'oa_delta': review['selected']['oa'] - baseline_review['selected']['oa']})
                    state['runs'].append({'seed': seed, 'method': method, 'sampling': mode,
                                          'wall_seconds': time.monotonic() - before, 'review': review,
                                          'checkpoint_sha256': result['checkpoint_sha256'],
                                          'sidecar_sha256': result['sidecar_sha256']})
                    assert review['backbone_changed'] and review['heads_changed']
                    save()
            del source, target, controls
        require_frozen_code(code)
        assert all(hash_file(path) == value for path, value in frozen.items())
        assert len(state['pairs']) == 4 and len(state['runs']) == 8 and len(state['sources']) == 2
        state.update(status='passed', completed_source_updates=20000, completed_target_updates=40000)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(ended_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
        save()


if __name__ == '__main__':
    main()
