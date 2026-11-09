"""Bounded PaviaU spatial adaptation on frozen role releases."""
import argparse
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
from scipy.io import loadmat
import torch

from evaluation.g4_target import independent_trainval
from joint_training import _atomic_torch_save, _restore_rng, _rng_state, _run_lock, _tree_fingerprint
from loadData.data_read import read_band_metadata
from loadData.spatial_tiles import tile_slice, validate_tile_plan
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from train_spatial import SpatialConfig
from transfer.g3_training import Scene, _forward, _optimizer, _update, state_hash

TARGET = 'paviaU'
CLASSES = 9
STEPS = 12
EVENTS = (1, 4, 7, 10, 12)
SEED = 64
CPU = torch.device('cpu')
SHARED = ('adapter.', 'CNN_denoise.', 'CNN_Branch.', 'HGNN_Branch.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_release(release, plan, support_seed, k):
    validate_tile_plan(plan)
    fields = {'scene', 'plan_hash', 'support', 'validation', 'train_role', 'validation_role', 'release_hash'}
    if (set(release) != fields or release['scene'] != TARGET or release['plan_hash'] != plan['plan_hash']
            or release['train_role'] != 'train' or release['validation_role'] != 'val'
            or release['release_hash'] != digest({n: v for n, v in release.items() if n != 'release_hash'})):
        raise ValueError('Spatial role release identity differs')
    shape = tuple(plan['shape'])
    role_ids = np.full(shape, -1, dtype=np.int32)
    result = {}
    seen = set()
    for key, role, budget, seed in (('support', 'train', k, support_seed), ('validation', 'val', 5, 3031)):
        row = release[key]
        ids, labels = row['indices'], row['labels']
        if (set(row) != {'seed', 'per_class', 'indices', 'labels'} or row['seed'] != seed
                or row['per_class'] != budget or len(ids) != CLASSES * budget or len(labels) != len(ids)
                or any(type(i) is not int or not 0 <= i < math.prod(shape) for i in ids)
                or ids != sorted(set(ids)) or seen.intersection(ids)
                or any(type(c) is not int or not 1 <= c <= CLASSES for c in labels)
                or any(labels.count(c) != budget for c in range(1, CLASSES + 1))):
            raise ValueError('Spatial role label budget or coordinates differ')
        seen.update(ids)
        tiles = [t for t in plan['tiles'] if t['partition'] == role]
        for tile in tiles:
            role_ids[tile_slice(tile)] = tile['id']
        allowed = {t['id'] for t in tiles}
        if not set(role_ids.ravel()[ids]).issubset(allowed):
            raise ValueError('Released label escapes the registered spatial role')
        labels_grid = np.zeros(shape, dtype=np.int64)
        labels_grid.ravel()[ids] = labels
        result[role] = [(t, labels_grid[tile_slice(t)].copy()) for t in tiles
                        if (labels_grid[tile_slice(t)] > 0).any()]
    return result


def initialize(config, source_path=None, source_result=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = config.research.build_model({TARGET: CLASSES})
    backbone = model.backbone
    fresh = backbone.state_dict()
    if backbone.graph_mode != 'learned' or backbone.norm_type != 'domain_bn':
        raise ValueError('Expected learned graph and fresh target DomainBN')
    graph = {n: v for n, v in fresh.items() if n.startswith('learnable_graph.')}
    shared = {n for n in fresh if n.startswith(SHARED) and '.norms.' not in n}
    if len(graph) != 6 or len(shared) != 29 or len(fresh) - len(graph) - len(shared) != 25:
        raise ValueError('The registered backbone parameter contract changed')
    copied = []
    if source_path is not None:
        source = load_model_checkpoint(source_path, device='cpu')
        result = read(source_result)
        tasks = {'xuzhou', 'whu-honghu'}
        if (source.backbone.graph_mode != 'static' or source.backbone.norm_type != 'domain_bn'
                or source.backbone.region_residual_mode != 'none' or set(source.heads) != tasks
                or set(result['per_dataset_updates']) != tasks or result['status'] != 'completed'):
            raise ValueError('Source roles or architecture differ')
        old = source.backbone.state_dict()
        eligible = {n for n in old if n.startswith(SHARED) and '.norms.' not in n}
        if eligible != shared:
            raise ValueError('Shared source tensor names differ')
        for name in sorted(shared):
            a, b = old[name], fresh[name]
            if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all():
                raise ValueError('Invalid shared tensor: ' + name)
            fresh[name] = a
            copied.append(name)
        backbone.load_state_dict(fresh, strict=True)
        if any(not torch.equal(backbone.state_dict()[n], old[n]) for n in copied):
            raise ValueError('Shared weights were not copied exactly')
    return model, {
        'copied_keys': copied, 'fresh_graph_keys': sorted(graph),
        'graph_sha256': _tree_fingerprint(graph),
        'fresh_domain_sha256': _tree_fingerprint({n: v for n, v in fresh.items() if '.norms.' in n}),
        'fresh_heads_sha256': _tree_fingerprint({n: v for n, v in model.state_dict().items() if not n.startswith('backbone.')}),
        'initial_state_sha256': state_hash(model),
    }


@torch.no_grad()
def validation(model, scene, tiles):
    model.eval()
    names = ('fused', 'graph', 'cnn')
    scores = {n: [] for n in names}
    truth = []
    for tile, local in tiles:
        if tile['partition'] != 'val':
            raise ValueError('Validation can only access validation tiles')
        local = torch.from_numpy(local.reshape(-1))
        valid = local > 0
        truth.append(local[valid] - 1)
        heads = _forward(model, scene, tile)[:3]
        for name, head in zip(names, heads):
            scores[name].append(head[valid].cpu())
    labels = torch.cat(truth)
    metrics, probabilities = {}, {}
    for name in names:
        p = torch.cat(scores[name])
        predicted = p.argmax(1)
        cm = torch.bincount(labels * CLASSES + predicted, minlength=CLASSES**2).reshape(CLASSES, CLASSES)
        recall = cm.diag().double() / cm.sum(1)
        metrics[name] = {'confusion': cm.tolist(), 'recall': recall.tolist(), 'oa': float(cm.diag().sum().double() / len(labels)),
                         'aa': float(recall.mean()), 'nll': float(-p[torch.arange(len(labels)), labels].clamp_min(1e-8).log().mean())}
        probabilities[name] = p
    independent_trainval({'train': probabilities, 'val': probabilities},
                         {'train_labels': labels, 'val_labels': labels, 'class_num': CLASSES},
                         {'train': metrics, 'val': metrics})
    return metrics, probabilities


def run_case(registration, index, out, until, code):
    case = registration['cases'][index]
    pins = {Path(n): h for n, h in registration['pins'].items()}

    def guard():
        require_frozen_code(code)
        if any(hash_file(p) != h for p, h in pins.items()):
            raise ValueError('A frozen input changed')

    out = external_directory(out)
    if any(out == p or out in p.parents for p in pins):
        raise ValueError('Output contains a frozen input')
    guard()
    plan = read(registration['plan'])
    role = load_release(read(case['release']), plan, case['support_seed'], case['k'])
    cube = loadmat(registration['image'], variable_names=['paviaU'])['paviaU']
    scene = Scene.prepare(TARGET, cube, plan, read_band_metadata(TARGET, cube.shape[-1]), CLASSES)
    if scene.stats != read(registration['statistics']):
        raise ValueError('Train-only statistics differ from frozen spatial preflight')
    config = SpatialConfig(registration['config'])
    source = registration['source'] if case['method'] == 'shared_transfer' else None
    if case['method'] not in ('scratch', 'shared_transfer'):
        raise ValueError('Unknown initialization method')
    model, init = initialize(config, source['weight'] if source else None, source['result'] if source else None)
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=20)
    rng = np.random.default_rng(SEED)
    sequence = rng.integers(len(role['train']), size=STEPS).tolist()
    identity = digest({'registration': registration, 'case': index, 'code': code['commit'], 'init': init})
    out.mkdir(parents=True, exist_ok=True)
    state_path = out/'state.pt'
    with _run_lock(out):
        model.training_config = {'spatial_p4_identity': identity, 'plan_hash': plan['plan_hash'],
                                 'statistics_hash': scene.stats['statistics_hash'], 'initialization': init}
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            if state['identity'] != identity:
                raise ValueError('Resume identity differs')
            fingerprints = state.pop('fingerprints')
            if fingerprints != {n: _tree_fingerprint(state[n]) for n in ('model', 'optimizer', 'scheduler', 'rng', 'trace', 'history', 'best')}:
                raise ValueError('Resume state integrity differs')
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            _restore_rng(state['rng'], CPU)
            cursor, trace, history, best = (state[n] for n in ('cursor', 'trace', 'history', 'best'))
            if (not 0 <= cursor <= STEPS or [r['step'] for r in trace] != list(range(1, cursor+1))
                    or [r['step'] for r in history] != [i for i in EVENTS if i <= cursor] or (cursor and not best)):
                raise ValueError('Resume cursor and event history differ')
        else:
            torch.manual_seed(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            cursor, trace, history, best = 0, [], [], None
        start = cursor
        if cursor > until:
            raise ValueError('Requested stop precedes authoritative state')
        for i in range(cursor, until):
            tile, local = role['train'][sequence[i]]
            if tile['partition'] != 'train':
                raise ValueError('Optimization can only access training tiles')
            graph_before = {n: p.detach().clone() for n, p in model.backbone.learnable_graph.named_parameters()}
            loss = _update(model, scene, tile, local, config, optimizer)
            graph = dict(model.backbone.learnable_graph.named_parameters())
            changed = sum(int(torch.count_nonzero(p.detach() - graph_before[n])) for n, p in graph.items())
            grad = sum(float(p.grad.detach().abs().sum()) for p in graph.values() if p.grad is not None)
            if not changed or not math.isfinite(grad) or grad <= 0:
                raise ValueError('Learned graph did not participate in optimization')
            trace.append({'step': i+1, 'tile': tile['id'], 'loss': loss, 'graph_changed_elements': changed,
                          'graph_gradient_l1': grad, 'lr': [g['lr'] for g in optimizer.param_groups]})
            if i+1 in EVENTS:
                metrics, scores = validation(model, scene, role['val'])
                nll = metrics['fused']['nll']
                scheduler.step(nll)
                history.append({'step': i+1, 'metrics': metrics})
                if best is None or nll < best['nll']:
                    best = {'step': i+1, 'nll': nll, 'scores': scores,
                            'model': {n: v.detach().clone() for n, v in model.state_dict().items()}}
            state = {'identity': identity, 'cursor': i+1, 'trace': trace, 'history': history, 'best': best,
                     'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'rng': _rng_state(CPU)}
            state['fingerprints'] = {n: _tree_fingerprint(state[n]) for n in ('model','optimizer','scheduler','rng','trace','history','best')}
            _atomic_torch_save(state, state_path)
        guard()
        if until == STEPS:
            model.load_state_dict(best['model'], strict=True)
            metrics, scores = validation(model, scene, role['val'])
            if (_tree_fingerprint(scores) != _tree_fingerprint(best['scores'])
                    or metrics != next(e['metrics'] for e in history if e['step'] == best['step'])):
                raise ValueError('Selected checkpoint probabilities changed after strict reload')
            report = {'identity': identity, 'initialization': init, 'updates': STEPS,
                      'best_step': best['step'], 'metrics': metrics, 'raw_gt_loaded': False,
                      'query_accessed': False, 'state_sha256': hash_file(state_path)}
            if (out/'result.json').exists():
                if read(out/'result.json') != report:
                    raise ValueError('Completed result differs')
            else:
                write_json(out/'result.json', report)
        return {'updates_this_call': until-start, 'cursor': until, 'identity': identity}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--case', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--until', type=int, choices=(2, STEPS), required=True)
    args = p.parse_args()
    if hash_file(args.registration) != args.sha256:
        raise ValueError('Registration SHA256 differs')
    registration = read(args.registration)
    code = code_snapshot()
    if code['commit'] != registration['commit'] or not 0 <= args.case < len(registration['cases']):
        raise ValueError('Unregistered code or case')
    paths = [registration[x] for x in ('image', 'plan', 'statistics')]
    paths += [case['release'] for case in registration['cases']]
    source = registration['source']
    paths += [source['weight'], source['weight']+'.json', source['result']]
    if set(paths) - set(registration['pins']):
        raise ValueError('Registration has unpinned inputs')
    print(json.dumps(run_case(registration, args.case, args.out, args.until, code)), flush=True)


if __name__ == '__main__':
    main()
