"""Bounded P4 spatial adaptation on frozen fixed and development role releases."""
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
from loadData.observation_sampling import subset_statistics
from loadData.dataset_registry import DATASETS
from loadData.spatial_tiles import tile_slice, validate_tile_plan
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from train_spatial import SpatialConfig
from transfer.g3_training import Scene, _forward, _optimizer, _update, state_hash
from transfer import p4_coverage, p4_exposure, p4_guarded, p4_update_evidence as updates

TARGETS = {'paviaU', 'xuzhou', p4_guarded.SCENE}
SOURCE_TASKS = TARGETS
STEPS = 12
EVENTS = (1, 4, 7, 10, 12)
SEED = 64
CPU = torch.device('cpu')
SHARED = ('adapter.', 'CNN_denoise.', 'CNN_Branch.', 'HGNN_Branch.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def development_tiles(development, plan):
    validate_tile_plan(plan)
    if (development['scene'] != 'xuzhou' or development['parent_plan_hash'] != plan['plan_hash']
            or development['shape'] != plan['shape'] or len(development['roles']) != len(plan['tiles'])
            or development['development_roles_hash'] != digest({n:v for n,v in development.items()
                                                                 if n != 'development_roles_hash'})):
        raise ValueError('Development role identity differs')
    result = {'train': [], 'val': []}
    for original, row in zip(plan['tiles'], development['roles']):
        if {n:v for n,v in row.items() if n!='role'} != original:
            raise ValueError('Development role changes registered geometry')
        role = row['role']
        if original['partition'] == 'train' and role in ('devtrain', 'devval'):
            result['train' if role == 'devtrain' else 'val'].append(original)
        elif (original['partition'], role) not in (('val','unused'), ('test','test')):
            raise ValueError('Development roles can only use original training tiles')
    if not all(result.values()) or [t['id'] for t in result['val']] != development['heldout_tile_ids']:
        raise ValueError('Development validation tile manifest differs')
    return result


def fit_development_statistics(cube, plan, development):
    tiles = development_tiles(development, plan)['train']
    return fit_role_statistics(cube, plan, tiles, development['development_roles_hash'])


def fit_role_statistics(cube, plan, tiles, roles_hash, *, positions=None):
    if cube.ndim != 3 or list(cube.shape[:2]) != plan['shape']:
        raise ValueError('Development observation shape differs')
    low, high = [], []
    for tile in tiles:
        values = cube[tile_slice(tile)]
        if positions is not None:
            values = values[:, :, positions]
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite development training observation')
        low.append(values.min(axis=(0,1)))
        high.append(values.max(axis=(0,1)))
    stats = {'schema_version': 1, 'plan_hash': plan['plan_hash'], 'fit_partition': 'train',
             'fit_role': 'devtrain', 'development_roles_hash': roles_hash,
             'minimum': np.min(low,axis=0).tolist(), 'maximum': np.max(high,axis=0).tolist()}
    stats['statistics_hash'] = digest(stats)
    return stats


def load_release(release, plan, support_seed, k, development=None, *, guarded_tiles=None):
    validate_tile_plan(plan)
    fields = {'scene', 'support', 'validation', 'release_hash'}
    if development is None:
        fields |= {'plan_hash', 'train_role', 'validation_role'}
        contract = (release.get('scene') == 'paviaU' and release.get('plan_hash') == plan['plan_hash']
                    and release.get('train_role') == 'train' and release.get('validation_role') == 'val')
        role_tiles = {r:[t for t in plan['tiles'] if t['partition']==r] for r in ('train','val')}
    else:
        fields.add('development_roles_hash')
        scene = p4_guarded.SCENE if guarded_tiles is not None else 'xuzhou'
        contract = (release.get('scene') == scene
                    and release.get('development_roles_hash') == development['development_roles_hash'])
        role_tiles = development_tiles(development, plan) if guarded_tiles is None else guarded_tiles
    if (set(release) != fields or not contract or support_seed not in (512,2048) or k not in (1,5)
            or release['release_hash'] != digest({n: v for n, v in release.items() if n != 'release_hash'})):
        raise ValueError('Spatial role release identity differs')
    shape = tuple(plan['shape'])
    classes = DATASETS[release['scene']][2]
    role_ids = np.full(shape, -1, dtype=np.int32)
    result = {}
    seen = set()
    for key, role, budget, seed in (('support', 'train', k, support_seed), ('validation', 'val', 5, 3031)):
        row = release[key]
        ids, labels = row['indices'], row['labels']
        if (set(row) != {'seed', 'per_class', 'indices', 'labels'} or row['seed'] != seed
                or row['per_class'] != budget or len(ids) != classes * budget or len(labels) != len(ids)
                or any(type(i) is not int or not 0 <= i < math.prod(shape) for i in ids)
                or ids != sorted(set(ids)) or seen.intersection(ids)
                or any(type(c) is not int or not 1 <= c <= classes for c in labels)
                or any(labels.count(c) != budget for c in range(1, classes + 1))):
            raise ValueError('Spatial role label budget or coordinates differ')
        seen.update(ids)
        tiles = role_tiles[role]
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


def initialize(config, source_path=None, source_result=None, *, coverage_arm=None, model_seed=SEED):
    target = config.values['dataset']
    if target not in TARGETS:
        raise ValueError('Unknown P4 spatial target')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = config.research.build_model({target: DATASETS[target][2]})
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
        if coverage_arm is not None and source.backbone.coverage_modulation is not None:
            raise ValueError('Matched coverage requires an E1-off source checkpoint')
        result = read(source_result)
        tasks = SOURCE_TASKS - {target}
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
    initialization = {
        'copied_keys': copied, 'fresh_graph_keys': sorted(graph),
        'graph_sha256': _tree_fingerprint(graph),
        'fresh_domain_sha256': _tree_fingerprint({n: v for n, v in fresh.items() if '.norms.' in n}),
        'fresh_heads_sha256': _tree_fingerprint({n: v for n, v in model.state_dict().items() if not n.startswith('backbone.')}),
        'initial_state_sha256': state_hash(model),
    }
    if coverage_arm is not None:
        model, coverage = p4_coverage.initialize(model, config, coverage_arm)
        initialization.update(coverage=coverage, base_initial_state_sha256=initialization['initial_state_sha256'],
                              initial_state_sha256=state_hash(model))
    return model, initialization


@torch.no_grad()
def validation(model, scene, tiles, allowed_tiles, view=None, *, metadata_missing=False):
    model.eval()
    names = ('fused', 'graph', 'cnn')
    scores = {n: [] for n in names}
    truth = []
    options = {'metadata_missing': True} if metadata_missing else {}
    for tile, local in tiles:
        if tile['id'] not in allowed_tiles:
            raise ValueError('Validation can only access validation tiles')
        local = torch.from_numpy(local.reshape(-1))
        valid = local > 0
        truth.append(local[valid] - 1)
        heads = _forward(model, scene, tile, observation_view=view(tile) if view else None, **options)[:3]
        for name, head in zip(names, heads):
            scores[name].append(head[valid.to(head.device)].cpu())
    labels = torch.cat(truth)
    metrics, probabilities = {}, {}
    for name in names:
        p = torch.cat(scores[name])
        predicted = p.argmax(1)
        classes = scene.classes
        cm = torch.bincount(labels * classes + predicted, minlength=classes**2).reshape(classes, classes)
        recall = cm.diag().double() / cm.sum(1)
        metrics[name] = {'confusion': cm.tolist(), 'recall': recall.tolist(), 'oa': float(cm.diag().sum().double() / len(labels)),
                         'aa': float(recall.mean()), 'nll': float(-p[torch.arange(len(labels)), labels].clamp_min(1e-8).log().mean())}
        probabilities[name] = p
    independent_trainval({'train': probabilities, 'val': probabilities},
                         {'train_labels': labels, 'val_labels': labels, 'class_num': scene.classes},
                         {'train': metrics, 'val': metrics})
    return metrics, probabilities


def run_case(registration, index, out, until, code, *, coverage_arm=None, runtime=None):
    if ('full_budget_policy' in registration) != (runtime is not None):
        raise ValueError('Full budget cases require their explicit runtime contract')
    target = CPU if runtime is None else torch.device(runtime['numeric']['device'])
    audited = registration.get('parameter_audit') == updates.SCHEMA
    timing = updates.Timing(target) if audited else None
    coverage_enabled = 'coverage' in registration
    if coverage_enabled != (coverage_arm is not None) or (coverage_enabled and coverage_arm not in p4_coverage.ARMS):
        raise ValueError('Coverage registration requires an explicitly registered coverage arm')
    if coverage_enabled:
        if 'exposure' not in registration:
            raise ValueError('Matched coverage requires a guarded exposure plan')
        if (registration.get('commit') != code['commit']
                or type(index) is not int or not 0 <= index < len(registration['cases'])):
            raise ValueError('Unregistered coverage code or case')
    case = registration['cases'][index]
    if runtime is not None and (runtime['case']['index'] != index or runtime['case']['arm'] != coverage_arm
                                or any(runtime['case'][key] != case[key] for key in ('support_seed', 'k', 'method'))):
        raise ValueError('Full budget runtime case roles differ')
    pins = {Path(n): h for n, h in registration['pins'].items()}
    paths = [registration[x] for x in ('image', 'plan', 'statistics')]
    if 'development' in registration:
        paths.append(registration['development'])
    if 'guarded' in registration:
        paths.extend(registration['guarded'].values())
    if 'exposure' in registration:
        paths.extend(registration['exposure'].values())
    paths += [row['release'] for row in registration['cases']]
    source = registration['source']
    paths += [source['weight'], source['weight'] + '.json', source['result']]
    if set(map(Path, paths)) - pins.keys():
        raise ValueError('Registration has unpinned inputs')

    def guard():
        require_frozen_code(code)
        if any(hash_file(p) != h for p, h in pins.items()):
            raise ValueError('A frozen input changed')

    out = external_directory(out)
    if any(out == p.resolve() or out in p.resolve().parents for p in pins):
        raise ValueError('Output contains a frozen input')
    guard()
    plan = read(registration['plan'])
    target = registration['config']['dataset']
    development = read(registration['development']) if 'development' in registration else None
    guarded = development is not None and development.get('kind') == p4_guarded.KIND
    expected_target = p4_guarded.SCENE if guarded else ('xuzhou' if development else 'paviaU')
    if target != expected_target or guarded != ('guarded' in registration):
        raise ValueError('Target/role protocol differs')
    guarded_plan, guarded_tiles = p4_guarded.bind_inputs(registration, plan, development) if guarded else (None, None)
    role = load_release(read(case['release']), plan, case['support_seed'], case['k'], development,
                        guarded_tiles=guarded_tiles)
    steps, events, exposure = STEPS, EVENTS, None
    if 'exposure' in registration:
        if not guarded:
            raise ValueError('Exposure plan requires guarded HongHu')
        frozen, exposure = p4_exposure.bind(registration, case, role['train'])
        steps, events = frozen['steps'], frozen['validation_steps']
        lookup = {tile['id']: (tile, local) for tile, local in role['train']}
        sequence = [lookup[i] for i in exposure['tile_sequence']]
        if type(until) is not int or not 1 <= until <= steps:
            raise ValueError('Requested stop exceeds the registered exposure budget')
    else:
        if until not in (2, STEPS):
            raise ValueError('Legacy spatial stops must be 2 or 12')
        rng = np.random.default_rng(SEED)
        sequence = [role['train'][i] for i in rng.integers(len(role['train']), size=STEPS).tolist()]
    _, variable, classes, _ = DATASETS[target]
    cube = loadmat(registration['image'], variable_names=[variable])[variable]
    bands = read_band_metadata(target, cube.shape[-1])
    view = None
    coverage = None
    if coverage_enabled:
        coverage = p4_coverage.bind(registration, frozen, bands, guarded_plan['plan_hash'], coverage_arm)
        record = coverage['records']['train']
        stats = read(registration['statistics'])
        selected = subset_statistics(stats, bands, record)
        allowed = guarded_tiles
        actual = fit_role_statistics(cube, guarded_plan, allowed['train'], development['development_roles_hash'],
                                     positions=record['selected_positions'])
        if (stats.get('fit_role') != 'devtrain' or stats.get('development_roles_hash') != guarded_plan['plan_hash']
                or any(actual[key] != selected[key] for key in ('minimum', 'maximum'))):
            raise ValueError('Retained train-only statistics differ from frozen spatial preflight')
        scene = Scene(target, cube, guarded_plan, bands, classes, stats)
        view = p4_coverage.observation_provider(scene, allowed, coverage)
    elif guarded:
        allowed = guarded_tiles
        stats = fit_role_statistics(cube, guarded_plan, allowed['train'], development['development_roles_hash'])
        scene = Scene(target, cube, guarded_plan, bands, classes, stats)
        view = p4_guarded.observation_provider(scene, allowed)
    elif development:
        allowed = development_tiles(development, plan)
        stats = fit_development_statistics(cube, plan, development)
        scene = Scene(target, cube, plan, bands, classes, stats)
    else:
        allowed = {r:[t for t in plan['tiles'] if t['partition']==r] for r in ('train','val')}
        scene = Scene.prepare(target, cube, plan, bands, classes)
    train_ids = {t['id'] for t in allowed['train']}
    val_ids = {t['id'] for t in allowed['val']}
    if coverage is None and scene.stats != read(registration['statistics']):
        raise ValueError('Train-only statistics differ from frozen spatial preflight')
    config = SpatialConfig(registration['config'])
    source = registration['source'] if case['method'] == 'shared_transfer' else None
    if case['method'] not in ('scratch', 'shared_transfer'):
        raise ValueError('Unknown initialization method')
    model_seed = SEED if runtime is None else runtime['case']['model_seed']
    device = target
    model, init = initialize(config, source['weight'] if source else None, source['result'] if source else None,
                             coverage_arm=coverage_arm, model_seed=model_seed)
    if runtime is not None:
        if init != runtime['case']['initialization']:
            raise ValueError('Full budget initialization differs from the registered CPU state')
        model.to(device)
    coverage_parameters = ({n: p for n, p in model.named_parameters() if n.startswith(p4_coverage.PREFIX)}
                           if coverage is not None else {})
    forward_options = {'metadata_missing': True} if coverage is not None and coverage['metadata'] == 'missing' else {}
    optimizer = _optimizer(model, config)
    if audited:
        layout = updates.layout(model, optimizer)
        layout_sha = runtime['case']['parameter_layout_sha256']
        if digest(layout) != layout_sha:
            raise ValueError('Adam parameter layout differs from registration')
        updates.bind_names(optimizer, layout)
        recorder = updates.Recorder(model, optimizer, layout)
    fields = updates.fields(registration)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=20)
    binding = {'registration': registration, 'case': index, 'code': code['commit'], 'init': init}
    if runtime is not None:
        binding['runtime'] = runtime
    identity = digest(binding)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out/'state.pt'
    if runtime is None:
        lock = _run_lock(out)
    else:
        from transfer.p4_runtime_device import lease

        lock = lease(out / '.worker.lock')
    with lock:
        model.training_config = {'spatial_p4_identity': identity, 'plan_hash': scene.plan['plan_hash'],
                                 'statistics_hash': scene.stats['statistics_hash'], 'initialization': init}
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            if state['identity'] != identity or state.get('runtime') != runtime:
                raise ValueError('Resume identity differs')
            fingerprints = state.pop('fingerprints')
            if fingerprints != {n: _tree_fingerprint(state[n]) for n in fields}:
                raise ValueError('Resume state integrity differs')
            if audited:
                updates.check(model, optimizer, state, layout_sha)
                recorder.counts = state['parameter_audit']['counts']
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            _restore_rng(state['rng'], device)
            cursor, trace, history, best = (state[n] for n in ('cursor', 'trace', 'history', 'best'))
            if (not 0 <= cursor <= steps or [r['step'] for r in trace] != list(range(1, cursor+1))
                    or [r['tile'] for r in trace] != [tile['id'] for tile, _ in sequence[:cursor]]
                    or [r['step'] for r in history] != [i for i in events if i <= cursor]
                    or bool(history) != (best is not None)):
                raise ValueError('Resume cursor and event history differ')
        else:
            torch.manual_seed(model_seed)
            random.seed(model_seed)
            np.random.seed(model_seed)
            cursor, trace, history, best = 0, [], [], None
        start = cursor
        if cursor > until:
            raise ValueError('Requested stop precedes authoritative state')
        if timing:
            timing.startup_seconds = timing.mark() - timing.started
        for i in range(cursor, until):
            started = timing.mark() if timing else None
            tile, local = sequence[i]
            if tile['id'] not in train_ids:
                raise ValueError('Optimization can only access training tiles')
            graph_before = {n: p.detach().clone() for n, p in model.backbone.learnable_graph.named_parameters()}
            observation = view(tile) if view else None
            before = {n: p.detach().clone() for n, p in coverage_parameters.items()}
            update_start = timing.begin_update() if timing else None
            if coverage is None:
                loss = _update(model, scene, tile, local, config, optimizer, observation_view=observation)
            else:
                loss, coverage_info = _update(model, scene, tile, local, config, optimizer,
                                              observation_view=observation, return_coverage_info=True,
                                              **forward_options)
            if timing:
                update_end = timing.mark()
                memory = timing.update_memory()
            if coverage is not None:
                observed = p4_coverage.update_evidence(coverage_parameters, before, coverage_info,
                                                       observation, coverage)
            graph = dict(model.backbone.learnable_graph.named_parameters())
            changed = sum(int(torch.count_nonzero(p.detach() - graph_before[n])) for n, p in graph.items())
            grad = sum(float(p.grad.detach().abs().sum()) for p in graph.values() if p.grad is not None)
            if not math.isfinite(grad) or (not audited and (not changed or grad <= 0)):
                raise ValueError('Learned graph did not participate in optimization')
            trace.append({'step': i+1, 'tile': tile['id'], 'loss': loss, 'graph_changed_elements': changed,
                          'graph_gradient_l1': grad, 'lr': [g['lr'] for g in optimizer.param_groups]})
            if coverage is not None:
                trace[-1]['coverage'] = observed
            if audited:
                trace[-1]['parameter_flags'] = recorder.take_step()
            validation_start = timing.mark() if timing else None
            if i+1 in events:
                metrics, scores = validation(model, scene, role['val'], val_ids, view, **forward_options)
                nll = metrics['fused']['nll']
                scheduler.step(nll)
                history.append({'step': i+1, 'metrics': metrics})
                if best is None or nll < best['nll']:
                    best = {'step': i+1, 'nll': nll, 'scores': scores,
                            'model': {n: v.detach().clone() for n, v in model.state_dict().items()}}
            checkpoint_start = timing.mark() if timing else None
            state = {'identity': identity, 'cursor': i+1, 'trace': trace, 'history': history, 'best': best,
                     'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'rng': _rng_state(device)}
            if runtime is not None:
                state['runtime'] = runtime
            if audited:
                state['parameter_audit'] = recorder.snapshot()
            state['fingerprints'] = {n: _tree_fingerprint(state[n]) for n in fields}
            _atomic_torch_save(state, state_path)
            if timing:
                end = timing.mark()
                timing.steps.append({'step': i+1, 'preparation_seconds': update_start - started,
                    'train_with_hooks_seconds': update_end - update_start,
                    'trace_evidence_seconds': (validation_start if i+1 in events else checkpoint_start) - update_end,
                    'validation_seconds': checkpoint_start - validation_start if i+1 in events else 0.0,
                    'checkpoint_seconds': end - checkpoint_start, 'end_to_end_step_seconds': end - started,
                    'train_peak_memory': memory})
        guard()
        if until == steps:
            reload_start = timing.mark() if timing else None
            model.load_state_dict(best['model'], strict=True)
            metrics, scores = validation(model, scene, role['val'], val_ids, view, **forward_options)
            if (_tree_fingerprint(scores) != _tree_fingerprint(best['scores'])
                    or metrics != next(e['metrics'] for e in history if e['step'] == best['step'])):
                raise ValueError('Selected checkpoint probabilities changed after strict reload')
            report = {'identity': identity, 'initialization': init, 'updates': steps,
                      'best_step': best['step'], 'metrics': metrics, 'raw_gt_loaded': False,
                      'query_accessed': False, 'state_sha256': hash_file(state_path)}
            if exposure is not None:
                report['exposure'] = {key: exposure[key] for key in (
                    'first_all_classes_step', 'label_exposure_at_events')}
            if coverage is not None:
                report['coverage'] = {**coverage, 'arm': coverage_arm,
                                      'pair_sha256': digest({'registration': registration, 'case': index,
                                                             'base': init['base_initial_state_sha256']})}
            if runtime is not None:
                report['runtime'] = runtime
                report['coverage']['pair_sha256'] = runtime['case']['pair_sha256']
            if (out/'result.json').exists():
                if read(out/'result.json') != report:
                    raise ValueError('Completed result differs')
            else:
                write_json(out/'result.json', report)
            if timing:
                timing.reload_seconds = timing.mark() - reload_start
        result = {'updates_this_call': until-start, 'cursor': until, 'identity': identity}
        if timing:
            result['timing'] = timing.result()
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--case', type=int, required=True)
    p.add_argument('--coverage-arm', choices=p4_coverage.ARMS)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--until', type=int, required=True)
    args = p.parse_args()
    if hash_file(args.registration) != args.sha256:
        raise ValueError('Registration SHA256 differs')
    registration = read(args.registration)
    code = code_snapshot()
    if code['commit'] != registration['commit'] or not 0 <= args.case < len(registration['cases']):
        raise ValueError('Unregistered code or case')
    result = run_case(registration, args.case, args.out, args.until, code, coverage_arm=args.coverage_arm)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
