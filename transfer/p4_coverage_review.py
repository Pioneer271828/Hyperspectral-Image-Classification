"""Audit frozen HongHu coverage artifacts and replay released development validation."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat
import torch

from evaluation.classification_metrics import metrics_from_confusion
from joint_training import _tree_fingerprint
from loadData.data_read import read_band_metadata
from loadData.dataset_registry import DATASETS
from loadData.observation_sampling import subset_statistics
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from research_runtime import code_snapshot, hash_file, require_frozen_code, tensor_hash, write_json
from train_spatial import SpatialConfig
from transfer import p4_coverage as coverage, p4_coverage_batch as batch, p4_guarded as guarded
from transfer.g2_contract import read_pinned
from transfer.g3_training import Scene, _forward, state_hash
from transfer.p4_spatial import initialize, load_release


HEADS = ('fused', 'graph', 'cnn')
PERTURBATIONS = ('missing', 'shift10nm')
CLASS_IDS = list(range(1, 23))


def execution_pins(scope, execution, root, scope_sha):
    if (execution['status'] != 'completed' or execution['scope_sha256'] != scope_sha
            or execution['commit'] != scope['commit'] or execution['raw_gt_loaded'] is not False
            or execution['query_accessed'] is not False or execution['optimizer_updates'] != scope['optimizer_updates']
            or [row['case'] for row in execution['rows']] != scope['cases']):
        raise ValueError('Incomplete or mismatched 40-case execution')
    files = {f'{case["id"]}/{part}/{name}' for case in scope['cases']
             for part in ('full', 'split') for name in ('state.pt', 'result.json')}
    files |= {f'{case["id"]}/{phase}.{suffix}' for case in scope['cases']
              for phase, *_ in batch.phases(scope) for suffix in ('log', 'json')}
    if set(execution['artifacts']) != files or (root / 'failure.json').exists():
        raise ValueError('Incomplete or unexpected batch artifact inventory')
    return {str(root / relative): sha for relative, sha in execution['artifacts'].items()}


def read_case(scope, row, root, pins):
    case = row['case']
    folder = root / case['id']
    states, results = [], []
    for part in ('full', 'split'):
        path = folder / part
        state = torch.load(path / 'state.pt', map_location='cpu', weights_only=True)
        result_path = path / 'result.json'
        result = read_pinned(result_path, pins[str(result_path)])
        fingerprints = {k: _tree_fingerprint(state[k]) for k in batch.FIELDS}
        if (state['fingerprints'] != fingerprints or state['cursor'] != scope['steps']
                or result['updates'] != scope['steps'] or result['state_sha256'] != pins[str(path / 'state.pt')]
                or result['identity'] != state['identity'] or result['coverage']['arm'] != case['arm']
                or result['raw_gt_loaded'] is not False or result['query_accessed'] is not False):
            raise ValueError('Case arm, budget or artifact fingerprint differs')
        states.append(state)
        results.append(result)
    if (states[0]['fingerprints'] != states[1]['fingerprints']
            or {k: v for k, v in results[0].items() if k != 'state_sha256'}
            != {k: v for k, v in results[1].items() if k != 'state_sha256'}):
        raise ValueError('Full and split states or selected results differ')
    if len(row['calls']) != 4 or row['updates'] != 2 * scope['steps']:
        raise ValueError('Case actual update accounting differs')
    for call, (phase, part, cursor, updates) in zip(row['calls'], batch.phases(scope)):
        event_path = folder / f'{phase}.json'
        log = folder / f'{phase}.log'
        expected = {'cursor': cursor, 'updates_this_call': updates, 'identity': results[0]['identity']}
        if (call != read_pinned(event_path, pins[str(event_path)]) or call['phase'] != phase
                or call['argv'] != batch.invocation(scope, case, root, part, cursor)
                or call['exit_code'] != 0 or call['result'] != expected or call['log_sha256'] != pins[str(log)]
                or json.loads(log.read_text(encoding='utf-8').splitlines()[-1]) != expected):
            raise ValueError('Case CLI or continuation receipt differs')
        if phase == 'first':
            if call['partial'] != {'cursor': cursor, 'validation_events': 0, 'has_selected_state': False}:
                raise ValueError('Split path crossed first validation')
        elif call['state_sha256'] != pins[str(folder / part / 'state.pt')]:
            raise ValueError('CLI state hash differs from final artifact')
    return states[0], results[0]


def check_exposure(state, result, release, cohort, plan):
    counts = {tile['id']: np.bincount(labels.ravel(), minlength=23)[1:]
              for tile, labels in release['train']}
    witness = cohort['witness_tiles']
    if not set(witness) <= counts.keys() or len(witness) != len(set(witness)):
        raise ValueError('Exposure witness accesses unreleased supervision')
    rng = np.random.default_rng(plan['sampler_seed'])
    sequence = rng.permutation(sorted(witness)).tolist()
    sequence += rng.choice(sorted(counts), size=plan['steps'] - len(sequence), replace=True).tolist()
    events = plan['validation_steps']
    cumulative = np.cumsum([counts[tile] for tile in sequence], axis=0)
    full = np.flatnonzero((cumulative > 0).all(axis=1))
    if not full.size or full[0] + 1 > events[0]:
        raise ValueError('Incomplete class exposure before validation')
    evidence = {'first_all_classes_step': int(full[0]) + 1,
                'label_exposure_at_events': {str(i): cumulative[i - 1].tolist() for i in events}}
    if (cohort['tile_sequence'] != sequence or [row['tile'] for row in state['trace']] != sequence
            or [row['step'] for row in state['trace']] != list(range(1, plan['steps'] + 1))
            or any(cohort[k] != v for k, v in evidence.items()) or result['exposure'] != evidence):
        raise ValueError('Recorded sequence or class exposure differs')
    return evidence


def chosen_event(state, result, steps):
    history = state['history']
    if [event['step'] for event in history] != steps or any(
            not math.isfinite(e['metrics']['fused']['nll']) for e in history):
        raise ValueError('Validation event sequence or NLL differs')
    chosen = min(history, key=lambda e: e['metrics']['fused']['nll'])
    if (state['best']['step'] != chosen['step'] or result['best_step'] != chosen['step']
            or state['best']['nll'] != chosen['metrics']['fused']['nll'] or result['metrics'] != chosen['metrics']):
        raise ValueError('Selected state violates first minimum fused validation NLL')
    return chosen


def metrics(scores, labels, saved=None):
    """Recompute from released labels and replayed probabilities via NumPy."""
    result = {}
    for name, probabilities in scores.items():
        p = probabilities.numpy()
        if (p.shape != (len(labels), 22) or not np.isfinite(p).all() or (p < 0).any()
                or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)):
            raise ValueError('Invalid replayed validation probabilities')
        cm = np.zeros((22, 22), dtype=np.int64)
        np.add.at(cm, (labels - 1, p.argmax(1)), 1)
        if cm.sum(1).tolist() != [5] * 22:
            raise ValueError('Released validation support differs')
        m = metrics_from_confusion(cm, CLASS_IDS)
        recall = [entry['recall'] for entry in m['per_class']]
        nll = float(-np.log(np.maximum(p[np.arange(len(p)), labels - 1].astype(np.float64), 1e-8)).mean())
        if saved is not None:
            v = saved[name]
            if (cm.tolist() != v['confusion'] or recall != v['recall'] or abs(m['oa'] - v['oa']) > 1e-12
                    or abs(m['aa'] - v['aa']) > 1e-12 or abs(nll - v['nll']) > 1e-6):
                raise ValueError('Independent metrics differ from the producer')
        result[name] = {'oa': m['oa'], 'aa': m['aa'], 'kappa': m['kappa'], 'macro_f1': m['macro_f1'],
                        'nll': nll, 'recall': recall, 'confusion': cm.tolist(),
                        'zero_recall_classes': [c for c, v in zip(CLASS_IDS, recall) if v == 0]}
    return result


def inputs(reg, plan, exposure):
    geometry = read_pinned(Path(reg['development']), reg['pins'][reg['development']])
    scene_plan, allowed = guarded.bind_inputs(reg, plan, geometry)
    _, variable, classes, _ = DATASETS[guarded.SCENE]
    cube = loadmat(reg['image'], variable_names=[variable])[variable]
    bands = read_band_metadata(guarded.SCENE, cube.shape[-1])
    frozen = coverage.bind(reg, exposure, bands, scene_plan['plan_hash'], 'none')
    positions = [i for i, band_id in enumerate(bands.band_ids) if band_id % 2 == 0 and bands.valid[i]]
    if any(r['selected_positions'] != positions for r in frozen['records'].values()):
        raise ValueError('Actual retained columns differ from the registered stride')
    stats = read_pinned(Path(reg['statistics']), reg['pins'][reg['statistics']])
    retained = subset_statistics(stats, bands, frozen['records']['train'])
    minima, maxima = [], []
    for tile in allowed['train']:
        values = np.asarray(cube[tile_slice(tile)][:, :, positions], dtype=np.float64)
        minima.append(values.min(axis=(0, 1)))
        maxima.append(values.max(axis=(0, 1)))
    if np.min(minima, axis=0).tolist() != retained['minimum'] or np.max(maxima, axis=0).tolist() != retained['maximum']:
        raise ValueError('Independent retained train statistics differ')
    scene = Scene(guarded.SCENE, cube, scene_plan, bands, classes, stats)
    return geometry, allowed, scene, frozen


def validation_views(scene, allowed, frozen, release):
    """Keep each authorized tile independent and retain exactly its labeled rows."""
    permission = {tile['id']: tile for tile in allowed['val']}
    views, labels, coordinates = [], [], []
    record = frozen['records']['val']
    for tile, local in release['val']:
        if permission.get(tile['id']) != tile:
            raise ValueError('Validation contains an unauthorized tile')
        rows = np.flatnonzero(local.ravel())
        nominal = guarded.retained_view(scene, tile, {'sampling': record, 'metadata': 'nominal'})
        if nominal.bands.fingerprint != frozen['forward_metadata_sha256']:
            raise ValueError('Nominal validation metadata differs')
        shifted = guarded.model_metadata(nominal.bands, 'shift10nm')
        views.append((tile, rows, nominal, type(nominal)(nominal.observation, shifted)))
        coordinates.extend(nominal.observation.global_ids.ravel()[rows].tolist())
        labels.extend(local.ravel()[rows].tolist())
    if len(set(coordinates)) != 110 or np.bincount(labels, minlength=23).tolist() != [0] + [5] * 22:
        raise ValueError('Validation coordinates or class budget differs')
    return views, np.asarray(labels), digest({'coordinates': coordinates, 'labels': labels})


def released_roles(reg, plan, geometry, allowed, case):
    path = reg['cases'][case['index']]['release']
    released = read_pinned(Path(path), reg['pins'][path])
    return load_release(released, plan, case['support_seed'], case['k'], geometry, guarded_tiles=allowed)


@torch.no_grad()
def replay(model, scene, views, mode, arm):
    scores = {head: [] for head in HEADS}
    audit = []
    model.eval()
    for tile, rows, nominal, shifted in views:
        view = shifted if mode == 'shift10nm' else nominal
        outputs = _forward(model, scene, tile, observation_view=view, metadata_missing=(mode == 'missing'))
        for name, probability in zip(HEADS, outputs[:3]):
            scores[name].append(probability[rows].cpu())
        metadata = None if mode == 'missing' else view.bands.fingerprint
        info = outputs[3].get('coverage')
        if arm == 'none':
            if info is not None:
                raise ValueError('Coverage-off model invoked the modulator')
        elif (info is None or info['metadata_hash'] != metadata
                or (mode == 'missing' and (info['usable_anchors'] or info['fallback_reason'] != 'missing_metadata'))):
            raise ValueError('Replayed perturbation metadata or fallback differs')
        audit.append({'tile': tile['id'], 'bands': view.observation.spectral.shape[-1], 'metadata_sha256': metadata,
                      'fallback': None if info is None else info['fallback_reason'],
                      'usable_anchors': None if info is None else info['usable_anchors']})
    return {name: torch.cat(rows) for name, rows in scores.items()}, audit


def participation(trace, initial, record):
    params = initial['coverage']['parameters']
    rows = [event['coverage'] for event in trace]
    expected = (record['record_hash'], len(record['selected_positions']), record['selected_metadata_hash'])
    if any((r['sampling_sha256'], r['bands'], r['metadata_sha256']) != expected for r in rows):
        raise ValueError('Actual retained training band identity differs')
    if not params:
        if any(set(row) != {'sampling_sha256', 'bands', 'metadata_sha256'} for row in rows):
            raise ValueError('None arm contains active coverage evidence')
        return {'parameter_count': 0, 'parameters': {}, 'usable_anchor_counts': [], 'fallbacks': []}
    if len(params) != 4 or sum(params.values()) != 194 or any(row['parameters'].keys() != params.keys() for row in rows):
        raise ValueError('Coverage parameter identities differ')
    evidence = {}
    for key, count in params.items():
        observations = [row['parameters'][key] for row in rows]
        gradients = [o['gradient_l1'] for o in observations if o['gradient_l1'] is not None]
        changed = [o['changed_elements'] for o in observations]
        if (any(not math.isfinite(g) or g < 0 for g in gradients)
                or any(type(v) is not int or not 0 <= v <= count for v in changed)):
            raise ValueError('Invalid recorded parameter participation')
        evidence[key] = {'max_gradient_l1': max(gradients, default=0), 'max_changed_elements': max(changed),
                         'steps_with_change': sum(v > 0 for v in changed)}
    return {'parameter_count': 194, 'parameters': evidence,
            'usable_anchor_counts': [r['usable_anchors'] for r in rows], 'fallbacks': [r['fallback_reason'] for r in rows],
            'nonparticipating_tensors': [key for key, v in evidence.items()
                                         if not v['max_gradient_l1'] or not v['max_changed_elements']]}


def paired(rows):
    grouped = {}
    for row in rows:
        case = row['case']
        key = case['support_seed'], case['k'], case['method']
        grouped.setdefault(key, {})[case['arm']] = row
    for group in grouped.values():
        if set(group) != set(coverage.ARMS):
            raise ValueError('Missing matched arm in review')
        baseline = group['none']
        for row in group.values():
            for field in ('pair_sha256', 'common_initialization', 'exposure', 'validation_population_sha256'):
                if baseline[field] != row[field]:
                    raise ValueError('Matched arm pairing differs: ' + field)
            row['versus_none'] = {}
            for head in HEADS:
                a, b = row['heads'][head], baseline['heads'][head]
                row['versus_none'][head] = {**{n: a[n] - b[n] for n in ('oa', 'aa', 'kappa', 'macro_f1', 'nll')},
                    'recall': [x - y for x, y in zip(a['recall'], b['recall'])],
                    'declines': [c for c, x, y in zip(CLASS_IDS, a['recall'], b['recall']) if x < y]}


def review(scope_path, scope_sha, execution_path, execution_sha, out, diagnostics, code):
    tick = perf_counter()
    scope, reg = batch.load_scope(scope_path, scope_sha, code)
    execution_path = Path(execution_path).resolve()
    execution = read_pinned(execution_path, execution_sha)
    root = execution_path.parent
    out, diagnostics = [batch.sibling(path, scope_path) for path in (out, diagnostics)]
    if out == diagnostics or out.exists() or diagnostics.exists():
        raise ValueError('Review and diagnostics require distinct new outputs')
    artifacts = execution_pins(scope, execution, root, scope_sha)
    pins = {**scope['protected_files'], **artifacts, str(scope_path): scope_sha,
            scope['registration']: scope['registration_sha256'], str(execution_path): execution_sha}
    batch.check_pins(pins)
    plan, exposure = [read_pinned(Path(path), reg['pins'][path])
                      for path in (reg['plan'], reg['exposure']['plan'])]
    geometry, allowed, scene, frozen = inputs(reg, plan, exposure)
    cohorts = {(row['support_seed'], row['k']): row for row in exposure['cohorts']}
    releases, populations = {}, {}
    results, perturbations = [], []
    config = SpatialConfig(reg['config'])
    for row in execution['rows']:
        case = row['case']
        index, arm = case['index'], case['arm']
        state, result = read_case(scope, row, root, artifacts)
        release_path = reg['cases'][index]['release']
        if release_path not in releases:
            releases[release_path] = released_roles(reg, plan, geometry, allowed, case)
            populations[release_path] = validation_views(scene, allowed, frozen, releases[release_path])
        cohort = cohorts[case['support_seed'], case['k']]
        counts = check_exposure(state, result, releases[release_path], cohort, exposure)
        event = chosen_event(state, result, scope['validation_steps'])
        source = reg['source'] if case['method'] == 'shared_transfer' else None
        source_args = (source['weight'], source['result']) if source else ()
        model, initialization = initialize(config, *source_args, coverage_arm=arm)
        identity = digest({'registration': reg, 'case': index, 'code': code['commit'], 'init': initialization})
        pair_id = digest({'registration': reg, 'case': index, 'base': initialization['base_initial_state_sha256']})
        if (initialization != result['initialization'] or result['identity'] != identity
                or result['coverage'] != {**frozen, 'arm': arm, 'pair_sha256': pair_id}):
            raise ValueError('Case initialization, arm or registration identity differs')
        model.load_state_dict(state['best']['model'], strict=True)
        before = state_hash(model)
        views, labels, population = populations[release_path]
        scores, _ = replay(model, scene, views, 'nominal', arm)
        if any(not torch.equal(scores[name], state['best']['scores'][name]) for name in HEADS):
            raise ValueError('Nominal selected probabilities differ after reload')
        heads = metrics(scores, labels, event['metrics'])
        diagnostic = {}
        for mode in PERTURBATIONS:
            probs, audit = replay(model, scene, views, mode, arm)
            readout = metrics(probs, labels)
            diagnostic[mode] = {'heads': readout, 'tile_metadata': audit,
                'probability_sha256': {name: tensor_hash(p) for name, p in probs.items()},
                'versus_nominal_oa': {name: readout[name]['oa'] - heads[name]['oa'] for name in HEADS}}
        if state_hash(model) != before:
            raise ValueError('Validation replay mutated selected state')
        perturbations.append({'case': case, 'validation_population_sha256': population,
                              'selected_model_sha256': before, 'conditions': diagnostic})
        trace = state['trace']
        losses = [r['loss']['total'] for r in trace]
        results.append({'case': case, 'pair_sha256': pair_id, 'validation_population_sha256': population,
            'common_initialization': {k: v for k, v in initialization.items() if k not in ('coverage', 'initial_state_sha256')},
            'exposure': counts, 'heads': heads, 'selected_step': event['step'], 'selected_model_sha256': before,
            'validation_history': [{'step': e['step'], 'fused_nll': e['metrics']['fused']['nll']}
                                   for e in state['history']],
            'loss_first_two_mean': float(np.mean(losses[:2])), 'loss_last_two_mean': float(np.mean(losses[-2:])),
            'loss_by_step': losses, 'phase_wall_seconds': {e['phase']: e['wall_seconds'] for e in row['calls']},
            'training_coverage': participation(trace, initialization, frozen['records']['train'])})
    if len({row['validation_population_sha256'] for row in results}) != 1:
        raise ValueError('Cases did not share the same fixed validation coordinates and labels')
    paired(results)
    batch.check_pins(pins)
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    diagnostics.mkdir(exist_ok=False)
    diag_path = diagnostics / 'metadata.json'
    write_json(diag_path, {'status': 'passed', 'commit': code['commit'], 'scope_sha256': scope_sha,
        'selected_models': len(perturbations), 'validation_heads': len(perturbations) * 6,
        'updates': 0, 'query_accessed': False, 'retained_band_ids': frozen['records']['val']['selected_band_ids'],
        'rows': perturbations})
    path = out / 'receipt.json'
    write_json(path, {'status': 'passed', 'commit': code['commit'], 'scope_sha256': scope_sha,
        'execution_sha256': execution_sha, 'metadata_diagnostics_sha256': hash_file(diag_path),
        'cases': len(results), 'exact_full_split_pairs': len(results), 'completed_resume_updates': 0,
        'validation_heads': len(results) * 3, 'class_recalls': len(results) * 3 * 22,
        'updates': 0, 'query_accessed': False, 'raw_gt_loaded': False,
        'retained_band_ids': frozen['records']['val']['selected_band_ids'], 'protected_files': pins,
        'wall_seconds': perf_counter() - tick, 'rows': results})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scope', 'execution'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        raise ValueError('Unexpected review commit')
    path = review(args.scope, args.scope_sha256, args.execution, args.execution_sha256,
                  args.out, args.diagnostics, code)
    print(json.dumps({'path': str(path), 'sha256': hash_file(path)}), flush=True)


if __name__ == '__main__':
    main()
