"""Read-only fixed-label fitting and retained-band replay of accepted P4 states."""
import argparse
from pathlib import Path
import re
import subprocess
from time import perf_counter

import numpy as np
from scipy.io import loadmat
import torch

from joint_training import _rng_state, _tree_fingerprint
from loadData.data_read import read_band_metadata
from loadData.dataset_registry import DATASETS
from loadData.observation_sampling import replay, sample
from loadData.spatial_protocol import PartitionObservation
from model.research_config import digest
from research_runtime import ROOT, code_snapshot, hash_file, require_frozen_code, tensor_hash, write_json
from train_spatial import SpatialConfig
from transfer import p4_coverage as coverage, p4_coverage_batch as batch
from transfer import p4_coverage_review as prior, p4_guarded as guarded
from transfer.g2_contract import read_pinned
from transfer.g3_training import Scene, _forward, state_hash
from transfer.p4_frozen_band_probe import selected_view
from transfer.p4_spatial import initialize


POLICY = 'wp05_frozen_fit_observations_v1'
PRODUCER = '2ce582ee48b7456f88a9818cb8764af4f322fe24'
PARENT_PINS = {
    'scope': '846170ccb95d78b5831fce388ef62810c7dc3b5e1b993e19f0d54f9fb6ee2993',
    'execution': '17b85ddf515f10f768cd5b02128b6176efd63d438005f404b409813cf0f29b5e',
    'audit': 'be33181c6e6ea69c243558e6a7320ebed36de81e5461fb5a1b51a4affc66679a',
    'metadata': 'f3eae04d52c48fcb773d56bbc1ea6b1aab4359e98b5b206b9e9313f8e5bd79ff',
}
HEADS = prior.HEADS
STAGES = ('initial', 'selected', 'last')
REFERENCE = 'training_stride'
RANDOM_SEED = 3032
CPU = torch.device('cpu')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def runtime_blobs(producer, consumer):
    """Pin every producer Python/config blob, including transitive runtime code."""
    require(all(re.fullmatch('[0-9a-f]{40}', sha) for sha in (producer, consumer)), 'Full Git identities required')
    def files(commit):
        result = {}
        raw = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '-z', commit])
        for item in raw.decode('utf-8').split('\0'):
            if item:
                entry, name = item.split('\t', 1)
                if name.endswith('.py') or (name.startswith('configs/') and name.endswith('.json')):
                    result[name] = entry.split()[2]
        return result

    old, current = files(producer), files(consumer)
    require(bool(old), 'Producer runtime inventory is empty')
    differences = sorted(path for path in old if current.get(path) != old[path])
    require(not differences, 'Producer runtime blobs differ: ' + ', '.join(differences))
    return old


def parent_documents(links, code, *, synthetic=False):
    require_frozen_code(code)
    require(set(links) == set(PARENT_PINS), 'Unexpected parent document roles')
    if not synthetic:
        require({key: row['sha256'] for key, row in links.items()} == PARENT_PINS,
                'Unaccepted real producer SHA256')
    docs = {key: read_pinned(Path(row['path']), row['sha256']) for key, row in links.items()}
    scope, execution, audit, metadata = (docs[key] for key in PARENT_PINS)
    prod = scope['commit']
    require(type(synthetic) is bool and scope['synthetic'] is synthetic
            and (synthetic or (prod == PRODUCER and prod != code['commit'])),
            'Producer and consumer identities differ from the registered contract')
    require(scope['scope_hash'] == digest({k: v for k, v in scope.items() if k != 'scope_hash'}),
            'Producer scope identity differs')
    root = Path(links['execution']['path']).parent
    artifacts = prior.execution_pins(scope, execution, root, links['scope']['sha256'])
    reg_path = scope['registration']
    reg = read_pinned(Path(reg_path), scope['registration_sha256'])
    require(all(scope[k] == v for k, v in batch.manifest(reg, synthetic=synthetic).items()),
            'Producer case matrix differs from the original registration')
    pins = {**scope['protected_files'], **artifacts, reg_path: scope['registration_sha256'],
            links['scope']['path']: links['scope']['sha256'], links['execution']['path']: links['execution']['sha256']}
    require(audit['protected_files'] == pins and audit['status'] == 'passed'
            and audit['commit'] == metadata['commit'] == prod and metadata['status'] == 'passed'
            and audit['scope_sha256'] == metadata['scope_sha256'] == links['scope']['sha256']
            and audit['execution_sha256'] == links['execution']['sha256']
            and audit['metadata_diagnostics_sha256'] == links['metadata']['sha256']
            and audit['cases'] == audit['exact_full_split_pairs'] == metadata['selected_models'] == 40
            and audit['validation_heads'] == 120 and audit['class_recalls'] == 2640
            and metadata['validation_heads'] == 240 and audit['completed_resume_updates'] == 0
            and audit['updates'] == metadata['updates'] == 0 and audit['raw_gt_loaded'] is False
            and audit['query_accessed'] is metadata['query_accessed'] is False,
            'Accepted audit or metadata diagnostics do not bind the producer')
    require([row['case'] for row in audit['rows']] == [row['case'] for row in metadata['rows']] == scope['cases'],
            'Accepted audit does not cover the full ordered case matrix')
    pins.update({links[key]['path']: links[key]['sha256'] for key in ('audit', 'metadata')})
    batch.check_pins(pins)
    return scope, reg, audit, metadata, root, pins


def fixed_population(tiles, allowed, role, width, budget):
    require(role in ('train', 'val'), 'Only released train/devval roles are permitted')
    permitted = {t['id']: t for t in allowed[role]}
    layout, labels, coordinates = [], [], []
    for tile, local in tiles:
        require(permitted.get(tile['id']) == tile and tile['partition'] == role,
                'Fixed labels escape the authorized role')
        y0, y1, x0, x1 = tile['box']
        require(local.shape == (y1 - y0, x1 - x0) and np.all((0 <= local) & (local <= 22)),
                'Invalid released tile labels')
        iy, ix = np.mgrid[y0:y1, x0:x1]
        rows = np.flatnonzero(local.ravel())
        labels.extend(local.ravel()[rows].tolist())
        coordinates.extend((iy * width + ix).ravel()[rows].tolist())
        layout.append((tile, rows))
    require(len(set(coordinates)) == 22 * budget and np.bincount(labels, minlength=23).tolist() == [0] + [budget] * 22,
            'Fixed label budget or coordinate uniqueness differs')
    return layout, np.asarray(labels), {'sha256': digest({'coordinates': coordinates, 'labels': labels}),
        'labels': len(labels), 'per_class': budget, 'tiles': len(layout)}


def stub(bands, plan_hash, role='val'):
    return PartitionObservation(role, np.zeros((1, 1, len(bands.band_ids)), dtype=np.float32),
                                np.zeros((1, 1, 2), dtype=np.float32),
                                np.zeros((1, 1), dtype=np.int64), plan_hash)


def freeze_conditions(bands, plan_hash, reference):
    observation = stub(bands, plan_hash)
    view = replay(observation, bands, reference)
    train_ids = set(view.bands.band_ids)
    valid = [(b, w) for b, w, use in zip(bands.band_ids, bands.wavelengths, bands.valid) if use]
    low, high = min(w for _, w in valid), max(w for _, w in valid)
    middle = low + (high - low) / 2
    physical = bands.wavelength_unit == 'nm' and bands.wavelength_quality != 'index_only'
    choices = {
        REFERENCE: (None, None, True),
        'odd_stride': ({'mode': 'uniform_stride', 'stride': 2, 'offset': 1}, None, any(b % 2 for b, _ in valid)),
        'stride4': ({'mode': 'uniform_stride', 'stride': 4, 'offset': 0}, None, any(b % 4 == 0 for b, _ in valid)),
        'full': ({'mode': 'none'}, None, True),
        'low_half': ({'mode': 'range_crop', 'lower': low, 'upper': middle, 'unit': 'nm'}, None, physical),
        'high_half': ({'mode': 'range_crop', 'lower': float(np.nextafter(middle, np.inf)),
                       'upper': high, 'unit': 'nm'}, None, physical and high > middle),
        'random50': ({'mode': 'random_subset', 'keep_fraction': .5}, RANDOM_SEED, True),
    }
    conditions, seen = {}, {}
    for name, (config, seed, available) in choices.items():
        if not available:
            conditions[name] = {'status': 'unavailable', 'reason': 'no eligible retained bands or physical centers',
                                'configuration': config, 'seed': seed}
            continue
        if name == REFERENCE:
            record = reference
        else:
            _, record = sample(observation, bands, config, np.random.default_rng(seed) if seed is not None else None)
        ids = set(record['selected_band_ids'])
        key = tuple(record['selected_positions'])
        conditions[name] = {'status': 'ready', 'sampling': record, 'metadata': 'nominal', 'seed': seed,
            'same_bands_as': seen.get(key), 'band_count': len(ids), 'train_intersection': len(ids & train_ids),
            'train_union': len(ids | train_ids), 'added': len(ids - train_ids), 'removed': len(train_ids - ids)}
        seen.setdefault(key, name)
    return conditions


def model_tensor_hash(tensors):
    return digest({key: tensor_hash(value) for key, value in tensors.items()})


def load_state(scope, case, root, pins, accepted, metadata):
    folder = root / case['id'] / 'full'
    path = folder / 'state.pt'
    report_path = folder / 'result.json'
    result = read_pinned(report_path, pins[str(report_path)])
    require(hash_file(path) == pins[str(path)] == result['state_sha256'], 'Selected or last state file changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(state['cursor'] == result['updates'] == scope['steps'] and result['identity'] == state['identity']
            and result['query_accessed'] is result['raw_gt_loaded'] is False
            and result['coverage']['arm'] == case['arm'] and result['coverage']['pair_sha256'] == accepted['pair_sha256']
            and result['exposure'] == accepted['exposure'],
            'Case role, selected state, or update budget differs')
    require(state['fingerprints'] == {key: _tree_fingerprint(state[key]) for key in batch.FIELDS},
            'Producer state fingerprint differs')
    choice = prior.chosen_event(state, result, scope['validation_steps'])
    selected, last = (model_tensor_hash(d) for d in (state['best']['model'], state['model']))
    require(choice['step'] == accepted['selected_step'] and
            selected == accepted['selected_model_sha256'] == metadata['selected_model_sha256'],
            'Case selected model differs from the accepted audit')
    require(choice['step'] != scope['steps'] or selected == last, 'Selected terminal state differs from last')
    manifest = {'path': str(path), 'state_sha256': pins[str(path)], 'result_sha256': pins[str(report_path)],
                'identity': state['identity'], 'selected_step': choice['step'],
                'initial': result['initialization']['initial_state_sha256'], 'selected': selected, 'last': last}
    return manifest, state, result


def context(links, code, synthetic):
    source, reg, accepted, metadata, root, pins = parent_documents(links, code, synthetic=synthetic)
    blobs = runtime_blobs(source['commit'], code['commit'])
    plan = read_pinned(Path(reg['plan']), pins[reg['plan']])
    geometry = read_pinned(Path(reg['development']), pins[reg['development']])
    stats = read_pinned(Path(reg['statistics']), pins[reg['statistics']])
    scene_plan, allowed = guarded.bind_inputs(reg, plan, geometry)
    bands = read_band_metadata(guarded.SCENE, len(stats['minimum']))
    train, val = (reg['coverage']['records'][role] for role in ('train', 'val'))
    train_bands = replay(stub(bands, scene_plan['plan_hash'], 'train'), bands, train).bands
    conditions = freeze_conditions(bands, scene_plan['plan_hash'], val)
    require(list(train_bands.band_ids) == val['selected_band_ids'] == accepted['retained_band_ids']
            == metadata['retained_band_ids'], 'Original training/validation band identities differ')
    cases, populations, releases = [], {}, {}
    for case, old, perturb in zip(source['cases'], accepted['rows'], metadata['rows']):
        release_path = reg['cases'][case['index']]['release']
        if release_path not in populations:
            release = prior.released_roles(reg, plan, geometry, allowed, case)
            populations[release_path] = {
                role: fixed_population(release[role], allowed, role, scene_plan['shape'][1], k)
                for role, k in (('train', case['k']), ('val', 5))}
            releases[release_path] = release
        data = populations[release_path]
        require(data['val'][2]['sha256'] == old['validation_population_sha256']
                == perturb['validation_population_sha256'], 'Fixed development validation population differs')
        manifest, state, result = load_state(source, case, root, pins, old, perturb)
        exposures = {row['tile'] for row in state['trace']}
        unique = np.zeros(22, dtype=np.int64)
        for tile, local in releases[release_path]['train']:
            if tile['id'] in exposures:
                unique += np.bincount(local.ravel(), minlength=23)[1:]
        cases.append({'case': case, 'state': manifest, 'release': release_path,
                      'pair_sha256': old['pair_sha256'], 'exposure': result['exposure'],
                      'unique_exposed_train_labels_by_class': unique.tolist(),
                      'populations': {role: value[2] for role, value in data.items()}})
    require(len({row['populations']['val']['sha256'] for row in cases}) == 1,
            'Cases do not share the same released development validation')
    frozen = {'policy': POLICY, 'synthetic': synthetic, 'producer_commit': source['commit'],
              'consumer_commit': code['commit'], 'consumer_source_hash': code['source_hash'],
              'parents': links, 'producer_runtime_blobs': blobs, 'protected_files': pins,
              'metadata_reused_sha256': links['metadata']['sha256'], 'conditions': conditions, 'cases': cases,
              'stages': list(STAGES), 'fit_population_roles': ['train', 'val'],
              'metric_policy': 'three_head_fixed_rows_float64_nll_entropy_margin_and_classification_v1',
              'nominal_observation_sha256': val['record_hash'], 'raw_gt_loaded': False, 'query_accessed': False}
    frozen['scope_hash'] = digest(frozen)
    return frozen, reg, scene_plan, allowed, bands, stats, populations, source, accepted, metadata, root


def prepare(links, out, code, *, synthetic=False):
    frozen, *_ = context(links, code, synthetic)
    out = batch.sibling(out, links['scope']['path'])
    batch.check_pins(frozen['protected_files'])
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    path = out / 'scope.json'
    write_json(path, frozen)
    return path


def load_context(path, sha, code):
    path = Path(path).resolve()
    frozen = read_pinned(path, sha)
    require(frozen['consumer_commit'] == code['commit'], 'Consumer checkout SHA differs')
    loaded = context(frozen['parents'], code, frozen['synthetic'])
    require(loaded[0] == frozen, 'Frozen diagnostic case, label, condition or seed differs')
    return loaded


def fixed_views(scene, allowed, layout, condition, role):
    require(role in ('train', 'val'), 'Only released train/devval observations are permitted')
    permitted = {tile['id']: tile for tile in allowed[role]}
    require(all(permitted.get(tile['id']) == tile for tile, _ in layout), 'Observation role access denied')
    return [(tile, rows, selected_view(scene, tile, permitted, condition) if role == 'val'
             else guarded.retained_view(scene, tile, condition)) for tile, rows in layout]


def capture(model, scene, tile, view, arm):
    graph = model.backbone.learnable_graph
    require(not model.training and graph is not None and 'build_graph' not in graph.__dict__,
            'A private eval model with its learned graph is required')
    original = graph.build_graph
    capture_rows = []

    def observed(*args, **kwargs):
        state = original(*args, **kwargs)
        capture_rows.append({'q_sha256': tensor_hash(state.Q), 'a_sha256': tensor_hash(state.A0),
                             'q_nnz': state.Q._nnz(), 'a_nnz': state.A0._nnz()})
        return state

    graph.build_graph = observed
    try:
        outputs = _forward(model, scene, tile, observation_view=view)
    finally:
        del graph.build_graph
    require(len(capture_rows) == 1, 'Expected one fresh Q/A graph per tile')
    info = outputs[3].get('coverage')
    require((info is None) == (arm == 'none') and (info is None or info['metadata_hash'] == view.bands.fingerprint),
            'Actual model coverage identity differs')
    return outputs[:3], {'tile': tile['id'], **capture_rows[0], 'bands': view.observation.spectral.shape[-1],
        'metadata_sha256': view.bands.fingerprint, 'coverage_active': info is not None,
        'usable_anchors': None if info is None else info['usable_anchors'],
        'fallback_reason': None if info is None else info['fallback_reason']}


@torch.no_grad()
def forward_rows(model, scene, views, arm):
    scores = {head: [] for head in HEADS}
    graphs = []
    for tile, rows, view in views:
        outputs, graph = capture(model, scene, tile, view, arm)
        graphs.append(graph)
        for head, probabilities in zip(HEADS, outputs):
            scores[head].append(probabilities[rows].cpu())
    return {key: torch.cat(value) for key, value in scores.items()}, graphs


def score(values, labels):
    p = values.double()
    target = torch.as_tensor(labels, dtype=torch.long) - 1
    require(tuple(p.shape) == (len(target), 22) and bool(torch.isfinite(p).all() and (p >= 0).all())
            and torch.allclose(p.sum(1), torch.ones(len(target), dtype=torch.float64), atol=1e-6, rtol=0),
            'Invalid fixed-row probabilities')
    predicted = p.argmax(1)
    cm = torch.bincount(target * 22 + predicted, minlength=484).reshape(22, 22)
    correct, actual, guessed = cm.diag().double(), cm.sum(1).double(), cm.sum(0).double()
    require(bool((actual > 0).all()), 'Class support differs')
    recall = correct / actual
    oa = float(correct.sum() / len(target))
    chance = float(actual @ guessed / len(target) ** 2)
    true_p = p[torch.arange(len(target)), target]
    two = torch.topk(p, 2, dim=1).values
    return {'oa': oa, 'aa': float(recall.mean()), 'kappa': (oa - chance) / (1 - chance),
            'macro_f1': float((2 * correct / (actual + guessed)).mean()), 'recall': recall.tolist(),
            'confusion': cm.tolist(), 'nll': float(-true_p.clamp_min(1e-8).log().mean()),
            'prediction_histogram': guessed.long().tolist(),
            'entropy_mean': float(-(p * p.clamp_min(1e-300).log()).sum(1).mean()),
            'true_probability_mean': float(true_p.mean()), 'margin_mean': float((two[:, 0] - two[:, 1]).mean())}


def deltas(observed, reference, new_metrics, old_metrics):
    results = {}
    for h in HEADS:
        delta = (observed[h].double() - reference[h].double()).abs()
        a, b = new_metrics[h], old_metrics[h]
        results[h] = {'probability_max_abs': float(delta.max()), 'probability_mean_abs': float(delta.mean()),
            'prediction_changes': int((observed[h].argmax(1) != reference[h].argmax(1)).sum()),
            **{key: a[key] - b[key] for key in ('oa', 'aa', 'nll', 'kappa', 'macro_f1')},
            'recall': [x - y for x, y in zip(a['recall'], b['recall'])]}
    return results


def add_readout(entry, prefix, observed, labels, arrays):
    scores, graphs = observed
    for head, p in scores.items():
        arrays[f'{prefix}__{head}'] = p.numpy()
    entry[prefix] = {'heads': {head: score(p, labels) for head, p in scores.items()}, 'graphs': graphs}
    return scores, entry[prefix]['heads']


def diagnose_case(row, reg, source, old, perturb, root, pins, scene, views, labels, conditions):
    case, manifest = row['case'], row['state']
    arm = case['arm']
    _, state, result = load_state(source, case, root, pins, old, perturb)
    source_args = (reg['source']['weight'], reg['source']['result']) if case['method'] == 'shared_transfer' else ()
    with torch.random.fork_rng(devices=[]):
        model, initialization = initialize(SpatialConfig(reg['config']), *source_args, coverage_arm=arm)
    require(initialization == result['initialization'] and state_hash(model) == manifest['initial'],
            'Reconstructed initialization differs from the producer')
    model.eval()
    entry, arrays, cached, replayed, fitting = {}, {}, {}, {}, {}
    calls = {'fit': 0, 'observation': 0, 'bypass': 0}
    for stage in STAGES:
        if stage != 'initial':
            model.load_state_dict(state['best']['model'] if stage == 'selected' else state['model'], strict=True)
        before = state_hash(model)
        require(before == manifest[stage], 'Loaded stage identity differs')
        for role in ('train', 'val'):
            key = (before, role)
            if key not in cached:
                cached[key] = forward_rows(model, scene, views[role][REFERENCE], arm)
                calls['fit'] += len(views[role][REFERENCE])
            fitting[stage, role] = add_readout(entry, f'fit__{stage}__{role}', cached[key], labels[role], arrays)
        require(state_hash(model) == before, 'Fixed-point replay changed model state or DomainBN')
    model.load_state_dict(state['best']['model'], strict=True)
    before = state_hash(model)
    observed = cached[before, 'val']
    require(all(torch.equal(observed[0][h], state['best']['scores'][h]) for h in HEADS),
            'Registered observation did not replay accepted probabilities bitwise')
    prior.metrics(observed[0], labels['val'], result['metrics'])
    condition_cache = {REFERENCE: observed}
    for name, condition in conditions.items():
        if condition['status'] == 'ready':
            alias = condition['same_bands_as']
            if name != REFERENCE and alias is None:
                condition_cache[name] = forward_rows(model, scene, views['val'][name], arm)
                calls['observation'] += len(views['val'][name])
            replayed[name] = add_readout(entry, f'obs__{name}', condition_cache[alias or name], labels['val'], arrays)
    intervention = None
    if arm != 'none':
        def unchanged_latent(_module, args, _output):
            return args[0]

        hook = model.backbone.coverage_modulation.register_forward_hook(unchanged_latent)
        try:
            bypass = forward_rows(model, scene, views['val'][REFERENCE], arm)
        finally:
            hook.remove()
        calls['bypass'] = len(views['val'][REFERENCE])
        bypass = add_readout(entry, 'bypass', bypass, labels['val'], arrays)
        reference = replayed[REFERENCE]
        intervention = deltas(reference[0], bypass[0], reference[1], bypass[1])
    require(state_hash(model) == before, 'Observation diagnostics changed model state or DomainBN')
    comparisons = {n: deltas(v[0], replayed[REFERENCE][0], v[1], replayed[REFERENCE][1]) for n, v in replayed.items()}
    fit_change = {f'{stage}__{role}': deltas(fitting[stage, role][0], fitting['initial', role][0],
                 fitting[stage, role][1], fitting['initial', role][1])
                  for stage in ('selected', 'last') for role in ('train', 'val')}
    return {'case': case, 'readouts': entry, 'versus_training_observation': comparisons,
            'fit_versus_initial': fit_change, 'affine_versus_same_model_bypass': intervention,
            'last_fit_reused_selected': manifest['last'] == manifest['selected'],
            'actual_tile_forwards': calls}, arrays, replayed


def run(scope_path, sha, out, code):
    tick = perf_counter()
    rng = _tree_fingerprint(_rng_state(CPU))
    frozen, reg, scene_plan, allowed, bands, stats, populations, source, old, metadata, root = load_context(scope_path, sha, code)
    out = batch.sibling(out, scope_path)
    require(not out.exists(), 'Diagnostic output already exists')
    _, variable, classes, _ = DATASETS[guarded.SCENE]
    cube = loadmat(reg['image'], variable_names=[variable])[variable]
    require(cube.shape == (*scene_plan['shape'], len(bands.band_ids)), 'Actual image/band shape differs')
    scene = Scene(guarded.SCENE, cube, scene_plan, bands, classes, stats)
    train_condition = {'status': 'ready', 'metadata': 'nominal', 'sampling': reg['coverage']['records']['train']}
    view_cache = {}
    entries = []
    artifacts = {}
    pair = None
    ready = {key: value for key, value in frozen['conditions'].items() if value['status'] == 'ready'}
    out.mkdir(exist_ok=False)
    for row, accepted, perturb in zip(frozen['cases'], old['rows'], metadata['rows']):
        release = row['release']
        if release not in view_cache:
            data = populations[release]
            view_cache[release] = {'train': {REFERENCE: fixed_views(scene, allowed, data['train'][0],
                                                                     train_condition, 'train')},
                'val': {name: fixed_views(scene, allowed, data['val'][0], condition, 'val')
                        for name, condition in ready.items() if condition['same_bands_as'] is None}}
        labels = {role: data[1] for role, data in populations[release].items()}
        entry, arrays, observed = diagnose_case(row, reg, source, accepted, perturb, root,
                    frozen['protected_files'], scene, view_cache[release], labels, frozen['conditions'])
        if row['case']['arm'] == 'none':
            pair_reference, pair = observed, row['pair_sha256']
        require(pair == row['pair_sha256'], 'Missing matched none case')
        entry['versus_none'] = {name: deltas(v[0], pair_reference[name][0], v[1], pair_reference[name][1])
                                for name, v in observed.items()}
        case_id = row['case']['id']
        raw, report = out / f'{case_id}.npz', out / f'{case_id}.json'
        np.savez_compressed(raw, **arrays)
        write_json(report, entry)
        artifacts.update({p.name: hash_file(p) for p in (raw, report)})
        entries.append({'case': row['case'], 'actual_tile_forwards': entry['actual_tile_forwards'],
                        'last_fit_reused_selected': entry['last_fit_reused_selected']})
    batch.check_pins({**frozen['protected_files'], str(Path(scope_path).resolve()): sha})
    require(_tree_fingerprint(_rng_state(CPU)) == rng, 'Diagnostic run changed global RNG state')
    require_frozen_code(code)
    path = out / 'receipt.json'
    write_json(path, {'status': 'passed', 'scope_sha256': sha, 'producer_commit': frozen['producer_commit'],
        'consumer_commit': code['commit'], 'rows': entries, 'artifacts': artifacts,
        'fit_heads': len(entries) * 18, 'observation_heads': len(entries) * len(ready) * 3,
        'observation_class_recalls': len(entries) * len(ready) * 3 * 22,
        'bypass_heads': sum(r['case']['arm'] != 'none' for r in entries) * 3,
        'bitwise_selected_reference_heads': len(entries) * 3, 'protected_producer_files': len(old['protected_files']),
        'rng_sha256': rng, 'updates': 0, 'raw_gt_loaded': False, 'query_accessed': False,
        'wall_seconds': perf_counter() - tick})
    return path


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    prep.add_argument('--synthetic', action='store_true')
    for key in PARENT_PINS:
        prep.add_argument('--' + key, type=Path, required=True)
        prep.add_argument('--' + key + '-sha256', required=True)
    execute = modes.add_parser('run')
    execute.add_argument('--scope', type=Path, required=True)
    execute.add_argument('--scope-sha256', required=True)
    for mode in (prep, execute):
        mode.add_argument('--expected-commit', required=True)
        mode.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require(code['commit'] == args.expected_commit, 'Unexpected diagnostic consumer commit')
    if args.mode == 'prepare':
        links = {k: {'path': str(getattr(args, k).resolve()), 'sha256': getattr(args, k + '_sha256')} for k in PARENT_PINS}
        path = prepare(links, args.out, code, synthetic=args.synthetic)
    else:
        path = run(args.scope, args.scope_sha256, args.out, code)
    print(json.dumps({'path': str(path), 'sha256': hash_file(path)}), flush=True)


if __name__ == '__main__':
    main()
