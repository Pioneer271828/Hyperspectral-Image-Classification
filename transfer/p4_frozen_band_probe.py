"""Read-only spectral perturbations of accepted guarded HongHu selected states."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.io import loadmat

from evaluation.classification_metrics import metrics_from_confusion
from loadData.data_read import read_band_metadata
from loadData.observation_sampling import ObservationView, sample, subset_statistics
from loadData.spatial_protocol import PartitionObservation, forward_partition
from model.observation_coverage import CoverageConfig, describe
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, tensor_hash, write_json
from train_spatial import SpatialConfig
from transfer import p4_guarded
from transfer.g3_training import Scene, _forward, state_hash
from transfer.observation_policy import default_conditions
from transfer.p4_spatial import fit_role_statistics, initialize, load_release, read


HEADS = ('fused', 'graph', 'cnn')
SOURCE_COMMIT = '83457d65ab190f3df88ed28e3c673ae3ce43f15f'
VALIDATION_SEED = 3031


def freeze_conditions(scene):
    bands = scene.bands
    configs = default_conditions(bands)
    low, high = min(bands.wavelengths), max(bands.wavelengths)
    configs.update({name: {'mode': 'range_crop', 'lower': low + a * (high - low),
                           'upper': low + b * (high - low), 'unit': bands.wavelength_unit}
                    for name, a, b in (('lower50', 0, .5), ('upper50', .5, 1), ('middle50', .25, .75))})
    stub = PartitionObservation('val', np.zeros((1, 1, len(bands.band_ids)), dtype=np.float32),
                                np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.int64),
                                scene.plan['plan_hash'])
    records = {}
    for name, config in configs.items():
        seed = int(digest({'seed': VALIDATION_SEED, 'name': name})[:16], 16)
        _, record = sample(stub, bands, config, np.random.default_rng(seed))
        records[name] = {'sampling': record, 'metadata': 'nominal', 'reference': 'full'}
    for mode in ('shift10nm', 'missing'):
        records['stride_' + mode] = dict(records['stride'], metadata=mode, reference='stride')
    return records


def selected_view(scene, tile, allowed, condition):
    if tile['partition'] != 'val' or allowed.get(tile['id']) != tile:
        raise ValueError('Spectral probe can access released validation tiles only')
    record = condition['sampling']
    if record['partition'] != 'val':
        raise ValueError('Spectral sampling record must bind validation')
    statistics = subset_statistics(scene.stats, scene.bands, record)
    positions = record['selected_positions']
    metadata = scene.bands.subset(positions)
    if metadata.fingerprint != record['selected_metadata_hash']:
        raise ValueError('Selected band metadata differs')
    y0, y1, x0, x1 = tile['box']
    # Slice raw retained columns first. Deleted values never enter normalization or graph building.
    tile_values = scene.cube[y0:y1, x0:x1]
    values = np.array(tile_values if len(positions) == tile_values.shape[-1] else tile_values[:, :, positions],
                      dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite retained observation')
    low, high = (np.asarray(statistics[key], dtype=np.float32) for key in ('minimum', 'maximum'))
    spectral = (values - low) / np.where(high > low, high - low, 1)
    h, w = scene.plan['shape']
    yy, xx = np.mgrid[y0:y1, x0:x1]
    coordinates = np.stack((yy / max(h - 1, 1), xx / max(w - 1, 1)), axis=-1).astype(np.float32)
    observation = PartitionObservation('val', spectral, coordinates, yy * w + xx, scene.plan['plan_hash'])
    mode = condition['metadata']
    if mode == 'shift10nm':
        if metadata.wavelength_unit != 'nm':
            raise ValueError('Wavelength shift requires physical coordinates')
        metadata = replace(metadata, wavelengths=tuple(value + 10 for value in metadata.wavelengths))
    elif mode not in ('nominal', 'missing'):
        raise ValueError('Unknown metadata perturbation')
    return ObservationView(observation, metadata)


def capture(model, scene, tile, view, *, missing):
    graph = model.backbone.learnable_graph
    if model.training or graph is None or 'build_graph' in graph.__dict__:
        raise ValueError('Expected a private evaluation model with an unwrapped learned graph')
    original = graph.build_graph
    observed = []

    def build(*args, **kwargs):
        state = original(*args, **kwargs)
        observed.append({'q_sha256': tensor_hash(state.Q), 'a_sha256': tensor_hash(state.A0),
                         'q_nnz': state.Q._nnz(), 'a_nnz': state.A0._nnz()})
        return state

    graph.build_graph = build
    try:
        if missing:
            result = forward_partition(model, scene.dataset, view.observation, wavelengths=None)
        else:
            result = _forward(model, scene, tile, observation_view=view)
    finally:
        del graph.build_graph
    if len(observed) != 1:
        raise RuntimeError('Expected one freshly built graph per observation')
    return result[:3], observed[0]


@torch.no_grad()
def evaluate(model, scene, tiles, condition):
    allowed = {tile['id']: tile for tile, _ in tiles}
    scores = {head: [] for head in HEADS}
    truth, graphs = [], []
    for tile, local in tiles:
        labels = torch.from_numpy(local.reshape(-1))
        valid = labels > 0
        truth.append(labels[valid] - 1)
        view = selected_view(scene, tile, allowed, condition)
        missing = condition['metadata'] == 'missing'
        heads, graph = capture(model, scene, tile, view, missing=missing)
        graphs.append(dict(graph, tile=tile['id']))
        for name, probabilities in zip(HEADS, heads):
            scores[name].append(probabilities[valid].cpu())
    target = torch.cat(truth)
    if torch.bincount(target, minlength=scene.classes).tolist() != [5] * scene.classes:
        raise ValueError('Validation label budget differs')
    adapter = model.backbone.adapter
    metadata = None if missing else view.bands
    support = describe(metadata, adapter._anchors(torch.device('cpu'), torch.float32),
                       CoverageConfig(wavelength_min=adapter.wavelength_min,
                                      wavelength_max=adapter.wavelength_max))
    return (target, {head: torch.cat(values) for head, values in scores.items()}, graphs,
            {'usable_anchors': int(support.usable.sum()), 'total_anchors': support.usable.numel(),
             **support.diagnostics})


def compare(target, probabilities, reference, classes):
    if (probabilities.shape != (len(target), classes) or not torch.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not torch.allclose(probabilities.sum(1), torch.ones(len(target)), rtol=0, atol=2e-6)):
        raise ValueError('Invalid probe probabilities')
    predictions = probabilities.argmax(1)
    changed = predictions != reference.argmax(1)
    counts = torch.bincount(target * classes + predictions, minlength=classes**2).reshape(classes, classes)
    return {**metrics_from_confusion(counts.numpy(), list(range(1, classes + 1))),
            'nll': float(-probabilities[torch.arange(len(target)), target].clamp_min(1e-8).log().mean()),
            'changed_predictions': int(changed.sum()),
            'changed_by_class': torch.bincount(target[changed], minlength=classes).tolist(),
            'mean_probability_l1': float((probabilities - reference).abs().sum(1).mean())}


def run(training, audit_path, audit_sha, out, code):
    started = perf_counter()
    require_frozen_code(code)
    if hash_file(audit_path) != audit_sha:
        raise ValueError('Independent short-test audit changed')
    audit = read(audit_path)
    if (audit['status'] != 'passed' or audit['tested_commit'] != SOURCE_COMMIT
            or audit['case_pairs'] != 8 or audit['query_accessed'] or audit['raw_gt_loaded']):
        raise ValueError('Accepted guarded short-test audit required')
    pins = {Path(path): sha for path, sha in audit['protected_files'].items()}
    registration_path = training / 'registration.json'
    if registration_path not in pins or hash_file(registration_path) != pins[registration_path]:
        raise ValueError('Audit does not bind this registration')
    registration = read(registration_path)
    if registration['commit'] != SOURCE_COMMIT or registration['config']['dataset'] != p4_guarded.SCENE:
        raise ValueError('Guarded HongHu production identity differs')
    pins[audit_path] = audit_sha
    if any(pins.get(Path(path)) != sha for path, sha in registration['pins'].items()):
        raise ValueError('Original training input is unbound')
    if any(out == path or out in path.parents or path.parent == out for path in pins):
        raise ValueError('Output overlaps protected inputs')
    if any(hash_file(path) != sha for path, sha in pins.items()):
        raise ValueError('A protected short-test file changed')

    plan = read(registration['plan'])
    geometry = read(registration['development'])
    guarded, roles = p4_guarded.bind_inputs(registration, plan, geometry)
    cube = loadmat(registration['image'], variable_names=['WHU_Hi_HongHu'])['WHU_Hi_HongHu']
    bands = read_band_metadata(p4_guarded.SCENE, cube.shape[-1])
    stats = fit_role_statistics(cube, guarded, roles['train'], guarded['plan_hash'])
    if stats != read(registration['statistics']):
        raise ValueError('Guarded train-only statistics changed')
    scene = Scene(p4_guarded.SCENE, cube, guarded, bands, 22, stats)
    conditions = freeze_conditions(scene)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'registration.json',
               {'scope': 'frozen short-checkpoint development sensitivity', 'source_commit': SOURCE_COMMIT,
                'code': code['commit'], 'audit_sha256': audit_sha, 'statistics_sha256': stats['statistics_hash'],
                'condition_seed': VALIDATION_SEED, 'conditions': conditions,
                'inputs': {str(path): sha for path, sha in pins.items()}})
    rows = []
    for i, case in enumerate(registration['cases']):
        folder = training / f'case-{i}-full'
        state_path, result_path = folder / 'state.pt', folder / 'result.json'
        if state_path not in pins or result_path not in pins:
            raise ValueError('Selected state has no independent audit binding')
        result = read(result_path)
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        if (state['cursor'] != 12 or state['identity'] != result['identity']
                or state['best']['step'] != result['best_step'] or result['state_sha256'] != pins[state_path]):
            raise ValueError('Short-test state or selected step differs')
        model, _ = initialize(SpatialConfig(registration['config']))
        model.load_state_dict(state['best']['model'], strict=True)
        if model.backbone.coverage_modulation is not None:
            raise ValueError('The selected guarded control must have coverage modulation disabled')
        model.eval()
        before = state_hash(model)
        tiles = load_release(read(case['release']), plan, case['support_seed'], case['k'], geometry,
                             guarded_tiles=roles)['val']
        details, saved = {}, {}
        for name, condition in conditions.items():
            target, scores, graphs, coverage = evaluate(model, scene, tiles, condition)
            if name == 'full':
                if any(not torch.equal(scores[head], state['best']['scores'][head]) for head in HEADS):
                    raise ValueError('Full-band selected probabilities differ from accepted short test')
                baseline_target = target
            elif not torch.equal(target, baseline_target):
                raise ValueError('Condition changed released validation coordinates')
            saved[name] = scores
            ref = condition['reference']
            ref_graphs = graphs if name == 'full' else details[ref]['graphs']
            details[name] = {
                'heads': {head: compare(target, scores[head], saved[ref][head], scene.classes) for head in HEADS},
                'graphs': graphs, 'coverage': coverage, 'reference': ref,
                'changed_q_tiles': sum(a['q_sha256'] != b['q_sha256'] for a, b in zip(graphs, ref_graphs)),
                'changed_a_tiles': sum(a['a_sha256'] != b['a_sha256'] for a, b in zip(graphs, ref_graphs)),
            }
        if state_hash(model) != before:
            raise RuntimeError('Frozen model state changed')
        scores_path = out / f'case-{i}-scores.pt'
        torch.save({'labels': baseline_target, 'conditions': saved}, scores_path)
        rows.append({'case': i, 'seed': case['support_seed'], 'k': case['k'], 'method': case['method'],
                     'selected_step': result['best_step'], 'conditions': details,
                     'scores_sha256': hash_file(scores_path), 'model_state_sha256': before})
    require_frozen_code(code)
    if any(hash_file(path) != sha for path, sha in pins.items()):
        raise RuntimeError('A protected input changed during probing')
    write_json(out / 'receipt.json',
               {'status': 'passed', 'registration_sha256': hash_file(out / 'registration.json'),
                'elapsed_seconds': perf_counter() - started, 'rows': rows, 'optimizer_updates': 0,
                'raw_gt_loaded': False, 'query_accessed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'audit', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--audit-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        raise ValueError('Unexpected spectral probe commit')
    run(args.training.resolve(), args.audit.resolve(), args.audit_sha256, external_directory(args.out), code)


if __name__ == '__main__':
    main()
