"""Audit sealed caches/checkpoints and independently recompute every prediction.

Default mode performs zero fitting updates. --replay-fits adds the registered
3840-update independent optimizer replay and obeys the current training pause.
Original-image/feature provenance is checked separately by audit_features.py.
"""
import argparse
import hashlib
import hmac
import itertools
from pathlib import Path

import numpy as np

from c8_bridge import canonical, sha, digest, read, write, protocol, fresh, require_learning_enabled
from independent import score, probability, metrics, sensitivity_reference, pair_reference
from independent import assert_close, replay, centroid_reference, resamples, interval


def cache(prefix):
    prefix = Path(prefix)
    meta = read(prefix.with_suffix('.json'))
    path = prefix.parent / meta['cache_name']
    if path.parent != prefix.parent or digest(path) != meta['cache_sha256']:
        raise ValueError('Frozen feature-cache digest/path differs')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    return data, meta


def validate_package(experiment, producer, *, replay_fits=False):
    root = Path(experiment).resolve()
    cfg = protocol()
    config_hash = sha(canonical(cfg))
    seal = read(root/'fitting-complete.json')
    if seal['source'] != producer or seal['protocol_sha256'] != config_hash or not seal['all_fitting_complete_before_any_development_array']:
        raise ValueError('Full fitting seal required before development evaluation')
    evaluation = read(root/'evaluation01/evaluation.json')
    if evaluation['producer'] != producer or evaluation['protocol_sha256'] != config_hash or evaluation['selection_used_target_development'] or evaluation['query_reads']:
        raise ValueError('Evaluation identity, selection or access contract differs')
    key = (root/'evaluation01/pseudonymization.key').read_bytes()
    if len(key) != 32:
        raise ValueError('Private alias key differs')
    files, stages, updates = {}, {}, 0
    replay_updates = 0
    for source, target in cfg['directions']:
        for stage in ('source', 'target'):
            folder = root/f'{stage}-{source}-to-{target}'
            manifest = read(folder/'manifest.json')
            phase = 'source_fit' if stage == 'source' else 'target_adapt'
            expected_updates = 384 if stage == 'source' else 1536
            if not manifest['complete'] or manifest['source'] != producer or manifest['protocol_sha256'] != config_hash or manifest['direction'] != [source, target] or manifest['stage'] != stage or manifest['denied_opens'] or not manifest['no_development_array_phase_available']:
                raise ValueError('Fitting stage identity/role/seal differs')
            X, frame = cache(folder/'features')
            if frame['phase'] != phase or frame['original_role'] != 'devtrain' or frame['scene'] != (source if stage == 'source' else target) or len(X['y']) != 10 or np.bincount(X['y'], minlength=2).tolist() != [5, 5]:
                raise ValueError('Fitting labels or feature role differs')
            stages[str(folder)] = X
            stage_updates = 0
            expected_registry = {(rep, seed, arm) for rep in ('raw_distribution', 'legacy_scene_distribution') for seed in cfg['seeds'] for arm in (('source_fit',) if stage == 'source' else ('target_adapt', 'target_scratch64', 'target_scratch128'))}
            if stage == 'source':
                expected_registry |= {(rep, None, 'centroid') for rep in ('raw_distribution', 'legacy_scene_distribution')}
            actual_registry = set()
            for entry in manifest['models']:
                path = folder/entry['file']
                if path.parent != folder or digest(path) != entry['sha256']:
                    raise ValueError('Checkpoint digest/path differs')
                model = read(path)
                actual_registry.add((entry['representation'], entry['seed'], entry['arm']))
                if model['source'] != producer or model['protocol_sha256'] != config_hash or model['fit_role'] != 'devtrain' or model['fit_points'] != 10 or model['feature_cache_sha256'] != frame['cache_sha256'] or model['selection'] not in ('fixed_terminal_no_development_selection', 'closed_form_no_development_selection'):
                    raise ValueError('Endpoint fitting identity differs')
                kind = entry['kind']
                count = 0 if kind == 'centroid' else (128 if entry['arm'] == 'target_scratch128' else 64)
                if model['optimizer_updates'] != count or entry['optimizer_updates'] != count:
                    raise ValueError('Registered per-endpoint update count differs')
                if replay_fits:
                    start = None
                    if entry['arm'] == 'target_adapt':
                        parent_path = root/f'source-{source}-to-{target}'/model['parent_source']['file']
                        if digest(parent_path) != model['parent_source']['sha256']:
                            raise ValueError('Adaptation source ancestry differs')
                        start = read(parent_path)['theta']
                    ref = centroid_reference(X[entry['representation']], X['y']) if kind == 'centroid' else replay(X[entry['representation']], X['y'], seed=entry['seed'], steps=count, start=start)
                    assert_close(model['theta'], ref, atol=1e-10, rtol=1e-9, path='independent coefficient replay')
                    replay_updates += count
                files[str(path)] = (model, entry, folder)
                stage_updates += count
            if actual_registry != expected_registry or len(manifest['models']) != len(expected_registry) or stage_updates != expected_updates or manifest['optimizer_updates'] != stage_updates:
                raise ValueError('Endpoint registry or stage update total differs')
            updates += stage_updates
    if len(files) != 52 or updates != 3840:
        raise ValueError('Full fixed endpoint coverage required')
    frames = {scene: cache(root/'evaluation01'/f'features-{scene}')[0] for scene in cfg['directions'][0]}
    rows = evaluation['rows']
    if len(rows) != 68 or len({r['id'] for r in rows}) != 68:
        raise ValueError('All 68 distinct registered evaluations required')
    expected_evaluations = set()
    for path, (model, entry, folder) in files.items():
        source, target = model['direction']
        cases = [(target, entry['arm'])] if folder.name.startswith('target-') else [(source, 'source_same_device' if entry['kind'] == 'logistic' else 'centroid_same_device'), (target, 'source_fit_transfer' if entry['kind'] == 'logistic' else 'centroid_transfer')]
        expected_evaluations.update((path, scene, method) for scene, method in cases)
    actual_evaluations = set()
    argmax_changes = 0
    for row in rows:
        model, entry, _ = files[row['model_file']]
        actual_evaluations.add((row['model_file'], row['scene'], row['method']))
        frame = frames[row['scene']]
        assert_close(row['point_ids'], frame['point_ids'].tolist())
        assert_close(row['parent_ids'], frame['parent_ids'].tolist())
        assert_close(row['truth'], frame['y'].tolist())
        if len(row['truth']) != 10 or row['metrics']['class_denominators'] != [5, 5] or row['model_sha256'] != entry['sha256']:
            raise ValueError('Evaluation support/checkpoint identity differs')
        z = [score(model['theta'], x) for x in frame[row['representation']]]
        p = [probability(v) for v in z]
        pred = [int(v > 0) for v in z]
        argmax_changes += sum(a != b for a, b in zip(row['predictions'], pred))
        assert_close(row['logits'], z)
        assert_close(row['probabilities'], p)
        assert_close(row['predictions'], pred)
        assert_close(row['metrics'], metrics(row['truth'], z, p, pred))
        assert_close(row['parent_sensitivity'], sensitivity_reference(row))
        for kind, values, aliases in [('point', row['point_ids'], row['point_aliases']), ('parent', row['parent_ids'], row['parent_aliases'])]:
            expected = [hmac.new(key, canonical([kind, row['scene'], int(v)]), hashlib.sha256).hexdigest()[:20] for v in values]
            assert_close(aliases, expected)
    if actual_evaluations != expected_evaluations or argmax_changes:
        raise ValueError('Evaluation registry or independent decisions differ')
    by_id = {r['id']: r for r in rows}
    comparisons = evaluation['paired_comparisons']
    if len(comparisons) != 64 or len({(p['A'], p['B']) for p in comparisons}) != 64:
        raise ValueError('All 64 paired comparisons required')
    for pair in comparisons:
        assert_close(pair, pair_reference(by_id[pair['A']], by_id[pair['B']]))
    directions = evaluation['direction_differences']
    if len(directions) != 8:
        raise ValueError('All eight directional summaries required')
    for result in directions:
        groups = [[next(r for r in rows if r['direction'] == direction and r['representation'] == result['representation'] and r['method'] == result['method'] and r['seed'] == seed) for seed in cfg['seeds']] for direction in cfg['directions']]
        correct = [[math_mean([int(row['predictions'][i] == row['truth'][i]) for row in group]) for i in range(10)] for group in groups]
        differences = [sum(w*v for w, v in zip(wa, correct[0]))/sum(wa)-sum(w*v for w, v in zip(wb, correct[1]))/sum(wb) for wa, wb in itertools.product(resamples(groups[0][0]['parent_ids']), resamples(groups[1][0]['parent_ids']))]
        expected = dict(OA_A_minus_B=math_mean(correct[0])-math_mean(correct[1]), per_seed_differences=[a['metrics']['OA']-b['metrics']['OA'] for a, b in zip(*groups)], observed_parent_cross_product_draws=len(differences), descriptive_range=interval(differences), point_pairing=False, causal_sensor_interpretation=False)
        assert_close(result, expected)
    return dict(scope='Independent cached-feature/checkpoint/prediction/metric audit; original-image provenance is a separate gate', passed=True, checkpoints=52, evaluations=68, prediction_presentations=680, unique_development_points=20, paired_comparisons=64, direction_summaries=8, producer_optimizer_updates=updates, independent_replay_updates=replay_updates, optimizer_replay_performed=replay_fits, original_image_feature_provenance_checked=False, argmax_changes=argmax_changes, query_reads=0)


def math_mean(values):
    import math
    return math.fsum(values)/len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--producer-commit', required=True)
    p.add_argument('--experiment', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--replay-fits', action='store_true')
    args = p.parse_args()
    if args.replay_fits:
        require_learning_enabled()
    out = fresh(args.expected_commit, args.out)
    result = validate_package(args.experiment, args.producer_commit, replay_fits=args.replay_fits)
    result.update(auditor=args.expected_commit, producer=args.producer_commit)
    write(out, result)
    print(__import__('json').dumps(result))


if __name__ == '__main__':
    main()
