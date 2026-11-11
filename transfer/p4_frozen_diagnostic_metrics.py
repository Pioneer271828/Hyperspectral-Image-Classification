"""Independently recompute P4 fixed-point and observation results from saved probabilities."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from evaluation.classification_metrics import metrics_from_confusion
from research_runtime import code_snapshot, hash_file, require_frozen_code, write_json
from transfer import p4_coverage_batch as batch, p4_frozen_diagnostics as diag
from transfer.g2_contract import read_pinned


METRICS = ('oa', 'aa', 'kappa', 'macro_f1', 'nll', 'entropy_mean', 'true_probability_mean', 'margin_mean')
DELTA = ('oa', 'aa', 'kappa', 'macro_f1', 'nll', 'probability_max_abs', 'probability_mean_abs', 'prediction_changes')
CLASSES = list(range(1, 23))


def recompute(values, labels):
    p = values.astype(np.float64)
    diag.require(p.shape == (len(labels), 22) and np.isfinite(p).all() and (p >= 0).all()
                 and np.allclose(p.sum(1), 1, rtol=0, atol=1e-6), 'Invalid independent probabilities')
    confusion = np.zeros((22, 22), dtype=np.int64)
    np.add.at(confusion, (labels - 1, p.argmax(1)), 1)
    m = metrics_from_confusion(confusion, CLASSES)
    diag.require(not m['missing_class_ids'], 'Independent label support differs')
    true = p[np.arange(len(p)), labels - 1]
    top = np.sort(p, axis=1)[:, -2:]
    return {**{k: m[k] for k in ('oa', 'aa', 'kappa', 'macro_f1')},
        'recall': [row['recall'] for row in m['per_class']], 'confusion': confusion.tolist(),
        'nll': float(-np.log(np.maximum(true, 1e-8)).mean()), 'prediction_histogram': confusion.sum(0).tolist(),
        'entropy_mean': float(-(p * np.log(np.maximum(p, 1e-300))).sum(1).mean()),
        'true_probability_mean': float(true.mean()), 'margin_mean': float(np.diff(top, axis=1).mean())}


def difference(current, baseline, a, b):
    delta = np.abs(current.astype(np.float64) - baseline.astype(np.float64))
    return {'probability_max_abs': float(delta.max()), 'probability_mean_abs': float(delta.mean()),
        'prediction_changes': int(np.count_nonzero(current.argmax(1) != baseline.argmax(1))),
        **{key: a[key] - b[key] for key in ('oa', 'aa', 'kappa', 'macro_f1', 'nll')},
        'recall': (np.array(a['recall']) - b['recall']).tolist()}


def equivalent(expected, recorded):
    return expected.keys() == recorded.keys() and all(
        math.isclose(value, recorded[k], rel_tol=0, abs_tol=1e-12) if isinstance(value, float) else value == recorded[k]
        for k, value in expected.items())


def inventory(frozen, receipt, sha):
    cases = [row['case'] for row in frozen['cases']]
    names = {f"{case['id']}.{ext}" for case in cases for ext in ('json', 'npz')}
    ready = {k: v for k, v in frozen['conditions'].items() if v['status'] == 'ready'}
    diag.require(receipt['status'] == 'passed' and receipt['scope_sha256'] == sha
        and receipt['producer_commit'] == frozen['producer_commit'] and receipt['consumer_commit'] == frozen['consumer_commit']
        and [row['case'] for row in receipt['rows']] == cases and set(receipt['artifacts']) == names
        and receipt['fit_heads'] == 720 and receipt['observation_heads'] == 40 * len(ready) * 3
        and receipt['observation_class_recalls'] == receipt['observation_heads'] * 22
        and receipt['bypass_heads'] == 96 and receipt['bitwise_selected_reference_heads'] == 120
        and receipt['raw_gt_loaded'] is receipt['query_accessed'] is False and receipt['updates'] == 0,
        'Incomplete or mismatched diagnostic execution')
    return ready


def check_graphs(graphs, layout, record, arm):
    diag.require([row['tile'] for row in graphs] == [tile['id'] for tile, _ in layout], 'Graph tile order differs')
    for row in graphs:
        active = arm != 'none'
        diag.require(row['bands'] == len(record['selected_positions'])
            and row['metadata_sha256'] == record['selected_metadata_hash'] and row['coverage_active'] is active
            and (isinstance(row['usable_anchors'], int) and row['usable_anchors'] >= 0
                 if active else row['usable_anchors'] is None and row['fallback_reason'] is None)
            and all(len(row[name]) == 64 and all(c in '0123456789abcdef' for c in row[name])
                    for name in ('q_sha256', 'a_sha256'))
            and min(row['q_nnz'], row['a_nnz']) > 0,
            'Graph or coverage evidence differs from the registered observation')


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_recall(values):
    return {f'recall_{class_id}': v for class_id, v in zip(CLASSES, values)}


def overview(rows, keys, values):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{**dict(zip(keys, key)), 'cases': len(group), 'metrics': {
        metric: {'min': min(data), 'max': max(data), 'mean': mean(data)}
        for metric in values if (data := [row[metric] for row in group])}}
        for key, group in groups.items()]


def verify(scope, sha, receipt_path, receipt_sha, out, code):
    tick = perf_counter()
    frozen, reg, _, _, _, _, populations, source, accepted, metadata, producer_root = diag.load_context(scope, sha, code)
    receipt_path = Path(receipt_path).resolve()
    receipt = read_pinned(receipt_path, receipt_sha)
    ready = inventory(frozen, receipt, sha)
    root = receipt_path.parent
    out = batch.sibling(out, scope)
    diag.require(not out.exists(), 'Independent review output already exists')
    pins = {**frozen['protected_files'], str(Path(scope).resolve()): sha, str(receipt_path): receipt_sha,
            **{str(root / name): value for name, value in receipt['artifacts'].items()}}
    batch.check_pins(pins)
    metrics_rows, delta_rows = [], []
    totals = {'fit': 0, 'observation': 0, 'bypass': 0}
    references, pair = {}, None
    for row, observed, old, perturb in zip(frozen['cases'], receipt['rows'], accepted['rows'], metadata['rows']):
        case = row['case']
        keys = {k: case[k] for k in ('id', 'support_seed', 'k', 'method', 'arm')}
        case_id, arm = case['id'], case['arm']
        layout = populations[row['release']]
        groups = {f'fit__{stage}__{role}': (role, reg['coverage']['records'][role])
                  for stage in diag.STAGES for role in ('train', 'val')}
        groups.update({f'obs__{name}': ('val', condition['sampling']) for name, condition in ready.items()})
        if arm != 'none':
            groups['bypass'] = ('val', reg['coverage']['records']['val'])
        result_path = root / f'{case_id}.json'
        result = read_pinned(result_path, pins[str(result_path)])
        diag.require(result['case'] == case and set(result['readouts']) == set(groups), 'Diagnostic readout roles differ')
        with np.load(root / f'{case_id}.npz', allow_pickle=False) as archive:
            diag.require(set(archive.files) == {f'{prefix}__{h}' for prefix in groups for h in diag.HEADS},
                         'Independent probability array inventory differs')
            probs = {name: archive[name] for name in archive.files}
        _, saved, _ = diag.load_state(source, case, producer_root, pins, old, perturb)
        same_state = row['state']['selected'] == row['state']['last']
        expected_forwards = {
            'fit': len({row['state'][s] for s in diag.STAGES}) * sum(len(layout[r][0]) for r in ('train', 'val')),
            'observation': sum(name != diag.REFERENCE and c['same_bands_as'] is None for name, c in ready.items())
                           * len(layout['val'][0]),
            'bypass': len(layout['val'][0]) if arm != 'none' else 0,
        }
        diag.require(result['last_fit_reused_selected'] is observed['last_fit_reused_selected'] is same_state
            and result['actual_tile_forwards'] == observed['actual_tile_forwards'] == expected_forwards,
            'Physical forward accounting differs from exact-state/observation deduplication')
        for label, count in expected_forwards.items():
            totals[label] += count
        readouts = {}
        for prefix, (role, sampling) in groups.items():
            actual = result['readouts'][prefix]
            check_graphs(actual['graphs'], layout[role][0], sampling, arm)
            diag.require(set(actual['heads']) == set(diag.HEADS), 'Three-head metric inventory differs')
            for h in diag.HEADS:
                name = f'{prefix}__{h}'
                readouts[name] = m = recompute(probs[name], layout[role][1])
                diag.require(equivalent(m, actual['heads'][h]), 'Independent metrics differ: ' + name)
                metrics_rows.append({**keys, 'readout': prefix, 'head': h, 'labels': layout[role][2]['labels'],
                    **{key: m[key] for key in METRICS}, 'zero_recall_classes': m['recall'].count(0),
                    **flatten_recall(m['recall'])})
        for h in diag.HEADS:
            ref = probs[f'obs__{diag.REFERENCE}__{h}']
            diag.require(np.array_equal(ref, saved['best']['scores'][h].numpy()) and
                         np.array_equal(ref, probs[f'fit__selected__val__{h}']),
                         'Accepted nominal probability replay differs')
            if same_state:
                for role in ('train', 'val'):
                    diag.require(np.array_equal(probs[f'fit__selected__{role}__{h}'], probs[f'fit__last__{role}__{h}']),
                                 'Exact-state reuse returned different probabilities')
            for name, condition in ready.items():
                alias = condition['same_bands_as']
                if alias is not None:
                    diag.require(np.array_equal(probs[f'obs__{name}__{h}'], probs[f'obs__{alias}__{h}']),
                                 'Identical retained positions returned different probabilities')

        def compare(mode, readout, ref_readout, record, reference_probs=probs, reference_metrics=readouts):
            diag.require(set(record) == set(diag.HEADS), 'Three-head delta inventory differs')
            for head in diag.HEADS:
                a, b = f'{readout}__{head}', f'{ref_readout}__{head}'
                d = difference(probs[a], reference_probs[b], readouts[a], reference_metrics[b])
                diag.require(equivalent(d, record[head]), 'Independent probability delta differs: ' + a)
                delta_rows.append({**keys, 'comparison': mode, 'readout': readout, 'head': head,
                    **{name: d[name] for name in DELTA}, **flatten_recall(d['recall'])})

        if arm == 'none':
            references, pair = (probs, readouts), row['pair_sha256']
        diag.require(row['pair_sha256'] == pair and set(result['versus_none']) == set(ready)
                     and set(result['versus_training_observation']) == set(ready),
                     'Independent observation or matched-none inventory differs')
        for name in ready:
            compare('versus_training_observation', f'obs__{name}', f'obs__{diag.REFERENCE}',
                    result['versus_training_observation'][name])
            compare('versus_none', f'obs__{name}', f'obs__{name}', result['versus_none'][name], *references)
        for stage in ('selected', 'last'):
            for role in ('train', 'val'):
                compare('fit_versus_initial', f'fit__{stage}__{role}', f'fit__initial__{role}',
                        result['fit_versus_initial'][f'{stage}__{role}'])
        if arm != 'none':
            compare('affine_versus_bypass', f'obs__{diag.REFERENCE}', 'bypass', result['affine_versus_same_model_bypass'])
        else:
            diag.require(result['affine_versus_same_model_bypass'] is None, 'None arm unexpectedly used an affine bypass')
    diag.require(len(metrics_rows) == sum(receipt[k] for k in ('fit_heads', 'observation_heads', 'bypass_heads')),
                 'Independent metric totals differ')
    batch.check_pins(pins)
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    csv_write(out / 'metrics.csv', metrics_rows)
    csv_write(out / 'comparisons.csv', delta_rows)
    fit = [r for r in metrics_rows if r['readout'].startswith('fit__')]
    obs = [r for r in metrics_rows if r['readout'].startswith('obs__')]
    write_json(out / 'overview.json', {
        'fit_by_support_and_initialization': overview(fit, ('support_seed', 'k', 'method', 'readout', 'head'),
            (*METRICS, 'recall_21', 'zero_recall_classes')),
        'observations_by_arm': overview(obs, ('arm', 'readout', 'head'), (*METRICS, 'recall_21', 'zero_recall_classes')),
        'probability_comparisons': overview(delta_rows, ('comparison', 'readout', 'arm', 'head'), (*DELTA, 'recall_21')),
    })
    path = out / 'receipt.json'
    write_json(path, {'status': 'passed', 'producer_commit': frozen['producer_commit'], 'consumer_commit': code['commit'],
        'scope_sha256': sha, 'diagnostic_sha256': receipt_sha, 'cases': len(frozen['cases']),
        'fit_heads': receipt['fit_heads'], 'observation_heads': receipt['observation_heads'],
        'bypass_heads': receipt['bypass_heads'], 'fit_class_recalls': receipt['fit_heads'] * 22,
        'observation_class_recalls': receipt['observation_class_recalls'],
        'comparison_heads': len(delta_rows), 'actual_tile_forwards': totals, 'protected_files': len(pins),
        'artifacts': {name: hash_file(out / name) for name in ('metrics.csv', 'comparisons.csv', 'overview.json')},
        'updates': 0, 'query_accessed': False, 'raw_gt_loaded': False, 'wall_seconds': perf_counter() - tick})
    return path


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scope', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    diag.require(code['commit'] == args.expected_commit, 'Unexpected independent metric consumer commit')
    path = verify(args.scope, args.scope_sha256, args.receipt, args.receipt_sha256, args.out, code)
    print(json.dumps({'path': str(path), 'sha256': hash_file(path)}), flush=True)


if __name__ == '__main__':
    main()
