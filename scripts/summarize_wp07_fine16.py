"""Read-only WP12-facing reduction of both preregistered scene packages."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, data):
    require(data, 'Empty output table')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def summarize(base, out):
    require(not sys.flags.optimize, 'Acceptance refuses optimized Python')
    require(not out.exists(), 'Use new summary output')
    primary, adverse, protected56, responses, costs = [], [], [], [], []
    scenes = {}
    for scene, steps, views in [('paviaU', 60, 82), ('xuzhou', 62, 10)]:
        folder = base/scene
        plan = read(folder/'pair/plan.json')
        result = read(folder/'results/summary.json')
        require(plan['optimizer_updates'] == 12*steps == result['optimizer_updates'], 'Budget differs')
        require(len(plan['cases']) == 12 and len(result['selected_steps']) == 12, 'Missing cases')
        require(all(int(v) in (0, steps//2, steps) for v in result['selected_steps'].values()), 'Selected step differs')
        for row in result['primary_pairs']:
            primary.append({'scene': scene, **row, 'points_per_band': 45})
        endpoints = rows(folder/'results/endpoints.csv')
        classes = rows(folder/'results/per-class-correct.csv')
        require(len(endpoints) == 144 and len(classes) == 1296, 'Endpoint/class denominators differ')
        require(all(int(r['support']) == 5 for r in classes), 'Class denominator differs')
        for row in rows(folder/'results/paired-per-class.csv'):
            if row['candidate'] == 'content' and row['baseline'] == 'mean':
                if row['protected'] == 'False':
                    adverse.append({'scene': scene, **row})
                if scene == 'xuzhou' and int(row['class_id']) in (5, 6):
                    protected56.append({'scene': scene, **row})
        for row in rows(folder/'results/resident-cost.csv'):
            costs.append({'scene': scene, **row})
        count, invariant = 0, 0
        for cf in sorted(folder.glob('counterfactual-*/**/receipt.json')):
            receipt = read(cf)
            require(receipt['status'] == 'passed' and receipt['optimizer_updates'] == 0, 'Counterfactual failed')
            for item in receipt['cases']:
                source = Path(item['path'])
                require(hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256'], 'Changed counterfactual source')
                packet = read(source)
                require(packet['checkpoint_unchanged'] and packet['optimizer_updates'] == 0, 'State changed')
                seed = int(packet['case'].split('/')[1])
                for endpoint, controls in packet['comparisons'].items():
                    reference = controls['content']
                    require(len(reference['response']) == views, 'Response denominator differs')
                    for control, value in controls.items():
                        require(value['response'] == reference['response'], 'Same graph branch inputs/states changed across coefficient controls')
                        require(len(value['validation']['rows']) == views, 'View roster differs')
                        for a, b in zip(reference['validation']['rows'], value['validation']['rows'], strict=True):
                            require(a['heads']['cnn'] == b['heads']['cnn'] and a['released'] == b['released']
                                    and a['observation'] == b['observation'] and a['graph']['range_identity'] == b['graph']['range_identity'], 'Input/CNN/frozen support invariant failed')
                            if control != 'content':
                                invariant += 1
                    for index, row in enumerate(reference['response']):
                        event = reference['validation']['rows'][index]
                        responses.append({'scene': scene, 'seed': seed, 'endpoint': endpoint,
                            'tile': event['tile'], 'band': event['band'], **row})
                    count += 1
        require(count == 6 and invariant == 3*2*views*3, 'Missing counterfactual model/endpoints')
        scenes[scene] = {'real_updates': 12*steps, 'effect_gate_passed': result['primary_effect_passed'],
            'selected_steps': result['selected_steps'], 'same_weight_invariant_pairs': invariant,
            'response_rows': 3*2*views, 'metric_endpoints': 144, 'class_rows': 1296,
            'unique_spatial_releases': 1, 'validation_points_per_band': 45,
            'scope': 'short development comparison, no independent confirmation or cross-scale radius attribution'}
    out.mkdir()
    write(out/'primary-pairs.csv', primary)
    # Preserve a header even when there are no adverse class pairs.
    if adverse:
        write(out/'adverse-classes.csv', adverse)
    write(out/'xuzhou-classes-5-6.csv', protected56)
    write(out/'graph-response.csv', responses)
    write(out/'resident-cost.csv', costs)
    result = {'status': 'passed', 'real_optimizer_updates': 1464, 'diagnostic_optimizer_updates': 0,
        'scenes': scenes, 'adverse_class_pairs_including_duplicate_selected_last': len(adverse),
        'strict_nonzero_response_rows': sum(r['strict_nonzero_delta_cells'] > 0 for r in responses),
        'response_rows': len(responses), 'raw_ground_truth_accessed': False,
        'scope': 'WP12 readable evidence only; no compression/deployment or converged accuracy claim'}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.base.resolve(), args.out.resolve())))
