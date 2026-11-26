"""Capture only the missing selected mean-coefficient point probabilities; zero updates."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.environ['WP07_GRAPH_SOURCE'])

import torch
from research_runtime import code_snapshot, require_frozen_code, tensor_hash
from scripts import run_native_released_points as worker
from scripts.run_wp07_b_pair import observation_only_resources
from transfer import wp07_fine16_training as training, wp07_graph_range as graph
from transfer import registered_development as dev
from transfer.native_scale_campaign import read, packet
from transfer import native_released_campaign as campaign
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale_audit import _exact


def pin(path):
    return dev._pin(path.resolve(strict=True))


def clean(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x.item()))


def main():
    graph.require_checks()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'core', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--expected-production-commit', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    code = code_snapshot()
    require_frozen_code(code)
    graph.require(code['commit'] == args.expected_production_commit and args.root in args.out.parents,
                  'Unexpected frozen producer or output root')
    graph.require(os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu and not args.out.exists(),
                  'Use the assigned GPU and a fresh attribution output')
    args.out.mkdir()
    cases_dir = args.out/'_private'
    cases_dir.mkdir()
    plans = {scene: read(pin(args.core/scene/'pair/plan.json'), args.root) for scene in ('paviaU', 'xuzhou')}
    program = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dev._json_new(args.out/'registration.json', {'producer': code['commit'], 'program_sha256': program,
        'plans': {scene: pin(args.core/scene/'pair/plan.json') for scene in plans},
        'models': 6, 'endpoint': 'selected_only_same_as_last', 'coefficient': 'mean',
        'optimizer_updates': 0, 'new_labels': 0, 'points_per_band': 45,
        'purpose': 'missing per-point granularity for preregistered A-C=(A-B)+(B-C), not repeated four-control analysis'})
    results, descriptors = [], {}
    with observation_only_resources(), worker.arithmetic(args.root, args.gpu) as execution, torch.no_grad():
        for scene, plan in plans.items():
            graph.require(plan['producer'] == code['commit'], 'Core plan producer differs')
            entries = packet(plan['cache'], args.root)['entries']
            receipt = read(plan['preflight'], args.root)
            for case in (c for c in plan['cases'] if c['arm'] == 'content'):
                graph.require(case['gpu'] == args.gpu, 'Original checkpoint used another physical device')
                dest = Path(case['destination'])
                spec_pin = pin(dest/'registration.json')
                spec = read(spec_pin, args.root)
                state_pin = read(pin(dest/'completed.json'), args.root)['checkpoint']
                state = packet(state_pin, args.root)
                graph.require(state['selected']['step'] == spec['steps']
                              and _exact(state['selected']['state'], state['model']), 'Do not add duplicate or changed endpoint')
                provider = training.BankProvider(args.root, receipt, case['seed'])
                run = training.RangeRun(spec, provider, entries, code['commit'], execution)
                initial = {n: v.detach().clone() for n, v in run.model.named_parameters()
                    if n.startswith(('backbone.range_content.', 'backbone.learnable_graph.feature_norm.'))}
                run.restore(state, spec['steps'])
                run.model.eval()
                before = run.pack()
                weights = dict(run.model.named_parameters())
                changes = [{'tensor': n, 'initial_hash': tensor_hash(v), 'trained_hash': tensor_hash(weights[n]),
                    'delta_l2': float((weights[n]-v).norm()), 'initial_l2': float(v.norm()),
                    'trained_l2': float(weights[n].norm())} for n, v in initial.items()]
                records = []

                def observe(event, actual, output, targets):
                    descriptor = clean({'metadata': dataclasses.asdict(actual.view.bands),
                        'actual_band_count': len(actual.view.bands.band_ids),
                        'original_positions': actual.receipt['source']['positions']})
                    key = scene+'/'+event['band']
                    if key in descriptors:
                        graph.require(descriptors[key] == descriptor, 'Band declaration changed across views/models')
                    descriptors[key] = descriptor
                    y0, _, x0, _ = actual.lattice.box
                    width = actual.lattice.scene_shape[1]
                    grid_width = actual.lattice.shape[1]
                    graph.require(actual.lattice.factor == 1, 'Attribution is factor1 only')
                    for head, tensor in zip(training.HEADS, output[:3], strict=True):
                        probabilities = tensor.detach().cpu().numpy()
                        for ordinal, (pixel, label) in enumerate(targets.points):
                            y, x = divmod(pixel, width)
                            cell = (y-y0)*grid_width+(x-x0)
                            records.append({'tile': event['tile'], 'released_ordinal': ordinal,
                                'native_index': cell, 'head': head, 'band': event['band'],
                                'class_id': label, 'probabilities': probabilities[cell].astype(float).tolist()})

                with run.selector_intervention('mean'):
                    measured = training.evaluate_validation(run.model, spec['validation'], spec['steps'],
                        lambda e: run._materialize('devval', e), run._forward, released=provider,
                        detailed=True, observe=observe)
                metrics = campaign.scoring(measured, 9)
                old_path = args.core/scene/'counterfactual-gpu3-output'/(case['id'].replace('/', '-')+'.json')
                old = read(pin(old_path), args.root)['comparisons']['selected']['mean']
                graph.require(metrics == old['metrics'] and len(records) == 270, 'Existing mean intervention or point denominator differs')
                for a, b in zip(measured['rows'], old['validation']['rows'], strict=True):
                    graph.require(a['heads'] == b['heads'] and a['observation'] == b['observation']
                                  and a['released'] == b['released'] and a['graph']['range_identity'] == b['graph']['range_identity'],
                                  'Existing mean inference inputs/outputs differ')
                independent = read(pin(dest/'independent.json'), args.root)['endpoints']['selected']['original_point_predictions']
                key = lambda r: (r['tile'], r['released_ordinal'], r['head'], r['band'])
                reference = {key(r): r for r in independent}
                for r in records:
                    graph.require(r['class_id'] == reference[key(r)]['class_id'], 'Point labels differ')
                    if r['head'] == 'cnn':
                        graph.require(r['probabilities'] == reference[key(r)]['probabilities'], 'CNN point probabilities changed under coefficients')
                graph.require(_exact(before, run.pack()) and hashlib.sha256(Path(state_pin['path']).read_bytes()).hexdigest() == state_pin['sha256'],
                              'Inference changed model/optimizer/RNG or original file')
                gradients = [{'step': r['step'], 'band': r['band'], **next(iter(r['fresh'].values()))} for r in state['history']]
                result = dev._json_new(cases_dir/(case['id'].replace('/', '-')+'.json'), {
                    'scene': scene, 'seed': case['seed'], 'producer': code['commit'], 'program_sha256': program,
                    'registration': spec_pin, 'checkpoint': state_pin, 'checkpoint_state': state_hash(run.model),
                    'original_mean_counterfactual': pin(old_path), 'points': records, 'metrics': metrics,
                    'parameter_changes': changes, 'gate_history': gradients, 'optimizer_updates': 0,
                    'checkpoint_model_optimizer_RNG_unchanged': True, 'CNN_points_bitwise_invariant': True})
                results.append(result)
                print(json.dumps({'case': case['id'], 'point_rows': len(records), 'optimizer_updates': 0}), flush=True)
    graph.require(len(results) == 6, 'Missing selected model')
    dev._json_new(args.out/'band-descriptors.json', descriptors)
    result = dev._json_new(args.out/'receipt.json', {'status': 'passed', 'producer': code['commit'],
        'program_sha256': program, 'cases': results, 'point_rows': 1620, 'optimizer_updates': 0,
        'classifier_head_new_training': False, 'new_labels': 0, 'query_accessed': False, 'raw_GT_opened': False})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
