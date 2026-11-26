"""One predeclared old fine16 checkpoint, original GPU, existing full/even only; zero updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from joint_training import _rng_state
from research_runtime import code_snapshot, require_frozen_code, tensor_hash
from scripts.probe_wp07_observed_conditions import Provider, pin
from scripts.run_wp07_b_pair import observation_only_resources
from scripts.run_wp07_condition_integration import table
from scripts import run_native_released_points as worker
from transfer import wp07_exact_positions as position
from transfer import wp07_observed_conditions as conditions
from transfer import wp07_fine16_training as frozen
from transfer import wp07_graph_range as graph
from transfer import native_released_campaign as campaign
from transfer import registered_development as dev
from transfer.native_scale_campaign import read, packet
from transfer.native_scale_training import evaluate_validation
from transfer.xuzhou_multiscale_audit import _exact


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'core', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--gpu', required=True)
    a = parser.parse_args()
    code = code_snapshot(); require_frozen_code(code)
    graph.require(code['commit'] == a.expected_commit and os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu, 'Wrong source or GPU')
    a.out.mkdir(); private = a.out/'_private'; private.mkdir()
    plan = read(pin(a.core/'xuzhou/pair/plan.json'), a.root)
    case = next(c for c in plan['cases'] if c['seed'] == 512 and c['arm'] == 'content')
    graph.require(case['gpu'] == a.gpu, 'Use the original physical GPU only')
    dest = Path(case['destination'])
    spec = read(pin(dest/'registration.json'), a.root)
    checkpoint = read(pin(dest/'completed.json'), a.root)['checkpoint']
    state = packet(checkpoint, a.root)
    historical = read(pin(dest/'independent.json'), a.root)['endpoints']['selected']['original_point_predictions']
    graph.require(state['selected']['step'] == spec['steps'] and _exact(state['selected']['state'], state['model']), 'Endpoint must be the previously audited selected=last')
    receipt = read(plan['preflight'], a.root)
    records = []
    with observation_only_resources(), worker.arithmetic(a.root, a.gpu) as execution, torch.no_grad():
        graph.require(execution.numeric == spec['numeric'], 'Historical arithmetic identity changed')
        provider = Provider(a.root, receipt, 512)
        entries = packet(plan['cache'], a.root)['entries']
        engine = frozen.RangeRun(spec, provider, entries, spec['producer'], execution)
        engine.restore(state, spec['steps']); model = engine.model; model.eval()
        original = {n:tensor_hash(t) for n,t in model.state_dict().items()}
        rng = _rng_state(execution.device)
        conditions.install_encoders(model, spec['module_seed'] + 41017)
        position.extend(model, spec['module_seed'] + 27183)
        graph.require(_exact(rng, _rng_state(execution.device)), 'Encoder insertion changed CPU/CUDA RNG')
        inserted = {n:tensor_hash(t) for n,t in model.state_dict().items()}

        def evaluate(representation, control, *, neutral=False):
            provider.override = control
            points = []

            def observe(event, actual, output, targets):
                y0, _, x0, _ = actual.lattice.box
                width, gw = actual.lattice.scene_shape[1], actual.lattice.shape[1]
                factor = actual.lattice.factor
                for head, probabilities in zip(frozen.HEADS, output[:3], strict=True):
                    for ordinal, (pixel, label) in enumerate(targets.points):
                        y, x = divmod(pixel, width)
                        cell = ((y-y0)//factor)*gw + (x-x0)//factor
                        points.append(dict(tile=event['tile'], released_ordinal=ordinal, head=head,
                            band=event['band'], class_id=label, probabilities=probabilities[cell].cpu().double().tolist()))

            with position.representation(model, representation), conditions.selection_policy(model, 'content', 'joint'):
                value = evaluate_validation(model, spec['validation'], spec['steps'],
                    lambda event: engine._materialize('devval', event), engine._forward,
                    released=provider, detailed=True, observe=observe)
            graph.require(len(points) == 270, 'Original point denominator changed')
            return dict(representation=representation, control=control, neutral=neutral,
                        metrics=campaign.scoring(value, 9), validation=value, points=points)

        reference = evaluate('exact_mask', 'missing')
        key = lambda p:(p['tile'], p['released_ordinal'], p['head'], p['band'])
        old = {key(p):p for p in historical}
        graph.require(all(p['probabilities'] == old[key(p)]['probabilities'] and p['class_id'] == old[key(p)]['class_id'] for p in reference['points']), 'Missing graft differs from historical probabilities')
        records.append(reference)
        for representation in position.REPRESENTATIONS:
            for control in ('matched', 'swap_band', 'unknown_wavelength', 'missing_gsd'):
                record = evaluate(representation, control)
                for row, base in zip(record['validation']['rows'], reference['validation']['rows'], strict=True):
                    graph.require((row['observation'], row['released'], row['graph']['range_identity'], row['heads']['cnn']) ==
                                  (base['observation'], base['released'], base['graph']['range_identity'], base['heads']['cnn']),
                                  'Descriptor changed content, points, frozen Q/A or other classifier branch')
                records.append(record)
            group = {r['control']:r for r in records if r['representation'] == representation and r['control'] != 'missing'}
            graph.require(group['matched']['points'] == group['unknown_wavelength']['points'] == group['missing_gsd']['points'], 'Unknown physical metadata promoted')
        model.backbone.range_metadata.weight.zero_()
        neutral = evaluate('exact_mask', 'matched', neutral=True)
        graph.require(neutral['points'] == reference['points'], 'Neutral exact-mask graft differs')
        records.append(neutral)
        graph.require(all(tensor_hash(model.state_dict()[n]) == value for n,value in original.items()), 'Old tensor changed')
        graph.require(all(tensor_hash(model.state_dict()[n]) == value for n,value in inserted.items() if not n.startswith('backbone.range_metadata.')), 'Inserted encoder changed in inference')
        graph.require(_exact(rng, _rng_state(execution.device)), 'Probe changed RNG')
        graph.require(hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest() == checkpoint['sha256'], 'Original checkpoint bytes changed')
    rows = []
    for record in records:
        for head, bands in record['metrics'].items():
            for band, metrics in bands.items():
                base = reference['metrics'][head][band]
                rows.append(dict(scene='xuzhou', seed=512, representation=record['representation'], control=record['control'],
                    neutral=record['neutral'], head=head, band=band, count=metrics['count'], oa=metrics['oa'], nll=metrics['nll'],
                    oa_delta=metrics['oa']-base['oa'], nll_delta=metrics['nll']-base['nll'], condition_parameters_trained=False))
    table(a.out/'endpoints.csv', rows)
    dev._json_new(private/'point-records.json', dict(checkpoint=checkpoint, original_producer=spec['producer'],
        executor=code['commit'], records=records, optimizer_updates=0, condition_parameters_trained=False))
    dev._json_new(a.out/'summary.json', dict(status='passed', executor=code['commit'], source_hash=code['source_hash'],
        checkpoint=checkpoint, preflight=plan['preflight'], physical_gpu=a.gpu, original_tensors_and_file_unchanged=True,
        historical_missing_and_neutral_bitwise_equal_vectors=270, source_checkpoint_models=1,
        records=len(records), endpoint_confusions=len(rows), insertion_and_inference_CPU_CUDA_RNG_unchanged=True,
        geometry_and_CNN_branch_exact=True, real_optimizer_updates=0, new_real_views_materialized=0,
        condition_parameters_trained=False, query_accessed=False, raw_GT_opened=False,
        scope='Fixed-seed untrained-condition insertion on one old graph-content model; existing full/even views only; no efficacy claim'))


if __name__ == '__main__':
    main()
