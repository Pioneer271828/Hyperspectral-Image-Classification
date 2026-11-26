"""Frozen exact-position synthetic matrix; reuse the B phase executor and independent point auditor."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from scripts.run_native_released_points import arithmetic
from scripts.run_wp07_condition_integration import save, table
from transfer import registered_development as dev
from transfer import wp07_exact_positions as position
from transfer import wp07_position_run as matrix
from transfer import wp07_condition_run as previous
from transfer import wp07_fine16_training as phases
from transfer import wp07_graph_range as graph
from transfer.g3_training import state_hash
from transfer.native_scale_campaign import read, packet
from transfer.xuzhou_multiscale_audit import _exact


def metadata(path, value, root):
    if path.exists():
        pin = dev._pin(path)
        graph.require(read(pin, root) == value, 'Saved metadata differs')
        return pin
    return dev._json_new(path, value)


@torch.no_grad()
def controls(engine, provider):
    rows = []
    engine.model.eval()
    before = state_hash(engine.model)
    rng = torch.get_rng_state().clone()
    for event in engine.registered['validation']:
        actual, _, _ = engine._materialize('devval', event)
        for scope in ('both', 'cnn', 'graph'):
            for mode in ('mean', 'content', 'metadata', 'joint'):
                policy = scope + '.' + mode
                with engine.policy_intervention(policy):
                    original, trace, _ = engine._forward(actual, 'devval', 'matched')
                    for intervention in ('paired', 'missing'):
                        with provider.intervention(intervention):
                            changed, changed_trace, _ = engine._forward(actual, 'devval', 'matched')
                        exact = [torch.equal(x, y) for x, y in zip(original[:3], changed[:3])]
                        graph.require(trace['range_identity'] == changed_trace['range_identity'], 'Descriptor altered Q/A')
                        if mode in ('mean', 'content') or (intervention == 'paired' and provider.representation in ('summary8', 'balanced_summary') and event['band'].startswith('interior')):
                            graph.require(all(exact), 'Condition-negative prediction changed')
                        if scope != 'both':
                            graph.require(exact[1 if scope == 'cnn' else 2], 'Other classifier branch changed')
                        rows.append(dict(tile=event['tile'], band=event['band'], policy=policy,
                                         intervention=intervention, fused_exact=exact[0], graph_exact=exact[1], cnn_exact=exact[2],
                                         maximum_probability_delta=max(float((x-y).abs().max()) for x, y in zip(original[:3], changed[:3])),
                                         selector_weights_exact=trace['weights'] == changed_trace['weights'], QA_exact=True))
            for mode, fallback in (('metadata', 'mean'), ('joint', 'content')):
                with engine.policy_intervention(scope + '.' + mode), provider.intervention('missing'):
                    missing, _, _ = engine._forward(actual, 'devval', 'matched')
                with engine.policy_intervention(scope + '.' + fallback):
                    reference, _, _ = engine._forward(actual, 'devval', 'matched')
                graph.require(all(torch.equal(x, y) for x, y in zip(missing[:3], reference[:3])), 'Missing fallback differs')
    graph.require(state_hash(engine.model) == before and torch.equal(rng, torch.get_rng_state()), 'Read-only controls changed model or RNG')
    return rows


def fixtures(out, code, representations=position.REPRESENTATIONS):
    out.mkdir(exist_ok=True)
    header = metadata(out/'registration.json', dict(producer=code['commit'], source_hash=code['source_hash'],
        classes=[9, 22], seeds=[64, 512, 2048], representations=list(representations),
        arms=list(matrix.ARMS), synthetic_cases=48, steps=24, routes=['full', '12+12', 'completed0'],
        maximum_synthetic_optimizer_updates=2304, maximum_real_optimizer_updates=0,
        positions={k:list(v) for k,v in position.POSITION_SETS.items()}, selection='first_min_fused_NLL_0_12_24'), out)
    endpoints, classes_table, participation, costs, control_rows, cases = [], [], [], [], [], []
    new_updates = reused_updates = 0
    started = time.perf_counter()
    with arithmetic(out, None) as execution:
        for classes in (9, 22):
            reference = matrix.FixtureProvider(classes, 'summary8')
            cachepath = out/f'c{classes}-geometry.pt'
            if cachepath.exists():
                cachepin = dev._pin(cachepath)
                entries = packet(cachepin, out)['entries']
            else:
                entries = matrix.cache(reference, code['commit'])
                cachepin = save(cachepath, {'entries': entries})
            initials = defaultdict(set)
            for seed in (64, 512, 2048):
                for representation in representations:
                    provider = matrix.FixtureProvider(classes, representation)
                    for arm in matrix.ARMS:
                        case = f'c{classes}-s{seed}-{representation}-{arm}'
                        dest = out/case
                        dest.mkdir(exist_ok=True)
                        spec = matrix.contract(provider, seed, arm, code['commit'], execution.numeric, cachepin,
                                               {'synthetic': True, 'registration': header})
                        specpin = metadata(dest/'registration.json', spec, out)
                        initials[seed].add(spec['initialization']['state'])
                        states = {}
                        for phase, prior in (('full', None), ('prefix', None), ('resume', 'prefix'), ('completed', 'resume')):
                            path = dest/(phase + '.pt')
                            if path.exists():
                                receipt = read(dev._pin(dest/(phase+'.json')), out)
                                states[phase] = packet(receipt['checkpoint'], out)
                                reused_updates += receipt['updates']
                            else:
                                receipt = phases.execute(spec, phase, path, provider, entries, code['commit'], execution,
                                    prior=None if prior is None else states[prior], run_factory=matrix.PositionRun)
                                states[phase] = packet(dev._pin(path), out)
                                receipt['checkpoint'] = dev._pin(path)
                                dev._json_new(dest/(phase+'.json'), receipt)
                                new_updates += receipt['updates']
                        graph.require(_exact(states['full'], states['resume']) and _exact(states['full'], states['completed']), 'Saved route states differ')
                        auditpath = dest/'independent.json'
                        if auditpath.exists():
                            audit = read(dev._pin(auditpath), out)
                        else:
                            audit = phases.audit(spec, states['full'], provider, entries, code['commit'], execution,
                                                 cost=seed == 64, run_factory=matrix.PositionRun, point_context=('profile',))
                            dev._json_new(auditpath, audit)
                        engine = matrix.PositionRun(spec, provider, entries, code['commit'], execution)
                        engine.restore(states['full'], 24)
                        controlpath = dest/'controls.json'
                        if controlpath.exists():
                            observed = read(dev._pin(controlpath), out)
                        else:
                            observed = controls(engine, provider)
                            dev._json_new(controlpath, observed)
                        base = dict(case=case, classes=classes, seed=seed, representation=representation, policy=arm)
                        control_rows.extend({**base, **row} for row in observed)
                        for endpoint, value in audit['endpoints'].items():
                            for head, bands in value['metrics'].items():
                                for band, metrics in bands.items():
                                    common = dict(**base, endpoint=endpoint, head=head, band=band)
                                    endpoints.append(dict(**common, **{k:metrics[k] for k in ('count', 'oa', 'aa', 'nll')}))
                                    for i, cls in enumerate(metrics['per_class']):
                                        classes_table.append(dict(**common, class_id=cls['class_id'], support=cls['support'],
                                                                  correct=metrics['confusion_matrix'][i][i], recall=cls['recall']))
                        for row in audit['cost']:
                            costs.append({**base, **{k:v for k,v in row.items() if k != 'samples_ms'}})
                        initial = position.initialize(provider.baseline(seed), spec['module_seed']).state_dict()
                        named = dict(engine.model.named_parameters())
                        for name, tensor in named.items():
                            events = [next(r for r in event['parameter_participation'] if r['tensor'] == name) for event in states['full']['history']]
                            participation.append(dict(**base, tensor=name, allocated=tensor.numel(),
                                final_changed=int(torch.count_nonzero(tensor.detach().cpu() - initial[name])),
                                nonzero_gradient_element_step_sum=sum(e['nonzero_gradient_elements'] for e in events)))
                        cases.append(dict(**base, registration=specpin, selected_step=audit['selected_step'],
                                          allocated_parameters=spec['parameters'], full_split_completed_exact=True,
                                          synthetic_updates=48, real_updates=0))
                        print(json.dumps({'case':case, 'status':'passed'}), flush=True)
            graph.require(all(len(values) == 1 for values in initials.values()), 'Representation or policy changed common initialization')
    for name, rows in (('endpoints',endpoints), ('per-class',classes_table), ('parameter-participation',participation), ('resident-cost',costs), ('controls',control_rows)):
        table(out/(name+'.csv'), rows)
    dev._json_new(out/'summary.json', dict(status='passed', producer=code['commit'], source_hash=code['source_hash'],
        cases=cases, synthetic_optimizer_updates=2304, new_updates_this_invocation=new_updates,
        reused_committed_updates=reused_updates, real_optimizer_updates=0,
        confusion_matrices_recounted=len(endpoints), elapsed_seconds=time.perf_counter()-started,
        scope='Synthetic engineering; neither independent real accuracy nor sensor confirmation'))


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--representation-family', choices=('legacy', 'balanced'), default='legacy')
    a = p.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    graph.require(code['commit'] == a.expected_commit, 'Wrong frozen source')
    graph.require(Path(__file__).resolve().parents[1] not in a.out.resolve().parents, 'Output inside checkout')
    fixtures(a.out, code, position.REPRESENTATIONS if a.representation_family == 'legacy' else position.BALANCED_REPRESENTATIONS)


if __name__ == '__main__':
    main()
