"""Review accepted P4 fitting ledgers and their independently replayed predictions."""
from collections import defaultdict
from fractions import Fraction
from statistics import mean
from pathlib import Path

import numpy as np

from evaluation.spatial_prediction import paired_difference
from evaluation.wp12_cost_primitives import timing_summary
from research_runtime import hash_file, require_frozen_code, write_json
from transfer import p4_coverage_batch as batch, p4_runtime_device as device, p4_three_arm_contract as contract
from transfer.g2_contract import read_pinned


HEADS = ('fused', 'graph', 'cnn')
SCHEMA = 'wp05_honghu_three_arm_analysis_v1'


def slope(steps, values):
    """Least-squares loss change per 100 updates on the last fifth of observations."""
    width = max(2, (len(values) + 4) // 5)
    axis = np.asarray(steps[-width:], dtype=np.float64)
    axis -= axis.mean()
    return float(100 * (axis @ np.asarray(values[-width:], dtype=np.float64)) / (axis @ axis))


def fitting_evidence(state, receipts, sequence, chosen, observations, resume):
    trace, history = state['trace'], state['history']
    nll = [e['metrics']['fused']['nll'] for e in history]
    totals = [r['loss']['total'] for r in trace]
    terms = set(trace[0]['loss'])
    device.require(all(r['loss'].keys() == terms for r in trace), 'Training loss components changed during fitting')
    timings = [row for call in receipts for row in call['timing']['steps']]
    device.require([row['step'] for row in timings] == list(range(1, len(sequence) + 1)),
                   'Missing synchronized fitting step timings')
    peaks = [row['train_peak_memory'] for row in timings if row['train_peak_memory'] is not None]
    for name, observation in observations.items():
        device.require(set(observation['heads']) == set(HEADS), f'Missing three heads for {name}')
        for head in observation['heads'].values():
            support = [sum(row) for row in head['confusion']]
            device.require(support == [5] * len(support), 'Fitting development class supports differ')
            head.update(class_support=support,
                        zero_recall_classes=[j for j, recall in enumerate(head['recall'], 1) if recall == 0])
    return {
        'selected_is_last': chosen['step'] == state['cursor'],
        'train_loss': {'first': trace[0]['loss'], 'last': trace[-1]['loss'],
                       'terminal_slope_per_100_steps': slope(range(1, len(trace) + 1), totals)},
        'validation_nll': {'first': nll[0], 'last': nll[-1],
                           'terminal_slope_per_100_steps': slope([e['step'] for e in history], nll),
                           'events': [{'step': e['step'], 'fused_nll': e['metrics']['fused']['nll'],
                                       'oa': {head: e['metrics'][head]['oa'] for head in HEADS}} for e in history]},
        'resume': resume,
        'cost': {'worker_wall_seconds': sum(call['wall_seconds'] for call in receipts),
                 'train_step': timing_summary([1000*r['train_with_hooks_seconds'] for r in timings],
                                              repeats=len(timings), blocks=1),
                 'phase_seconds': {name: sum(row[name] for row in timings) for name in (
                     'preparation_seconds', 'train_with_hooks_seconds', 'trace_evidence_seconds',
                     'validation_seconds', 'checkpoint_seconds', 'end_to_end_step_seconds')},
                 'train_peak_cuda': ({key: max(row[key] for row in peaks) for key in peaks[0]} if peaks else None)},
    }


def paired_outcomes(rows, *, train_support_per_class=5):
    cohort = defaultdict(dict)
    for row in rows:
        case = row['case']
        device.require(case['model_seed'] not in cohort[case['arm']], 'Duplicate fitted seed')
        cohort[case['arm']][case['model_seed']] = row['observations']
    device.require(set(cohort) == set(contract.ARMS)
                   and all(set(arm) == set(contract.SEEDS) for arm in cohort.values()),
                   'The complete nine-case fitting matrix is required for paired outcomes')

    def head(arm, seed, condition):
        return cohort[arm][seed][condition]['heads']['fused']

    def fractions(arm, condition, metric):
        return {seed: Fraction(str(head(arm, seed, condition)[metric])) for seed in contract.SEEDS}

    def primary(arm):
        return {seed: mean(Fraction(str(head(arm, seed, name)['oa'])) for name in contract.UNSEEN)
                for seed in contract.SEEDS}

    comparisons = {}
    for baseline in ('none', 'anchor_only'):
        left, right = primary('continuous'), primary(baseline)
        paired = paired_difference({seed: float(val) for seed, val in left.items()},
                                   {seed: float(val) for seed, val in right.items()}, contract.SEEDS)
        deltas = [left[seed] - right[seed] for seed in contract.SEEDS]
        primary_passed = mean(deltas) > 0 and sum(delta > 0 for delta in deltas) >= contract.PROTECTION['minimum_positive_seeds']
        rows_by_condition = {}
        for condition in contract.CONDITIONS:
            effects, protected = {}, True
            for metric in ('oa', 'aa'):
                left, right = (fractions(arm, condition, metric) for arm in ('continuous', baseline))
                delta = 100 * mean(left[seed] - right[seed] for seed in contract.SEEDS)
                effects[metric] = paired_difference({seed: float(val) for seed, val in left.items()},
                                                    {seed: float(val) for seed, val in right.items()}, contract.SEEDS)
                protected &= delta >= -Fraction(str(contract.PROTECTION['max_oa_aa_drop_pp']))
            recalls = [head('continuous', seed, condition)['recall'] for seed in contract.SEEDS]
            references = [head(baseline, seed, condition)['recall'] for seed in contract.SEEDS]
            recall_pp = [100 * mean(Fraction(str(a)) - Fraction(str(b)) for a, b in zip(now, old, strict=True))
                         for now, old in zip(zip(*recalls, strict=True), zip(*references, strict=True), strict=True)]
            failed = [i for i, delta in enumerate(recall_pp, 1)
                      if train_support_per_class <= contract.PROTECTION['max_small_train_support']
                      and delta < -contract.PROTECTION['max_small_recall_drop_pp']]
            rows_by_condition[condition] = {**effects, 'mean_recall_delta_pp': [float(v) for v in recall_pp],
                'small_classes_below_threshold': failed, 'passed': not failed and protected}
        comparisons[baseline] = {'primary': paired, 'primary_passed': primary_passed, 'conditions': rows_by_condition,
            'passed': primary_passed and all(row['passed'] for row in rows_by_condition.values())}
    return comparisons


def review(scope_path, scope_sha, execution, acceptance_path, acceptance_sha, out, code):
    """Authenticate the full accepted matrix before publishing a fresh result."""
    from transfer import p4_full_budget_campaign as campaign, p4_full_budget_registration as registration

    scope, _ = registration.load(scope_path, scope_sha, code)
    fit = scope.get('fit')
    device.require(fit is not None and fit['variant'] != 'long_synthetic',
                   'A complete three-arm matrix is required for fitting review')
    contract.registered_for_dispatch(scope, execution)
    execution = batch.sibling(execution, scope_path)
    acceptance_path = Path(acceptance_path).resolve()
    device.require(acceptance_path == execution / 'acceptance.json', 'Fitting acceptance path differs')
    receipt = read_pinned(acceptance_path, acceptance_sha)
    device.require(receipt['status'] == 'passed' and receipt['scope_sha256'] == scope_sha
                   and receipt['producer_commit'] == code['commit'] and receipt['fit_variant'] == fit['variant']
                   and receipt['cases'] == 9 and receipt['pairs'] == 3
                   and receipt['optimizer_updates'] == scope['budget']['planned_optimizer_updates']
                   and receipt['audit_optimizer_updates'] == 0
                   and receipt['raw_gt_loaded'] is receipt['query_accessed'] is False,
                   'Fitting review requires the complete accepted nine-case matrix')
    with device.lease(execution / '.coordinator.lock'):
        rows, pins = [], {**receipt['protected_files'], str(acceptance_path): acceptance_sha}
        expected_audits = {str(campaign.bound_folder(execution, case) / 'audit.json') for case in scope['cases']}
        device.require(set(receipt['case_audits']) == expected_audits
                       and all(pins.get(k) == v for k, v in receipt['case_audits'].items()),
                       'Fitting audit inventory differs')
        batch.check_pins(pins)
        for case in scope['cases']:
            folder = campaign.bound_folder(execution, case)
            row = campaign.accepted(folder, scope, case, scope_sha)
            device.require(row['class_count'] == fit['classes'] and row['observations'].keys() == set(contract.CONDITIONS)
                           and row['coverage_participation']['parameter_count'] == (0 if case['arm'] == 'none' else 194),
                           'Fitting diagnostic scope or parameter capacity differs')
            rows.append(row)
        support = {row['populations']['train']['per_class'] for row in rows}
        device.require(support == {contract.ROLE[1]}, 'Fitting train support differs')
        comparisons = paired_outcomes(rows, train_support_per_class=contract.ROLE[1])
        # The readouts come from the independent worker replay, with one fixed selected state per case.
        result = {'schema': SCHEMA, 'status': 'passed', 'scope_sha256': scope_sha,
                  'acceptance_sha256': acceptance_sha, 'producer_commit': code['commit'],
                  'synthetic': scope['synthetic'], 'classes': fit['classes'], 'optimizer_updates': 0,
                  'verified_training_updates': sum(r['optimizer_updates'] for r in rows),
                  'last_step_selections': sum(row['fitting']['selected_is_last'] for row in rows),
                  'primary_conditions': list(contract.UNSEEN), 'protection': contract.PROTECTION,
                  'comparisons': comparisons,
                  'hypothesis_supported': (all(row['passed'] for row in comparisons.values())
                                           if fit['research_inference'] else None),
                  'rows': rows, 'raw_gt_loaded': False, 'query_accessed': False}
        batch.check_pins(pins)
        require_frozen_code(code)
        out = batch.sibling(out, scope_path)
        out.mkdir(exist_ok=False)
        path = out / 'review.json'
        write_json(path, result)
    return path
