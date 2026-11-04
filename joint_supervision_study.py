"""Frozen four-cell supervision experiment; no adaptive budget or seed selection."""
import json
from pathlib import Path
from joint_supervision import MODES, SUPERVISION, supervision_settings
from joint_training import options

EXPECTED = {
    'protocol': 'm4_supervision_factorial_full_v1', 'modes': list(MODES),
    'supervision': list(SUPERVISION), 'training_seeds': [64, 512, 2048],
    'updates_per_case': 1503, 'new_optimizer_updates': 18036,
    'reuse_historical_training': False,
    'selection': 'unchanged_calibrated_full_observation_validation',
    'graph': 'fixed_original_hypergraph',
    'finalization': 'exact_state_same_classes_probability_atol_1e-5_v1',
    'candidate': ['x_residual', 'primary_only'],
    'comparators': [['none', 'primary_only'], ['x_residual', 'legacy_three_heads']],
    'gate': {'macro_oa_mean_delta_gt': 0, 'positive_seed_count_gte': 2,
             'each_dataset_aa_floor': -.005, 'honghu_oa_floor': -.005,
             'small_class_max_train_labels': 25, 'each_small_class_recall_floor': -.02}}


def load_study(path='configs/experiments/m4_supervision_full.json'):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value != EXPECTED:
        raise ValueError('Modified preregistration; create a new protocol')
    return value


def case_settings(study, plan, base, mode, supervision, seed):
    if (study != EXPECTED or mode not in MODES or supervision not in SUPERVISION
            or type(seed) is not int or seed not in study['training_seeds']):
        raise ValueError('Unregistered study/case')
    return supervision_settings(options(plan, 'full', seed, base), mode, supervision)


def select_case(launch, source, study, mode, supervision, seed):
    if study != EXPECTED or launch['code'] != source or launch['study'] != study:
        raise ValueError('Wrong launch source or protocol')
    expected = {(m, s, n) for m in MODES for s in SUPERVISION for n in study['training_seeds']}
    cases = launch['cases']
    keys = [(c['mode'], c['supervision'], c['seed']) for c in cases]
    if len(keys) != 12 or set(keys) != expected or len({c['out'] for c in cases}) != 12:
        raise ValueError('Complete distinct twelve-case launch required')
    if any(type(c['seed']) is not int for c in cases) or (mode, supervision, seed) not in expected:
        raise ValueError('Unregistered case identity')
    return next(c for c in cases if (c['mode'], c['supervision'], c['seed']) == (mode, supervision, seed))
