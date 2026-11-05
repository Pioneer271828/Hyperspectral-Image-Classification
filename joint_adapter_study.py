"""Preregistered existing VCA mode comparison; no combination or budget search."""
import json
from pathlib import Path
from joint_adapter_modes import MODES, adapter_settings
from joint_training import options

EXPECTED = {
    'protocol': 'm2_joint_adapter_modes_full_v1', 'modes': list(MODES), 'training_seeds': [64,512,2048],
    'updates_per_case': 1503, 'new_optimizer_updates': 13527, 'reuse_historical_training': False,
    'supervision': 'legacy_three_heads', 'residual': 'none', 'observation': 'full',
    'selection': 'unchanged_calibrated_full_observation_validation',
    'metadata_quality': 'provided_approximate_not_assumed_measured_no_fwhm',
    'capacity': 'identical_registered_parameters_uniform_attention_inactive',
    'finalization': 'exact_state_same_classes_probability_atol_1e-5_v1',
    'candidate': 'wavelength', 'comparators': ['index','uniform'],
    'gate': {'macro_oa_mean_delta_gt':0,'positive_seed_count_gte':2,'each_dataset_aa_floor':-.005,
             'honghu_oa_floor':-.005,'small_class_max_train_labels':25,'each_small_class_recall_floor':-.02}}


def load_study(path='configs/experiments/m2_joint_full.json'):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value != EXPECTED: raise ValueError('Modified M2-J2 preregistration')
    return value


def case_settings(study, plan, base, mode, seed):
    if study != EXPECTED or mode not in MODES or type(seed) is not int or seed not in study['training_seeds']:
        raise ValueError('Unregistered M2 case')
    return adapter_settings(options(plan,'full',seed,base),mode)


def select_case(launch, source, study, mode, seed):
    if study != EXPECTED or launch['code'] != source or launch['study'] != study:
        raise ValueError('Wrong M2 source or protocol')
    expected = {(m,s) for m in MODES for s in study['training_seeds']}
    cases = launch['cases']; keys = [(c['mode'],c['seed']) for c in cases]
    if (len(cases) != 9 or set(keys) != expected or len({c['out'] for c in cases}) != 9
            or any(type(c['seed']) is not int for c in cases) or type(seed) is not int or (mode,seed) not in expected):
        raise ValueError('Complete distinct nine-case launch required')
    return next(c for c in cases if (c['mode'],c['seed']) == (mode,seed))
