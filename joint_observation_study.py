"""Registered M1-J2 budgets and label-blind fixed observation evaluation bank."""
import json
from pathlib import Path
import numpy as np
import torch

from joint_training import options
from joint_observation import freeze_settings, _stub, _metadata_check
from loadData.observation_sampling import sample, replay
from loadData.data_read import read_band_metadata
from model.research_config import digest

EXPECTED = {
    'protocol': 'm1_joint_paired_full_v1', 'presets': ['full', 'alternate75'],
    'training_seeds': [64, 512, 2048], 'sampling_seed_offset': 8128,
    'updates_per_case': 1503, 'evaluation_seed': 9026,
    'conditions': {'full': {'mode': 'none'}, 'random75': {'mode': 'random_subset', 'keep_fraction': .75},
                   'random50': {'mode': 'random_subset', 'keep_fraction': .5},
                   'stride2': {'mode': 'uniform_stride', 'stride': 2, 'offset': 0}},
    'evaluation_calibration': 'label_free_condition_matched_scene_population',
    'selection': 'unchanged_full_observation_validation', 'graph': 'fixed_full_spectrum_derived',
    'new_optimizer_updates': 9018, 'reuse_historical_training': False}


def load_study(path='configs/experiments/m1_joint_full.json'):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value != EXPECTED:
        raise ValueError('Modified M1-J2 preregistration; create a new protocol')
    return value


def case_settings(study, base_plan, base, items, provenance, preset, seed):
    if study != EXPECTED or preset not in study['presets'] or type(seed) is not int or seed not in study['training_seeds']:
        raise ValueError('Unregistered study/case')
    metadata = {n: read_band_metadata(n, i['bands']) for n, i in items.items()}
    return freeze_settings(options(base_plan, 'full', seed, base), items, metadata,
        preset=preset, sampling_seed=study['sampling_seed_offset']+seed,
        plan_hash=provenance['prepared_manifest_sha256'])


def evaluation_bank(study, metadata, plan_hash):
    if study != EXPECTED:
        raise ValueError('Unregistered evaluation bank')
    bank = {}
    for condition, config in study['conditions'].items():
        bank[condition] = {}
        for name, meta in metadata.items():
            seed = int(digest([study['protocol'], study['evaluation_seed'], condition, name])[:16], 16)
            _, record = sample(_stub(meta, plan_hash), meta, config, np.random.default_rng(seed))
            bank[condition][name] = record
    return bank


def condition_view(item, metadata, record, plan_hash):
    _metadata_check(item, metadata)
    selected = replay(_stub(metadata, plan_hash), metadata, record).bands
    positions = record['selected_positions']
    if positions == list(range(item['bands'])):
        return item
    tensor = item['net_input']
    index = torch.tensor(positions, device=tensor.device)
    view = dict(item)
    view.update(net_input=torch.cat((tensor[..., :-2].index_select(-1, index), tensor[..., -2:]), -1),
                bands=len(selected.band_ids),
                wavelengths=torch.as_tensor(item['wavelengths'], device=tensor.device).index_select(0, index),
                fwhm=None if item['fwhm'] is None else torch.as_tensor(item['fwhm'], device=tensor.device).index_select(0,index))
    return view
