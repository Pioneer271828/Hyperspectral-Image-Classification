"""Frozen, label-blind M1 views for the calibrated joint benchmark.

Full-scene normalization and full-spectrum-derived Q/A are retained. This is
observation augmentation, not an unseen-sensor or spatial-inductive protocol.
"""
import copy
from dataclasses import asdict

import numpy as np
import torch

from joint_baseline import NAMES, _item_fingerprint
from joint_calibrated_selection import calibrated_settings, enabled, load_selected
from joint_training import schedule_for, train_step, validation_snapshot
from loadData.observation_sampling import BandMetadata, sample, replay
from loadData.spatial_protocol import PartitionObservation
from model.research_config import digest

VERSION = 'joint_real_band_views_v1'
PRESETS = {'full': [{'mode': 'none'}],
           'alternate75': [{'mode': 'none'}, {'mode': 'random_subset', 'keep_fraction': .75}]}


def _stub(metadata, plan_hash):
    # Only freeze band indices. No scene pixels or labels enter the sampler.
    return PartitionObservation('train', np.zeros((1, 1, len(metadata.band_ids)), np.float32),
                                np.zeros((1, 1, 2), np.float32), np.zeros((1, 1), np.int64), plan_hash)


def _metadata_check(item, meta):
    if (item['net_input'].ndim != 3 or item['net_input'].shape[-1] != len(meta.band_ids) + 2
            or item['bands'] != len(meta.band_ids)):
        raise ValueError('Joint view requires B spectral and two coordinate channels')
    wave = torch.as_tensor(item['wavelengths']).detach().cpu().double()
    if not torch.equal(wave, torch.tensor(meta.wavelengths, dtype=torch.float64)):
        raise ValueError('Wavelength metadata differs from the prepared input')
    widths = item['fwhm']
    if (widths is None) != (meta.fwhm is None):
        raise ValueError('FWHM presence differs')
    if widths is not None and not torch.equal(torch.as_tensor(widths).detach().cpu().double(),
                                             torch.tensor(meta.fwhm, dtype=torch.float64)):
        raise ValueError('FWHM metadata differs')
    if not all(meta.valid):
        raise ValueError('Joint baseline must be prefiltered to valid source bands')
    # The inherited full-scene calibration path still uses legacy auto units.
    # Reject ambiguous cases rather than silently changing validation semantics.
    auto_wave = 'normalized' if wave.max() <= 2 and wave.min() >= -.5 else 'nm'
    if auto_wave != meta.wavelength_unit:
        raise ValueError('Metadata unit is ambiguous for inherited calibration')
    if widths is not None:
        auto_width = 'normalized' if max(meta.fwhm) <= 2 else 'nm'
        if auto_width != meta.fwhm_unit:
            raise ValueError('FWHM unit is ambiguous for inherited calibration')


def _records(base, metadata, preset, seed, plan_hash):
    rng = np.random.default_rng(seed)
    configs = PRESETS[preset]
    records = []
    for i, name in enumerate(schedule_for(base)):
        _, record = sample(_stub(metadata[name], plan_hash), metadata[name],
                           configs[i % len(configs)], rng)
        records.append({'step': i + 1, 'dataset': name, 'observation': record})
    return records


def freeze_settings(settings, items, metadata, *, preset, sampling_seed, plan_hash):
    """Store the entire auditable schedule in settings BEFORE any model update."""
    if (preset not in PRESETS or type(sampling_seed) is not int or sampling_seed < 0
            or not isinstance(plan_hash, str) or not plan_hash or 'joint_observation' in settings):
        raise ValueError('Explicit registered preset, nonnegative seed and plan identity required')
    base = calibrated_settings(settings)
    if (base['config']['model']['region_residual_mode'] != 'none'
            or base['config']['model'].get('coverage_mode', 'none') != 'none'):
        raise ValueError('Do not combine M1 with another research factor')
    if tuple(items) != NAMES or tuple(metadata) != NAMES:
        raise ValueError('Three ordered task metadata entries required')
    for name in NAMES:
        if any(k.startswith('test_') for k in items[name]):
            raise ValueError('Test labels are not accepted')
        _metadata_check(items[name], metadata[name])
    policy = {'version': VERSION, 'preset': preset, 'sampling_seed': sampling_seed,
              'plan_hash': plan_hash, 'base_settings_hash': digest(base),
              'metadata': {n: asdict(metadata[n]) for n in NAMES},
              'inputs': {n: _item_fingerprint(items[n]) for n in NAMES},
              'records': _records(base, metadata, preset, sampling_seed, plan_hash),
              'calibration_and_selection': 'unchanged_full_observation',
              'graph_context': 'fixed_full_spectrum_derived',
              'normalization': 'select_existing_full_scene_normalized_columns'}
    # Canonical JSON-compatible lists avoid tuple/list differences after resume.
    import json
    policy = json.loads(json.dumps(policy, allow_nan=False))
    policy['policy_hash'] = digest(policy)
    return {**base, 'joint_observation': policy}


class JointObservationRuntime:
    train_step = staticmethod(train_step)
    validation_snapshot = staticmethod(validation_snapshot)

    def __init__(self, settings):
        self.settings = copy.deepcopy(settings)
        self.policy = self.settings['joint_observation']
        self.metadata = {n: BandMetadata(**m) for n, m in self.policy['metadata'].items()}
        self.validate(settings)

    def validate(self, settings):
        if settings != self.settings or not enabled(settings):
            raise ValueError('Joint observation settings mismatch or missing calibration')
        p = self.policy
        base = {k: v for k, v in settings.items() if k != 'joint_observation'}
        if (p.get('version') != VERSION or p.get('policy_hash') != digest({k: v for k, v in p.items() if k != 'policy_hash'})
                or p.get('base_settings_hash') != digest(base) or p.get('preset') not in PRESETS
                or base['config']['model']['region_residual_mode'] != 'none'
                or base['config']['model'].get('coverage_mode', 'none') != 'none'):
            raise ValueError('Frozen observation policy identity mismatch')
        expected = _records(base, self.metadata, p['preset'], p['sampling_seed'], p['plan_hash'])
        if p['records'] != expected:
            raise ValueError('Frozen observation sequence differs from registered recipe')
        return True

    def validate_inputs(self, items):
        if tuple(items) != NAMES or {n: _item_fingerprint(items[n]) for n in NAMES} != self.policy['inputs']:
            raise ValueError('Observation input/graph/label identity mismatch')
        for name in NAMES:
            _metadata_check(items[name], self.metadata[name])

    def build_model(self, config, counts):
        return config.build_model(counts)

    def view(self, index, name, item):
        if type(index) is not int or not 0 <= index < len(self.policy['records']):
            raise ValueError('Global update index outside frozen schedule')
        event = self.policy['records'][index]
        if event['dataset'] != name:
            raise ValueError('Dataset differs from frozen update schedule')
        meta = self.metadata[name]
        record = event['observation']
        selected = replay(_stub(meta, self.policy['plan_hash']), meta, record).bands
        positions = record['selected_positions']
        if positions == list(range(len(meta.band_ids))):
            return item, selected, record
        tensor = item['net_input']
        index_tensor = torch.tensor(positions, device=tensor.device)
        result = dict(item)
        result['net_input'] = torch.cat((tensor[..., :-2].index_select(-1, index_tensor), tensor[..., -2:]), -1)
        result['bands'] = len(positions)
        result['wavelengths'] = torch.as_tensor(item['wavelengths'], device=tensor.device).index_select(0, index_tensor)
        result['fwhm'] = (None if item['fwhm'] is None else
                          torch.as_tensor(item['fwhm'], device=tensor.device).index_select(0, index_tensor))
        return result, selected, record

    def train_step_at(self, index, model, optimizer, name, item, config):
        view, meta, record = self.view(index, name, item)
        def forward_view(model, name, item):
            return model(name, item['net_input'], item['Q'], item['A'],
                         wavelengths=item['wavelengths'], fwhm=item['fwhm'],
                         superpixel_scale=item['superpixel_scale'], wavelength_unit=meta.wavelength_unit,
                         fwhm_unit=meta.fwhm_unit or meta.wavelength_unit)
        result = train_step(model, optimizer, name, view, config, forward_fn=forward_view)
        result['observation'] = {'record_hash': record['record_hash'], 'bands': view['bands'],
                                 'source_bands': len(self.metadata[name].band_ids)}
        return result

    def load_selected(self, path, items, raw_source_path, *, device='cpu'):
        self.validate_inputs(items)
        model = load_selected(path, items, raw_source_path, device=device)
        saved = model.training_config['joint_runtime'].get('joint_observation')
        if saved != self.policy:
            raise ValueError('Checkpoint observation policy mismatch')
        return model
