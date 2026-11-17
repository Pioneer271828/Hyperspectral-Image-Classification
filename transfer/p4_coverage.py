"""Register the five existing E1 arms for one matched guarded HongHu observation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from joint_training import _tree_fingerprint
from loadData.data_read import read_band_metadata
from loadData.observation_sampling import sample
from loadData.spatial_protocol import PartitionObservation
from model.coverage_config import CONDITIONS, COVERAGE_DEFAULTS, coverage_options
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from train_spatial import SpatialConfig
from transfer import p4_exposure, p4_guarded
from transfer.g2_contract import read_pinned


POLICY = 'p4_guarded_matched_coverage_v1'
ARMS = ('none', *sorted(CONDITIONS))
METADATA_MODES = ('nominal', 'missing', 'shift10nm')
PREFIX = 'backbone.coverage_modulation.'
SELECTION = 'first_minimum_fused_validation_nll'


def validate_base(config):
    model = config.research.model
    if coverage_options(asdict(model)) != COVERAGE_DEFAULTS or model.adapter_attention_mode != 'wavelength':
        raise ValueError('Matched coverage requires the wavelength baseline with default E1 settings')


def freeze(registration, exposure, bands, *, observation, seed, metadata):
    if (registration['config']['dataset'] != p4_guarded.SCENE or 'guarded' not in registration
            or exposure['policy'] != p4_exposure.POLICY
            or exposure['geometry_sha256'] != registration['pins'][registration['development']]
            or exposure['plan_hash'] != digest({k: v for k, v in exposure.items() if k != 'plan_hash'})):
        raise ValueError('Matched coverage requires a pinned guarded HongHu exposure plan')
    if (metadata not in METADATA_MODES or type(seed) is not int or not 0 <= seed < 2**32
            or bands.wavelength_unit != 'nm' or bands.wavelength_quality == 'index_only'):
        raise ValueError('Expected explicit physical band selection, metadata mode and uint32 seed')
    records = {}
    for role in ('train', 'val'):
        stub = PartitionObservation(role, np.zeros((1, 1, len(bands.band_ids)), dtype=np.float32),
                                    np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.int64),
                                    registration['geometry_hash'])
        view, records[role] = sample(stub, bands, observation, np.random.default_rng(seed))
    effective = p4_guarded.model_metadata(view.bands, metadata)
    return {'policy': POLICY, 'arms': list(ARMS), 'records': records, 'observation': observation,
            'observation_seed': seed, 'metadata': metadata, 'source_metadata_sha256': bands.fingerprint,
            'forward_metadata_sha256': None if metadata == 'missing' else effective.fingerprint,
            'exposure_plan_sha256': registration['pins'][registration['exposure']['plan']],
            'steps': exposure['steps'], 'validation_steps': exposure['validation_steps'], 'selection': SELECTION}


def bind(registration, exposure, bands, geometry_hash, arm):
    if arm not in ARMS:
        raise ValueError('Select an explicitly registered coverage arm')
    frozen = registration['coverage']
    if registration.get('geometry_hash') != geometry_hash:
        raise ValueError('Coverage geometry differs from the registered observation')
    expected = freeze(registration, exposure, bands, observation=frozen['observation'],
                      seed=frozen['observation_seed'], metadata=frozen['metadata'])
    if frozen != expected:
        raise ValueError('Coverage arms, observation metadata or exposure budget differs')
    return expected


def observation_provider(scene, allowed, frozen):
    registered = {tile['id']: tile for tiles in allowed.values() for tile in tiles}
    views = {}

    def view(tile):
        if registered.get(tile['id']) != tile:
            raise ValueError('Coverage observation escapes the development roles')
        if tile['id'] not in views:
            selected = p4_guarded.retained_view(scene, tile, {
                'sampling': frozen['records'][tile['partition']], 'metadata': frozen['metadata']})
            actual = None if frozen['metadata'] == 'missing' else selected.bands.fingerprint
            if actual != frozen['forward_metadata_sha256']:
                raise ValueError('Forward coverage metadata differs from the registration')
            views[tile['id']] = selected
        return views[tile['id']]

    return view


def initialize(base, config, arm):
    """Add only the identity-initialized E1 state to the accepted P4 base."""
    validate_base(config)
    if arm not in ARMS:
        raise ValueError('Unknown matched coverage arm')
    original = base.state_dict()
    model = base
    if arm != 'none':
        values = config.to_dict()
        values['model'].update(coverage_mode='affine', coverage_condition=arm)
        with torch.random.fork_rng(devices=[]):
            model = SpatialConfig(values).research.build_model({p4_guarded.SCENE: p4_exposure.CLASSES})
        state = model.state_dict()
        if (not original.keys() <= state.keys()
                or any((state[key].shape, state[key].dtype) != (value.shape, value.dtype)
                       for key, value in original.items())
                or any(not key.startswith(PREFIX) for key in state.keys() - original.keys())):
            raise ValueError('Coverage may add only E1 tensors to the P4 initialization')
        model.load_state_dict({**state, **original}, strict=True)
    final = model.state_dict()
    params = {key: value.numel() for key, value in model.named_parameters() if key.startswith(PREFIX)}
    if (sum(params.values()) != (0 if arm == 'none' else 194)
            or any(not torch.equal(final[key], value) for key, value in original.items())):
        raise ValueError('Coverage initialization changed the common P4 state or registered capacity')
    module = model.backbone.coverage_modulation
    return model, {'arm': arm, 'parameters': params, 'parameter_count': sum(params.values()),
                   'operator': None if module is None else module.export_config(),
                   'state_sha256': _tree_fingerprint({key: final[key] for key in params})}


def update_evidence(params, before, coverage, view, frozen):
    record = frozen['records']['train']
    metadata_hash = frozen['forward_metadata_sha256']
    bands = view.observation.spectral.shape[-1]
    if bands != len(record['selected_positions']) or list(view.bands.band_ids) != record['selected_band_ids']:
        raise ValueError('Actual retained band identity differs')
    observed = {'sampling_sha256': record['record_hash'], 'bands': bands, 'metadata_sha256': metadata_hash}
    if not params:
        if coverage is not None:
            raise ValueError('Coverage-off arm unexpectedly invoked E1')
        return observed
    if coverage is None:
        raise ValueError('Enabled arm did not report E1 diagnostics')
    usable = coverage['usable_anchors']
    reason = coverage['fallback_reason']
    if (coverage['metadata_hash'] != metadata_hash
            or (frozen['metadata'] == 'missing' and (usable or reason != 'missing_metadata'))):
        raise ValueError('Actual E1 metadata identity or fallback differs')
    tensors = {}
    for key, parameter in params.items():
        grad = parameter.grad
        changed = int(torch.count_nonzero(parameter.detach() != before[key]))
        if ((usable > 0) != (grad is not None) or (grad is not None and not torch.isfinite(grad).all())
                or (not usable and changed)):
            raise ValueError('Coverage gradient/update violates the E1 support gate: ' + key)
        tensors[key] = {'gradient_l1': None if grad is None else float(grad.detach().double().abs().sum()),
                        'changed_elements': changed}
    return {**observed, 'fallback_reason': reason, 'usable_anchors': usable, 'parameters': tensors}


def prepare(parent, parent_sha, *, commit, observation, seed, metadata):
    registration = read_pinned(parent, parent_sha)
    if 'coverage' in registration or 'exposure' not in registration or 'guarded' not in registration:
        raise ValueError('Expected a guarded exposure registration with no coverage arms')
    validate_base(SpatialConfig(registration['config']))
    pins = registration['pins']
    geometry_path = registration['development']
    stats_path = registration['statistics']
    exposure_path = registration['exposure']['plan']
    geometry = read_pinned(Path(geometry_path), pins[geometry_path])
    stats = read_pinned(Path(stats_path), pins[stats_path])
    exposure = read_pinned(Path(exposure_path), pins[exposure_path])
    bands = read_band_metadata(p4_guarded.SCENE, len(stats['minimum']))
    registration.update(commit=commit, geometry_hash=geometry['development_roles_hash'])
    registration['coverage'] = freeze(registration, exposure, bands, observation=observation,
                                      seed=seed, metadata=metadata)
    return registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--observation', type=json.loads, required=True, help='Existing band sampler JSON config')
    parser.add_argument('--observation-seed', type=int, required=True)
    parser.add_argument('--metadata', choices=METADATA_MODES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        raise ValueError('Unexpected coverage registration commit')
    require_frozen_code(code)
    registration = prepare(args.registration, args.sha256, commit=code['commit'],
                           observation=args.observation, seed=args.observation_seed, metadata=args.metadata)
    require_frozen_code(code)
    out = external_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'registration.json', registration)


if __name__ == '__main__':
    main()
