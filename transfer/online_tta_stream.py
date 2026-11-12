"""Pinned observation-only protocol for resumable E4b adaptation."""
from dataclasses import fields
from io import BytesIO
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from evaluate_tta import ARRAYS, MODES, _observation
from evaluation.classification_metrics import class_order
from loadData.observation_sampling import BandMetadata
from loadData.data_read import read_band_metadata
from model.research_config import digest
from research_runtime import external_directory, tensor_hash
from transfer.registered_development import _load_source, _read

VERSION = 'wp09_online_tta_v1'
SETTINGS = ('task', 'mode', 'reset', 'scale', 'lr', 'rank', 'seed')
ORDERS = ('original', 'reverse', 'seed64')
ADMITTED_SOURCE = 'wp09_known_source_admitted_v1'
ADMISSION_HASHES = ('authority_sha256', 'source_receipt_sha256', 'preparation_sha256',
                    'context_registration_sha256', 'dataset_evidence_sha256')
PROVENANCE_FIELDS = ('kind source producer_commit exporter_commit selected_step class_ids bands_hash '
                     'observations baseline_hashes')


def optional_pin(path, checksum):
    if path is None and checksum is None:
        return None
    if path is None or checksum is None:
        raise ValueError('Supply both the independently approved path and SHA-256')
    return dict(path=str(Path(path).resolve()), sha256=checksum)


def fields_are(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f'Expected exactly these fields: {names}')


def parse_json(raw):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON field: {key}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f'Invalid JSON number: {value}')

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def validate_pin(pin):
    fields_are(pin, 'path sha256')
    if not isinstance(pin['path'], str) or str(external_directory(pin['path'])) != pin['path']:
        raise ValueError('Pinned path must be canonical and external')
    require_hash(pin['sha256'])


def read_pin(pin):
    validate_pin(pin)
    return _read(pin)


def require_hash(value, length=64):
    if not isinstance(value, str) or not re.fullmatch(f'[0-9a-f]{{{length}}}', value):
        raise ValueError('Expected full hexadecimal identity')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value):
        raise ValueError('Invalid stream identifier')


def order_frames(frames, order):
    if order not in ORDERS:
        raise ValueError('Unsupported TTA stream order')
    if order == 'reverse':
        frames = list(reversed(frames))
    elif order == 'seed64':
        frames = [frames[i] for i in np.random.default_rng(64).permutation(len(frames))]
    ordered, previous, episode = [], None, 0
    for frame in frames:
        boundary = (frame['scene'], frame['segment'])
        if boundary != previous:
            episode += 1
        ordered.append(dict(frame, episode=episode))
        previous = boundary
    return ordered


def frame_budget(mode, count):
    return dict(predictions=count, optimizer_updates=count if mode in ('affine', 'adapter') else 0,
                statistics_updates=count if mode == 'statistics' else 0)


def load_observation(pin, bands):
    with np.load(BytesIO(read_pin(pin)), allow_pickle=False) as archive:
        if set(archive.files) != ARRAYS or len(archive.files) != len(ARRAYS):
            raise ValueError('Only observation arrays may enter TTA')
        data = dict(archive)
    wavelengths = np.asarray(bands.wavelengths, dtype=np.float32)
    if not np.array_equal(data['wavelengths_nm'], wavelengths):
        raise ValueError('Observation wavelength identity changed')
    return _observation(data, 'cpu')


class StreamRegistration:
    def __init__(self, pin, *, admission=None):
        self.pin = pin
        reg = parse_json(read_pin(pin))
        fields_are(reg, 'version commit source_hash runtime source provenance settings class_ids '
                        'bands observations frames order budget')
        if reg['version'] != VERSION:
            raise ValueError('Unsupported TTA registration version')
        require_hash(reg['commit'], 40)
        require_hash(reg['source_hash'])
        fields_are(reg['runtime'], 'python numpy torch threads interop deterministic mkldnn cpu')
        fields_are(reg['source'], 'weights sidecar')
        if admission is not None and admission != reg['provenance']:
            raise ValueError('Registration does not bind the independently approved source admission')
        provenance = parse_json(read_pin(reg['provenance']))
        kind = provenance.get('kind')
        self.admission = None
        if kind == ADMITTED_SOURCE:
            if admission is None:
                raise ValueError('Supply the operator-approved source admission path and SHA-256')
            fields_are(provenance, PROVENANCE_FIELDS + ' ' + ' '.join(ADMISSION_HASHES)
                       + ' scope task norm scale windows sample_ids training_class_counts annotation_costs')
            for name in ADMISSION_HASHES:
                require_hash(provenance[name])
            if provenance['scope'] not in ('synthetic', 'registered_development') or provenance['norm'] != 'group_norm':
                raise ValueError('Unsupported admitted known-source scope or normalization')
            self.admission = provenance
        else:
            if kind not in ('synthetic', 'e4b_pavia_development') or admission is not None:
                raise ValueError('Unsupported source provenance; registered development needs a verified source admission')
            fields_are(provenance, PROVENANCE_FIELDS + ' origin_hash')
            require_hash(provenance['origin_hash'])
        for name in ('producer_commit', 'exporter_commit'):
            require_hash(provenance[name], 40)
        if type(provenance['selected_step']) is not int or provenance['selected_step'] < 0:
            raise ValueError('Invalid source selection step')
        fields_are(reg['settings'], ' '.join(SETTINGS))
        settings = reg['settings']
        task = settings['task']
        identifier(task)
        if (settings['mode'] not in MODES or settings['reset'] not in ('episodic', 'continual')
                or type(settings['scale']) is not int or settings['scale'] < 1
                or type(settings['rank']) is not int or settings['rank'] < 1
                or type(settings['seed']) is not int or not 0 <= settings['seed'] < 2**32
                or type(settings['lr']) not in (int, float) or not math.isfinite(settings['lr'])
                or settings['lr'] <= 0):
            raise ValueError('Invalid registered TTA settings')
        if not isinstance(reg['class_ids'], list):
            raise ValueError('Explicit ordered head classes required')
        ids = class_order(reg['class_ids'])
        if provenance['class_ids'] != reg['class_ids']:
            raise ValueError('Head class order differs from provenance')
        fields_are(reg['bands'], ' '.join(field.name for field in fields(BandMetadata)))
        self.bands = BandMetadata(**reg['bands'])
        if self.bands.wavelength_unit != 'nm' or not all(self.bands.valid):
            raise ValueError('E4b requires registered physical bands with no invalid channels')
        if kind == 'e4b_pavia_development':
            if (task != 'paviaU' or tuple(ids) != tuple(range(1, 10))
                    or self.bands != read_band_metadata(task, 103)):
                raise ValueError('E4b PaviaU physical-band provenance differs')
        if (provenance['source'] != reg['source'] or provenance['bands_hash'] != self.bands.fingerprint
                or provenance['observations'] != reg['observations']):
            raise ValueError('Source or observation provenance differs')
        if self.admission is not None:
            if (provenance['task'] != task or provenance['scale'] != settings['scale']
                    or provenance['selected_step'] < 1):
                raise ValueError('Task, scale or fixed-final-step source admission differs')
            if settings['mode'] == 'statistics':
                raise ValueError('An admitted GroupNorm source has no supported running statistics')
        for item in reg['source'].values():
            validate_pin(item)
        if reg['source']['sidecar']['path'] != reg['source']['weights']['path'] + '.json':
            raise ValueError('Source sidecar does not belong to source weights')
        metadata = parse_json(read_pin(reg['source']['sidecar']))
        if metadata['class_counts'].get(task) != len(ids):
            raise ValueError('Head class order differs from checkpoint')
        if self.admission is not None and (metadata['class_counts'] != {task: len(ids)}
                or metadata['backbone_kwargs']['norm_type'] != provenance['norm']
                or metadata['training_config']['selected_step'] != provenance['selected_step']):
            raise ValueError('Admitted model architecture or selection step differs')
        read_pin(reg['source']['weights'])
        observations = reg['observations']
        if not isinstance(observations, dict) or not observations:
            raise ValueError('Registered observation table required')
        for key, item in observations.items():
            identifier(key)
            validate_pin(item)
        if self.admission is not None and (not isinstance(provenance['windows'], dict)
                or provenance['windows'].keys() != observations.keys()):
            raise ValueError('Observations must match admitted development windows')
        baseline = provenance['baseline_hashes']
        if not isinstance(baseline, dict) or (provenance['kind'] != 'synthetic' and set(baseline) != set(observations)):
            raise ValueError('Physical development observations need an independent frozen baseline')
        for hashes in baseline.values():
            fields_are(hashes, 'fused graph cnn')
            for value in hashes.values():
                require_hash(value)
        frames = reg['frames']
        if not isinstance(frames, list) or not frames:
            raise ValueError('Explicit nonempty TTA stream required')
        seen = set()
        for frame in frames:
            fields_are(frame, 'frame_id scene segment observation')
            for name in frame:
                identifier(frame[name])
            if frame['frame_id'] in seen or frame['observation'] not in observations:
                raise ValueError('Duplicate frame ID or unknown observation')
            if self.admission is not None:
                window = provenance['windows'][frame['observation']]
                if (window['role'] != 'development_evaluation' or window['scene'] != frame['scene']
                        or frame['segment'] != frame['observation']):
                    raise ValueError('Frame lies outside its admitted development scene or segment')
            seen.add(frame['frame_id'])
        if self.admission is not None and {f['observation'] for f in frames} != observations.keys():
            raise ValueError('Registered stream must use every admitted window')
        self.frames = order_frames(frames, reg['order'])
        budget = frame_budget(settings['mode'], len(frames))
        if (reg['budget'] != budget or any(type(v) is not int for v in reg['budget'].values())):
            raise ValueError('TTA budget differs from registered frames and mode')
        self.value, self.settings, self.provenance = reg, settings, provenance
        self.population = digest(sorted(
            [dict(frame, observation_sha256=observations[frame['observation']]['sha256']) for frame in frames],
            key=lambda row: row['frame_id']))

    def source_model(self):
        model = _load_source(self.value['source']['weights'], self.value['source']['sidecar'])
        task = self.settings['task']
        if (model.backbone.graph_mode != 'learned' or model.backbone.coverage_modulation is not None
                or self.settings['rank'] >= model.heads[task].in_features
                or any(not torch.isfinite(v).all() for v in model.state_dict().values())):
            raise ValueError('Unsupported E4b source checkpoint')
        return model

    def observation_identity(self, frame):
        return self.value['observations'][frame['observation']]['sha256']

    def observation(self, key):
        return load_observation(self.value['observations'][key], self.bands)

    @staticmethod
    def fingerprints(probabilities):
        return {key: tensor_hash(value) for key, value in probabilities.items()}
