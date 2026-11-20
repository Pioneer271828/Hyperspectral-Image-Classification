"""Operator-pinned data scope and supported producer identities for WP09."""
from dataclasses import fields
from hashlib import sha256
import subprocess

from evaluation.open_set import _semantics
from evaluation.open_set_producer import ROLES
from evaluation.registered_open_set import _deferred
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from research_runtime import ROOT
from transfer.online_tta_stream import fields_are, identifier, parse_json, read_pin, require_hash, validate_pin


AUTHORITY_KIND = 'wp09_known_development_authority_v1'
SCOPES = ('synthetic', 'registered_development')


def read_authority(pin):
    """Validate the separately trusted root without opening any label array."""
    authority = parse_json(read_pin(pin))
    fields_are(authority, 'kind scope context_registration dataset_evidence preparation_commit training_commit bands windows')
    if authority['kind'] != AUTHORITY_KIND or authority['scope'] not in SCOPES:
        raise ValueError('Expected operator-approved known development authority and explicit data scope')
    for name in ('preparation_commit', 'training_commit'):
        require_hash(authority[name], 40)
    registration = authority['context_registration']
    context = parse_json(read_pin(registration))
    fields_are(context, 'version kind known unknown dataset observations labels')
    if context['version'] != 1 or context['kind'] != 'known_full_context_preparation_v1':
        raise ValueError('Authority needs the original known-only three-role registration')
    identifier(context['dataset'])
    _semantics(context['known'], context['unknown'])
    _deferred(context['labels'])
    observations = context['observations']
    if not isinstance(observations, list) or len(observations) != len(ROLES):
        raise ValueError('Authority requires all three original region roles')
    paths = [pin['path'], registration['path'], context['labels']['path']]
    for role, row in zip(ROLES, observations, strict=True):
        fields_are(row, 'role scene_uid scene_shape origin superpixel_scale input')
        identifier(row['scene_uid'])
        if (row['role'] != role or not _int_pair(row['scene_shape'], 2)
                or not _int_pair(row['origin'], 0) or type(row['superpixel_scale']) is not int
                or not 4 <= row['superpixel_scale'] <= 4096):
            raise ValueError('Authority needs the original ordered spatial roles and geometry')
        validate_pin(row['input'])
        paths.append(row['input']['path'])
    evidence = authority['dataset_evidence']
    validate_pin(evidence)
    paths.append(evidence['path'])
    if len(set(paths)) != len(paths) or not read_pin(evidence):
        raise ValueError('Data evidence, preparation roles and deferred labels require distinct paths')
    fields_are(authority['bands'], ' '.join(field.name for field in fields(BandMetadata)))
    bands = BandMetadata(**authority['bands'])
    if bands.wavelength_unit != 'nm' or not all(bands.valid):
        raise ValueError('Source and development need the same ordered physical nm bands')
    windows = authority['windows']
    if not isinstance(windows, dict) or not windows:
        raise ValueError('Pre-registered development windows required')
    for name, box in windows.items():
        identifier(name)
        if (not isinstance(box, list) or len(box) != 4 or not _int_pair(box[:2], 0)
                or not _int_pair(box[2:], 0) or box[1] - box[0] < 2 or box[3] - box[2] < 2):
            raise ValueError('Invalid local development window bounds')
    return authority, context, bands


def _int_pair(values, minimum):
    return (isinstance(values, list) and len(values) == 2
            and all(type(i) is int and minimum <= i < 2**31 for i in values))


def verify_producer(code, expected_commit, paths):
    fields_are(code, 'commit branch dirty source_hash files')
    if (code['commit'] != expected_commit or code['dirty'] is not False
            or not isinstance(code['files'], dict) or digest(code['files']) != code['source_hash']):
        raise ValueError('Producer receipt does not bind the pre-registered clean code identity')
    for path in paths:
        try:
            raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{expected_commit}:{path}'],
                                          stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise ValueError(f'Supported producer source unavailable at {expected_commit}: {path}') from error
        if sha256(raw.replace(b'\r\n', b'\n')).hexdigest() != code['files'].get(path):
            raise ValueError('Producer receipt file identity differs from its frozen Git source')
