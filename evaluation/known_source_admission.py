"""Authenticate the known-source producer and export only authorized TTA observations."""
import argparse
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_tta import HEADS, _observation
from evaluation.known_context_preparation import _load as load_context
from evaluation.known_development_authority import read_authority, verify_producer
from evaluation.open_set import _array_hash
from evaluation.open_set_producer import ROLES, _observation as load_prepared
from evaluation.registered_open_set import _code, _output
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from transfer.g3_training import build_model, state_hash
from transfer.online_tta_stream import (ADMITTED_SOURCE, fields_are, parse_json, read_pin, validate_pin)
from transfer.registered_development import _json_new, _pin


def _distinct(pins):
    for pin in pins:
        validate_pin(pin)
    paths = [pin['path'] for pin in pins]
    if len(set(paths)) != len(paths):
        raise ValueError('Producer artifacts and all reserved roles must be distinct')


def _preparation(prepared, authority, context, bands, reserved):
    value = parse_json(read_pin(prepared))
    fields_are(value, 'status version code registration inputs known unknown dataset observations labels statistics '
                      'statistics_hash checks full_context_known_only preprocessing_source_only training_history_verified '
                      'evaluation_labels_read query optimizer_updates limits')
    raw_pin = authority['context_registration']
    if (value['status'] != 'passed' or value['version'] != 1 or value['registration'] != raw_pin
            or value['full_context_known_only'] is not True or value['preprocessing_source_only'] is not True
            or value['training_history_verified'] is not False or value['evaluation_labels_read'] is not False
            or value['query'] != 0 or value['optimizer_updates'] != 0
            or any(value[k] != context[k] for k in ('known', 'unknown', 'dataset', 'labels'))):
        raise ValueError('Unsupported preparation receipt or changed semantic and deferred roles')
    verify_producer(value['code'], authority['preparation_commit'],
                    ('evaluation/known_context_preparation.py', 'scripts/prepare_open_set_contexts.py'))
    if not isinstance(value['observations'], list) or len(value['observations']) != len(ROLES):
        raise ValueError('Preparation receipt must release the three original observation roles')
    outputs = []
    for raw, packet in zip(context['observations'], value['observations'], strict=True):
        fields_are(packet, 'role scene_uid origin superpixel_scale input')
        if any(packet[k] != raw[k] for k in ('role', 'scene_uid', 'origin', 'superpixel_scale')):
            raise ValueError('Prepared role or spatial identity changed')
        outputs.append(packet['input'])
    _distinct([*reserved, *outputs])
    original, arrays, contexts, inputs = load_context(raw_pin)
    if original != context or value['inputs'] != inputs:
        raise ValueError('Preparation input roles differ from the authorization')
    known = [row['id'] for row in context['known']]
    source = arrays[0]['spectral']
    low, high = source.min((0, 1)), source.max((0, 1))
    scale = np.where(high > low, high - low, np.float32(1))
    stats = dict(minimum=low.tolist(), maximum=high.tolist(), fit_role='source_train',
                 fit_pixels=int(np.prod(source.shape[:2])), source_pin=context['observations'][0]['input'],
                 clipping=False)
    if (not np.isfinite(scale).all() or value['statistics'] != stats
            or value['statistics_hash'] != digest(stats)):
        raise ValueError('Source-only normalization does not reproduce the preparation')
    expected_waves = np.asarray(bands.wavelengths, dtype=np.float32)
    packets, checks = [], []
    for row, original_arrays, spatial, pin in zip(context['observations'], arrays, contexts, outputs, strict=True):
        role = row['role']
        packet = load_prepared(pin, role, known)
        r0, r1, c0, c1 = spatial['box']
        sh, sw = spatial['scene_shape']
        yy, xx = np.mgrid[r0:r1, c0:c1]
        coordinates = np.stack((yy / (sh - 1), xx / (sw - 1)), axis=-1).astype(np.float32)
        x = np.concatenate(((original_arrays['spectral'] - low) / scale, coordinates), axis=-1)
        if (not np.array_equal(packet['x'], x) or not np.array_equal(packet['wavelengths_nm'], expected_waves)
                or not np.array_equal(original_arrays['wavelengths_nm'], expected_waves)
                or not np.array_equal(original_arrays['indices'], packet['indices'])):
            raise ValueError('Prepared observations must reproduce geometry, support and ordered physical bands')
        annotation = original_arrays.get('context_labels')
        if annotation is not None and not np.array_equal(packet['labels'], annotation.ravel()[packet['indices']]):
            raise ValueError('Prepared known support labels differ')
        # The producer's evidence preserves the original indices dtype.
        checks.append(dict(role=role, context=spatial, annotated_pixels=0 if annotation is None else annotation.size,
            annotation_sha256=None if annotation is None else _array_hash(annotation),
            selected_pixels=len(packet['indices']), array_hashes={
                key: _array_hash(original_arrays['indices'] if key == 'indices' else
                                 annotation.ravel()[packet['indices']] if key == 'labels' else data)
                for key, data in packet.items()}))
        packets.append(packet)
    if checks != value['checks']:
        raise ValueError('Preparation receipt support or array evidence differs')
    return value, packets, contexts, checks


def _verified_model(receipt, trial, prepared, source_packet, known, source):
    budget = trial['updates']
    if (type(budget) is not int or budget < 1 or type(trial['seed']) is not int or not 0 <= trial['seed'] < 2**32
            or type(receipt['selected_step']) is not int or receipt['selected_step'] != budget
            or type(receipt['optimizer_updates']) is not int or receipt['optimizer_updates'] != budget
            or receipt['selection'] != 'fixed_final_step' or not isinstance(receipt['loss'], list)
            or len(receipt['loss']) != budget
            or any(type(loss) not in (int, float) or not math.isfinite(loss) for loss in receipt['loss'])
            or receipt['source_annotated_pixels'] != prepared['checks'][0]['annotated_pixels']
            or receipt['source_supervised_pixels'] != len(source_packet['indices'])):
        raise ValueError('Source update budget, fixed final step or annotated/support counts differ')
    weights, sidecar = source['weights'], source['sidecar']
    if sidecar['path'] != weights['path'] + '.json':
        raise ValueError('Actual adjacent source checkpoint sidecar required')
    metadata = parse_json(read_pin(sidecar))
    config = parse_json(read_pin(trial['config']))
    task = prepared['dataset']
    model, _ = build_model(config, task, len(known), trial['seed'], 'cpu')
    expected_metadata = model.export_config() | {'training_config': dict(
        phase='known_source', optimizer_updates=budget, seed=trial['seed'], selected_step=budget,
        selection='fixed_final_step', registration=receipt['registration'],
        context_statistics_hash=prepared['statistics_hash'])}
    if (metadata != json.loads(json.dumps(expected_metadata)) or state_hash(model) != receipt['initial_state_sha256']
            or model.backbone.graph_mode != 'learned' or model.backbone.norm_type != 'group_norm'
            or model.backbone.extra_channels != 2 or model.backbone.graph_assignment_mode != 'learned'
            or model.backbone.graph_adjacency_mode != 'learned'
            or any(getattr(model.backbone, name) is not None for name in (
                'interaction', 'region_residual', 'coverage_modulation', 'graph_router', 'edge_updater'))):
        raise ValueError('Checkpoint architecture, source registration or fixed-step metadata differs')
    declared = parse_json(read_pin(receipt['training_declaration']))
    if declared != dict(version=1, kind='open_set_training_semantics_declaration_v1',
            checkpoint_sha256=weights['sha256'], sidecar_sha256=sidecar['sha256'],
            dataset=task, known=prepared['known'], source_training_class_ids=known,
            unknown_training_excluded=True):
        raise ValueError('Known source training declaration or class columns differ')
    saved = torch.load(BytesIO(read_pin(weights)), map_location='cpu', weights_only=True)
    expected = model.state_dict()
    if (not isinstance(saved, dict) or saved.keys() != expected.keys()
            or any(not torch.is_tensor(item) or item.dtype != expected[name].dtype
                   or item.shape != expected[name].shape or not torch.isfinite(item).all()
                   for name, item in saved.items())):
        raise ValueError('Source weights do not fit the exact FP32 producer architecture')
    model.load_state_dict(saved, strict=True)
    if (state_hash(model) != receipt['final_state_sha256']
            or receipt['initial_state_sha256'] == receipt['final_state_sha256']):
        raise ValueError('Actual source weight state does not match the trained final step')
    model.backbone.adapter.cache_attention = False
    return model.eval()


def _development_windows(windows, packet, context):
    x = packet['x']
    h, w = x.shape[:2]
    r, _, c, _ = context['box']
    rows, cols = np.divmod(packet['indices'], w)
    sample_ids = [json.dumps([context['scene_uid'], r + int(y), c + int(z)], separators=(',', ':'))
                  for y, z in zip(rows, cols, strict=True)]
    occupied, assigned = np.zeros((h, w), bool), np.zeros(len(rows), bool)
    values, identities = {}, {}
    for name, (a, b, d, e) in windows.items():
        if b > h or e > w or occupied[a:b, d:e].any():
            raise ValueError('Development windows must stay inside the authorized role and be disjoint')
        occupied[a:b, d:e] = True
        mask = (rows >= a) & (rows < b) & (cols >= d) & (cols < e)
        if not mask.any():
            raise ValueError('Every development window requires pre-registered evaluation samples')
        assigned |= mask
        identities[name] = dict(role=ROLES[-1], scene=context['scene_uid'], scene_shape=context['scene_shape'],
                                box=[r + a, r + b, c + d, c + e], label_positions=np.flatnonzero(mask).tolist(),
                                local_indices=((rows[mask] - a) * (e - d) + cols[mask] - d).tolist())
        view = x[a:b, d:e]
        values[name] = dict(spectral=view[..., :-2], coordinates=view[..., -2:],
                            wavelengths_nm=packet['wavelengths_nm'])
    if not assigned.all():
        raise ValueError('Development windows omit pre-registered evaluation samples')
    return values, identities, sample_ids


def _verified_source(authority_pin, source_receipt):
    authority, context, bands = read_authority(authority_pin)
    base = [authority_pin, authority['dataset_evidence'], authority['context_registration'], context['labels'],
            *[item['input'] for item in context['observations']]]
    _distinct([*base, source_receipt])
    receipt = parse_json(read_pin(source_receipt))
    fields_are(receipt, 'status code registration inputs checkpoint sidecar training_declaration initial_state_sha256 '
                        'final_state_sha256 source_annotated_pixels source_supervised_pixels optimizer_updates '
                        'selected_step selection loss calibration_read evaluation_read query')
    if (receipt['status'] != 'completed' or receipt['calibration_read'] is not False
            or receipt['evaluation_read'] is not False or receipt['query'] != 0):
        raise ValueError('Completed source-only training receipt required')
    verify_producer(receipt['code'], authority['training_commit'],
                    ('evaluation/known_source_training.py', 'scripts/train_known_source.py',
                     'evaluation/known_development_authority.py', 'transfer/g3_training.py'))
    source = dict(weights=receipt['checkpoint'], sidecar=receipt['sidecar'])
    base += [source_receipt, receipt['registration'], receipt['training_declaration'], *source.values()]
    _distinct(base)
    trial = parse_json(read_pin(receipt['registration']))
    fields_are(trial, 'version kind authority prepared config seed updates')
    if trial['version'] != 2 or trial['kind'] != 'known_source_authorized_v2' or trial['authority'] != authority_pin:
        raise ValueError('Source training was not pre-registered under this data authority and scope')
    base += [trial['prepared'], trial['config']]
    _distinct(base)
    prepared, packets, contexts, checks = _preparation(trial['prepared'], authority, context, bands, base)
    outputs = [row['input'] for row in prepared['observations']]
    expected_inputs = [trial['prepared'], authority['context_registration'], context['observations'][0]['input'],
                       outputs[0], receipt['registration'], trial['config'], authority_pin, authority['dataset_evidence']]
    if receipt['inputs'] != expected_inputs:
        raise ValueError('Source training receipt inputs differ from the authorized source role')
    values, windows, sample_ids = _development_windows(authority['windows'], packets[-1], contexts[-1])
    classes = [row['id'] for row in prepared['known']]
    model = _verified_model(receipt, trial, prepared, packets[0], classes, source)
    task, scale = context['dataset'], context['observations'][-1]['superpixel_scale']
    identity = dict(kind=ADMITTED_SOURCE, scope=authority['scope'], source=source,
        producer_commit=authority['training_commit'], authority_sha256=authority_pin['sha256'],
        source_receipt_sha256=source_receipt['sha256'], preparation_sha256=trial['prepared']['sha256'],
        context_registration_sha256=authority['context_registration']['sha256'],
        dataset_evidence_sha256=authority['dataset_evidence']['sha256'], selected_step=receipt['selected_step'],
        class_ids=classes, bands_hash=bands.fingerprint, task=task, norm='group_norm', scale=scale,
        windows=windows, sample_ids=sample_ids,
        training_class_counts=[int(np.count_nonzero(packets[0]['labels'] == label)) for label in classes],
        annotation_costs=dict(source_pixels=checks[0]['annotated_pixels'], calibration_pixels=checks[1]['annotated_pixels'],
                              source_supervised=checks[0]['selected_pixels'], calibration_selected=checks[1]['selected_pixels'],
                              development_selected=checks[2]['selected_pixels']))
    return identity, values, model, [*base, *outputs]


def verify_admission(reg, authority_pin, source_receipt):
    """Revalidate the producer chain and admitted observation bytes before offline scoring."""
    if reg.admission['source_receipt_sha256'] != source_receipt['sha256']:
        raise ValueError('Approved parent source training receipt differs from source admission')
    identity, values, _, _ = _verified_source(authority_pin, source_receipt)
    if any(reg.admission[key] != value for key, value in identity.items()):
        raise ValueError('Source admission differs from the verified producer, support or development windows')
    for name, arrays in values.items():
        expected = _observation(arrays, 'cpu')
        if not all(torch.equal(a, b) for a, b in zip(expected, reg.observation(name), strict=True)):
            raise ValueError('Admitted observation differs from original authorized development window')


def admit_source(authority_pin, source_receipt, *, out, expected_commit):
    """Read verified training inputs, then release label-free observations for TTA."""
    code = _code(expected_commit)
    identity, values, model, pins = _verified_source(authority_pin, source_receipt)
    output = _output(out, pins)
    source_state = state_hash(model)
    baseline = {}
    with torch.inference_mode():
        for name, item in values.items():
            x, waves = _observation(item, 'cpu')
            probabilities = model(identity['task'], x, wavelengths=waves, wavelength_unit='nm',
                                  superpixel_scale=identity['scale'])
            if any(not torch.isfinite(p).all() for p in probabilities):
                raise ValueError('Nonfinite admitted baseline predictions')
            baseline[name] = {head: tensor_hash(p) for head, p in zip(HEADS, probabilities, strict=True)}
    if state_hash(model) != source_state:
        raise ValueError('Frozen admission baselines changed source state')
    require_frozen_code(code)
    output.mkdir(parents=True, exist_ok=False)
    observations = {}
    for name, item in values.items():
        path = output / f'{name}.npz'
        with path.open('xb') as stream:
            np.savez(stream, **item)
        observations[name] = _pin(path)
    return _json_new(output / 'admission.json', dict(identity, exporter_commit=code['commit'],
                     observations=observations, baseline_hashes=baseline))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('authority', 'source-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    authority = dict(path=str(args.authority.resolve()), sha256=args.authority_sha256)
    source = dict(path=str(args.source_receipt.resolve()), sha256=args.source_receipt_sha256)
    print(json.dumps(admit_source(authority, source, out=args.out, expected_commit=args.expected_commit)))


if __name__ == '__main__':
    main()
