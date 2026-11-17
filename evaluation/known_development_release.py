"""Bind a completed WP09 stream to its original deferred development label release."""
import json
from pathlib import Path

import numpy as np

from evaluation.open_set import _ids, _labels
from model.research_config import canonical_json
from transfer.online_tta_stream import fields_are, parse_json, read_pin, validate_pin


RELEASE_KIND = 'wp09_known_development_label_release_v1'


def reserve_label(pin, label_path):
    validate_pin(pin)
    if pin['path'] == label_path:
        raise ValueError('Deferred label payload cannot be used as pre-scoring metadata or input')


def reserve_label_directory(path, label_path):
    if Path(label_path).is_relative_to(path):
        raise ValueError('Deferred label payload must be outside prediction and replay directories')


def _sample_mapping(reg, authority, context):
    admitted = reg.admission
    role = context['observations'][-1]
    annotation = admitted['annotation_costs']
    fields_are(annotation, 'source_pixels calibration_pixels source_supervised calibration_selected development_selected')
    counts = admitted['training_class_counts']
    classes = [row['id'] for row in context['known']]
    if (admitted['scope'] != authority['scope'] or admitted['task'] != context['dataset']
            or admitted['class_ids'] != classes or admitted['scale'] != role['superpixel_scale']
            or admitted['context_registration_sha256'] != authority['context_registration']['sha256']
            or admitted['dataset_evidence_sha256'] != authority['dataset_evidence']['sha256']
            or admitted['windows'].keys() != authority['windows'].keys()
            or any(type(n) is not int or n < 1 for n in annotation.values())
            or not isinstance(counts, list) or len(counts) != len(classes)
            or any(type(n) is not int or n < 1 for n in counts) or sum(counts) != annotation['source_supervised']
            or annotation['source_pixels'] < annotation['source_supervised']
            or annotation['calibration_pixels'] < annotation['calibration_selected']):
        raise ValueError('Approved development scope, semantic classes or training support differs')
    sample_ids = _ids(admitted['sample_ids'], annotation['development_selected'])
    r0, c0 = role['origin']
    sh, sw = role['scene_shape']
    used = set()
    for name, (a, b, d, e) in authority['windows'].items():
        item = admitted['windows'][name]
        fields_are(item, 'role scene scene_shape box label_positions local_indices')
        height, width = b - a, e - d
        if (item['role'] != role['role'] or item['scene'] != role['scene_uid'] or item['scene_shape'] != [sh, sw]
                or item['box'] != [r0 + a, r0 + b, c0 + d, c0 + e] or r0 + b > sh or c0 + e > sw):
            raise ValueError('Approved development window role or scene geometry differs')
        positions, indices = item['label_positions'], item['local_indices']
        if (not isinstance(positions, list) or not isinstance(indices, list) or not positions
                or len(positions) != len(indices) or any(type(i) is not int for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Invalid pre-registered development sample mapping')
        for position, index in zip(positions, indices, strict=True):
            if (type(position) is not int or not 0 <= position < len(sample_ids) or position in used
                    or type(index) is not int or not 0 <= index < height * width):
                raise ValueError('Invalid or duplicate pre-registered development sample')
            row, col = divmod(index, width)
            sample = json.dumps([role['scene_uid'], r0 + a + row, c0 + d + col], separators=(',', ':'))
            if sample_ids[position] != sample:
                raise ValueError('Development sample ID is outside the approved geometry')
            used.add(position)
    if len(used) != len(sample_ids):
        raise ValueError('Approved development samples must all be covered exactly once')
    return sample_ids


def release_metadata(reg, authority_pin, source_receipt, authority, context, run_proofs):
    if (reg.admission is None or reg.admission['authority_sha256'] != authority_pin['sha256']
            or reg.value['bands'] != authority['bands']):
        raise ValueError('Source admission and operator-approved development authority differ')
    sample_ids = _sample_mapping(reg, authority, context)
    return dict(version=1, kind=RELEASE_KIND, scope=authority['scope'], authority=authority_pin,
                admission=reg.value['provenance'], source_receipt=source_receipt,
                role='development_evaluation', population=reg.population,
                class_ids=reg.value['class_ids'], labels=context['labels'], sample_count=len(sample_ids),
                training_class_counts=reg.admission['training_class_counts'], runs=run_proofs)


def released_labels(pin, expected, reg):
    """The only authorized target-label read, after the release and every run pass."""
    reserve_label(pin, expected['labels']['path'])
    try:
        release = parse_json(read_pin(pin))
    except FileNotFoundError as error:
        raise ValueError('Provide the pinned development release metadata produced by --freeze-release') from error
    if canonical_json(release) != canonical_json(expected):
        raise ValueError('Development label release differs from the authorized role, samples or frozen runs')
    try:
        packet = parse_json(read_pin(expected['labels']))
    except FileNotFoundError as error:
        raise ValueError(f"Authorized development label payload has not been released: {expected['labels']['path']}") from error
    fields_are(packet, 'version role sample_ids labels')
    if (packet['version'] != 1 or packet['role'] != release['role']
            or packet['sample_ids'] != reg.admission['sample_ids']):
        raise ValueError('Released label payload role or registered sample order differs')
    values = _labels(packet['labels'], release['sample_count'])
    if not np.isin(values, [0, *release['class_ids']]).all():
        raise ValueError('Closed-set development release contains a class outside the known columns')
    by_window = {}
    for key, window in reg.admission['windows'].items():
        r0, r1, c0, c1 = window['box']
        array = np.zeros((r1 - r0) * (c1 - c0), dtype=np.int64)
        array[window['local_indices']] = values[window['label_positions']]
        by_window[key] = array
    return {frame['frame_id']: by_window[frame['observation']] for frame in reg.frames}
