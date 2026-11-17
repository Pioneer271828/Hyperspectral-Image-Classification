"""Prepare sparse HongHu development labels from an audited guarded cut."""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from loadData.dataset_registry import DATASETS
from loadData.spatial_tiles import tile_slice, validate_tile_plan
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.g2_contract import read_pinned, sha
from transfer.guarded_tile_feasibility import MINIMUM, SCENE, _counts, _cuts


SUPPORT_SEEDS = (512, 2048)
VALIDATION_SEED = 3031


def guarded_roles(plan, blocker, witness):
    validate_tile_plan(plan)
    if plan['gap'] < 1 or witness['status'] != 'feasible':
        raise ValueError('A guarded feasible cut is required')
    focus = next((t for t in plan['tiles'] if t['id'] == blocker['only_tile']), None)
    if focus is None or focus['partition'] != 'train':
        raise ValueError('Cut parent must be an original training tile')
    geometry = (witness['axis'], witness['cut'], witness['first_box'], witness['second_box'])
    if geometry not in _cuts(focus['box'], plan['gap']):
        raise ValueError('Cut escapes its parent or alters the buffer')
    eligible = {t['id'] for t in plan['tiles'] if t['partition'] == 'train'} - {focus['id']}
    released_val = witness['validation_blocks']
    val = set(released_val)
    if (len(val) != len(released_val) or not val <= eligible | {'first', 'second'}
            or len(val & {'first', 'second'}) != 1):
        raise ValueError('Validation witness violates original training membership')
    roles = []
    next_id = len(plan['tiles'])
    for tile in plan['tiles']:
        if tile['id'] == focus['id']:
            for offset, (key, box) in enumerate(zip(('first', 'second'), geometry[2:])):
                roles.append(dict(id=next_id + offset, parent_tile=tile['id'], box=box,
                                  partition='train', role='devval' if key in val else 'devtrain'))
        else:
            partition = tile['partition']
            role = ('devval' if tile['id'] in val else 'devtrain') if partition == 'train' else (
                'test' if partition == 'test' else 'unused')
            roles.append(dict(tile, role=role))
    return roles


def _sample(indices, values, classes, seed, count):
    rng = np.random.default_rng(seed)
    selected = sorted((int(idx), c) for c in range(1, classes + 1)
                      for idx in rng.permutation(np.sort(indices[values == c]))[:count])
    return {'seed': seed, 'per_class': count,
            'indices': [idx for idx, _ in selected], 'labels': [c for _, c in selected]}


def prepare_labels(labels, plan, classes, blocker, witness):
    if list(labels.shape) != plan['shape']:
        raise ValueError('Label array does not match the parent plan')
    roles = guarded_roles(plan, blocker, witness)
    result = {}
    for role, field in (('devtrain', 'train_counts'), ('devval', 'validation_counts')):
        ids, values = [], []
        for tile in roles:
            if tile['role'] != role:
                continue
            block = labels[tile_slice(tile)]
            _counts(block, classes)
            y0, y1, x0, x1 = tile['box']
            yy, xx = np.mgrid[y0:y1, x0:x1]
            ids.append((yy * labels.shape[1] + xx).ravel())
            values.append(block.ravel())
        indices, classes_at_indices = np.concatenate(ids), np.concatenate(values)
        counts = _counts(classes_at_indices, classes)
        if counts.tolist() != witness[field] or np.any(counts < MINIMUM):
            raise ValueError('Actual development coverage differs from audited witness')
        result[role] = (indices, classes_at_indices)
    val = _sample(*result['devval'], classes, VALIDATION_SEED, MINIMUM)
    train = {}
    for seed in SUPPORT_SEEDS:
        for k in (1, MINIMUM):
            train[seed, k] = _sample(*result['devtrain'], classes, seed, k)
    return roles, train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'labels', 'scan-registration', 'scan-receipt', 'audit'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        raise ValueError('Unexpected guarded role preparation commit')
    require_frozen_code(code)
    plan = read_pinned(args.plan, args.plan_sha256)
    scan = read_pinned(args.scan_registration, args.scan_registration_sha256)
    scan_receipt = read_pinned(args.scan_receipt, args.scan_receipt_sha256)
    audit = read_pinned(args.audit, args.audit_sha256)
    witness = audit['chosen_geometry_for_followup']
    audit_pins = set(audit['protected_files'].values())
    if (audit['status'] != 'passed' or scan_receipt['status'] != 'feasible_within_scope'
            or scan_receipt['registration_sha256'] != args.scan_registration_sha256
            or not {args.scan_registration_sha256, args.scan_receipt_sha256} <= audit_pins
            or witness not in scan_receipt['candidates'] or scan['scene'] != SCENE
            or scan['plan_sha256'] != args.plan_sha256 or scan['labels_sha256'] != args.labels_sha256):
        raise ValueError('Independent guarded cut audit does not bind the supplied inputs')
    if hash_file(args.labels) != sha(args.labels_sha256):
        raise ValueError('Original label file changed')
    roles = guarded_roles(plan, scan['blocker'], witness)
    out = external_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    registration = {'commit': code['commit'], 'scene': SCENE, 'plan_sha256': args.plan_sha256,
                    'labels_sha256': args.labels_sha256, 'audit_sha256': args.audit_sha256,
                    'scan_receipt_sha256': args.scan_receipt_sha256,
                    'support_seeds': list(SUPPORT_SEEDS), 'support_k': [1, MINIMUM],
                    'validation_seed': VALIDATION_SEED, 'validation_k': MINIMUM,
                    'roles': roles}
    write_json(out / 'registration.json', registration)
    _, key, classes, _ = DATASETS[SCENE]
    labels = loadmat(args.labels, variable_names=[key + '_gt'])[key + '_gt']
    _, support, validation = prepare_labels(labels, plan, classes, scan['blocker'], witness)
    for name in ('plan', 'labels', 'scan_registration', 'scan_receipt', 'audit'):
        if hash_file(getattr(args, name)) != getattr(args, name + '_sha256'):
            raise RuntimeError('Frozen role input changed')
    require_frozen_code(code)
    geometry = {'schema_version': 1, 'kind': 'guarded_honghu_development_v1', 'scene': SCENE,
                'parent_plan_hash': plan['plan_hash'], 'shape': plan['shape'],
                'roles': roles, 'audit_sha256': args.audit_sha256,
                'selection_scope': 'oracle-assisted original training geometry'}
    geometry['development_roles_hash'] = digest(geometry)
    write_json(out / 'roles.json', geometry)
    released = []
    for (seed, k), training in support.items():
        name = f'support{seed}-k{k}.json'
        record = {'scene': SCENE, 'development_roles_hash': geometry['development_roles_hash'],
                  'support': training, 'validation': validation}
        record['release_hash'] = digest(record)
        write_json(out / name, record)
        released.append({'file': name, 'sha256': hash_file(out / name), 'seed': seed, 'k': k})
    write_json(out / 'receipt.json',
               {'status': 'prepared', 'registration_sha256': hash_file(out / 'registration.json'),
                'roles_sha256': hash_file(out / 'roles.json'), 'releases': released,
                'raw_gt_array_decoded_in_curator': True, 'original_val_test_values_examined': False,
                'optimizer_updates': 0})


if __name__ == '__main__':
    main()
