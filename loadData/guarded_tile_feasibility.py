"""Check whether a guarded cut can rescue class coverage within original training tiles."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat
from scipy.optimize import Bounds, LinearConstraint, milp

from loadData.dataset_registry import DATASETS
from loadData.spatial_tiles import tile_slice, validate_tile_plan
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.g2_contract import read_pinned, sha


MINIMUM = 5
MINIMUM_SIDE = 4
SCENE = 'whu-honghu'


def _cuts(box, gap):
    for axis in ('row', 'column'):
        start_index = 0 if axis == 'row' else 2
        start, end = box[start_index:start_index + 2]
        for cut in range(start + MINIMUM_SIDE, end - gap - MINIMUM_SIDE + 1):
            first, second = list(box), list(box)
            first[start_index + 1] = cut
            second[start_index] = cut + gap
            yield axis, cut, first, second


def _counts(labels, classes):
    if labels.dtype.kind not in 'iuf' or not np.isfinite(labels).all():
        raise ValueError('Invalid labels in original training region')
    if (labels < 0).any() or (labels > classes).any() or (labels != np.floor(labels)).any():
        raise ValueError('Invalid class IDs in original training region')
    return np.bincount(labels.astype(np.int64).ravel(), minlength=classes + 1)[1:]


def _assign(matrix):
    """Return a checked validation subset, an infeasibility result, or an unresolved result."""
    total = matrix.sum(axis=1)
    if np.any(total < 2 * MINIMUM):
        return {'status': 'insufficient_retained_class_counts'}
    size = matrix.shape[1]
    answer = milp(np.ones(size), integrality=np.ones(size), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, MINIMUM, total - MINIMUM),
                  options={'time_limit': 30.0})
    if answer.status != 0:
        return {'status': 'solver_infeasible' if answer.status == 2 else 'unresolved',
                'solver_status': int(answer.status)}
    selected = np.rint(answer.x).astype(np.int64)
    val = matrix @ selected
    train = total - val
    if (not np.isin(selected, (0, 1)).all() or not np.allclose(answer.x, selected)
            or np.any(val < MINIMUM) or np.any(train < MINIMUM)):
        raise RuntimeError('Solver assignment violates class coverage')
    return {'status': 'feasible', 'validation_indices': np.flatnonzero(selected).tolist(),
            'train_counts': train.tolist(), 'validation_counts': val.tolist()}


def scan_guarded_cuts(labels, plan, classes, blocker):
    """Read class values only inside original training blocks, with no role release."""
    validate_tile_plan(plan)
    if list(labels.shape) != plan['shape'] or plan['gap'] < 1:
        raise ValueError('Expected the original spatial geometry and a positive guard')
    class_id, tile_id = blocker['class_id'], blocker['only_tile']
    if type(class_id) is not int or not 1 <= class_id <= classes or type(tile_id) is not int:
        raise ValueError('Invalid blocker')
    training = [tile for tile in plan['tiles'] if tile['partition'] == 'train']
    counts = [_counts(labels[tile_slice(tile)], classes) for tile in training]
    index = class_id - 1
    occupied = [(tile['id'], int(count[index])) for tile, count in zip(training, counts)
                if count[index]]
    if occupied != [(tile_id, blocker['available'])]:
        raise ValueError('The previously documented single-tile class no longer matches')

    others = [tile for tile in training if tile['id'] != tile_id]
    other_counts = [count for tile, count in zip(training, counts) if tile['id'] != tile_id]
    focus = next(tile for tile in training if tile['id'] == tile_id)
    result = []
    for axis, cut, first, second in _cuts(focus['box'], plan['gap']):
        left, right = [_counts(labels[tile_slice({'box': box})], classes) for box in (first, second)]
        row = {'axis': axis, 'cut': cut, 'first_box': first, 'second_box': second,
               'blocker_counts': [int(left[index]), int(right[index])]}
        if min(row['blocker_counts']) < MINIMUM:
            result.append(dict(row, status='blocker_below_minimum'))
            continue
        answer = _assign(np.stack([*other_counts, left, right], axis=1))
        if answer['status'] == 'feasible':
            labels_for_indices = [tile['id'] for tile in others] + ['first', 'second']
            answer['validation_blocks'] = [
                labels_for_indices[i] for i in answer.pop('validation_indices')
            ]
        result.append(dict(row, **answer))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'labels', 'previous-registration', 'previous-receipt'):
        parser.add_argument('--' + name, required=True, type=Path)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        raise ValueError('Unexpected feasibility checker commit')
    require_frozen_code(code)
    plan = read_pinned(args.plan, args.plan_sha256)
    validate_tile_plan(plan)
    previous = read_pinned(args.previous_registration, args.previous_registration_sha256)
    receipt = read_pinned(args.previous_receipt, args.previous_receipt_sha256)
    if (previous['plan_sha256'] != args.plan_sha256 or previous['gt_sha256'] != args.labels_sha256
            or receipt['registration_sha256'] != args.previous_registration_sha256
            or receipt['status'] != 'solver_infeasible' or len(receipt['singleton_tile_classes']) != 1):
        raise ValueError('Previous whole-tile failure does not bind these inputs')
    if hash_file(args.labels) != sha(args.labels_sha256):
        raise ValueError('Label file changed')

    _, key, classes, _ = DATASETS[SCENE]
    blocker, = receipt['singleton_tile_classes']
    out = external_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    registration = {'commit': code['commit'], 'scene': SCENE, 'plan_sha256': args.plan_sha256,
                    'labels_sha256': args.labels_sha256,
                    'previous_registration_sha256': args.previous_registration_sha256,
                    'previous_receipt_sha256': args.previous_receipt_sha256,
                    'blocker': blocker, 'gap': plan['gap'], 'minimum_side': MINIMUM_SIDE,
                    'minimum_per_class_in_each_role': MINIMUM,
                    'rule': 'all horizontal and vertical guarded cuts of the single blocker tile',
                    'label_values_scope': 'original_train_tiles_only'}
    write_json(out / 'registration.json', registration)
    start = perf_counter()
    labels = loadmat(args.labels, variable_names=[key + '_gt'])[key + '_gt']
    candidates = scan_guarded_cuts(labels, plan, classes, blocker)
    for name in ('plan', 'labels', 'previous_registration', 'previous_receipt'):
        if hash_file(getattr(args, name)) != getattr(args, name + '_sha256'):
            raise RuntimeError('Frozen feasibility input changed')
    require_frozen_code(code)
    statuses = [row['status'] for row in candidates]
    status = ('feasible_within_scope' if 'feasible' in statuses else
              'unresolved' if 'unresolved' in statuses else 'infeasible_within_scope')
    write_json(out / 'receipt.json',
               {'status': status, 'registration_sha256': hash_file(out / 'registration.json'),
                'candidates': candidates, 'elapsed_seconds': perf_counter() - start,
                'raw_gt_array_decoded_in_curator': True, 'original_val_test_values_examined': False,
                'releases_published': 0, 'optimizer_updates': 0})


if __name__ == '__main__':
    main()
