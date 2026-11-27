"""Reconstruct frozen C8 features through the existing C6 independent NPY reader.

Run only against a completed, sealed experiment. This is not another data reader
or an alternate point-selection protocol. No model fitting occurs here.
"""
import argparse
from pathlib import Path

import numpy as np

from c8_bridge import PairInputs, fresh, write, digest, read, protocol
from audit import cache
from independent import descriptor_reference, assert_close
from audit_inputs import independent_npy


def audit_original_features(expected, experiment, c7_root, inputs, plan):
    root = Path(experiment)
    if not read(root/'fitting-complete.json')['all_fitting_complete_before_any_development_array'] or not (root/'evaluation01/evaluation.json').is_file():
        raise ValueError('Completed frozen fitting and development evaluation required')
    gate = PairInputs(expected_commit=expected, run_root=c7_root, inputs=inputs, plan_file=plan, direction=('xuzhou', 'whu-honghu'))
    reference = {}
    read_tiles = set()
    for scene in ('xuzhou', 'whu-honghu'):
        bank = gate.banks[scene]
        for role in ('devtrain', 'devval'):
            values = {rep: [] for rep in ('raw_distribution', 'legacy_scene_distribution')}
            points, parents, labels = [], [], []
            for tile, selected in sorted(gate._selected[scene][role].items()):
                row = bank.rows[tile]
                box = row['box']
                shape = (box[1]-box[0], box[3]-box[2], bank.pin['shape'][2])
                path = Path(inputs)/scene/f'tile-{tile}.npy'
                # The existing C7 production gate validates the NPY pin before
                # the existing C6 independent reader interprets its bytes.
                phase = ('source_fit' if role == 'devtrain' else 'source_development') if scene == 'xuzhou' else ('target_adapt' if role == 'devtrain' else 'target_development')
                admitted = gate.batch(phase, tile)
                if digest(path) != admitted['original_npy_sha256']:
                    raise ValueError('Original selected tile digest differs')
                raw = independent_npy(path, shape)
                low = np.asarray(bank.registration['statistics']['minimum'], np.float32)
                high = np.asarray(bank.registration['statistics']['maximum'], np.float32)
                normalized = np.empty_like(raw)
                for band in range(shape[2]):
                    normalized[:, :, band] = (raw[:, :, band]-low[band])/(high[band]-low[band] if high[band] > low[band] else 1)
                for point in selected:
                    pid = point['point_id']
                    if row['role'] != role or point['parent_id'] != row['parent_tile'] or [pid, point['original_label']] not in bank.samples[tile]:
                        raise ValueError('Frozen role, parent or released label differs')
                    yy, xx = divmod(pid, bank.pin['shape'][1])
                    yy -= box[0]
                    xx -= box[2]
                    if not 0 <= yy < shape[0] or not 0 <= xx < shape[1]:
                        raise ValueError('Original point outside its authorized tile')
                    for rep, image in [('raw_distribution', raw), ('legacy_scene_distribution', normalized)]:
                        patch = np.empty((1, shape[2], 7, 7), np.float32)
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                patch[0, :, dy+3, dx+3] = image[min(max(yy+dy, 0), shape[0]-1), min(max(xx+dx, 0), shape[1]-1)]
                        values[rep].append(descriptor_reference(patch)[0])
                    points.append(pid)
                    parents.append(point['parent_id'])
                    labels.append(point['common_id'])
                read_tiles.add((scene, tile))
            reference[scene, role] = dict(**values, point_ids=points, parent_ids=parents, y=labels)
    caches = []
    for source, target in protocol()['directions']:
        caches += [(root/f'source-{source}-to-{target}'/'features', source, 'devtrain'), (root/f'target-{source}-to-{target}'/'features', target, 'devtrain')]
    caches += [(root/'evaluation01'/f'features-{scene}', scene, 'devval') for scene in ('xuzhou', 'whu-honghu')]
    for prefix, scene, role in caches:
        data, meta = cache(prefix)
        if meta['scene'] != scene or meta['original_role'] != role:
            raise ValueError('Cache scene/role differs')
        for field, expected_values in reference[scene, role].items():
            assert_close(data[field], expected_values, atol=1e-8, rtol=1e-7, path=field)
    if len(read_tiles) != 17:
        raise ValueError('Frozen seventeen-tile scope differs')
    return dict(passed=True, scope='Original C7 membership, C6 independent NPY bytes, per-band frozen normalization, clipped context and independent distribution descriptors', selected_tiles=17, unique_labeled_points=40, feature_caches_checked=6, new_label_memberships=0, optimizer_updates=0, query_reads=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('expected-commit', 'experiment', 'c7-root', 'inputs', 'plan', 'out'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    out = fresh(args.expected_commit, args.out)
    result = audit_original_features(args.expected_commit, args.experiment, args.c7_root, args.inputs, args.plan)
    write(out, result)
    print(__import__('json').dumps(result))


if __name__ == '__main__':
    main()
