"""Connect the actual custody release and calibration CLI without confirmation labels."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from evaluation.wp11_confirmation import build_requests, confirmation_environment, unused_tiles
from inference.route_controls import request_hash
from model.research_config import digest
from research_runtime import code_snapshot
from scripts.prepare_wp11_gate import MODEL_FILES, LIGHT_PINS, run
from scripts.prepare_wp11_label_custody import AUTHORITY, RELEASE, ROLES, package, plan_roles
from scripts.run_wp11_calibration import run as calibrate
from tests.test_wp11_calibration import read, save
from tests.test_wp11_confirmation import file_pin, synthetic_package
from transfer.g3_training import state_hash


def synthetic_registration(root, code, *, with_design=False, prepared=None):
    if prepared is None:
        base, (full, light, scene, record, _, _, _, _) = synthetic_package(root)
        source = read(base)
    else:
        source, full, light, scene, record = prepared
    tiles = {t['tile_id']: t['file'] for t in source['tiles']}
    available = unused_tiles(scene.plan, record)
    for i in available:
        if i not in tiles:
            box = scene.plan['tiles'][i]['box']
            path = root / f'observation-{i}.npy'
            np.save(path, scene.cube[box[0]:box[1], box[2]:box[3]], allow_pickle=False)
            tiles[i] = file_pin(path)
    obs = save(root, 'observations', dict(status='candidates_prepared', scene=scene.dataset,
               raw_labels_read=False, plan_sha256=scene.plan['plan_hash'], development_sha256=record['validation_hash'],
               tiles=[dict(tile_id=i, box=scene.plan['tiles'][i]['box'], file=tiles[i]) for i in available]))
    plan_spec = save(root, 'plan-spec', dict(geometry=save(root, 'geometry', scene.plan),
                     development=save(root, 'dev-record', record), observations=obs))
    plan = plan_roles(plan_spec, root / 'plan', code)
    role_plan = file_pin(root / 'plan/role-plan.json')
    spec = {k: source[k] for k in (*MODEL_FILES, *LIGHT_PINS)}
    if "matched_release" in source:
        spec["matched_release"] = source["matched_release"]
    classes = len(record["classes"])
    spec.update(role_plan=role_plan, scale=12, benchmark=dict(seed=64, warmups=1, pairs=20),
                settings=dict(name='synthetic custody handoff', criterion='max_probability', thresholds=[0.],
                              min_coverage=1., min_class_support=1, max_drop=dict(oa=1., aa=1., recall=1.)))
    for role in ROLES:
        ids = [b['tile_id'] for b in plan['blocks'] if b['role'] == role]
        released = [dict(tile_id=i, indices=list(range(classes)) if prepared is not None or i == ids[0] else [],
                         labels=list(range(1, classes + 1)) if prepared is not None or i == ids[0] else []) for i in ids]
        sparse = save(root, f'{role}-source', dict(policy=RELEASE, role=role, plan_sha256=role_plan['sha256'], blocks=released))
        authority = save(root, f'{role}-authority', dict(policy=AUTHORITY, status='authorized', role=role,
                         plan_sha256=role_plan['sha256'], source_sha256=sparse['sha256'],
                         scope='held_aside_sparse_development_only', query_access=False))
        out = root / role
        package(save(root, f'{role}-spec', dict(plan=role_plan, role=role, authority=authority, source=sparse)), out, code)
        spec[role] = {key: file_pin(out / f'{key}.json') for key in ('commitment', 'blocks')}
    spec['calibration'].update({key: file_pin(root / 'calibration' / f'{key}.json') for key in ('labels', 'receipt')})
    if with_design:
        requests = build_requests(scene, read(spec['confirmation']['blocks']), spec['scale'])
        fixed = save(root, 'planned-fixed', dict(format='dy-fixed-path-v1', input_hash=request_hash(requests),
                     full_state=state_hash(full), light_state=state_hash(light.model), min_coverage=1.,
                     routes=[dict(id=request['id'], route='full') for request in requests]))
        spec['confirmation_design'] = save(root, 'planned-design', dict(
            fixed=fixed, environment_hash=digest(confirmation_environment()), cost_interval_ms=[.001, 10000.],
            ece_bins=2, nll_epsilon=1e-8, relative_tolerances=dict(p50_ms=.1, p95_ms=.1)))
    return spec


class GateRegistrationTests(unittest.TestCase):
    def test_blind_registration_and_actual_calibration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            code = code_snapshot()
            spec = synthetic_registration(root, code)
            for role in ROLES:
                (root / f'{role}-source.json').unlink()
            for key in ('labels', 'receipt'):
                (root / 'confirmation' / f'{key}.json').unlink()
            calibration_labels = root / 'calibration/labels.json'
            calibration_receipt = root / 'calibration/receipt.json'
            withheld = {p: p.read_bytes() for p in (calibration_labels, calibration_receipt)}
            for p in withheld:
                p.unlink()
            result = run(save(root, 'gate-spec', spec), root / 'registered', code)
            self.assertEqual((result['calibration_tiles'], result['confirmation_tiles']), (1, 1))
            self.assertFalse(result['held_aside_labels_read'])
            handoff = read(result['confirmation'])
            self.assertEqual(handoff['confirmation_labels_sha256'], read(spec['confirmation']['commitment'])['labels_sha256'])
            self.assertNotIn('confirmation/labels.json', str(handoff))
            self.assertNotIn('support', read(spec['confirmation']['commitment']))
            for path, data in withheld.items():
                path.write_bytes(data)
            self.assertEqual(calibrate(result['calibration'], root / 'frozen', code)['status'], 'adaptive_frozen')

    def test_rejects_cross_role_swaps_and_wrong_deferred_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            code = code_snapshot()
            spec = synthetic_registration(root, code)
            swapped = dict(spec, confirmation=spec['calibration'].copy())
            for key in ('labels', 'receipt'):
                swapped['confirmation'].pop(key)
            with self.assertRaisesRegex(ValueError, 'Role-specific'):
                run(save(root, 'swapped', swapped), root / 'bad', code)
            for key in ('labels', 'receipt'):
                calibration = dict(spec['calibration'], **{key: dict(spec['calibration'][key], sha256='0'*64)})
                with self.assertRaisesRegex(ValueError, f'{key} pin differs'):
                    run(save(root, f'wrong-{key}', dict(spec, calibration=calibration)), root / 'bad', code)
            self.assertFalse((root / 'bad').exists())
