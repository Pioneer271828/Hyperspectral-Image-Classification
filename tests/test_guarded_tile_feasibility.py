import unittest

import numpy as np

from loadData.data_read import read_band_metadata
from loadData.spatial_tiles import make_tile_plan, tile_slice
from model.research_config import digest
from transfer.g3_training import Scene
from transfer.guarded_spatial_roles import guarded_roles, prepare_labels
from transfer.guarded_tile_feasibility import scan_guarded_cuts
from transfer.p4_guarded import observation_provider, validate_geometry
from transfer.p4_spatial import fit_role_statistics, load_release


class GuardedTileFeasibilityTest(unittest.TestCase):
    def setUp(self):
        self.plan = make_tile_plan((48, 64), tile_size=16, gap=2)
        self.train = [tile for tile in self.plan['tiles'] if tile['partition'] == 'train']
        self.labels = np.zeros(self.plan['shape'], dtype=np.int16)
        self.target = self.train[0]
        for tile in self.train[1:3]:
            self.labels[tile_slice(tile)][0, :5] = 1
        self.blocker = {'class_id': 2, 'only_tile': self.target['id'], 'available': 10}
        target = self.labels[tile_slice(self.target)]
        target[0, :5] = 2
        target[0, -5:] = 2

    def scan(self, labels=None):
        return scan_guarded_cuts(self.labels if labels is None else labels,
                                 self.plan, 2, self.blocker)

    def test_feasible_cut_and_heldout_labels_do_not_affect_scan(self):
        original = self.scan()
        poison = self.labels.copy()
        for tile in self.plan['tiles']:
            if tile['partition'] != 'train':
                poison[tile_slice(tile)] = 99
        self.assertEqual(self.scan(poison), original)
        self.assertEqual(len(original), 6)
        feasible, = [row for row in original if row['status'] == 'feasible']
        self.assertEqual(feasible['axis'], 'column')
        self.assertEqual(feasible['second_box'][2] - feasible['first_box'][3], 2)
        self.assertEqual(feasible['blocker_counts'], [5, 5])
        self.assertEqual(feasible['train_counts'], [5, 5])
        self.assertEqual(feasible['validation_counts'], [5, 5])

    def test_class_counts_must_hold_jointly(self):
        self.labels[tile_slice(self.train[2])][0, :5] = 0
        self.labels[tile_slice(self.train[1])][1, :5] = 1
        screened = [row for row in self.scan() if min(row['blocker_counts']) >= 5]
        self.assertEqual([row['status'] for row in screened], ['solver_infeasible'])

    def test_changed_blocker_is_rejected(self):
        self.labels[tile_slice(self.train[1])][1, 0] = 2
        with self.assertRaisesRegex(ValueError, 'single-tile class'):
            self.scan()

    def test_roles_holdout_untouched_and_sparse_budget_nested(self):
        witness, = [row for row in self.scan() if row['status'] == 'feasible']
        labels = self.labels.copy()
        for tile in self.plan['tiles']:
            if tile['partition'] != 'train':
                labels[tile_slice(tile)] = -1
        roles, support, val = prepare_labels(labels, self.plan, 2, self.blocker, witness)
        self.assertEqual(len(roles), len(self.plan['tiles']) + 1)
        self.assertEqual(len(val['indices']), 10)
        original_test = [t for t in self.plan['tiles'] if t['partition'] == 'test']
        self.assertEqual([dict(t, role='test') for t in original_test],
                         [t for t in roles if t['role'] == 'test'])
        for seed in (512, 2048):
            small, large = set(support[seed, 1]['indices']), set(support[seed, 5]['indices'])
            self.assertEqual((len(small), len(large)), (2, 10))
            self.assertTrue(small < large)
            self.assertFalse(large & set(val['indices']))

    def test_modified_witness_and_external_role_are_rejected(self):
        witness, = [row for row in self.scan() if row['status'] == 'feasible']
        with self.assertRaisesRegex(ValueError, 'alters the buffer'):
            guarded_roles(self.plan, self.blocker, dict(witness, cut=witness['cut'] + 1))
        holdout = next(t['id'] for t in self.plan['tiles'] if t['partition'] == 'test')
        with self.assertRaisesRegex(ValueError, 'original training membership'):
            guarded_roles(self.plan, self.blocker,
                          dict(witness, validation_blocks=[*witness['validation_blocks'], holdout]))

    def runtime_geometry(self):
        witness, = [row for row in self.scan() if row['status'] == 'feasible']
        geometry = {'schema_version': 1, 'kind': 'guarded_honghu_development_v1',
                    'scene': 'whu-honghu', 'parent_plan_hash': self.plan['plan_hash'],
                    'shape': self.plan['shape'], 'roles': guarded_roles(self.plan, self.blocker, witness)}
        geometry['development_roles_hash'] = digest(geometry)
        scan = {'scene': 'whu-honghu', 'blocker': self.blocker}
        audit = {'status': 'passed', 'chosen_geometry_for_followup': witness}
        runtime, allowed = validate_geometry(self.plan, geometry, scan, audit)
        return geometry, runtime, allowed

    def test_runtime_cannot_observe_superseded_parent_or_heldout_tiles(self):
        _, runtime, allowed = self.runtime_geometry()
        cube = np.arange(np.prod((*runtime['shape'], 3)), dtype=np.float32).reshape(*runtime['shape'], 3)
        stats = fit_role_statistics(cube, runtime, allowed['train'], runtime['plan_hash'])
        bands = read_band_metadata('whu-honghu', 3)
        view = observation_provider(Scene('whu-honghu', cube, runtime, bands, 22, stats), allowed)
        poisoned = np.full_like(cube, np.nan)
        for tiles in allowed.values():
            for tile in tiles:
                poisoned[tile_slice(tile)] = cube[tile_slice(tile)]
        self.assertEqual(fit_role_statistics(poisoned, runtime, allowed['train'], runtime['plan_hash']), stats)
        altered_view = observation_provider(Scene('whu-honghu', poisoned, runtime, bands, 22, stats), allowed)
        for tiles in allowed.values():
            for tile in tiles:
                self.assertIs(runtime['tiles'][tile['id']], tile)
                np.testing.assert_array_equal(view(tile).observation.spectral,
                                              altered_view(tile).observation.spectral)
        for tile in (self.target, next(t for t in self.plan['tiles'] if t['partition'] == 'test')):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                view(tile)

    def test_guarded_loader_rejects_support_from_validation_or_test(self):
        geometry, _, allowed = self.runtime_geometry()
        width = self.plan['shape'][1]

        def coords(tiles):
            return sorted(y * width + x for tile in tiles for y in range(tile['box'][0], tile['box'][1])
                          for x in range(tile['box'][2], tile['box'][3]))

        train, val = coords(allowed['train']), coords(allowed['val'])
        support = {'seed': 512, 'per_class': 1, 'indices': train[:22], 'labels': list(range(1, 23))}
        validation = {'seed': 3031, 'per_class': 5, 'indices': val[:110],
                      'labels': list(range(1, 23)) * 5}
        payload = {'scene': 'whu-honghu', 'development_roles_hash': geometry['development_roles_hash'],
                   'support': support, 'validation': validation}
        role = load_release(dict(payload, release_hash=digest(payload)), self.plan, 512, 1, geometry,
                            guarded_tiles=allowed)
        self.assertEqual(sum(np.count_nonzero(v) for _, v in role['val']), 110)
        heldout = [t for t in self.plan['tiles'] if t['partition'] == 'test']
        for forbidden in (val[110], coords(heldout)[0]):
            poisoned = dict(support, indices=sorted([*train[:21], forbidden]))
            record = dict(payload, support=poisoned)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                load_release(dict(record, release_hash=digest(record)), self.plan, 512, 1, geometry,
                             guarded_tiles=allowed)
