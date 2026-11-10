import unittest

import numpy as np

from loadData.spatial_tiles import make_tile_plan, tile_slice
from transfer.guarded_spatial_roles import guarded_roles, prepare_labels
from transfer.guarded_tile_feasibility import scan_guarded_cuts


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
