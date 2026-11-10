import unittest

import numpy as np

from loadData.guarded_tile_feasibility import scan_guarded_cuts
from loadData.spatial_tiles import make_tile_plan, tile_slice


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
