import unittest
from collections import deque
import numpy as np

from evaluation.spatial_diagnostics import spatial_support
from scripts.audit_spatial_support import split_masks
from loadData.fixed_splits import label_hash


class SpatialSupportTests(unittest.TestCase):
    def test_split_partition_guards(self):
        gt = np.array([[1, 0, 2, 1]])
        split = {'shape': [1, 4], 'label_hash': label_hash(gt),
                 'indices': {'train': [0], 'val': [2], 'test': [3]}}
        self.assertEqual(sum(int(m.sum()) for m in split_masks(gt, split).values()), 3)
        with self.assertRaises(ValueError):
            split_masks(gt, {**split, 'label_hash': 'wrong'})
        for bad in ([0], [1], [4], [-1], [3, 3], [2.5], []):
            with self.assertRaises(ValueError):
                split_masks(gt, {**split, 'indices': {**split['indices'], 'test': bad}})

    def test_geometry_and_scoring_are_distinct(self):
        gt = np.array([[1, 1, 2, 2]])
        score = np.array([[False, True, False, False]])
        strict = spatial_support(gt, [1, 2], geometry_mask=score, score_mask=score)
        whole = spatial_support(gt, [1, 2], geometry_mask=np.ones_like(score), score_mask=score)
        self.assertEqual(strict['widths']['1']['boundary']['count'], 0)
        self.assertEqual(whole['widths']['1']['boundary']['per_class_count'], [1, 0])
        self.assertEqual(whole['geometry_label_count_outside_scoring'], 3)
        self.assertFalse(whole['widths']['1']['boundary']['full_class_coverage'])

    def test_unknown_holes_and_diagonals_do_not_create_boundaries(self):
        gt = np.array([[1, 0], [0, 2]])
        mask = np.ones_like(gt, dtype=bool)
        result = spatial_support(gt, [1, 2], geometry_mask=mask, score_mask=mask)
        self.assertEqual(result['score_unlabeled_count'], 2)
        for row in result['widths'].values():
            self.assertFalse(row['boundary_evaluation_available'])
            self.assertEqual(row['observed_nonboundary']['count'], 2)
        gt = np.array([[1, 2, 0, 2, 2]])
        mask = np.ones_like(gt, dtype=bool)
        self.assertEqual(spatial_support(gt, [1, 2], geometry_mask=mask,
                                        score_mask=mask)['widths']['3']['boundary']['count'], 2)

    def test_width_and_class_order(self):
        gt = np.array([[1, 1, 1, 2, 2, 2]])
        mask = np.ones_like(gt, dtype=bool)
        result = spatial_support(gt, [2, 1, 3], geometry_mask=mask, score_mask=mask)
        for width in (1, 2, 3):
            row = result['widths'][str(width)]
            self.assertEqual(row['boundary']['per_class_count'], [width, width, 0])
            self.assertEqual(row['boundary']['count'] + row['observed_nonboundary']['count'], 6)

    def test_input_validation_and_no_mutation(self):
        gt = np.array([[1., np.nan], [2., 99.]])
        mask = np.array([[True, False], [True, False]])
        original = gt.copy()
        spatial_support(gt, [1, 2], geometry_mask=mask, score_mask=mask)
        np.testing.assert_array_equal(gt, original)
        with self.assertRaises(ValueError):
            spatial_support(gt, [1, 2], geometry_mask=mask, score_mask=~mask)
        with self.assertRaises(ValueError):
            spatial_support(gt, [1, 2], geometry_mask=mask.astype(int), score_mask=mask)
        for bad in (np.array([[1.5]]), np.array([[-1]]), np.array([[99]])):
            one = np.ones_like(bad, dtype=bool)
            with self.assertRaises(ValueError):
                spatial_support(bad, [1, 2], geometry_mask=one, score_mask=one)
        for widths in ((), (0,), (1, 1), (True,)):
            with self.assertRaises(ValueError):
                spatial_support(gt, [1, 2], geometry_mask=mask, score_mask=mask, widths=widths)

    def test_empty_scoring(self):
        gt = np.array([[1, 2]])
        empty = np.zeros_like(gt, dtype=bool)
        result = spatial_support(gt, [1, 2], geometry_mask=empty, score_mask=empty)
        self.assertEqual(result['scoring']['count'], 0)
        self.assertFalse(result['widths']['1']['boundary_evaluation_available'])

    def test_independent_graph_distance_oracle(self):
        rng = np.random.default_rng(1729)
        for _ in range(50):
            gt = rng.integers(0, 4, size=(7, 9))
            geometry = rng.random(gt.shape) > .2
            score = geometry & (rng.random(gt.shape) > .4)
            result = spatial_support(gt, [1, 2, 3], geometry_mask=geometry, score_mask=score)
            valid = {(r, c) for r in range(7) for c in range(9) if geometry[r, c] and gt[r, c]}
            def neighbors(p):
                r, c = p
                return {(r-1, c), (r+1, c), (r, c-1), (r, c+1)} & valid
            distance = {p: 0 for p in valid if any(gt[p] != gt[q] for q in neighbors(p))}
            queue = deque(distance)
            while queue:
                p = queue.popleft()
                for q in neighbors(p):
                    if q not in distance:
                        distance[q] = distance[p] + 1
                        queue.append(q)
            for width in (1, 2, 3):
                expected = [sum(score[p] and gt[p] == c and d < width for p, d in distance.items())
                            for c in (1, 2, 3)]
                self.assertEqual(result['widths'][str(width)]['boundary']['per_class_count'], expected)


if __name__ == '__main__':
    unittest.main()
