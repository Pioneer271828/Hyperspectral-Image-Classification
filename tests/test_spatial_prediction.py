import unittest
import numpy as np
from evaluation.spatial_prediction import spatial_prediction_report, paired_difference
from scripts.evaluate_frozen_m4_spatial import packet_arrays, validate_history
import torch


class SpatialPredictionTests(unittest.TestCase):
    def test_historical_m4_passed_schema_and_budget_guards(self):
        record = {'status': 'passed', 'trained_updates': 13527, 'new_updates_in_evaluation': 0,
                  'test_based_selection': False, 'code': {'commit': '98d2b75cff192f79f14d3ef9dda6687b2cc7bffe'}}
        support = {'status': 'completed'}
        validate_history(record, support)
        for field, value in [('status', 'running'), ('trained_updates', 1503),
                             ('new_updates_in_evaluation', 1), ('test_based_selection', True),
                             ('code', {'commit': 'other'})]:
            with self.assertRaises(ValueError): validate_history({**record, field: value}, support)
        with self.assertRaises(ValueError): validate_history(record, {'status': 'running'})

    def test_original_packet_order_is_validated_before_joint_sort(self):
        labels = np.array([[1, 2, 1]])
        packet = {'pixel_indices': torch.tensor([2, 0, 1]), 'labels': torch.tensor([1, 1, 2]),
                  'heads': {h: {'prediction': torch.tensor([2, 1, 1])} for h in ('cnn', 'graph', 'fused')}}
        ids, truth, heads = packet_arrays(packet, labels, [2, 0, 1])
        np.testing.assert_array_equal(ids, [0, 1, 2])
        np.testing.assert_array_equal(truth, [1, 2, 1])
        np.testing.assert_array_equal(heads['fused'], [1, 1, 2])
        with self.assertRaises(ValueError): packet_arrays(packet, labels, [0, 1, 2])
        packet['labels'][0] = 2
        with self.assertRaises(ValueError): packet_arrays(packet, labels, [2, 0, 1])

    def report(self, gt, heads, indices=None, geometry=None):
        gt = np.asarray(gt)
        if indices is None:
            indices = np.flatnonzero(gt > 0)
        if geometry is None:
            geometry = gt > 0
        return spatial_prediction_report(gt, [1, 2], pixel_indices=indices,
                                         truth=gt.ravel()[indices], heads=heads, geometry_mask=geometry)

    def test_hand_calculated_metrics_and_complementarity(self):
        heads = {'cnn': [1, 1, 1, 2], 'graph': [2, 2, 2, 2], 'fused': [1, 2, 2, 2]}
        r = self.report([[1, 1, 2, 2]], heads)
        b = r['widths']['1']['boundary']
        self.assertEqual(b['heads']['fused']['confusion_matrix'], [[0, 1], [0, 1]])
        self.assertEqual(b['heads']['fused']['oa'], .5)
        self.assertAlmostEqual(b['heads']['fused']['macro_f1'], 1/3)
        self.assertEqual(b['branches']['correction']['denominator'], 2)
        self.assertEqual(b['branches']['correction']['numerator'], 1)
        self.assertEqual(r['widths']['1']['observed_nonboundary']['heads']['fused']['oa'], 1)
        self.assertEqual(r['whole_score']['heads']['fused']['oa'], .75)

    def test_geometry_scope_and_unlabeled_are_distinct(self):
        gt = np.array([[1, 1, 2, 0]])
        heads = dict.fromkeys(('cnn', 'graph', 'fused'), [1])
        mask = np.array([[False, True, False, False]])
        strict = self.report(gt, heads, [1], mask)
        whole = self.report(gt, heads, [1])
        self.assertIsNone(strict['widths']['1']['boundary']['heads']['fused']['oa'])
        self.assertEqual(whole['widths']['1']['boundary']['heads']['fused']['oa'], 1)
        self.assertEqual(whole['widths']['1']['boundary']['heads']['fused']['missing_class_ids'], [2])

    def test_partition_conserves_every_confusion_entry(self):
        rng = np.random.default_rng(42)
        for _ in range(12):
            gt = rng.integers(0, 3, (8, 7))
            n = np.count_nonzero(gt)
            heads = {h: rng.integers(1, 3, n) for h in ('cnn', 'graph', 'fused')}
            r = self.report(gt, heads)
            for row in r['widths'].values():
                for h in heads:
                    combined = (np.array(row['boundary']['heads'][h]['confusion_matrix']) +
                                np.array(row['observed_nonboundary']['heads'][h]['confusion_matrix']))
                    np.testing.assert_array_equal(combined, r['whole_score']['heads'][h]['confusion_matrix'])

    def test_packet_order_truth_shapes_and_ids_rejected(self):
        gt = np.array([[1, 2]])
        heads = dict.fromkeys(('cnn', 'graph', 'fused'), [1, 2])
        for indices in ([1, 0], [0, 0], [0, 2], [-1, 1]):
            with self.assertRaises(ValueError):
                spatial_prediction_report(gt, [1, 2], pixel_indices=indices, truth=[1, 2],
                                          heads=heads, geometry_mask=gt > 0)
        for truth in ([2, 1], [1]):
            with self.assertRaises(ValueError):
                spatial_prediction_report(gt, [1, 2], pixel_indices=[0, 1], truth=truth,
                                          heads=heads, geometry_mask=gt > 0)
        for bad in ({'cnn': [1, 2]}, {**heads, 'graph': [1]}, {**heads, 'fused': [0, 2]}):
            with self.assertRaises(ValueError): self.report(gt, bad)

    def test_empty_outputs_remain_undefined_and_inputs_unchanged(self):
        gt = np.array([[1, 0, 2]])
        heads = {h: np.array([1, 2]) for h in ('cnn', 'graph', 'fused')}
        before = {h: p.copy() for h, p in heads.items()}
        r = self.report(gt, heads)
        for row in r['widths'].values():
            self.assertIsNone(row['boundary']['heads']['fused']['macro_f1'])
            self.assertIsNone(row['boundary']['branches']['correction']['rate'])
        for h in heads: np.testing.assert_array_equal(before[h], heads[h])
        empty = self.report(gt, dict.fromkeys(heads, []), np.array([], dtype=int))
        self.assertIsNone(empty['whole_score']['heads']['fused']['oa'])

    def test_paired_seed_identity_and_sample_sd(self):
        a = {2048: .6, 64: .8, 512: .7}
        b = {512: .5, 2048: .5, 64: .5}
        r = paired_difference(a, b)
        np.testing.assert_allclose(r['values_pp'], [30, 20, 10])
        self.assertAlmostEqual(r['mean_pp'], 20)
        self.assertAlmostEqual(r['sample_sd_pp'], 10)
        self.assertEqual(r['positive_seeds'], 3)
        for bad in ({64: .5}, {**a, 2048: None}, {**a, 2048: np.nan}, {**a, 2048: 1.1}):
            with self.assertRaises(ValueError): paired_difference(a, bad)
        r = paired_difference(dict.fromkeys(a), dict.fromkeys(a))
        self.assertIsNone(r['mean_pp'])


if __name__ == '__main__':
    unittest.main()
