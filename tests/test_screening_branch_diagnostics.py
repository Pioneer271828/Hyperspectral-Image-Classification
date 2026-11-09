"""Hand-calculated target attribution and frozen diagnostic instrumentation checks."""
import unittest
import json

import numpy as np
import torch

from scripts.diagnose_screening_branches import summarize, target_confusion, describe, collect_partition, prediction_equivalence
from train_spatial import SpatialConfig, evaluate
from loadData.spatial_tiles import make_tile_plan, fit_tile_statistics

torch.set_num_threads(1)


class TargetAttributionTests(unittest.TestCase):
    def test_fused_confusion_can_occur_when_both_auxiliary_heads_are_correct(self):
        labels = np.array([6, 6, 6, 6, 6, 6])
        predictions = {'cnn': np.array([6, 6, 2, 2, 2, 2]),
                       'graph': np.array([6, 2, 6, 2, 2, 2]),
                       'fused': np.array([2, 2, 2, 2, 6, 6])}
        result = summarize(labels, predictions, [2, 6])
        focus = result['target_confusion']
        self.assertEqual(focus['fused_confusion_count'], 4)
        self.assertEqual(list(focus['fused_confusion_aux_strata'].values()), [1, 1, 1, 1])
        self.assertEqual(focus['fused_confusion_both_aux_same_wrong'], 1)
        self.assertEqual(np.array(focus['joint_counts']).sum(), 6)
        self.assertEqual(result['branches']['harm']['rate'], 1)
        self.assertEqual(result['branches']['strata']['both_wrong']['fused_correct'], 2)

    def test_no_focus_support_and_masked_invalid_values_stay_undefined(self):
        labels = np.array([2, 6, 0])
        values = np.array([2., np.nan, -1.])
        mask = np.array([True, False, True])
        result = summarize(labels, {key: values for key in ('cnn', 'graph', 'fused')}, [2, 6], mask=mask)
        self.assertEqual(result['target_confusion']['support'], 0)
        self.assertIsNone(result['target_confusion']['fused_confusion_rate'])
        self.assertIsNone(result['branches_by_true_class']['6']['correction']['rate'])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            target_confusion(labels, {key: values for key in ('cnn', 'graph', 'fused')}, [2, 6], 6, 2)

    def test_gate_summary_is_descriptive_and_rejects_invalid_values(self):
        self.assertEqual(describe(np.ones(4))['mean'], 1.)
        self.assertIsNone(describe(np.array([]))['mean'])
        for bad in ([np.nan], [-.1], [1.1]):
            with self.assertRaises(ValueError): describe(bad)
        with self.assertRaises(ValueError):
            target_confusion(np.array([1]), {'cnn': np.array([1]), 'graph': np.array([1]), 'fused': np.array([1])}, [1], 1, 1)


class FrozenCollectionTests(unittest.TestCase):
    def test_roundoff_tolerance_never_accepts_changed_prediction_ids(self):
        reference = torch.tensor([[.2, .8]])
        actual = reference + torch.tensor([[1e-5, -1e-5]])
        self.assertLess(prediction_equivalence([actual], [reference])[0], 2e-5)
        with self.assertRaises(AssertionError):
            prediction_equivalence([reference + .001], [reference])
        with self.assertRaises(AssertionError):
            prediction_equivalence([torch.tensor([[.5000001, .4999999]])],
                                   [torch.tensor([[.4999999, .5000001]])])

    def test_requesting_gates_preserves_eval_outputs_state_and_coordinate_order(self):
        torch.manual_seed(12)
        config = SpatialConfig({'name': 'frozen_gate_fixture', 'dataset': 'paviaU',
            'model': {'graph_mode': 'learned', 'latent_channels': 8, 'adapter_embed_dim': 8, 'norm_type': 'group_norm'},
            'interaction': {'interaction_mode': 'gated'}, 'training': {'epochs': 0, 'steps_per_epoch': 2},
            'tile_size': 12, 'gap': 1})
        cube = np.random.default_rng(11).normal(size=(60, 60, 6)).astype(np.float32)
        labels = np.tile(np.arange(1, 10), 400).reshape(60, 60)
        plan = make_tile_plan((60, 60), 12, 1); stats = fit_tile_statistics(cube, plan)
        model = config.research.build_model({'paviaU': 9}).eval()
        expected = evaluate(model, cube, labels, plan, stats, 'test', 'paviaU', 9, None, 50)
        report, arrays, tiles = collect_partition(model, cube, labels, plan, stats, 'test', 'paviaU', None, 50, list(range(1, 10)))
        self.assertEqual(report['classification']['fused']['confusion_matrix'], expected['confusion'])
        self.assertAlmostEqual(report['fused_nll'], expected['nll'], places=6)
        self.assertTrue(report['state_unchanged'] and report['diagnostic_flag_equivalent'])
        np.testing.assert_array_equal(arrays['truth'], labels.reshape(-1)[arrays['global_ids']])
        self.assertEqual(len(set(arrays['global_ids'])), len(arrays['truth']))
        self.assertEqual(sum(t['count'] for t in tiles), len(arrays['truth']))
        for name in ('graph_to_cnn_gate', 'cnn_to_graph_gate_pixels'):
            self.assertEqual(arrays[name].shape, arrays['truth'].shape)
        self.assertEqual(len(model._forward_pre_hooks), 0)
        with self.assertRaises(ValueError):
            collect_partition(model.train(), cube, labels, plan, stats, 'test', 'paviaU', None, 50, list(range(1, 10)))


if __name__ == '__main__':
    unittest.main()
