"""Reversible interventions, simultaneous message directions and paired accounting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.cross_branch_interaction import BidirectionalSelectiveInteraction
from model.region_pixel_alignment import RegionPixelAlignment
from loadData.spatial_tiles import make_tile_plan, fit_tile_statistics
from train_spatial import SpatialConfig
from scripts.intervene_screening_messages import retain_messages, message_strengths, paired_effect, run_validation, check_equivalent

torch.set_num_threads(1)


class InterventionTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(8)
        layer = BidirectionalSelectiveInteraction(4, mode='gated', hidden_dim=3, strength=.1, chunk_size=0).eval()
        model = SimpleNamespace(training=False, backbone=SimpleNamespace(interaction=layer))
        return model, layer, torch.randn(6, 4), torch.randn(2, 4), torch.tensor([[1., 0.]]*3 + [[0., 1.]]*3)

    def test_disabled_targets_are_identity_and_enabled_direction_reads_old_snapshot(self):
        model, layer, pixels, regions, q = self.fixture()
        state = {key: value.clone() for key, value in layer.state_dict().items()}
        outputs = {}
        for condition in ('original', 'cnn_to_graph', 'graph_to_cnn', 'none'):
            with retain_messages(model, condition):
                outputs[condition] = layer.forward_aligned(pixels, regions, q, RegionPixelAlignment())[:2]
            self.assertEqual(message_strengths(model), {'graph_to_cnn': .1, 'cnn_to_graph': .1})
        torch.testing.assert_close(outputs['none'][0], pixels, rtol=0, atol=0)
        torch.testing.assert_close(outputs['none'][1], regions, rtol=0, atol=0)
        torch.testing.assert_close(outputs['cnn_to_graph'][0], pixels, rtol=0, atol=0)
        torch.testing.assert_close(outputs['graph_to_cnn'][1], regions, rtol=0, atol=0)
        torch.testing.assert_close(outputs['original'][0], outputs['graph_to_cnn'][0], rtol=0, atol=0)
        torch.testing.assert_close(outputs['original'][1], outputs['cnn_to_graph'][1], rtol=0, atol=0)
        self.assertFalse(torch.equal(outputs['original'][0], pixels))
        self.assertFalse(torch.equal(outputs['original'][1], regions))
        for key, value in layer.state_dict().items(): self.assertTrue(torch.equal(value, state[key]))

    def test_context_restores_after_error_and_nested_interventions(self):
        model, _, _, _, _ = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with retain_messages(model, 'cnn_to_graph'):
                self.assertEqual(message_strengths(model)['graph_to_cnn'], 0.)
                with retain_messages(model, 'none'):
                    self.assertEqual(message_strengths(model)['cnn_to_graph'], 0.)
                self.assertEqual(message_strengths(model)['cnn_to_graph'], .1)
                raise RuntimeError('injected')
        self.assertEqual(message_strengths(model), {'graph_to_cnn': .1, 'cnn_to_graph': .1})
        with self.assertRaises(ValueError):
            with retain_messages(model, 'invalid'): pass
        model.training = True
        with self.assertRaises(ValueError):
            with retain_messages(model, 'none'): pass

    def test_no_interaction_model_is_a_valid_negative_control(self):
        model = SimpleNamespace(training=False, backbone=SimpleNamespace(interaction=None))
        for condition in ('original', 'cnn_to_graph', 'graph_to_cnn', 'none'):
            with retain_messages(model, condition) as info:
                self.assertEqual(info['original_strengths'], info['effective_strengths'])
        self.assertIsNone(model.backbone.interaction)

    def test_paired_counts_and_support_mismatch(self):
        truth = np.array([1, 1, 2, 2])
        first = np.array([1, 2, 2, 1])-1
        second = np.array([2, 1, 2, 2])-1
        def payload(prediction):
            return {'truth': truth, 'global_ids': np.arange(4), 'tile_ids': np.zeros(4),
                    'probabilities': np.tile(np.eye(2)[prediction][:, None], (1, 3, 1))}
        result = paired_effect(payload(first), payload(second), focus_class=1, confused_as=2)
        for branch in result.values():
            self.assertEqual(branch['all']['corrected'], 2)
            self.assertEqual(branch['all']['harmed'], 1)
            self.assertEqual(branch['all']['oa_delta_pp'], 25.)
            self.assertEqual(branch['focus']['oa_delta_pp'], 0.)
        different = payload(second); different['global_ids'] = np.arange(4)[::-1]
        with self.assertRaises(ValueError): paired_effect(payload(first), different)
        with self.assertRaises(AssertionError): check_equivalent(payload(first), payload(second))
        missing = paired_effect(payload(first), payload(first), focus_class=6)
        self.assertIsNone(missing['fused']['focus']['oa_delta_pp'])
        json.dumps(missing, allow_nan=False)

    def test_full_collection_only_visits_validation_and_restores_model(self):
        torch.manual_seed(4)
        config = SpatialConfig({'name': 'intervention_fixture', 'dataset': 'paviaU',
            'model': {'graph_mode': 'learned', 'latent_channels': 8, 'adapter_embed_dim': 8, 'norm_type': 'group_norm'},
            'interaction': {'interaction_mode': 'gated'}, 'training': {'epochs': 0, 'steps_per_epoch': 2},
            'tile_size': 12, 'gap': 1})
        model = config.research.build_model({'paviaU': 9}).eval()
        cube = np.random.default_rng(4).normal(size=(60, 60, 6)).astype(np.float32)
        labels = np.tile(np.arange(1, 10), 400).reshape(60, 60)
        plan = make_tile_plan((60, 60), 12, 1); stats = fit_tile_statistics(cube, plan)
        from scripts.intervene_screening_messages import collect_partition
        partitions = []
        def guarded(*args, **kwargs):
            partitions.append(args[5]); self.assertEqual(args[5], 'val')
            return collect_partition(*args, **kwargs)
        before = message_strengths(model)
        with tempfile.TemporaryDirectory() as directory, patch('scripts.intervene_screening_messages.collect_partition', side_effect=guarded):
            summaries, original, checks = run_validation(model, cube, labels, plan, stats, None, 50,
                                                         list(range(1, 10)), Path(directory))
        self.assertEqual(partitions, ['val']*4)
        self.assertEqual(message_strengths(model), before)
        self.assertEqual(set(summaries), {'original', 'cnn_to_graph', 'graph_to_cnn', 'none'})
        self.assertEqual(len(checks), 4)
        self.assertEqual(summaries['original']['paired_vs_original']['fused']['all']['net_correct'], 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
