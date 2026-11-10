from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compression.int8 import IntegerMatrix
from compression.int8_checkpoint import load_int8, save_int8
from compression.precision_fallback import HEADS, LAYER_ORDER, register_ptq_fallback, run_ptq_fallback
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from test_int8_model import fixture, predict
from transfer.g3_training import state_hash


class PrecisionFallbackTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.scene, self.record = fixture()
        self.calibration = [tile['id'] for tile in self.record['roles'] if tile['role'] == 'devtrain'][:2]
        self.plan = self.register()

    def register(self):
        return register_ptq_fallback(self.reference, self.scene, self.record, self.calibration)

    def review(self, scene=None, plan=None):
        plan = self.plan if plan is None else plan
        return run_ptq_fallback(self.reference, self.scene if scene is None else scene, self.record,
                                plan=plan, expected_plan_sha256=digest(plan))

    def test_actual_integer_candidates_and_selected_model_reload(self):
        with torch.no_grad():
            for family in (self.reference.heads, self.reference.hgnn_heads, self.reference.cnn_heads):
                head = family[self.scene.dataset]
                head.weight.zero_()
                head.bias.copy_(torch.arange(head.out_features))
        self.plan = self.register()
        original = state_hash(self.reference)
        changed = deepcopy(self.scene)
        for tile in self.record['roles']:
            if tile['role'] in ('test', 'unused'):
                changed.cube[tile_slice(tile)] = np.nan
        model, result = self.review(scene=changed)
        self.assertEqual(result['selected_low_precision_paths'], list(LAYER_ORDER))
        self.assertEqual(result['protected_small_classes'], self.record['classes'])
        self.assertEqual(result['kept_fp32_paths'], [])
        self.assertFalse(result['query_labels_accessed'])
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertEqual(state_hash(self.reference), original)
        self.assertTrue(all(isinstance(model.get_submodule(path), IntegerMatrix) for path in LAYER_ORDER))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'selected.pt'
            pin = save_int8(model, path)
            restored = load_int8(path, pin)
            for actual, expected in zip(predict(restored, self.scene, self.record),
                                        predict(self.reference, self.scene, self.record), strict=True):
                self.assertTrue(torch.equal(actual, expected))

    def test_one_new_small_class_error_forces_layer_fallback_even_when_totals_improve(self):
        seen = []

        def score(model, scene, blocks, class_ids):
            truth = np.concatenate([values for _, _, values in blocks])
            paths = tuple(path for path in LAYER_ORDER if isinstance(model.get_submodule(path), IntegerMatrix))
            before = truth.copy()
            same_class = np.flatnonzero(truth == 1)
            other_class = np.flatnonzero(truth == 2)
            before[same_class[0]] = before[other_class[0]] = 3
            after = truth.copy()
            if paths == (LAYER_ORDER[0],):
                # Recover two prior mistakes but create one new error in the same small class.
                after[same_class[1]] = 3
            seen.append(paths)
            return {head: (before if not paths else after) for head in HEADS}

        with patch('compression.precision_fallback._predict', side_effect=score):
            model, result = self.review()
        self.assertEqual(seen, [(), (LAYER_ORDER[0],), (LAYER_ORDER[1],), LAYER_ORDER])
        first, second, joint = result['trials']
        self.assertLess(first['protection']['fused']['oa_drop'], 0.)
        self.assertEqual(first['protection']['fused']['per_class_recall_drop']['1'], 0.)
        self.assertEqual(first['protection']['fused']['new_small_class_errors']['1'], 1)
        self.assertFalse(first['passed'])
        self.assertTrue(second['passed'])
        self.assertTrue(joint['protection']['fused']['passes_metric_guards'])
        self.assertFalse(joint['passed'])
        self.assertEqual(joint['failed_single_layers'], [LAYER_ORDER[0]])
        self.assertEqual(result['kept_fp32_paths'], [LAYER_ORDER[0]])
        self.assertTrue(torch.equal(model.get_submodule(LAYER_ORDER[0]).weight,
                                    self.reference.get_submodule(LAYER_ORDER[0]).weight))
        self.assertIsInstance(model.get_submodule(LAYER_ORDER[1]), IntegerMatrix)

    def test_none_passes_preserves_float_and_rejects_guard_or_validation_drift(self):
        def score(model, scene, blocks, class_ids):
            truth = np.concatenate([values for _, _, values in blocks])
            if any(isinstance(model.get_submodule(path), IntegerMatrix) for path in LAYER_ORDER):
                truth[truth == 1] = 2
            return {head: truth for head in HEADS}

        with patch('compression.precision_fallback._predict', side_effect=score):
            model, result = self.review()
        self.assertEqual(result['selected_low_precision_paths'], [])
        self.assertEqual(result['kept_fp32_paths'], list(LAYER_ORDER))
        self.assertEqual(state_hash(model), state_hash(self.reference))
        self.assertTrue(all(row['protection']['fused']['failed_recall_classes'] == [1] for row in result['trials']))
        modified = deepcopy(self.scene)
        tile = self.scene.plan['tiles'][self.plan['validation']['tiles'][0]['id']]
        modified.cube[tile_slice(tile)] += 0.5
        tampered = deepcopy(self.plan)
        tampered['guards']['fused_max_drop']['recall'] = 0.01
        with patch('compression.precision_fallback._predict', side_effect=AssertionError('must reject before inference')):
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                self.review(scene=modified)
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                self.review(plan=tampered)
