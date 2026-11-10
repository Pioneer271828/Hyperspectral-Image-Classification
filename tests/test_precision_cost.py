from collections import Counter
from copy import deepcopy
import unittest

import torch

from compression.int8_model import calibrate_model, calibration_contract, convert_model
from compression.precision_cost import HEADS, benchmark_precision_cost, prepare_resident_precision_tiles
from compression.precision_fallback import LAYER_ORDER, register_ptq_fallback
from compression.ternary_model import pack_model, ternarize_model
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from research_runtime import tensor_hash
from test_int8_model import fixture
from transfer.g3_training import _forward


class PrecisionCostTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.scene, self.record = fixture()
        ids = [tile['id'] for tile in self.record['roles'] if tile['role'] == 'devtrain'][:1]
        path = LAYER_ORDER[1]
        contract = calibration_contract(self.reference, self.scene, self.record, [path], ids)
        calibrated = calibrate_model(self.reference, self.scene, self.record, paths=[path],
                                     contract=contract, expected_hash=digest(contract))
        self.models = {'fp32': self.reference, 'ptq': convert_model(calibrated), 'ternary': pack_model(
            ternarize_model(self.reference, paths=LAYER_ORDER[:1], threshold_ratio=.05))}
        self.rows = register_ptq_fallback(self.reference, self.scene, self.record, ids)['validation']['tiles'][:2]

    def test_same_resident_tiles_and_real_integer_path_in_balanced_sweeps(self):
        tiles = prepare_resident_precision_tiles(self.scene, self.rows)
        called = []
        integer = self.models['ptq'].get_submodule(LAYER_ORDER[1])
        handle = integer.kernel.register_forward_hook(lambda module, args, result: called.append(result.dtype))
        try:
            result = benchmark_precision_cost(self.models, tiles, self.scene.dataset, warmups=1, repeats=12)
        finally:
            handle.remove()
        self.assertEqual(called, [torch.quint8] * (len(tiles) * (12 + 1 + 2)))
        self.assertEqual(set(Counter(tuple(row['order']) for row in result['samples']).values()), {2})
        self.assertEqual(result['models']['ptq']['precision_paths'], [LAYER_ORDER[1]])
        self.assertEqual(result['models']['ternary']['precision_paths'], [LAYER_ORDER[0]])
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertFalse(result['query_labels_accessed'])
        with torch.inference_mode():
            for arm, model in self.models.items():
                expected = [{head: tensor_hash(value) for head, value in zip(HEADS, _forward(
                    model, self.scene, self.scene.plan['tiles'][tile.tile_id])[:3], strict=True)} for tile in tiles]
                self.assertEqual(result['output_hashes'][arm], expected)

    def test_changed_inputs_parameters_and_unbalanced_order_are_rejected(self):
        altered = deepcopy(self.scene)
        tile = self.scene.plan['tiles'][self.rows[0]['id']]
        altered.cube[tile_slice(tile)] += 0.5
        with self.assertRaisesRegex(ValueError, 'tile bytes changed'):
            prepare_resident_precision_tiles(altered, self.rows)
        tiles = prepare_resident_precision_tiles(self.scene, self.rows)
        with self.assertRaisesRegex(ValueError, 'balanced single-thread repeats'):
            benchmark_precision_cost(self.models, tiles, self.scene.dataset, repeats=13)
        with torch.no_grad():
            self.models['ptq'].heads[self.scene.dataset].bias.add_(.1)
        with self.assertRaisesRegex(ValueError, 'outside its declared assignment'):
            benchmark_precision_cost(self.models, tiles, self.scene.dataset)
