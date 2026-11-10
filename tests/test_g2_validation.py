import hashlib
import json
import unittest

import numpy as np
import torch

from loadData.spatial_tiles import make_tile_plan
from test_g2_contract import fixture
from transfer.g2_observations import pack_source_observations
from transfer.g2_source import build_source_model, require_shared_heads
from transfer.g2_validation import evaluate_source_validation


class SourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.catalog, self.contract, _, _ = fixture()
        self.scene = 'paviaU'
        self.plan = make_tile_plan((24, 24), tile_size=8, gap=0)
        cube = np.random.default_rng(8).random((24, 24, 6), dtype=np.float32)
        waves = np.linspace(420, 850, 6, dtype=np.float32)
        artifact = self.catalog['scenes'][self.scene]['artifact_sha256']
        self.payloads = {}
        releases = self.contract['source_releases'][self.scene]
        for role in ('train', 'val'):
            self.payloads[role] = pack_source_observations(self.scene, artifact, self.plan, role, cube, waves)
            releases[role]['observation_sha256'] = hashlib.sha256(self.payloads[role]).hexdigest()
        self.val = next(tile for tile in self.plan['tiles'] if tile['partition'] == 'val')
        y, _, x, _ = self.val['box']
        i = y * self.plan['shape'][1] + x
        self.labels = json.dumps(dict(scene=self.scene, partition='val', plan_hash=self.plan['plan_hash'],
                                      observation_sha256=releases['val']['observation_sha256'],
                                      indices=[i, i + 1, i + 2], labels=[1, 2, 3])).encode()
        releases['val']['labels_sha256'] = hashlib.sha256(self.labels).hexdigest()
        base = {'model': {'graph_mode': 'learned', 'norm_type': 'group_norm'},
                'training': {'steps_per_epoch': 1}}
        self.model, _ = build_source_model(self.catalog, self.contract, base, seed=64)
        self.model.eval()

    def evaluate(self, scene=None, labels=None, observations=None):
        return evaluate_source_validation(
            self.catalog, self.contract, self.model, scene or self.scene, self.plan,
            self.payloads['train'], self.payloads['val'] if observations is None else observations,
            self.labels if labels is None else labels, superpixel_scale=16)

    def test_fixed_logits_oracle_and_only_registered_source_access(self):
        head = require_shared_heads(self.catalog, self.contract, self.model)
        values = ((2., 1.), (1., 2.), (0., 0.))
        with torch.no_grad():
            for family, bias in zip((self.model.heads, self.model.hgnn_heads, self.model.cnn_heads), values):
                family[head].weight.zero_()
                family[head].bias.copy_(torch.tensor(bias))
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        result = self.evaluate()
        self.assertEqual(result['scored_tiles'], [self.val['id']])
        for name, bias in zip(('fused', 'graph', 'cnn'), values):
            actual = result['metrics'][name]
            expected_nll = np.logaddexp(*bias) - np.mean(bias)
            self.assertAlmostEqual(actual['nll'], expected_nll, places=6)
            self.assertEqual(actual['count'], 2)
            expected_matrix = [[1, 0], [1, 0]] if bias[0] >= bias[1] else [[0, 1], [0, 1]]
            self.assertEqual(actual['confusion_matrix'], expected_matrix)
        self.assertEqual([row['phase'] for row in result['access']],
                         ['fit_statistics', 'validation_graph', 'source_select'])
        self.assertTrue(all(row['scene'] == self.scene for row in result['access']))
        self.assertTrue(all(row['labels_sha256'] is None for row in result['access'][:2]))
        self.assertEqual(result['access'][2]['labels_sha256'], hashlib.sha256(self.labels).hexdigest())
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_bad_roles_and_altered_release_rejected(self):
        with self.assertRaisesRegex(ValueError, 'registered source'):
            self.evaluate(scene=self.contract['target'])
        with self.assertRaisesRegex(ValueError, 'observation release hash'):
            self.evaluate(observations=self.payloads['train'])
        with self.assertRaisesRegex(ValueError, 'label release hash'):
            self.evaluate(labels=self.labels + b' ')
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'eval-mode'):
            self.evaluate()
