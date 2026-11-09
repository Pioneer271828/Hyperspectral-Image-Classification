import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from loadData.spatial_protocol import forward_partition
from loadData.spatial_tiles import fit_tile_statistics, make_tile_plan, observe_tile
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from test_g2_contract import fixture
from transfer.g2_source import build_source_model, require_shared_heads, source_labels


class SourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.catalog, self.contract, _, _ = fixture()
        self.plan = make_tile_plan((24, 24), tile_size=8, gap=0)
        self.scene = 'paviaU'
        self.contract['semantic_mapping'][self.scene] = {'6': 'semantic_a', '4': 'semantic_b'}

    def ids(self, partition):
        tile = next(tile for tile in self.plan['tiles'] if tile['partition'] == partition)
        y0, _, x0, _ = tile['box']
        start = y0 * self.plan['shape'][1] + x0
        return [start, start + 1, start + 2]

    def release(self, partition, **changes):
        frozen = self.contract['source_releases'][self.scene][partition]
        record = dict(scene=self.scene, partition=partition, plan_hash=self.plan['plan_hash'],
                      observation_sha256=frozen['observation_sha256'],
                      indices=self.ids(partition), labels=[4, 6, 3])
        record.update(changes)
        payload = json.dumps(record).encode()
        frozen['labels_sha256'] = hashlib.sha256(payload).hexdigest()
        return payload

    def load(self, phase, payload):
        return source_labels(self.catalog, self.contract, self.scene, phase, self.plan, payload)

    def test_semantic_order_and_unmapped_exclusion(self):
        for partition, phase in [('train', 'source_train'), ('val', 'source_select')]:
            labels, counts = self.load(phase, self.release(partition))
            self.assertEqual(labels.reshape(-1)[self.ids(partition)].tolist(), [2, 1, 0])
            self.assertEqual(counts, [1, 1])
            self.assertEqual(np.count_nonzero(labels), 2)
            self.assertFalse(labels.flags.writeable)

    def test_role_hash_and_observation_mismatch(self):
        valid = self.release('val')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.load('source_train', valid)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.load('source_select', valid + b' ')
        with self.assertRaisesRegex(ValueError, 'observation mismatch'):
            self.load('source_select', self.release('val', observation_sha256='a' * 64))
        with self.assertRaisesRegex(ValueError, 'observation mismatch'):
            self.load('source_select', self.release('val', partition='train'))

    def test_rehashed_coordinate_injection_and_missing_class_rejected(self):
        for partition in ('val', 'test'):
            with self.assertRaisesRegex(ValueError, 'escape'):
                self.load('source_train', self.release('train', indices=self.ids(partition)))
        with self.assertRaisesRegex(ValueError, 'lacks a shared class'):
            self.load('source_train', self.release('train', labels=[4, 4, 3]))

    def test_target_labels_refused_before_decoding(self):
        for scene in ('whu-honghu', 'whu-hanchuan', 'WHUL'):
            with self.assertRaisesRegex(ValueError, 'require a source'):
                source_labels(self.catalog, self.contract, scene, 'source_select', self.plan, b'invalid')
        with self.assertRaisesRegex(ValueError, 'require a source'):
            self.load('source_test', b'invalid')

    def test_shared_head_gradients_and_checkpoint(self):
        config = {'model': {'graph_mode': 'learned', 'norm_type': 'group_norm'},
                  'training': {'steps_per_epoch': 1}}
        model, _ = build_source_model(self.catalog, self.contract, config, seed=64)
        head = require_shared_heads(self.catalog, self.contract, model)
        cube = np.random.default_rng(64).random((24, 24, 6), dtype=np.float32)
        stats = fit_tile_statistics(cube, self.plan)
        tile = next(tile for tile in self.plan['tiles'] if tile['partition'] == 'train')
        obs = observe_tile(cube, self.plan, stats, tile)
        wave = np.linspace(420, 850, 6, dtype=np.float32)
        outputs = forward_partition(model, head, obs, wave, 16)[:3]
        loss = sum(-out[0, 0].log() - out[1, 1].log() for out in outputs)
        loss.backward()
        for family in (model.heads, model.hgnn_heads, model.cnn_heads):
            self.assertGreater(family[head].weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            save_model_checkpoint(model, checkpoint)
            loaded = load_model_checkpoint(checkpoint)
        self.assertEqual(require_shared_heads(self.catalog, self.contract, loaded), head)
        model.eval()
        with torch.no_grad():
            original = forward_partition(model, head, obs, wave, 16)
            restored = forward_partition(loaded, head, obs, wave, 16)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original[:3], restored[:3])))
        self.contract['semantic_classes'].reverse()
        with self.assertRaisesRegex(ValueError, 'registered shared'):
            require_shared_heads(self.catalog, self.contract, loaded)

    def test_domain_specific_normalization_rejected(self):
        config = {'model': {'graph_mode': 'learned', 'norm_type': 'domain_bn'},
                  'training': {'steps_per_epoch': 1}}
        with self.assertRaisesRegex(ValueError, 'scene-independent'):
            build_source_model(self.catalog, self.contract, config, seed=64)
