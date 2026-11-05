import unittest
import numpy as np
from transfer.tile_sampling import SupportTileSampler


class TileSamplingTests(unittest.TestCase):
    def fixture(self):
        tiles = [{'id': i, 'box': [0, 1, i * 2, i * 2 + 2]} for i in range(5)]
        roles = [{**t, 'role': 'devtrain' if t['id'] < 4 else 'devval'} for t in tiles]
        # Class 1 appears in three train tiles, class 2 only one; tile 3 is empty.
        return {'tiles': tiles}, roles, np.array([[1, 0, 1, 0, 1, 2, 0, 0, 2, 2]])

    def test_uniform_legacy_sequence_and_train_only_probabilities(self):
        plan, roles, labels = self.fixture()
        sampler = SupportTileSampler(plan, roles, labels, [1, 2])
        old = np.random.default_rng(2050)
        expected = [int(old.integers(3)) for _ in range(1000)]
        rng = np.random.default_rng(2050)
        self.assertEqual([sampler.draw(rng)[0]['id'] for _ in expected], expected)
        np.testing.assert_allclose(sampler.metadata['tile_probabilities'], [1/3]*3)
        np.testing.assert_allclose(sampler.metadata['expected_class_participation_per_update'], [1, 1/3])

    def test_class_first_distribution_anchor_and_heldout_invariance(self):
        plan, roles, labels = self.fixture()
        sampler = SupportTileSampler(plan, roles, labels, [1, 2], 'class_first')
        # Marginal tile probabilities: 1/6, 1/6, 1/6 + 1/2.
        np.testing.assert_allclose(sampler.metadata['tile_probabilities'], [1/6, 1/6, 2/3])
        np.testing.assert_allclose(sampler.metadata['expected_class_participation_per_update'], [1, 2/3])
        changed = labels.copy(); changed[0, 8:] = 0
        self.assertEqual(sampler.metadata, SupportTileSampler(plan, roles, changed, [1, 2], 'class_first').metadata)
        rng = np.random.default_rng(64); replay = np.random.default_rng(64)
        counts = np.zeros(3)
        for _ in range(20000):
            tile, anchor = sampler.draw(rng)
            self.assertEqual((tile, anchor), sampler.draw(replay))
            self.assertIn(anchor, labels[0, tile['box'][2]:tile['box'][3]])
            counts[tile['id']] += 1
        np.testing.assert_allclose(counts / counts.sum(), [1/6, 1/6, 2/3], atol=.015)

    def test_missing_class_or_unknown_mode_rejected(self):
        plan, roles, labels = self.fixture()
        with self.assertRaises(ValueError): SupportTileSampler(plan, roles, labels, [1, 2], 'unknown')
        labels[labels == 2] = 0
        with self.assertRaises(ValueError): SupportTileSampler(plan, roles, labels, [1, 2], 'class_first')
