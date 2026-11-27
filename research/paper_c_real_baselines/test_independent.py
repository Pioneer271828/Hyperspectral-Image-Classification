"""Numerical cross-checks on synthetic inputs only; never open real HSI arrays."""
import unittest

import numpy as np

import independent as ref
from features import transform
from models import fit, centroid, predict
from metrics import measures, sensitivity, paired


class IndependentArithmetic(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(901).uniform(-.4, .8, (10, 8))
        self.y = np.array([0]*5+[1]*5)

    def test_all_seeds_short_endpoints_and_adaptation_match_scalar_GD(self):
        for seed in (64, 512, 2048):
            for steps in (64, 128):
                model = fit(self.X, self.y, seed=seed, steps=steps)
                ref.assert_close(model['theta'], ref.replay(self.X, self.y, seed=seed, steps=steps), atol=1e-10, rtol=1e-9)
            source = fit(self.X, self.y, seed=seed, steps=64)
            target = fit(-self.X, self.y, seed=seed, steps=64, start=source['theta'])
            ref.assert_close(target['theta'], ref.replay(-self.X, self.y, seed=seed, steps=64, start=source['theta']), atol=1e-10, rtol=1e-9)

    def test_centroid_distances_and_coefficients_have_independent_reference(self):
        model = centroid(self.X, self.y)
        ref.assert_close(model['theta'], ref.centroid_reference(self.X, self.y))
        scores, _, predictions = predict(model, self.X)
        expected = [ref.score(model['theta'], row) for row in self.X]
        ref.assert_close(scores, expected)
        self.assertEqual(predictions.tolist(), [int(v > 0) for v in expected])

    def test_sort_interpolation_features_match_different_band_counts(self):
        rng = np.random.default_rng(902)
        for bands in (2, 17, 270, 436):
            patches = rng.normal(size=(2, bands, 7, 7)).astype('float32')
            ref.assert_close(transform(patches), ref.descriptor_reference(patches), atol=1e-8, rtol=1e-7)
        patches = np.ones((1, 12, 7, 7), 'float32')
        self.assertEqual(ref.descriptor_reference(patches).tolist(), [[0.]*8])

    def row(self, offset):
        z = np.linspace(-5, 5, 10)+offset
        p = [ref.probability(float(v)) for v in z]
        pred = [int(v > 0) for v in z]
        return dict(id=str(offset), scene='synthetic', truth=self.y.tolist(), logits=z.tolist(), probabilities=p, predictions=pred, point_ids=list(range(10)), parent_ids=[1]*3+[2]*4+[3]*3, metrics=measures(self.y, z, p, pred))

    def test_metrics_cluster_missingness_and_pair_deltas_are_independent(self):
        a, b = self.row(0), self.row(3)
        for row in (a, b):
            ref.assert_close(row['metrics'], ref.metrics(row['truth'], row['logits'], row['probabilities'], row['predictions']))
            ref.assert_close(sensitivity(row), ref.sensitivity_reference(row))
        ref.assert_close(paired(a, b), ref.pair_reference(a, b))
        self.assertEqual(len(list(ref.resamples(a['parent_ids']))), 27)

    def test_extreme_logits_and_zero_class_draws_remain_finite(self):
        y, z, p, pred = [0, 1], [-1000., 1000.], [0., 1.], [0, 1]
        ref.assert_close(measures(y, z, p, pred), ref.metrics(y, z, p, pred))
        expected = ref.metrics(y, z, p, pred, [2, 0])
        self.assertIsNone(expected['AA'])
        self.assertEqual(expected['recall'], [1., None])

    def test_nested_comparison_rejects_counts_undefined_values_and_scores(self):
        for actual, expected in [({'N': 9}, {'N': 10}), ([1.], [None]), ([.5001], [.5]), ({}, {'OA': .5})]:
            with self.assertRaises(ValueError):ref.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
