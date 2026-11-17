"""Boundary cases for the WP10 CUDA elementwise probability contract."""
import unittest

import torch

from scripts.review_wp10_tensorly_cuda import probability_tolerance


class ProbabilityToleranceTests(unittest.TestCase):
    def paired(self, values, *, shape=None):
        tensor = torch.tensor(values, dtype=torch.float64)
        if shape:
            tensor = tensor.reshape(shape)
        return [(tensor, tensor, tensor)]

    def test_near_zero_and_scale_dependent_bounds(self):
        reference = self.paired([0., .5, 1.], shape=(1, 3))
        actual = self.paired([2e-6, .5 + 5.9e-6, 1. + 10.9e-6], shape=(1, 3))
        for row in probability_tolerance(reference, actual).values():
            self.assertEqual(row["elements"], 3)
            self.assertEqual(row["exceeded"], 1)
            self.assertAlmostEqual(row["maximum_ratio"], 2.0)

    def test_shape_nonfinite_and_empty_rejected(self):
        reference = self.paired([0., .5], shape=(1, 2))
        for actual in (self.paired([0., float("nan")], shape=(1, 2)),
                       self.paired([0., float("inf")], shape=(1, 2)),
                       self.paired([0., .5], shape=(2, 1)), [], [(torch.zeros(1, 2),)]):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                probability_tolerance(reference, actual)
        with self.assertRaises(ValueError):
            probability_tolerance([], [])
