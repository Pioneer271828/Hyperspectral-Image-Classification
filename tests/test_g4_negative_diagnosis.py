"""Numeric edge cases for the post hoc spectral and Q diagnostics."""
import unittest
import numpy as np
import torch

from g4_negative_diagnosis import graph_statistics, spectral_statistics, spectrum


class NegativeDiagnosisTests(unittest.TestCase):
    def test_known_spectra(self):
        self.assertAlmostEqual(spectrum(np.eye(4))["participation_dimension"], 4)
        one = spectrum(np.ones((4, 4)))
        self.assertEqual((one["pca95"], one["pca99"]), (1, 1))
        self.assertAlmostEqual(one["participation_dimension"], 1)

    def test_constant_bands_and_invalid_sample(self):
        result = spectral_statistics(np.ones((5, 3)))
        self.assertEqual(result["constant_bands"], 3)
        self.assertEqual(result["correlation"]["pca95"], 0)
        self.assertIsNone(result["adjacent_band_correlation_p05_median"])
        for value in [np.array([[1, np.nan], [2, 3]]), np.zeros((1, 3))]:
            with self.assertRaises(ValueError):
                spectral_statistics(value)

    def test_duplicated_bands(self):
        x = np.arange(20, dtype=float)
        result = spectral_statistics(np.column_stack((x, x, np.ones_like(x))))
        self.assertEqual(result["constant_bands"], 1)
        self.assertAlmostEqual(result["correlation"]["participation_dimension"], 1)
        np.testing.assert_allclose(result["adjacent_band_correlation_p05_median"], [1, 1])

    def test_hard_assignment_guard_and_graph_degree(self):
        q = torch.sparse_coo_tensor([[0, 1, 2], [0, 0, 1]], torch.ones(3), (3, 2))
        obs = {"Q": q, "A": torch.tensor([[1., 1.], [1., 1.]])}
        region, result = graph_statistics(obs)
        np.testing.assert_array_equal(region, [0, 0, 1])
        self.assertEqual(result["mean_region_pixels"], 1.5)
        self.assertEqual(result["isolated_regions"], 0)
        obs["Q"] = torch.sparse_coo_tensor([[0, 1, 2], [0, 0, 1]], torch.full((3,), .5), (3, 2))
        with self.assertRaises(ValueError):
            graph_statistics(obs)


if __name__ == "__main__":
    unittest.main()
