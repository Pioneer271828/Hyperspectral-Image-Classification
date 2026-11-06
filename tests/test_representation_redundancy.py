"""Hand and independent NumPy oracles for frozen descriptive statistics."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluation.representation_redundancy import aligned_summary, matrix_summary
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, write_json
from scripts.analyze_representation_redundancy import ROOT, analyze


def example():
    c = torch.tensor([[0., 1.], [4., 2.], [8., 3.], [2., 4.], [5., 5.]], dtype=torch.float64)
    g = torch.tensor([[1., 4.], [3., 2.], [5., 0.], [0., 0.]], dtype=torch.float64)
    q = torch.tensor([[1., 0., 0., 0.], [.25, .75, 0., 0.], [0., 1., 0., 0.],
                      [0., 0., 2., 0.], [0., 0., 1., 0.]], dtype=torch.float64)
    return c, g, q


def manifest(root):
    c, g, q = example()
    path = root / "features.pt"
    torch.save({"cnn": torch.cat((c, torch.zeros(len(c), 62)), 1),
                "graph": torch.cat((g, torch.zeros(len(g), 62)), 1), "Q": q.to_sparse()}, path)
    checkpoint = root / "frozen-checkpoint.pt"; checkpoint.write_bytes(b"Hash only: deliberately not a torch checkpoint")
    spec = {"schema_version": 1, "split": "validation", "stage": "stage2",
        "feature_file": {"path": str(path), "sha256": file_hash(path)},
        "checkpoint": {"path": str(checkpoint), "sha256": file_hash(checkpoint)},
        "observation_sha256": "a" * 64, "model_commit": "b" * 40,
        "selection": {"maximum_regions": 100, "seed": 64, "variance_floor": 1e-24}}
    record = root / "manifest.json"; write_json(record, spec)
    return record, spec


class RepresentationTests(unittest.TestCase):
    def test_hand_rank_correlation_orthogonal_constants_and_missing_statistics(self):
        duplicated = torch.tensor([[-1., -1., 5.], [0., 0., 5.], [1., 1., 5.]])
        r = matrix_summary(duplicated)
        self.assertEqual(r["numerical_rank"], 1); self.assertAlmostEqual(r["effective_dimension"], 1.)
        self.assertAlmostEqual(r["covariance_eigenvalues"][0], 2.)
        self.assertEqual(r["valid_correlation_channels"], [0, 1]); self.assertEqual(r["correlation_pairs"], 1)
        self.assertAlmostEqual(r["mean_absolute_channel_correlation"], 1.)
        orthogonal = torch.tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
        r = matrix_summary(orthogonal)
        self.assertEqual(r["numerical_rank"], 2); self.assertAlmostEqual(r["effective_dimension"], 2.)
        self.assertEqual(r["mean_absolute_channel_correlation"], 0.)
        for x, reason in ((torch.ones(4, 3), "covariance_below_variance_floor"),
                          (torch.empty(0, 3), "fewer_than_two_samples"), (torch.ones(1, 3), "fewer_than_two_samples")):
            r = matrix_summary(x); self.assertEqual(r["reason"], reason)
            self.assertIsNone(r["effective_dimension"]); self.assertIsNone(r["mean_absolute_channel_correlation"])
            json.dumps(r, allow_nan=False)

    def test_independent_numpy_covariance_eigen_correlation_scaling_and_variance_floor(self):
        x = np.random.default_rng(731).normal(size=(40, 5)) * np.arange(1, 6)
        actual = matrix_summary(torch.from_numpy(x))
        eigen = np.linalg.eigvalsh(np.cov(x, rowvar=False, ddof=1))[::-1]
        np.testing.assert_allclose(actual["covariance_eigenvalues"], eigen, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual["explained_variance_ratio"], eigen / eigen.sum(), rtol=1e-12)
        self.assertAlmostEqual(actual["effective_dimension"], eigen.sum()**2 / (eigen**2).sum(), places=12)
        corr = np.corrcoef(x, rowvar=False)
        self.assertAlmostEqual(actual["mean_absolute_channel_correlation"], np.abs(corr[np.triu_indices(5, 1)]).mean(), places=12)
        scaled = matrix_summary(torch.from_numpy(x * 7))
        np.testing.assert_allclose(scaled["explained_variance_ratio"], actual["explained_variance_ratio"], atol=1e-14)
        self.assertAlmostEqual(scaled["effective_dimension"], actual["effective_dimension"], places=12)
        changed = matrix_summary(torch.from_numpy(x * np.array([100, 1, 1, 1, 1])))
        self.assertLess(changed["effective_dimension"], actual["effective_dimension"])
        below = matrix_summary(torch.from_numpy(x * 1e-20))
        self.assertIsNone(below["effective_dimension"])
        self.assertIsNotNone(matrix_summary(torch.from_numpy(x * 1e-20), variance_floor=0.)["effective_dimension"])

    def test_weighted_sparse_dense_alignment_zero_mass_common_sampling_and_rng_immutability(self):
        c, g, q = example(); originals = [x.clone() for x in (c, g, q)]
        rng, np_rng = torch.get_rng_state().clone(), np.random.get_state()
        all_rows = aligned_summary(c, g, q, maximum_regions=100, seed=19)
        self.assertEqual(all_rows["sampled_region_indices"], [0, 1, 2]); self.assertEqual(all_rows["zero_mass_regions"], 1)
        np.testing.assert_allclose(all_rows["sampled_region_masses"], [1.25, 1.75, 3.])
        pooled = (q.numpy().T @ c.numpy())[:3] / q.numpy().sum(0)[:3, None]
        expected = matrix_summary(torch.from_numpy(pooled))
        np.testing.assert_allclose(all_rows["cnn"]["covariance_eigenvalues"], expected["covariance_eigenvalues"], rtol=1e-12)
        cross = np.corrcoef(pooled.T, g.numpy()[:3].T)[:2, 2:]
        self.assertAlmostEqual(all_rows["cross_branch"]["mean_absolute_correlation"], np.abs(cross).mean(), places=12)
        self.assertEqual(all_rows["cross_branch"]["valid_channel_pairs"], 4)
        for membership in (q.to_sparse(), q.to_sparse_csr()):
            r = aligned_summary(c, g, membership, maximum_regions=100, seed=19)
            self.assertEqual(r["sampled_region_indices"], all_rows["sampled_region_indices"])
            for branch in ("cnn", "graph"):
                np.testing.assert_allclose(r[branch]["covariance_eigenvalues"], all_rows[branch]["covariance_eigenvalues"], atol=1e-12)
        selected = aligned_summary(c, g, q, maximum_regions=2, seed=19)
        self.assertEqual(selected["sampled_region_indices"], np.sort(np.random.default_rng(19).choice([0, 1, 2], 2, replace=False)).tolist())
        self.assertEqual(selected, aligned_summary(c, g, q, maximum_regions=2, seed=19))
        for value, before in zip((c, g, q), originals): torch.testing.assert_close(value, before, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        now = np.random.get_state(); self.assertEqual(now[0], np_rng[0]); np.testing.assert_array_equal(now[1], np_rng[1]); self.assertEqual(now[2:], np_rng[2:])
        empty = aligned_summary(c, g, torch.zeros_like(q), maximum_regions=2, seed=19)
        self.assertEqual(empty["sampled_regions"], 0); self.assertIsNone(empty["cross_branch"]["mean_absolute_correlation"])
        json.dumps(empty, allow_nan=False)

    def test_invalid_weights_shapes_features_and_policy_limits_rejected(self):
        c, g, q = example()
        for bad in (torch.ones(3), torch.ones(3, 0), torch.ones(3, 513), torch.ones(3, 2, dtype=torch.int64),
                    torch.full((3, 2), float("nan")), torch.full((3, 2), float("inf")), torch.tensor([[1e308], [-1e308]])):
            with self.assertRaises(ValueError): matrix_summary(bad)
        for floor in (-1, True, float("nan")):
            with self.assertRaises(ValueError): matrix_summary(c, variance_floor=floor)
        for bad in (q[:, :3], -q, torch.full_like(q, float("nan")), q.long()):
            with self.assertRaises(ValueError): aligned_summary(c, g, bad, maximum_regions=2, seed=19)
        for cap, seed in ((True, 1), (0, 1), (8193, 1), (2, True), (2, -1)):
            with self.assertRaises(ValueError): aligned_summary(c, g, q, maximum_regions=cap, seed=seed)

    def test_manifest_tensor_identity_no_checkpoint_load_nn_or_mutation_and_refusals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); record, spec = manifest(root)
            pins = {p: file_hash(p) for p in root.iterdir()}
            with patch("torch.nn.Module._call_impl", side_effect=AssertionError("No NN")), patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")), patch("torch.load", wraps=torch.load) as load:
                result = analyze(record, file_hash(record), root / "out")
                self.assertEqual(load.call_count, 1); self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
                self.assertEqual(load.call_args.args[0], spec["feature_file"]["path"])
            self.assertFalse(result["checkpoint_deserialized"]); self.assertFalse(result["extraction_provenance_verified"])
            self.assertEqual(result["files_verified_unchanged"], 3)
            for p, sha in pins.items(): self.assertEqual(file_hash(p), sha)
            for change in ({"split": "query"}, {"query_labels": "forbidden"}, {"stage": "head"},
                           {"feature_file": {**spec["feature_file"], "sha256": "0" * 64}}):
                bad = {**copy.deepcopy(spec), **change}; path = root / "bad.json"; write_json(path, bad)
                with self.assertRaises(ValueError): analyze(path, file_hash(path), root / "refused")
                self.assertFalse((root / "refused").exists())
            with self.assertRaises(ValueError): analyze(record, "0" * 64, root / "refused")
            with self.assertRaises(ValueError): analyze(record, file_hash(record), root / "out")
            with self.assertRaises(ValueError): analyze(record, file_hash(record), ROOT / "issue218-refused-output")
            tensor_path = Path(spec["feature_file"]["path"])
            features = torch.load(tensor_path, weights_only=True); features["query_labels"] = torch.zeros(1); torch.save(features, tensor_path)
            spec["feature_file"]["sha256"] = file_hash(tensor_path); write_json(record, spec)
            with self.assertRaises(ValueError): analyze(record, file_hash(record), root / "refused")

    def test_actual_synthetic_cli_and_no_training_query_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); record, _ = manifest(root); out = root / "cli"
            command = [sys.executable, "scripts/analyze_representation_redundancy.py", "--expected-commit", code_snapshot()["commit"],
                       "--manifest", str(record), "--manifest-sha256", file_hash(record), "--out", str(out)]
            result = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(result["sampled_regions"], 3)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["optimizer_updates"], 0); self.assertEqual(summary["model_forwards"], 0)
            for extra in (["--query"], ["--train"], ["--width", ".5"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
