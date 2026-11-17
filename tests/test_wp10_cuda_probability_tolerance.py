"""Boundary cases for the WP10 CUDA elementwise probability contract."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from scripts import review_wp10_tensorly_cuda as cli
from scripts.review_wp10_tensorly_cuda import probability_tolerance
from transfer import p4_runtime_device as p4


class ProbabilityToleranceTests(unittest.TestCase):
    def test_cuda_entry_uses_shared_lock_and_releases_after_failure(self):
        gpu = "GPU-11111111-1111-1111-1111-111111111111"
        sha = "a" * 40
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": gpu}):
            root = Path(tmp) / "runs"
            lease = root / ".p4_gpu_leases" / f"{gpu}.lock"
            lease.parent.mkdir(parents=True)
            args = ["--expected-commit", sha, "--gpu", gpu, "--runs", str(root),
                    "--legacy-checkout", tmp, "--cpu-initial", str(root / "initial.json"),
                    "--cpu-initial-sha256", "a" * 64, "--out", str(root / "new")]
            with patch.object(cli, "code_snapshot", return_value={"commit": sha}), \
                    patch.object(cli, "require_frozen_code"), \
                    patch.object(cli, "training_container_kind", return_value="lxc"), \
                    patch.object(torch, "get_num_threads", return_value=1), \
                    patch.object(torch.cuda, "device_count", return_value=1), \
                    patch.object(cli, "execute", side_effect=RuntimeError("measurement failure")) as worker:
                with p4.lease(lease), self.assertRaisesRegex(p4.ResourceUnavailable, "Active lease"):
                    cli.main(args)
                worker.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "measurement failure"):
                    cli.main(args)
                with p4.lease(lease):
                    pass

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
