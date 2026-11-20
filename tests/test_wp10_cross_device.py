"""Device admission, numerical tolerance and protection boundaries."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from inference import wp10_cross_device as measured, wp10_onednn_deployment as precision
from evaluation.wp12_wp10_cross_device import _numbers
from transfer.shared_gpu_lease import physical_lease


class CrossDeviceDeploymentTests(unittest.TestCase):
    def test_wrong_physical_uuid_is_rejected_before_any_device_access(self):
        want = "GPU-0c958696-3721-4009-628f-befb0ef9b867"
        visible = "GPU-b8dd997d-95e3-69ed-1952-e420e98bd236"
        with patch.dict("os.environ", CUDA_VISIBLE_DEVICES=visible):
            with patch("transfer.shared_gpu_lease.lease_path") as lock:
                with self.assertRaisesRegex(ValueError, "one physical GPU UUID"):
                    with physical_lease(want):
                        self.fail("mismatched physical GPU acquired")
                lock.assert_not_called()

    def test_cross_device_tolerance_is_independent_of_bitwise_and_argmax_guards(self):
        ref = torch.tensor([[.5, .5], [.9, .1]])
        got = torch.tensor([[.5, .5000003], [.9, .100003]])
        a, b = [[ref] * 3], [[got] * 3]
        results = measured.compare(a, b)
        self.assertEqual(results, _numbers(a, b))
        self.assertEqual({(row["elements"], row["changed"], row["exceeded"]) for row in results.values()},
                         {(4, 1, 1)})
        self.assertEqual(measured.accepted_arm("dense", "factor", {"cnn": {"passes_metric_guards": True}},
                                              evidence_passed=False), "dense")

    def test_protection_routes_only_if_all_heads_pass(self):
        guard = {"fused": {"passes_metric_guards": True}, "cnn": {"passes_metric_guards": False}}
        self.assertEqual(measured.accepted_arm("fp32", "qat-both", guard), "fp32")
        guard["cnn"]["passes_metric_guards"] = True
        self.assertEqual(measured.accepted_arm("fp32", "qat-first", guard), "qat-first")

    def test_foreign_integer_approval_is_rejected_before_open(self):
        with patch.object(precision, "require_frozen_code"), patch.object(precision.admission, "read") as read:
            with self.assertRaisesRegex(ValueError, "approved frozen QAT"):
                precision.preparation({"sha256": "0" * 64}, {"sha256": precision.OBSERVATION},
                                      SimpleNamespace(), Path("/unused"))
            read.assert_not_called()
