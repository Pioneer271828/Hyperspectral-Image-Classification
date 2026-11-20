"""Warm compressed deployment verifies its frozen observation and timed phase contract."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from compression.low_rank import LowRankMatrix
from evaluation.wp12_wp10_deployment import _samples, build
from inference.wp10_deployment_cost import REPEATS, _operator_evidence, _summary


class WarmDeploymentTests(unittest.TestCase):
    def test_dense_and_functional_factors_really_execute_fp32(self):
        dense = torch.nn.Conv2d(8, 8, 1)
        factors = LowRankMatrix(dense, 2)
        for layer, arms in ((dense, {"dense"}), (factors, {"right", "left"})):
            with self.subTest(arms=arms):
                runtime = SimpleNamespace(full=Mock(get_submodule=Mock(return_value=layer)))
                inputs = torch.ones(1, 8, 2, 2)
                with patch("inference.wp10_deployment_cost._sample", side_effect=lambda *_: layer(inputs)):
                    got = _operator_evidence(runtime, {"tiles": [1]}, [])
                self.assertEqual(got["point_convolutions"].keys(), arms)
                self.assertEqual(got["input_dtype"], "torch.float32")
                self.assertEqual({row["calls"] for row in got["point_convolutions"].values()}, {1})
                self.assertEqual(got["device"], "cpu")
                self.assertFalse(layer._forward_hooks)

    def test_warm_raw_phases_agree_with_summary_and_reject_cpu_claiming_cuda(self):
        raw = [dict(wall_ms=12., graph_ms=3., forward_excluding_graph_ms=5., forward_ms=8.,
                    output_materialization_and_hash_ms=1., read_verify_decode_ms=2.,
                    host_to_device_ms=None, cuda_peak_allocated_bytes=None, blocks=5, graph_calls=5,
                    rss_peak_kib=30)
               for _ in range(REPEATS)]
        row = dict(samples=raw, summary=_summary(raw, 3920))
        _samples(row, 5, 3920)
        row["summary"]["sequential_pixels_per_second"] = 99.
        with self.assertRaisesRegex(ValueError, "summary differs"):
            _samples(row, 5, 3920)
        raw[0]["cuda_peak_allocated_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "CPU stages"):
            _samples(row, 5, 3920)

    def test_reject_archive_approval_before_reading_artifacts(self):
        with patch("evaluation.wp12_wp10_deployment.admission.read") as read:
            with self.assertRaisesRegex(ValueError, "published archived evidence"):
                build({}, {})
        read.assert_not_called()
