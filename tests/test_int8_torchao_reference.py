"""Exercise frozen affine arithmetic with real native integer kernels."""

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from compression.int8 import CalibratedMatrix
from compression.int8_torchao_reference import TorchAOAffine, compare_layer
from research_runtime import write_json


SOURCE = os.environ.get("WP10_TORCHAO_SOURCE")


class DefaultInt8Tests(unittest.TestCase):
    def test_native_path_is_independent_of_optional_provider(self):
        layer = CalibratedMatrix(nn.Conv2d(8, 4, 1)).eval()
        x = torch.ones(1, 8, 3, 4)
        layer.observe(x, role="development_train")
        with patch("compression.int8_torchao_reference.util.find_spec", return_value=None):
            self.assertEqual(tuple(layer.freeze().convert()(x).shape), (1, 4, 3, 4))


@unittest.skipUnless(SOURCE, "Explicitly requested pinned TorchAO arithmetic")
class FrozenAffineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = TorchAOAffine(SOURCE)

    def test_fail_closed(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises((ValueError, OSError)):
                TorchAOAffine(path)
        with patch("compression.int8_torchao_reference.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "installation required"):
                TorchAOAffine(SOURCE)
        package = Path(self.oracle.ops.__file__)
        original = Path.read_bytes

        def corrupt(path):
            return original(path) + (b"\n" if path == package else b"")

        with patch.object(Path, "read_bytes", corrupt):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                TorchAOAffine(SOURCE)

    def test_frozen_layers_extremes_bias_range_and_dispatch(self):
        rng = torch.Generator().manual_seed(64)
        rows = {}
        for kind, width, outputs, bias in (("first", 64, 128, True), ("second", 128, 64, False),
                                            ("zero", 64, 128, True)):
            with self.subTest(kind=kind):
                torch.manual_seed(64)
                layer = nn.Conv2d(width, outputs, 1, bias=bias).eval()
                if kind == "zero":
                    with torch.no_grad():
                        layer.weight.zero_()
                        layer.bias.zero_()
                q = CalibratedMatrix(layer, qat=True).eval()
                calib = torch.randn(2, width, 4, 5, generator=rng)
                q.observe(calib if kind != "zero" else torch.zeros_like(calib), role="development_train")
                native = q.freeze().convert()
                x = (torch.randn(2, width, 5, 4, generator=rng)*3).transpose(2, 3)
                x[0, :, 0, 0] = 1e6
                x[0, :, 0, 1] = -1e6
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                    act_native, act, accum, bytes_native, bytes_oracle, native_out, real = compare_layer(native, x, self.oracle)
                calls = {e.key: e.count for e in profiler.key_averages()}
                self.assertEqual(calls["quantized::conv2d"], 1)
                self.assertTrue(torch.equal(act_native, act))
                self.assertEqual((int(act.min()), int(act.max())), (0, 127))
                weight = native.kernel.weight().int_repr().numpy().reshape(outputs, width)
                x_int = act.permute(0, 2, 3, 1).numpy().reshape(-1, width).astype(np.int64)
                zp = int(native.input_zero_point.item())
                expected = np.sum((x_int[:, None, :] - zp) * weight[None, :, :], axis=2)
                np.testing.assert_array_equal(accum.numpy(), expected)
                codes = int((bytes_native.to(torch.int16) - bytes_oracle.to(torch.int16)).abs().max())
                self.assertLessEqual(codes, 1)
                maximum = float((native_out-real).abs().max())
                self.assertLessEqual(maximum, native.kernel.scale * 1.0001)
                if kind == "zero":
                    self.assertTrue(torch.equal(native_out, torch.zeros_like(native_out)))
                for wrong in (x[:, :-1], torch.full_like(x, float("nan"))):
                    with self.assertRaises(ValueError):
                        compare_layer(native, wrong, self.oracle)
                rows[kind] = dict(input_elements=act.numel(), output_elements=bytes_native.numel(),
                    output_code_mismatches=int((bytes_native != bytes_oracle).count_nonzero()),
                    max_output_code_difference=codes, max_dequantized_error=maximum, dispatch=calls,
                    unbiased_or_biased="bias" if bias else "no_bias")
        report = os.environ.get("WP10_TORCHAO_FIXTURE_EVIDENCE")
        if report:
            write_json(Path(report), rows)
