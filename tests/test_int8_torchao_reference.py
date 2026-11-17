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
from compression.int8_torchao_onednn import OneDNNPointConv, convert_backend, load_backend, register_backend
from compression.int8_model import calibrate_model, calibration_contract, convert_model
from compression.precision_fallback import LAYER_ORDER
from compression.precision_recovery_audit import independent_predictions, save_package
from model.research_config import digest
from tests.test_precision_recovery import fixture
from transfer import matched_route_admission as admission, registered_development as dev
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
                backend = OneDNNPointConv(native)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as actual_kernel:
                    backend_codes = backend.codes(x)
                self.assertEqual({e.key: e.count for e in actual_kernel.key_averages()}["onednn::qconv2d_pointwise"], 1)
                self.assertLessEqual(int((backend_codes.int() - bytes_native.int()).abs().max()), 1)
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
                    oneDNN_output_code_mismatches=int((backend_codes != bytes_native).count_nonzero()),
                    unbiased_or_biased="bias" if bias else "no_bias")
        report = os.environ.get("WP10_TORCHAO_FIXTURE_EVIDENCE")
        if report:
            write_json(Path(report), rows)

    def test_three_arms_on_both_class_counts_and_reloaded_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            for classes in (9, 22):
                root = Path(tmp) / str(classes)
                root.mkdir()
                spec = admission.read(fixture(root / "inputs", classes))
                dense = dev._load_source(spec["weights"], spec["sidecar"])
                scene, record, _ = dev.load_development(spec["development"])
                roles = [r["id"] for r in record["roles"] if r["role"] == "devtrain"][:2]
                self.assertIs(convert_backend(dense, None), dense)
                for arm, paths in zip(("qat-first", "qat-second", "qat-both"),
                                      (LAYER_ORDER[:1], LAYER_ORDER[1:], LAYER_ORDER), strict=True):
                    with self.subTest(classes=classes, arm=arm):
                        calibration = calibration_contract(dense, scene, record, paths, roles, qat=True)
                        frozen = calibrate_model(dense, scene, record, paths=paths, contract=calibration,
                                                 expected_hash=digest(calibration))
                        native = convert_model(frozen)
                        package = save_package(native, arm, root / f"{arm}.pt")
                        pin = register_backend(package, root / f"{arm}.json", self.oracle)
                        with self.assertRaisesRegex(ValueError, "Attested|attested"):
                            load_backend(pin, None, enabled=True)
                        off = load_backend(pin, None)
                        with patch.object(CalibratedMatrix, "observe", side_effect=AssertionError("recalibration")):
                            model = load_backend(pin, self.oracle, enabled=True)
                            expected = independent_predictions(off, scene, record)
                            actual = independent_predictions(model, scene, record)
                        self.assertEqual(actual["predictions"], expected["predictions"])
                        self.assertEqual(len(actual["targets"]), 5*classes)
                        for head in expected["probabilities"]:
                            for a, b in zip(expected["probabilities"][head], actual["probabilities"][head], strict=True):
                                np.testing.assert_allclose(b, a, atol=1e-6, rtol=1e-5)
                        with patch("compression.int8_torchao_onednn.OneDNNPointConv", side_effect=ValueError("pack failed")):
                            with self.assertRaisesRegex(ValueError, "pack failed"):
                                convert_backend(off, self.oracle, enabled=True)
                            self.assertIs(convert_backend(off, None), off)
                        with self.assertRaises(ValueError):
                            load_backend(dict(pin, sha256="0"*64), self.oracle, enabled=True)
