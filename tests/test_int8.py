"""CPU quantization contracts with independent integer arithmetic oracles."""
import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from compression.int8 import CalibratedMatrix, IntegerMatrix

torch.set_num_threads(1)


def setup(conv=False, bias=True, qat=False):
    torch.manual_seed(512)
    layer = nn.Conv2d(8, 5, 1, bias=bias) if conv else nn.Linear(8, 5, bias=bias)
    x = torch.randn(2, 8, 3, 4) if conv else torch.randn(2, 3, 8)
    calibrated = CalibratedMatrix(layer, qat=qat)
    calibrated.observe(x, role='development_train')
    return layer, x, calibrated


def equal_state(test, a, b):
    test.assertEqual(set(a), set(b))
    for key in a:
        test.assertTrue(torch.equal(a[key], b[key]), key)


def integer_oracle(module, x):
    """NumPy int64 accumulation avoids using torch float/quantized layer ops."""
    weight = module.kernel.weight()
    w = weight.int_repr().numpy().astype(np.int64)
    scales = weight.q_per_channel_scales().numpy()
    zeros = weight.q_per_channel_zero_points().numpy()
    ins = float(module.input_scale)
    inz = int(module.input_zero_point)
    inp = np.clip(np.rint(x.numpy().astype(np.float64)/ins)+inz, 0, 127).astype(np.int64)-inz
    centered = w.reshape(w.shape[0], -1)-zeros[:, None]
    accum = (np.einsum('...i,oi->...o', inp, centered) if module.kind == 'linear'
             else np.einsum('nihw,oi->nohw', inp, centered))
    shape = (1, -1, 1, 1) if module.kind == 'conv1x1' else (-1,)
    out = accum * (ins*scales.reshape(shape))
    if module.kernel.bias() is not None:
        out += module.kernel.bias().detach().numpy().reshape(shape)
    codes = np.clip(np.rint(out/module.kernel.scale)+module.kernel.zero_point, 0, 255)
    return torch.tensor((codes-module.kernel.zero_point)*module.kernel.scale, dtype=torch.float32)


class Int8Tests(unittest.TestCase):
    def test_ptq_source_owned_dense_observation_and_frozen_no_updates(self):
        layer, x, cal = setup()
        before = copy.deepcopy(layer.state_dict())
        torch.testing.assert_close(cal.observe(x, role='source_train'), layer(x), rtol=0, atol=0)
        cal.freeze(); saved = copy.deepcopy(cal.state_dict())
        cal(x*1.3); cal(x*100)
        equal_state(self, saved, cal.state_dict())
        equal_state(self, before, layer.state_dict())
        self.assertIsNot(cal.layer.weight, layer.weight)

    def test_role_shape_nan_and_uncalibrated_guards_before_mutation(self):
        cal = CalibratedMatrix(nn.Linear(4, 3))
        x = torch.ones(2, 4)
        saved = copy.deepcopy(cal.state_dict())
        for role in ('validation', 'query', 'test', '', None):
            with self.assertRaises(ValueError): cal.observe(x, role=role)
        for bad in (torch.empty(0, 4), torch.full((2, 4), float('nan')),
                    torch.ones(2, 5), x.double(), x.to('meta')):
            with self.assertRaises(ValueError): cal.observe(bad, role='source_train')
        with self.assertRaises(ValueError): cal.freeze()
        with self.assertRaises(ValueError): cal(x)
        with self.assertRaises(ValueError): cal.convert()
        with self.assertRaises(ValueError): IntegerMatrix(cal)
        equal_state(self, saved, cal.state_dict())
        cal.observe(x, role='target_support'); cal.freeze()
        with self.assertRaises(ValueError): cal.observe(x, role='source_train')

    def test_finite_input_overflow_rolls_back_calibration(self):
        layer = nn.Linear(4, 3, bias=False)
        with torch.no_grad(): layer.weight.fill_(2.)
        cal = CalibratedMatrix(layer)
        saved = copy.deepcopy(cal.state_dict())
        with self.assertRaises(ValueError):
            cal.observe(torch.full((2, 4), 1e38), role='source_train')
        equal_state(self, saved, cal.state_dict())

    def test_integer_linear_conv_bias_and_outliers_numpy_oracle(self):
        for conv in (False, True):
            for bias in (False, True):
                _, x, cal = setup(conv, bias)
                quant = cal.freeze().convert()
                w = quant.kernel.weight()
                self.assertEqual(w.dtype, torch.qint8)
                self.assertEqual(w.qscheme(), torch.per_channel_affine)
                self.assertTrue((w.int_repr() >= -127).all())
                for multiplier in (0., 1., 100.):
                    probe = x*multiplier
                    actual = quant(probe)
                    # One output LSB permits backend tie-rounding differences only.
                    torch.testing.assert_close(actual, integer_oracle(quant, probe),
                                               rtol=0, atol=quant.kernel.scale*1.01)
                self.assertEqual(list(quant.parameters()), [])

    def test_weight_codes_independent_rounding_and_channel_scale(self):
        layer = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[0., 0., 0., 0.], [.01, -.02, .03, -.04],
                                             [1., -2., 3., -4.]]))
        cal = CalibratedMatrix(layer)
        cal.observe(torch.ones(2, 4), role='source_train'); q = cal.freeze().convert()
        scales = q.kernel.weight().q_per_channel_scales().numpy()
        # Fake quantization uses FP32 reciprocal multiplication. FP64 division
        # at a half-integer can cross a rounding tie and is a different oracle.
        inverse = np.reciprocal(scales.astype(np.float32))
        codes = np.clip(np.rint(layer.weight.detach().numpy()*inverse[:, None]), -127, 127)
        np.testing.assert_array_equal(codes, q.kernel.weight().int_repr().numpy())
        self.assertAlmostEqual(scales[2]/scales[1], 100., places=4)

    def test_exact_half_integer_round_to_even(self):
        layer = nn.Linear(7, 1, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[-2.5, -1.5, -.5, .5, 1.5, 2.5, 127.]]))
        cal = CalibratedMatrix(layer)
        cal.observe(torch.zeros(2, 7), role='source_train')
        q = cal.freeze().convert()
        np.testing.assert_array_equal(q.kernel.weight().int_repr().numpy(),
                                      [[-2, -2, 0, 0, 2, 2, 127]])

    def test_all_zero_and_constant_activations_finite_bias(self):
        for value in (0., 2., -2.):
            layer = nn.Linear(4, 3)
            with torch.no_grad(): layer.weight.zero_(); layer.bias.zero_()
            cal = CalibratedMatrix(layer)
            x = torch.full((2, 4), value)
            cal.observe(x, role='development_train'); quant = cal.freeze().convert()
            self.assertTrue(torch.equal(quant(x), torch.zeros(2, 3)))
            for fq in cal.quantizers():
                self.assertTrue((fq.scale > 0).all())

    def test_reject_layers_dtypes_hooks_and_integer_training(self):
        for layer in (nn.Conv2d(4, 4, 3), nn.Conv2d(4, 4, 1, groups=4),
                      nn.Linear(4, 4).double(), nn.ReLU()):
            with self.assertRaises(ValueError): CalibratedMatrix(layer)
        layer = nn.Linear(4, 4)
        hook = layer.register_forward_hook(lambda *args: None)
        with self.assertRaises(ValueError): CalibratedMatrix(layer)
        hook.remove()
        _, x, cal = setup(); q = cal.freeze().convert()
        with self.assertRaises(ValueError): q(x.requires_grad_())
        q.train()
        with self.assertRaises(ValueError): q(x.detach())

    def test_actual_quantized_kernels_in_profiler(self):
        for conv in (False, True):
            _, x, cal = setup(conv); q = cal.freeze().convert()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                q(x)
            names = {item.key for item in prof.key_averages()}
            self.assertIn('quantized::conv2d' if conv else 'quantized::linear', names)
            self.assertNotIn('aten::linear', names)
            self.assertNotIn('aten::conv2d', names)

    def test_integer_state_disk_bytes_exact_reload_and_payload_reduction(self):
        torch.manual_seed(64)
        layer = nn.Linear(512, 256)
        x = torch.randn(4, 512)
        cal = CalibratedMatrix(layer); cal.observe(x, role='source_train'); cal.freeze()
        q = cal.convert()
        dense_bytes, quant_bytes = io.BytesIO(), io.BytesIO()
        torch.save(layer.state_dict(), dense_bytes); torch.save(q.state_dict(), quant_bytes)
        self.assertLess(quant_bytes.tell(), dense_bytes.tell()*.4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'int8.pt'
            path.write_bytes(quant_bytes.getvalue())
            loaded = cal.convert(); loaded.load_state_dict(torch.load(path, weights_only=True))
        self.assertTrue(torch.equal(q(x), loaded(x)))

    def test_qat_master_gradient_changes_and_frozen_surrogate_agreement(self):
        layer, x, cal = setup(qat=True)
        before = cal.layer.weight.detach().clone()
        opt = torch.optim.SGD(cal.parameters(), lr=.01)
        opt.zero_grad(); cal.observe(x, role='development_train').square().mean().backward()
        self.assertTrue(torch.isfinite(cal.layer.weight.grad).all())
        self.assertGreater(float(cal.layer.weight.grad.abs().sum()), 0.)
        opt.step()
        self.assertFalse(torch.equal(before, cal.layer.weight))
        self.assertTrue(torch.equal(before, layer.weight))
        cal.observe(x, role='development_train'); cal.freeze()
        q = cal.convert()
        torch.testing.assert_close(cal(x), q(x), rtol=0, atol=q.kernel.scale*1.01)

    def test_equal_budget_fp32_qat_first_min_and_checkpoint_split_replay(self):
        torch.manual_seed(2048)
        base = nn.Linear(8, 5)
        train, valid = torch.randn(12, 8), torch.randn(7, 8)
        target, valtarget = torch.randn(12, 5), torch.randn(7, 5)
        for qat in (False, True):
            model = CalibratedMatrix(base, qat=True) if qat else copy.deepcopy(base)
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            history = []
            for step in range(12):
                opt.zero_grad()
                y = model.observe(train, role='source_train') if qat else model(train)
                (y-target).square().mean().backward(); opt.step()
                # A frozen copy evaluates; validation must never update training observers.
                evaluated = copy.deepcopy(model).freeze() if qat else model
                with torch.no_grad(): history.append(float((evaluated(valid)-valtarget).square().mean()))
                if step == 4:
                    packet = io.BytesIO()
                    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, packet)
                    packet.seek(0); saved = torch.load(packet, weights_only=True)
            selected = min(range(12), key=history.__getitem__)+1
            self.assertIn(selected, range(1, 13))
            resumed = CalibratedMatrix(base, qat=True) if qat else copy.deepcopy(base)
            resumed.load_state_dict(saved['model'])
            opt2 = torch.optim.Adam(resumed.parameters(), lr=.001)
            opt2.load_state_dict(saved['optimizer'])
            resumed_history = history[:5]
            for _ in range(7):
                opt2.zero_grad()
                y = resumed.observe(train, role='source_train') if qat else resumed(train)
                (y-target).square().mean().backward(); opt2.step()
                evaluated = copy.deepcopy(resumed).freeze() if qat else resumed
                with torch.no_grad(): resumed_history.append(float((evaluated(valid)-valtarget).square().mean()))
            equal_state(self, model.state_dict(), resumed.state_dict())
            self.assertEqual(history, resumed_history)
            self.assertEqual(selected, min(range(12), key=resumed_history.__getitem__)+1)


if __name__ == '__main__':
    unittest.main()
