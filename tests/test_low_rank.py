"""Explicit compression correctness and fair bounded recovery contracts."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from compression.low_rank import LowRankMatrix, factorize_model, spectrum
from compression.low_rank_checkpoint import save_low_rank, load_low_rank, sha256
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint

torch.set_num_threads(1)
DEVICE = os.environ.get('LR_DEVICE', 'cpu')
PATH = 'backbone.CNN_Branch.0.point_conv'


def fixture():
    torch.manual_seed(512)
    model = MultiDatasetVCASAHGCN({'demo': 3}, latent_channels=8, adapter_embed_dim=8,
        locality_window=None, norm_type='group_norm', graph_mode='static').to(DEVICE)
    x = torch.randn(6, 7, 14, device=DEVICE)
    wave = torch.linspace(450, 950, 12, device=DEVICE)
    q = nn.functional.one_hot(torch.arange(42, device=DEVICE) % 6, 6).float()
    a = torch.ones(6, 6, device=DEVICE) / 6
    return model, (x, q, a, wave)


def predict(model, data):
    x, q, a, wave = data
    return model('demo', x, q, a, wavelengths=wave)


class LowRankTests(unittest.TestCase):
    def test_full_rank_forward_input_gradient_bias_and_real_two_ops(self):
        for convolution in (False, True):
            for bias in (False, True):
                layer = (nn.Conv2d(7, 5, 1, bias=bias) if convolution else
                         nn.Linear(7, 5, bias=bias)).to(device=DEVICE, dtype=torch.float64)
                layer.eval()
                factor = LowRankMatrix(layer, 5, allow_expansion=True)
                x = torch.randn((2, 7, 3, 4) if convolution else (2, 3, 7),
                                dtype=torch.float64, device=DEVICE, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                target, result = layer(x), factor(y)
                torch.testing.assert_close(result, target, rtol=1e-11, atol=1e-12)
                target.square().sum().backward(); result.square().sum().backward()
                torch.testing.assert_close(x.grad, y.grad, rtol=1e-11, atol=1e-12)
                self.assertEqual(factor.training, False)
                self.assertIsNotNone(factor.left.grad); self.assertIsNotNone(factor.right.grad)
                self.assertEqual(sum(p.numel() for p in factor.parameters()), 5*(7+5)+(5 if bias else 0))
                fn = 'conv2d' if convolution else 'linear'
                with patch('compression.low_rank.F.' + fn, wraps=getattr(nn.functional, fn)) as call:
                    factor(y)
                    self.assertEqual(call.call_count, 2)

    def test_truncation_oracle_rank_budget_and_zero_spectrum(self):
        layer = nn.Linear(6, 6, bias=True).double().to(DEVICE)
        with torch.no_grad():
            layer.weight.copy_(torch.diag(torch.tensor([6, 5, 4, 3, 2, 1.], device=DEVICE)))
            layer.bias.fill_(.25)
        factor = LowRankMatrix(layer, 2)
        reconstructed = factor.left @ factor.right
        torch.testing.assert_close(reconstructed, torch.diag(layer.weight.diag()*
            torch.tensor([1, 1, 0, 0, 0, 0], device=DEVICE)), rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float((layer.weight-reconstructed).square().sum()), 30.)
        torch.testing.assert_close(factor(torch.zeros(1, 6, device=DEVICE, dtype=torch.float64)),
                                   layer.bias[None])
        self.assertEqual(spectrum(layer)['stable_rank'], 91/36)
        with torch.no_grad(): layer.weight.zero_()
        self.assertEqual(spectrum(layer)['participation_rank'], 0)
        self.assertEqual(LowRankMatrix(layer, 2).left.count_nonzero(), 0)
        with self.assertRaises(ValueError): LowRankMatrix(nn.Linear(64, 64), 48)

    def test_reject_invalid_shared_and_leave_source_unchanged(self):
        for layer, rank in [(nn.Conv2d(4, 4, 3), 1), (nn.Conv2d(4, 4, 1, groups=4), 1),
                            (nn.Linear(4, 4), True), (nn.Linear(4, 4), 0), (nn.Linear(4, 4), 5)]:
            with self.assertRaises(ValueError): LowRankMatrix(layer, rank)
        bad = nn.Linear(4, 4)
        with torch.no_grad(): bad.weight[0, 0] = float('nan')
        with self.assertRaises(ValueError): LowRankMatrix(bad, 1)
        shared = nn.Linear(4, 4)
        tied = nn.ModuleDict({'a': shared, 'b': shared})
        with self.assertRaises(ValueError): factorize_model(tied, {'a': 1})
        source = nn.Sequential(nn.Linear(6, 6), nn.ReLU())
        before = copy.deepcopy(source.state_dict())
        with self.assertRaises(ValueError): factorize_model(source, {'0': 2, '1': 1})
        for key in before: self.assertTrue(torch.equal(before[key], source.state_dict()[key]))

    def test_random_control_rng_isolation_empty_plan_and_dtype(self):
        layer = nn.Linear(8, 8).to(DEVICE)
        rng = torch.get_rng_state().clone()
        a = LowRankMatrix(layer, 2, init='random', seed=64)
        b = LowRankMatrix(layer, 2, init='random', seed=64)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(a.left, b.left))
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            c = LowRankMatrix(layer.to(dtype=dtype), 2)
            self.assertEqual(c.left.dtype, dtype)
            self.assertTrue(torch.isfinite(c.left).all())
        source = nn.Sequential(nn.Linear(8, 8)).to(DEVICE)
        data = torch.randn(2, 8, device=DEVICE)
        self.assertTrue(torch.equal(source(data), factorize_model(source, {})(data)))

    def test_model_full_rank_and_legacy_checkpoint_compatibility(self):
        source, data = fixture(); source.eval()
        rank = min(source.get_submodule(PATH).weight.shape[:2])
        candidate = factorize_model(source, {PATH: rank}, allow_expansion=True)
        for a, b in zip(predict(source, data), predict(candidate, data)):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy.pt'; save_model_checkpoint(source, path)
            restored = load_model_checkpoint(path, device=DEVICE)
            for a, b in zip(predict(source, data), predict(restored, data)):
                self.assertTrue(torch.equal(a, b))

    def test_compressed_checkpoint_exact_policy_hash_overwrite_and_trainability(self):
        source, data = fixture(); source.eval()
        source.heads['demo'].weight.requires_grad_(False)
        candidate = factorize_model(source, {PATH: 2})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'compressed.pt'; pin = save_low_rank(candidate, path)
            rng = torch.get_rng_state().clone()
            restored = load_low_rank(path, pin, device=DEVICE)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for a, b in zip(predict(candidate, data), predict(restored, data)):
                self.assertTrue(torch.equal(a, b))
            self.assertFalse(restored.heads['demo'].weight.requires_grad)
            with self.assertRaises(FileExistsError): save_low_rank(candidate, path)
            with self.assertRaises(ValueError): load_low_rank(path, '0'*64)
            with self.assertRaises((ValueError, RuntimeError, KeyError)):
                load_model_checkpoint(path, device=DEVICE, config={})
            with self.assertRaises(ValueError): factorize_model(candidate, {PATH: 1})
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
            packet = torch.load(path, weights_only=True)
            packet['dtype'] = 'float64'
            damaged = Path(tmp)/'wrong-dtype.pt'; torch.save(packet, damaged)
            with self.assertRaisesRegex(ValueError, 'dtype'):
                load_low_rank(damaged, sha256(damaged))

    def test_dense_svd_random_equal_budget_selection_and_split_resume(self):
        source, data = fixture()
        labels = torch.arange(42, device=DEVICE) % 3
        models = {'dense': copy.deepcopy(source), 'svd': factorize_model(source, {PATH: 2}),
                  'random': factorize_model(source, {PATH: 2}, init='random')}
        counts = {}
        for name, model in models.items():
            initial = copy.deepcopy(model.state_dict())
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            history = []; saved = None; best = float('inf'); selected = 0
            for step in range(12):
                model.train(); opt.zero_grad()
                loss = sum(nn.functional.nll_loss(p.clamp_min(1e-10).log(), labels) for p in predict(model, data))
                loss.backward(); opt.step(); model.eval()
                with torch.no_grad(): val = float(nn.functional.nll_loss(predict(model, data)[0].log(), labels))
                history.append(val)
                if val < best: best, selected = val, step+1
                if step == 4:
                    # Exercise the actual disk model and optimizer boundary, not deepcopy replay.
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp)/'split.pt'
                        if name == 'dense':
                            save_model_checkpoint(model, path)
                            restored = load_model_checkpoint(path, device=DEVICE)
                        else:
                            pin = save_low_rank(model, path)
                            restored = load_low_rank(path, pin, device=DEVICE)
                        opt_path = Path(tmp)/'optimizer.pt'
                        torch.save(opt.state_dict(), opt_path)
                        saved = (restored, torch.load(opt_path, weights_only=True, map_location=DEVICE))
            self.assertEqual(selected, min(range(12), key=history.__getitem__)+1)
            resumed, opt_state = saved
            opt2 = torch.optim.Adam(resumed.parameters(), lr=.001); opt2.load_state_dict(opt_state)
            for step in range(5, 12):
                resumed.train(); opt2.zero_grad()
                loss = sum(nn.functional.nll_loss(p.clamp_min(1e-10).log(), labels) for p in predict(resumed, data))
                loss.backward(); opt2.step()
            # CPU exact replay; independent native CUDA sparse trajectories are not promised.
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0 if DEVICE=='cpu' else 1e-5,
                                            atol=0 if DEVICE=='cpu' else 1e-6)
            if name != 'dense':
                self.assertFalse(torch.equal(initial[PATH+'.left'], model.state_dict()[PATH+'.left']))
                self.assertFalse(torch.equal(initial[PATH+'.right'], model.state_dict()[PATH+'.right']))
                counts[name] = sum(p.numel() for p in model.parameters())
        self.assertEqual(counts['svd'], counts['random'])


if __name__ == '__main__':
    unittest.main()
