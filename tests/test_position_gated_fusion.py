"""Independent scalar-fusion math, unchanged heads/loss, and strict paired restore."""
import copy
import io
import os
import unittest

import numpy as np
import torch

from model.branch_fusion import fixed_fusion
from model.position_gated_fusion import POLICY, PositionGatedFusion
from model.task_losses import probability_loss_components
from model.VCASAHGCN import MultiDatasetVCASAHGCN

DEVICE = os.environ.get("POSITION_FUSION_DEVICE", "cpu")
torch.set_num_threads(1)


def values(dtype=torch.float64):
    c = torch.tensor([[.1, -.7, .4], [1.2, .3, -.6], [-.9, .8, .2]], dtype=dtype, device=DEVICE)
    g = torch.tensor([[-.4, .2, .9], [.6, -.5, .7], [.3, -.6, -.1]], dtype=dtype, device=DEVICE)
    return c, g


def gate():
    m = PositionGatedFusion(3, enabled=True, device=DEVICE, dtype=torch.float64)
    with torch.no_grad():
        m.gate_weight.copy_(torch.tensor([[.2, -.1, .3, -.4, .15, .05]], device=DEVICE, dtype=torch.float64))
        m.gate_bias.fill_(-.2)
    return m


class PositionGatedFusionTests(unittest.TestCase):
    def test_default_off_exact_legacy_output_input_grad_rng_and_counts(self):
        for dtype in (torch.float32, torch.float64):
            c, g = values(dtype); c.requires_grad_(); g.requires_grad_()
            for alpha in (0., .35, .8, 1.):
                before = torch.get_rng_state().clone()
                cuda = torch.cuda.get_rng_state(0).clone() if DEVICE == "cuda" else None
                m = PositionGatedFusion(3, alph=alpha, device=DEVICE, dtype=dtype)
                self.assertEqual(sum(p.numel() for p in m.parameters()), 0)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                if cuda is not None: self.assertTrue(torch.equal(cuda, torch.cuda.get_rng_state(0)))
                actual, z = m(c, g, return_gate=True); expected = fixed_fusion(c, g, alpha)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(tuple(z.shape), (3, 1)); self.assertTrue(torch.all(z == alpha))
                a = torch.autograd.grad(actual.square().sum(), (c, g), retain_graph=True)
                b = torch.autograd.grad(expected.square().sum(), (c, g), retain_graph=True)
                for x, y in zip(a, b): torch.testing.assert_close(x, y, rtol=0, atol=0)
            m = PositionGatedFusion(64, enabled=True, device=DEVICE, dtype=dtype)
            self.assertEqual(sum(p.numel() for p in m.parameters()), 129)
            self.assertTrue(torch.all(m.gate_weight == 0))
            c64, g64 = c.repeat(1, 22)[:, :64], g.repeat(1, 22)[:, :64]
            torch.testing.assert_close(m(c64, g64), fixed_fusion(c64, g64), rtol=2e-6, atol=1e-7)
            before = torch.get_rng_state().clone()
            PositionGatedFusion(64, enabled=True, device=DEVICE, dtype=dtype)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_independent_numpy_fp64_formula_all_derivatives_and_finite_difference(self):
        c, g = values(); c.requires_grad_(); g.requires_grad_(); m = gate()
        w = m.gate_weight.detach().cpu().numpy().copy(); bias = m.gate_bias.detach().cpu().numpy().copy()
        cn, gn = c.detach().cpu().numpy(), g.detach().cpu().numpy()
        u = np.array([[.3, -.2, .7], [-.4, .5, .2], [.6, -.8, .1]])
        x = np.concatenate((cn, gn), axis=1)
        z = 1 / (1 + np.exp(-(x @ w.T + bias)))
        expected = z * cn + (1 - z) * gn
        actual, az = m(c, g, return_gate=True)
        np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(az.detach().cpu().numpy(), z, rtol=1e-12, atol=1e-12)
        (actual * torch.tensor(u, device=DEVICE)).sum().backward()
        ds = (u * (cn - gn)).sum(axis=1, keepdims=True) * z * (1 - z)
        oracle = (u * z + ds @ w[:, :3], u * (1 - z) + ds @ w[:, 3:], ds.T @ x, ds.sum(axis=0))
        for p, ref in zip((c, g, m.gate_weight, m.gate_bias), oracle):
            np.testing.assert_allclose(p.grad.cpu().numpy(), ref, rtol=1e-12, atol=1e-12)
        def independent_objective(cvalue, weights, bvalue):
            xvalue = np.concatenate((cvalue, gn), axis=1)
            score = xvalue @ weights.T + bvalue
            mixture = 1 / (1 + np.exp(-score))
            return ((mixture * cvalue + (1 - mixture) * gn) * u).sum()
        eps = 1e-6
        for index in range(6):
            wp, wm = w.copy(), w.copy(); wp[0, index] += eps; wm[0, index] -= eps
            finite = (independent_objective(cn, wp, bias) - independent_objective(cn, wm, bias)) / (2 * eps)
            self.assertAlmostEqual(finite, float(oracle[2][0, index]), places=8)
        finite = (independent_objective(cn, w, bias + eps) - independent_objective(cn, w, bias - eps)) / (2 * eps)
        self.assertAlmostEqual(finite, float(oracle[3][0]), places=8)
        cp, cm = cn.copy(), cn.copy(); cp[1, 2] += eps; cm[1, 2] -= eps
        finite = (independent_objective(cp, w, bias) - independent_objective(cm, w, bias)) / (2 * eps)
        self.assertAlmostEqual(finite, float(oracle[0][1, 2]), places=8)

    def test_convex_bounds_saturation_equal_inputs_swap_permutation_and_partitions(self):
        c, g = values(); m = gate(); actual, z = m(c, g, return_gate=True)
        self.assertTrue(torch.all((z >= 0) & (z <= 1)))
        self.assertTrue(torch.all(actual >= torch.minimum(c, g) - 1e-15))
        self.assertTrue(torch.all(actual <= torch.maximum(c, g) + 1e-15))
        order = torch.tensor([2, 0, 1], device=DEVICE)
        torch.testing.assert_close(m(c[order], g[order]), actual[order], rtol=1e-12, atol=1e-12)
        parts = torch.cat((m(c[:1], g[:1]), m(c[1:], g[1:])))
        torch.testing.assert_close(parts, actual, rtol=1e-12, atol=1e-12)
        swapped = copy.deepcopy(m)
        with torch.no_grad():
            swapped.gate_weight.copy_(-torch.cat((m.gate_weight[:, 3:], m.gate_weight[:, :3]), dim=1))
            swapped.gate_bias.copy_(-m.gate_bias)
        torch.testing.assert_close(swapped(g, c), actual, rtol=1e-12, atol=1e-12)
        same = m(c, c); torch.testing.assert_close(same, c, rtol=1e-15, atol=1e-15)
        for grad in torch.autograd.grad(same.sum(), tuple(m.parameters())): self.assertEqual(torch.count_nonzero(grad), 0)
        for bias, expected in ((80., c), (-80., g)):
            with torch.no_grad(): m.gate_weight.zero_(); m.gate_bias.fill_(bias)
            torch.testing.assert_close(m(c, g), expected, rtol=1e-15, atol=1e-15)

    def test_empty_invalid_inputs_parameters_version_and_score_overflow(self):
        c, g = values(); m = gate()
        empty, z = m(c[:0], g[:0], return_gate=True)
        self.assertEqual((tuple(empty.shape), tuple(z.shape)), ((0, 3), (0, 1)))
        for kw in ({"channels": True}, {"channels": 0}, {"enabled": 1}, {"alph": True}, {"alph": float("nan")},
                   {"alph": -1}, {"enabled": True, "alph": 0.}, {"enabled": True, "alph": 1.}, {"dtype": torch.float16}):
            with self.assertRaises(ValueError): PositionGatedFusion(**kw)
        for a, b in ((c[0], g), (c, g[:, :2]), (c.float(), g), (c.half(), g.half()),
                     (c.long(), g.long()), (c * float("nan"), g), (c, g * float("inf")), (None, g)):
            with self.assertRaises(ValueError): m(a, b)
        with self.assertRaises(ValueError): m(c, g, return_gate=1)
        for name in ("gate_weight", "gate_bias"):
            bad = gate()
            with torch.no_grad(): getattr(bad, name).fill_(float("nan"))
            with self.assertRaises(ValueError): bad(c, g)
        bad = gate(); bad.protocol_version.fill_(0)
        with self.assertRaises(ValueError): bad(c, g)
        bad = gate()
        with torch.no_grad(): bad.gate_weight.fill_(1e308)
        with self.assertRaisesRegex(ValueError, "scores"): bad(torch.ones_like(c) * 2, g.abs())
        if DEVICE == "cuda":
            with self.assertRaises(ValueError): m(c.cpu(), g)

    def test_paired_weights_only_roundtrip_strict_preflight_and_no_partial_restore(self):
        c, g = values()
        for enabled in (False, True):
            m = gate() if enabled else PositionGatedFusion(3, device=DEVICE, dtype=torch.float64)
            payload = m.export_payload(); stream = io.BytesIO(); torch.save(payload, stream); stream.seek(0)
            decoded = torch.load(stream, map_location="cpu", weights_only=True)
            restored = PositionGatedFusion.from_payload(decoded, device=DEVICE)
            torch.testing.assert_close(restored(c, g), m(c, g), rtol=0, atol=0)
            self.assertEqual(restored.export_config(), m.export_config())
            self.assertEqual(set(restored.state_dict()), set(m.state_dict()))
            if enabled:
                self.assertNotEqual(restored.gate_weight.data_ptr(), m.gate_weight.data_ptr())
                cpu = PositionGatedFusion.from_payload(decoded, device="cpu", dtype=torch.float64)
                torch.testing.assert_close(cpu(c.cpu(), g.cpu()), m(c, g).cpu(), rtol=1e-12, atol=1e-12)
        m = gate(); original = copy.deepcopy(m.state_dict())
        bad_states = []
        for change in ("missing", "extra", "version", "config", "shape", "nan", "dtype"):
            state = copy.deepcopy(original)
            if change == "missing": state.pop("protocol_version")
            if change == "extra": state["ignored"] = torch.tensor(0)
            if change == "version": state["protocol_version"].fill_(0)
            if change == "config": state["_extra_state"]["alph"] = .35
            if change == "shape": state["gate_weight"] = state["gate_weight"][:, :2]
            if change == "nan": state["gate_bias"].fill_(float("nan"))
            if change == "dtype": state["gate_weight"] = state["gate_weight"].float()
            bad_states.append(state)
        for state in bad_states:
            with self.assertRaises(ValueError): m.load_state_dict(state)
            for key in ("gate_weight", "gate_bias", "protocol_version"):
                torch.testing.assert_close(m.state_dict()[key], original[key], rtol=0, atol=0)
        for options in ({"strict": False}, {"assign": True}):
            with self.assertRaises(ValueError): m.load_state_dict(original, **options)
        for change in ("policy", "extra", "enabled"):
            config = m.export_config()
            if change == "policy": config["policy"] = "unknown"
            if change == "extra": config["ignored"] = 1
            if change == "enabled": config["enabled"] = 1
            with self.assertRaises(ValueError): PositionGatedFusion.from_config(config)
        payload = m.export_payload(); payload["config"]["alph"] = .35
        with self.assertRaises(ValueError): PositionGatedFusion.from_payload(payload)
        with self.assertRaises(ValueError): PositionGatedFusion.from_payload({"state_dict": original})
        with self.assertRaises(ValueError): PositionGatedFusion(3).load_state_dict(original)

    def test_original_three_heads_masked_13_losses_gate_gradients_and_synthetic_updates(self):
        devices = [0] if DEVICE == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(230)
            net = MultiDatasetVCASAHGCN({"a": 3}, latent_channels=8, adapter_embed_dim=8,
                graph_mode="learned", norm_type="group_norm").to(DEVICE)
            c = torch.randn(6, 64, device=DEVICE); g = torch.randn(6, 64, device=DEVICE)
            enabled = PositionGatedFusion(64, enabled=True, device=DEVICE)
            closed = PositionGatedFusion(64, device=DEVICE)
            old = net.predict_features("a", fixed_fusion(c, g), g, c)
            control = net.predict_features("a", closed(c, g), g, c)
            for a, b in zip(old, control): torch.testing.assert_close(a, b, rtol=0, atol=0)
            onehot = torch.eye(3, device=DEVICE).repeat(2, 1); mask = torch.ones_like(onehot); mask[-2:] = 0
            for output in (old, control):
                terms = probability_loss_components(output, onehot, mask)
                self.assertEqual(len(terms), 13)
                if output is old: reference = terms
                else:
                    for key in terms: torch.testing.assert_close(terms[key], reference[key], rtol=0, atol=0)
            head_params = [p for name, p in net.named_parameters() if name.startswith(("heads.", "hgnn_heads.", "cnn_heads."))]
            params = list(enabled.parameters()) + head_params
            optimizer = torch.optim.SGD(params, lr=.05)
            self.assertEqual(len({id(p) for p in params}), len(params))
            initial = [p.detach().clone() for p in params]
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                ci, gi = c.clone().requires_grad_(), g.clone().requires_grad_()
                f = enabled(ci, gi); outputs = net.predict_features("a", f, gi, ci)
                logits = net.predict_features("a", f, gi, ci, return_logits=True)
                for prob, raw in zip(outputs, logits):
                    torch.testing.assert_close(raw.softmax(-1), prob, rtol=0, atol=0)
                    torch.testing.assert_close(prob.sum(-1), torch.ones(6, device=DEVICE))
                terms = probability_loss_components(outputs, onehot, mask); self.assertTrue(torch.isfinite(terms["total"]))
                terms["total"].backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in enabled.parameters()), 0.)
                self.assertEqual(torch.count_nonzero(ci.grad[-2:]), 0); self.assertEqual(torch.count_nonzero(gi.grad[-2:]), 0)
                optimizer.step()
            self.assertTrue(all(torch.count_nonzero(a - b) > 0 for a, b in zip(params, initial)))
            optimizer.zero_grad(set_to_none=True)
            outputs = net.predict_features("a", enabled(c, g), g, c)
            probability_loss_components(outputs, onehot, mask, cnn_loss_weight=0., graph_branch_loss_weight=0.)["total"].backward()
            for group in (net.hgnn_heads["a"], net.cnn_heads["a"]):
                self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in group.parameters()))


if __name__ == "__main__":
    unittest.main()
