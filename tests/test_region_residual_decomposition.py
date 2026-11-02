"""Independent M4-A mathematics, autograd, controls and lifecycle checks."""
import io
import os
import unittest
from unittest.mock import patch

import torch

from model.region_residual_decomposition import RegionResidualDecomposition


class RegionResidualTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get('M4_DEVICE', 'cpu'))

    def tensor(self, values, **kwargs):
        return torch.tensor(values, device=self.device, dtype=torch.float64, **kwargs)

    def fixture(self):
        x = self.tensor([[1, -1], [3, 1], [4, 2], [10, -4]])
        q = self.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        return x, q

    def layer(self, mode='none', channels=2):
        return RegionResidualDecomposition(channels, mode).to(device=self.device, dtype=torch.float64)

    def test_hand_computed_hard_assignment_empty_region_and_residual_means(self):
        x, q = self.fixture()
        for assignment in (q, q.to_sparse()):
            result = self.layer('residual_only')(x, assignment, return_intermediates=True)
            expected_z = self.tensor([[2, 0], [4, 2], [10, -4], [0, 0]])
            expected_r = self.tensor([[-1, -1], [1, 1], [0, 0], [0, 0]])
            torch.testing.assert_close(result.region_features, expected_z, rtol=0, atol=0)
            torch.testing.assert_close(result.residual, expected_r, rtol=0, atol=0)
            torch.testing.assert_close(q.T @ result.residual, torch.zeros_like(expected_z), rtol=0, atol=0)
            torch.testing.assert_close(result.region_at_pixels + result.residual, x, rtol=0, atol=0)

    def test_soft_assignment_is_not_assumed_to_be_an_orthogonal_projection(self):
        x = self.tensor([[0, 2], [4, 6]])
        q = self.tensor([[0.75, 0.25], [0.25, 0.75]])
        result = self.layer()(x, q, return_intermediates=True)
        torch.testing.assert_close(result.region_features, self.tensor([[1, 3], [3, 5]]), rtol=0, atol=0)
        torch.testing.assert_close(result.region_at_pixels, self.tensor([[1.5, 3.5], [2.5, 4.5]]), rtol=0, atol=0)
        self.assertGreater(float((q.T @ result.residual).abs().max()), 0.5)
        torch.testing.assert_close(result.region_at_pixels + result.residual, x)

    def test_dense_coo_values_and_feature_and_logit_gradients_agree(self):
        results = []
        for sparse in (False, True):
            x = self.tensor([[1, -2], [0.2, 1], [3, -0.5]], requires_grad=True)
            logits = self.tensor([[0.1, 0.3], [0.5, -0.4], [-0.2, 0.8]], requires_grad=True)
            q = logits.softmax(1)
            layer = self.layer('x_residual')
            with torch.no_grad():
                layer.projection.weight[:, 2:].copy_(self.tensor([[0.4, 0.2], [-0.1, 0.5]]))
            out = layer(x, q.to_sparse() if sparse else q, return_intermediates=True)
            loss = out.cnn_input.square().sum() + 0.7 * out.region_features.square().sum()
            gradients = torch.autograd.grad(loss, (x, logits))
            for g in gradients:
                self.assertTrue(torch.isfinite(g).all())
                self.assertGreater(float(g.abs().max()), 1e-5)
            results.append((out, gradients))
        for name in ('cnn_input', 'region_features', 'region_at_pixels', 'residual'):
            torch.testing.assert_close(getattr(results[0][0], name), getattr(results[1][0], name), rtol=1e-9, atol=1e-10)
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9)

    def test_soft_logit_and_feature_numerical_gradcheck_dense_and_coo(self):
        for sparse in (False, True):
            x = self.tensor([[1, -2], [0.3, 1.2]], requires_grad=True)
            logits = self.tensor([[0.1, 0.4], [-0.2, 0.7]], requires_grad=True)
            layer = self.layer('residual_only')
            def function(features, scores):
                q = scores.softmax(1)
                out = layer(features, q.to_sparse() if sparse else q)
                return out.cnn_input, out.region_features
            self.assertTrue(torch.autograd.gradcheck(function, (x, logits), eps=1e-6, atol=2e-5, rtol=1e-4))

    def test_modes_initialization_counts_and_rng_neutrality(self):
        expected = {'none': 0, 'residual_only': 0, 'x_residual': 10,
                    'x_region': 10, 'capacity_control': 12}
        x, q = self.fixture()
        cpu_rng = torch.random.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(self.device).clone() if self.device.type == 'cuda' else None
        for mode, count in expected.items():
            layer = self.layer(mode)
            self.assertEqual(sum(p.numel() for p in layer.parameters()), count)
            out = layer(x, q)
            if mode != 'residual_only':
                torch.testing.assert_close(out.cnn_input, x, rtol=0, atol=0)
            else:
                self.assertGreater(float((out.cnn_input - x).abs().max()), 1)
            self.assertEqual(out.cnn_input.shape, x.shape)
            self.assertIsNone(out.region_at_pixels)
            self.assertIsNone(out.residual)
            self.assertEqual(out.diagnostics, {})
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        if gpu_rng is not None:
            self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state(self.device)))
        self.assertEqual(sum(p.numel() for p in self.layer('x_residual', 64).parameters()), 8256)
        self.assertEqual(sum(p.numel() for p in self.layer('capacity_control', 64).parameters()), 8320)

    def test_none_is_exact_object_and_does_not_broadcast_without_opt_in(self):
        x, q = self.fixture(); x.requires_grad_()
        with patch('model.region_residual_decomposition.broadcast_regions', side_effect=AssertionError('unneeded broadcast')):
            result = self.layer()(x, q)
        self.assertIs(result.cnn_input, x)
        result.cnn_input.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)

    def test_input_and_region_permutation_equivariance(self):
        x, q = self.fixture(); layer = self.layer('residual_only')
        rows = torch.tensor([2, 0, 3, 1], device=self.device)
        cols = torch.tensor([3, 1, 0, 2], device=self.device)
        a = layer(x, q); b = layer(x[rows], q[rows][:, cols].to_sparse())
        torch.testing.assert_close(b.cnn_input, a.cnn_input[rows])
        torch.testing.assert_close(b.region_features, a.region_features[cols])

    def test_low_mass_floor_and_zero_input_diagnostics(self):
        x = self.tensor([[2, -4], [4, 6]])
        q = self.tensor([[1 - 1e-8, 1e-8, 0], [1, 0, 0]])
        for assignment in (q, q.to_sparse()):
            result = self.layer()(x, assignment, return_intermediates=True, return_diagnostics=True)
            torch.testing.assert_close(result.region_features[1], x[0] * 0.01)
            self.assertEqual(result.diagnostics['clamped_positive_region_count'], 1)
            self.assertEqual(result.diagnostics['empty_region_count'], 1)
            zero = self.layer()(torch.zeros_like(x), assignment, return_diagnostics=True)
            self.assertEqual(zero.diagnostics['residual_relative_energy'], 0)

    def test_diagnostics_are_detached_and_exports_retain_autograd(self):
        x, q = self.fixture(); x.requires_grad_()
        out = self.layer()(x, q, return_intermediates=True, return_diagnostics=True)
        self.assertTrue(out.region_features.requires_grad)
        self.assertTrue(out.region_at_pixels.requires_grad)
        self.assertTrue(out.residual.requires_grad)
        self.assertTrue(all(isinstance(v, (str, int, float)) for v in out.diagnostics.values()))
        out.residual.square().sum().backward()
        self.assertGreater(float(x.grad.abs().max()), 0)

    def test_sparse_path_never_densifies_and_handles_duplicate_entries(self):
        x, q = self.fixture(); coo = q.to_sparse()
        duplicated = torch.sparse_coo_tensor(
            coo.indices().repeat(1, 2), coo.values().repeat(2) * 0.5,
            q.shape, device=self.device)
        with patch.object(torch.Tensor, 'to_dense', side_effect=AssertionError('dense Q allocated')):
            out = self.layer('residual_only')(x, duplicated)
        expected = self.tensor([[-1, -1], [1, 1], [0, 0], [0, 0]])
        torch.testing.assert_close(out.cnn_input, expected, rtol=0, atol=0)

    def test_no_stale_cache_or_input_mutation(self):
        x, q = self.fixture(); before_x=x.clone(); before_q=q.clone()
        layer = self.layer('residual_only'); first=layer(x, q).cnn_input.clone()
        torch.testing.assert_close(x, before_x); torch.testing.assert_close(q, before_q)
        q[1] = self.tensor([0, 1, 0, 0])
        second = layer(x, q).cnn_input
        self.assertGreater(float((second-first).abs().max()), 0)
        torch.testing.assert_close(second, self.layer('residual_only')(x, q).cnn_input)
        x[0] += 3
        torch.testing.assert_close(layer(x, q).cnn_input, self.layer('residual_only')(x, q).cnn_input)
        self.assertEqual(layer.state_dict(), {})

    def test_noncontiguous_and_float32_inputs(self):
        x, q = self.fixture(); x = x.T.contiguous().T
        self.assertFalse(x.is_contiguous())
        for sparse in (False, True):
            out = self.layer('residual_only').float()(x.float(), q.float().to_sparse() if sparse else q.float())
            self.assertEqual(out.cnn_input.dtype, torch.float32)
            torch.testing.assert_close(out.cnn_input, self.tensor([[-1, -1], [1, 1], [0, 0], [0, 0]]).float())

    def test_checkpoint_roundtrip_every_mode_and_repeated_training_graph(self):
        x, q = self.fixture()
        for mode in RegionResidualDecomposition.MODES:
            layer = self.layer(mode)
            with torch.no_grad():
                for p in layer.parameters(): p.add_(0.03)
            buffer = io.BytesIO()
            torch.save({'config': layer.export_config(), 'state_dict': layer.state_dict()}, buffer)
            buffer.seek(0); saved=torch.load(buffer, map_location=self.device, weights_only=True)
            restored=RegionResidualDecomposition(**saved['config']).to(device=self.device, dtype=x.dtype)
            restored.load_state_dict(saved['state_dict'], strict=True); restored.eval()
            torch.testing.assert_close(layer(x,q).cnn_input,restored(x,q).cnn_input,rtol=0,atol=0)
            for _ in range(2):
                features=x.clone().requires_grad_()
                restored(features,q).cnn_input.square().sum().backward()
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_projection_learns_to_use_residual_and_control_can_train(self):
        x, q = self.fixture()
        residual=self.layer('residual_only')(x,q).cnn_input.detach()
        for mode in ('x_residual', 'capacity_control'):
            layer=self.layer(mode)
            target=x+0.3*residual if mode=='x_residual' else x+0.2
            optimizer=torch.optim.Adam(layer.parameters(),lr=0.02)
            first=float((layer(x,q).cnn_input-target).square().mean())
            for _ in range(80):
                optimizer.zero_grad()
                loss=(layer(x,q).cnn_input-target).square().mean()
                loss.backward(); optimizer.step()
            last=float((layer(x,q).cnn_input-target).square().mean())
            self.assertLess(last,first*0.1)
            for p in layer.parameters(): self.assertTrue(torch.isfinite(p).all())
            if mode=='x_residual':
                self.assertGreater(float(layer.projection.weight[:,2:].abs().max()),0.01)
                q2=q.clone(); q2[1]=self.tensor([0,1,0,0])
                self.assertGreater(float((layer(x,q2).cnn_input-layer(x,q).cnn_input).abs().max()),1e-3)

    def test_rejects_malformed_shapes_values_layout_and_module_state(self):
        x,q=self.fixture()
        cases=[(x[:0],q[:0]),(x,q[:3]),(x,q[:,:0]),(x[:,0],q),
               (x,q*0.5),(x,q.long()),(x.half(),q.half()),(x.float(),q)]
        for sparse in (False,True):
            negative=q.clone(); negative[0]=self.tensor([1.1,-0.1,0,0])
            nan=q.clone(); nan[0,0]=float('nan')
            zero=q.clone(); zero[0]=0
            for bad in (negative,nan,zero): cases.append((x,bad.to_sparse() if sparse else bad))
        nonfinite=x.clone(); nonfinite[0,0]=float('inf'); cases.append((nonfinite,q))
        for features,assignment in cases:
            with self.subTest(shape=tuple(assignment.shape)),self.assertRaises(ValueError):
                self.layer()(features,assignment)
        with self.assertRaises(ValueError): self.layer()(x,q.to_sparse_csr())
        with self.assertRaises(ValueError): RegionResidualDecomposition(2,'bad')
        with self.assertRaises(ValueError): RegionResidualDecomposition(True)
        with self.assertRaises(ValueError): RegionResidualDecomposition(0)
        with self.assertRaisesRegex(ValueError,'move the M4 module'): self.layer('x_residual').float()(x,q)
        if self.device.type=='cuda':
            with self.assertRaises(ValueError): self.layer()(x,q.cpu())


if __name__ == '__main__':
    unittest.main()
