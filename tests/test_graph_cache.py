"""Dense graph cache regressions; GRAPH_CACHE_DEVICE=cuda runs the same contracts on GPU."""
import os
import unittest

import torch
from torch import nn

from model.VCASAHGCN import SharedHGCN


torch.set_num_threads(1)


class FreshConversionHGCN(SharedHGCN):
    """Reference with no conversion reuse; all graph convolutions stay identical."""

    def _sparse_graph(self, adjacency, dataset_name=None):
        if adjacency.is_sparse:
            return adjacency.coalesce()
        return adjacency.to(torch.float32).to_sparse().coalesce()


class GraphCacheTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('GRAPH_CACHE_DEVICE', 'cpu')
        torch.manual_seed(17)
        self.model = SharedHGCN(3, 5, 2, norm_type='group_norm').to(self.device).eval()
        self.reference = FreshConversionHGCN(3, 5, 2, norm_type='group_norm').to(self.device).eval()
        self.reference.load_state_dict(self.model.state_dict())
        self.x = torch.randn(2, 2, 3, device=self.device)
        self.q = torch.eye(4, device=self.device).to_sparse().coalesce()
        self.a = torch.tensor([[.8, .2, 0., 0.], [0., .7, .3, 0.],
                               [0., 0., .6, .4], [.1, 0., 0., .9]], device=self.device)

    def test_inplace_update_changes_outputs_and_matches_fresh_conversion(self):
        with torch.no_grad():
            before = self.model(self.x, self.q, self.a)
            self.a.copy_(self.a.roll(1, 1))
            actual = self.model(self.x, self.q, self.a)
            expected = self.reference(self.x, self.q, self.a)
        self.assertGreater((expected - before).abs().max().item(), 1e-5)
        torch.testing.assert_close(actual, expected)

    def test_warm_inference_then_eval_optimization_preserves_all_gradients(self):
        adjacency = nn.Parameter(self.a.clone())
        reference_adjacency = nn.Parameter(self.a.clone())
        pairs = ((self.model, adjacency), (self.reference, reference_adjacency))
        optimizers = [torch.optim.Adam([*model.parameters(), graph], lr=.001)
                      for model, graph in pairs]
        for _ in range(3):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                self.model(self.x, self.q, adjacency)
            inputs = [self.x.clone().requires_grad_() for _ in pairs]
            outputs = [model(x, self.q, graph) for (model, graph), x in zip(pairs, inputs)]
            for output in outputs:
                output.square().sum().backward()
            torch.testing.assert_close(outputs[0], outputs[1])
            self.assertIsNotNone(adjacency.grad)
            self.assertGreater(adjacency.grad.abs().sum().item(), 0)
            torch.testing.assert_close(adjacency.grad, reference_adjacency.grad)
            torch.testing.assert_close(inputs[0].grad, inputs[1].grad)
            for parameter, expected in zip(self.model.parameters(), self.reference.parameters()):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                torch.testing.assert_close(parameter.grad, expected.grad)
            for optimizer in optimizers:
                optimizer.step()
            torch.testing.assert_close(adjacency, reference_adjacency)
            for parameter, expected in zip(self.model.parameters(), self.reference.parameters()):
                torch.testing.assert_close(parameter, expected)
            self.assertEqual(len(self.model._sparse_graph_cache), 0)

    def test_same_storage_different_view_cannot_reuse_wrong_graph(self):
        transpose = self.a.t()
        self.assertEqual(self.a.data_ptr(), transpose.data_ptr())
        with torch.no_grad():
            first = self.model._sparse_graph(self.a)
            self.assertIs(first, self.model._sparse_graph(self.a))
            actual = self.model._sparse_graph(transpose)
            torch.testing.assert_close(actual.to_dense(), transpose)
            self.assertIsNot(actual, first)
            replacement = self.a.clone()
            replacement[0, 1] = .9
            torch.testing.assert_close(self.model._sparse_graph(replacement).to_dense(), replacement)

    def test_cache_is_bounded_and_not_persistent(self):
        keys = set(self.model.state_dict())
        with torch.no_grad():
            for size in (2, 4, 6, 8, 10):
                adjacency = torch.eye(size, device=self.device)
                sparse = self.model._sparse_graph(adjacency, str(size))
                self.assertIs(sparse, self.model._sparse_graph(adjacency, str(size)))
                self.assertEqual(len(self.model._sparse_graph_cache), 1)
        self.assertEqual(set(self.model.state_dict()), keys)

    def test_mode_device_dtype_and_parent_load_clear_cache(self):
        def warm():
            self.model.eval()
            with torch.no_grad():
                self.model._sparse_graph(self.a)
            self.assertEqual(len(self.model._sparse_graph_cache), 1)

        warm()
        self.model.train()
        self.assertFalse(self.model._sparse_graph_cache)
        with torch.no_grad():
            self.model._sparse_graph(self.a)
        self.assertFalse(self.model._sparse_graph_cache)
        warm()
        self.model.eval()
        self.assertFalse(self.model._sparse_graph_cache)
        warm()
        self.model.to('cpu')
        self.assertFalse(self.model._sparse_graph_cache)
        self.model.to(self.device)
        warm()
        self.model.double()
        self.assertFalse(self.model._sparse_graph_cache)
        self.model.float()
        parent = nn.ModuleDict({'graph': self.model})
        state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
        for assign in (False, True):
            warm()
            parent.load_state_dict(state, assign=assign)
            self.assertFalse(self.model._sparse_graph_cache)
            with torch.no_grad():
                torch.testing.assert_close(self.model(self.x, self.q, self.a),
                                           self.reference(self.x, self.q, self.a))

    def test_inference_tensors_without_versions_bypass_cache(self):
        with torch.inference_mode():
            adjacency = self.a.clone()
            self.model._sparse_graph(adjacency)
            adjacency.copy_(adjacency.roll(1, 1))
            torch.testing.assert_close(self.model._sparse_graph(adjacency).to_dense(), adjacency)
            self.assertFalse(self.model._sparse_graph_cache)

    def test_autocast_bypasses_cache_and_legacy_conversion_dtype_is_preserved(self):
        device_type = torch.device(self.device).type
        dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
        with torch.no_grad():
            self.model._sparse_graph(self.a)
            with torch.autocast(device_type, dtype=dtype):
                torch.testing.assert_close(self.model._sparse_graph(self.a).to_dense(), self.a)
                self.assertFalse(self.model._sparse_graph_cache)
            self.assertEqual(self.model._sparse_graph(self.a.double()).dtype, torch.float32)

    def test_sparse_path_preserves_values_and_gradients(self):
        with torch.no_grad():
            self.model._sparse_graph(self.a)
        logits = torch.randn(4, 4, device=self.device, requires_grad=True)
        reference_logits = logits.detach().clone().requires_grad_()
        graph = logits.softmax(-1).to_sparse().coalesce()
        reference_graph = reference_logits.softmax(-1).to_sparse().coalesce()
        actual = self.model(self.x, self.q, graph)
        expected = self.reference(self.x, self.q, reference_graph)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(logits.grad, reference_logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertFalse(self.model._sparse_graph_cache)


if __name__ == '__main__':
    unittest.main()
