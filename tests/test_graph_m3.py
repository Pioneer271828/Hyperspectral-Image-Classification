"""M3 contracts, controls, differentiability and label-free partition behavior."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from model.graph_state import GraphState
from model.learnable_superpixel_graph import LearnableSuperpixelGraph
from model.region_pixel_alignment import pool_regions, broadcast_regions


torch.set_num_threads(1)


def manual_state():
    Q = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]]).to_sparse().coalesce()
    return GraphState(Q, torch.eye(3).to_sparse().coalesce(),
                      torch.tensor([[.25, .5], [.75, .5], [.5, .5]]), (2, 2))


class GraphContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.builder = LearnableSuperpixelGraph(4, assignment_dim=8)
        self.features = torch.randn(7, 9, 4)

    def test_default_state_equals_legacy_outputs_losses_and_gradients(self):
        reference = copy.deepcopy(self.builder)
        original_x = self.features.clone().requires_grad_()
        state_x = self.features.clone().requires_grad_()
        Q, A, auxiliary = reference(original_x, 9)
        state = self.builder.build_graph(state_x, 9)
        torch.testing.assert_close(state.Q.to_dense(), Q.to_dense(), rtol=0, atol=0)
        torch.testing.assert_close(state.A0.to_dense(), A.to_dense(), rtol=0, atol=0)
        def objective(q, a, x, losses):
            regions = pool_regions(q, x.reshape(-1, 4))
            return broadcast_regions(q, torch.sparse.mm(a, regions)).square().mean() + sum(losses.values())
        losses = {k: v for k, v in auxiliary.items() if k.endswith('_loss')}
        objective(Q, A, original_x, losses).backward()
        objective(state.Q, state.A0, state_x, state.regularizers).backward()
        torch.testing.assert_close(original_x.grad, state_x.grad, rtol=0, atol=0)
        for (name, original), (_, current) in zip(reference.named_parameters(), self.builder.named_parameters()):
            self.assertIsNotNone(current.grad, name)
            torch.testing.assert_close(original.grad, current.grad, rtol=0, atol=0)
        _, _, converted = state.to_legacy()
        self.assertIs(converted['compactness_loss'], state.regularizers['compactness_loss'])

    def test_local_candidate_support_and_row_normalization(self):
        for h, w, scale in ((1, 1, 100), (1, 7, 4), (7, 1, 4), (7, 9, 9), (3, 4, .5)):
            state = self.builder.build_graph(torch.randn(h, w, 4), scale)
            gh, gw = state.grid_shape
            rows, cols = state.Q.indices()
            base_y = (rows // w) * gh // h
            base_x = (rows % w) * gw // w
            self.assertTrue(((cols // gw - base_y).abs() <= 1).all())
            self.assertTrue(((cols % gw - base_x).abs() <= 1).all())
            source, target = state.A0.indices()
            self.assertTrue(((source // gw - target // gw).abs() <= 1).all())
            self.assertTrue(((source % gw - target % gw).abs() <= 1).all())
            self.assertLessEqual(state.diagnostics['assignment_max_degree'], 9)
            self.assertLessEqual(state.diagnostics['graph_max_degree'], 9)
            self.assertEqual(state.region_count, gh * gw)
            state.validate()

    def test_radius_zero_identity_graph_and_empty_regions_are_finite(self):
        builder = LearnableSuperpixelGraph(4, assignment_radius=0, graph_radius=0)
        for q_mode in ('learned', 'fixed_grid'):
            state = builder.build_graph(torch.randn(1, 2, 4), .25, assignment_mode=q_mode)
            self.assertGreater(state.diagnostics['empty_region_count'], 0)
            torch.testing.assert_close(state.A0.to_dense(), torch.eye(state.region_count))
            regions = pool_regions(state.Q, torch.randn(2, 4))
            self.assertTrue(torch.isfinite(regions).all())
            self.assertTrue(all(torch.isfinite(v) for v in state.regularizers.values()))

    def test_task_objective_reaches_both_assignment_and_edge_scorers(self):
        x = self.features.clone().requires_grad_()
        state = self.builder.build_graph(x, 9)
        regions = pool_regions(state.Q, x.reshape(-1, 4))
        pixels = broadcast_regions(state.Q, torch.sparse.mm(state.A0, regions))
        # No graph regularizers: task-shaped downstream loss alone must reach Q/A.
        (pixels * torch.linspace(-1, 1, pixels.numel()).reshape_as(pixels)).sum().backward()
        for name in ('pixel_query', 'seed_key', 'graph_query', 'graph_key'):
            grad = getattr(self.builder, name).weight.grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_regularizers_live_diagnostics_detached_and_independent(self):
        state = self.builder.build_graph(self.features, 9)
        for loss in state.regularizers.values():
            self.assertTrue(loss.requires_grad)
        for name, loss in state.regularizers.items():
            self.builder.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            self.assertGreater(self.builder.pixel_query.weight.grad.abs().sum().item(), 0, name)
        detached = state.detached_diagnostics()
        for name, loss in detached['regularizer_values'].items():
            self.assertFalse(loss.requires_grad)
            loss.zero_()
            self.assertIsNot(loss, state.regularizers[name])
        injected = manual_state()
        source = torch.tensor(1., requires_grad=True)
        injected = GraphState(injected.Q, injected.A0, injected.region_positions, (2, 2),
                              regularizers={'demo_loss': source}, diagnostics={'nested': [source]})
        self.assertFalse(injected.diagnostics['nested'][0].requires_grad)
        self.assertTrue(injected.regularizers['demo_loss'].requires_grad)

    def test_fixed_controls_change_only_the_declared_component(self):
        states = {(q, a): self.builder.build_graph(self.features, 9, assignment_mode=q, adjacency_mode=a)
                  for q in ('learned', 'fixed_grid') for a in ('learned', 'fixed_local')}
        for q_mode in ('learned', 'fixed_grid'):
            torch.testing.assert_close(states[q_mode, 'learned'].Q.to_dense(), states[q_mode, 'fixed_local'].Q.to_dense())
        for a_mode in ('learned', 'fixed_local'):
            state = states['fixed_grid', a_mode]
            self.assertFalse(state.Q.requires_grad)
            self.assertEqual(state.Q._nnz(), 63)
            torch.testing.assert_close(state.Q.values(), torch.ones(63))
        torch.testing.assert_close(states['learned', 'fixed_local'].A0.to_dense(), states['fixed_grid', 'fixed_local'].A0.to_dense())
        first = states['fixed_grid', 'fixed_local']
        second = self.builder.build_graph(self.features * -11, 9, assignment_mode='fixed_grid', adjacency_mode='fixed_local')
        torch.testing.assert_close(first.Q.to_dense(), second.Q.to_dense(), rtol=0, atol=0)
        torch.testing.assert_close(first.A0.to_dense(), second.A0.to_dense(), rtol=0, atol=0)

    def test_fixed_q_learned_a_gradient_boundary(self):
        state = self.builder.build_graph(self.features, 9, assignment_mode='fixed_grid')
        regions = pool_regions(state.Q, self.features.reshape(-1, 4))
        torch.sparse.mm(state.A0, regions).square().sum().backward()
        self.assertIsNone(self.builder.pixel_query.weight.grad)
        self.assertIsNone(self.builder.seed_key.weight.grad)
        self.assertGreater(self.builder.graph_query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.builder.graph_key.weight.grad.abs().sum().item(), 0)

    def test_manual_injection_and_existing_consumers_with_empty_region(self):
        state = manual_state().validate()
        with patch.object(self.builder, '_build_assignment', side_effect=AssertionError('must not rebuild')):
            actual = self.builder.build_graph(torch.randn(2, 2, 4), 9, graph_state=state)
        self.assertIs(actual, state)
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], requires_grad=True)
        regions = pool_regions(actual.Q, x)
        torch.testing.assert_close(regions, torch.tensor([[2., 3.], [6., 7.], [0., 0.]]))
        y = broadcast_regions(actual.Q, torch.sparse.mm(actual.A0, regions))
        torch.testing.assert_close(y, regions[[0, 0, 1, 1]])
        y.sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_invalid_manual_states_and_builder_options_rejected(self):
        bad_states = []
        for change in ('Q_shape', 'negative', 'row_sum', 'dense', 'A_shape', 'position_nan', 'grid', 'loss_nan'):
            state = manual_state()
            if change == 'Q_shape': state.image_shape = (3, 2)
            if change == 'negative': state.Q = (-state.Q).coalesce()
            if change == 'row_sum': state.A0 = (state.A0 * 2).coalesce()
            if change == 'dense': state.Q = state.Q.to_dense()
            if change == 'A_shape': state.A0 = torch.eye(2).to_sparse().coalesce()
            if change == 'position_nan': state.region_positions[0, 0] = float('nan')
            if change == 'grid': state.grid_shape = (2, 2)
            if change == 'loss_nan': state.regularizers['bad'] = torch.tensor(float('nan'))
            bad_states.append((change, state))
        for name, state in bad_states:
            with self.subTest(name=name), self.assertRaises(ValueError): state.validate()
        for options in ({'assignment_mode': 'static'}, {'adjacency_mode': 'static'}, {'graph_state': object()},
                        {'graph_state': manual_state(), 'assignment_mode': 'fixed_grid'}):
            with self.assertRaises(ValueError): self.builder.build_graph(self.features, 9, **options)
        for scale in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): self.builder.build_graph(self.features, scale)
        with self.assertRaises(ValueError): self.builder.build_graph(torch.randn(0, 2, 4), 9)
        with self.assertRaises(ValueError): self.builder.build_graph(torch.randn(3, 2, 4), 9, graph_state=manual_state())

    def test_checkpoint_roundtrip_and_partition_input_independence(self):
        stream = io.BytesIO()
        torch.save(self.builder.state_dict(), stream)
        stream.seek(0)
        restored = LearnableSuperpixelGraph(4, assignment_dim=8)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        expected = self.builder.build_graph(self.features, 9)
        actual = restored.build_graph(self.features, 9)
        torch.testing.assert_close(expected.Q.to_dense(), actual.Q.to_dense(), rtol=0, atol=0)
        torch.testing.assert_close(expected.A0.to_dense(), actual.A0.to_dense(), rtol=0, atol=0)
        self.builder.build_graph(torch.randn(8, 11, 4) * 100, 4)
        again = self.builder.build_graph(self.features, 9)
        torch.testing.assert_close(expected.Q.to_dense(), again.Q.to_dense(), rtol=0, atol=0)
        torch.testing.assert_close(expected.A0.to_dense(), again.A0.to_dense(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
