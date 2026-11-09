"""E3 numerical, persistence and full-backbone acceptance, CPU by default."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from model.budgeted_graph_router import BudgetedGraphRouter
from model.graph_edge_update import GraphEdgeUpdater
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.VCASAHGCN import MultiDatasetVCASAHGCN


torch.set_num_threads(1)


def model(**kw):
    return MultiDatasetVCASAHGCN({"demo": 3}, latent_channels=8, adapter_embed_dim=8,
                                locality_window=None, norm_type="group_norm",
                                graph_mode="static", **kw)


def inputs():
    x = torch.randn(6, 7, 14)
    wave = torch.linspace(450, 950, 12)
    q = torch.nn.functional.one_hot(torch.arange(42) % 6, 6).float().to_sparse().coalesce()
    a = (torch.ones(6, 6) / 6).to_sparse().coalesce()
    return x, wave, q, a


def predict(net, data, diagnostics=False):
    x, wave, q, a = data
    return net(x, "demo", q, a, wavelengths=wave, return_graph_info=diagnostics,
               return_diagnostics=diagnostics)


class RouterTests(unittest.TestCase):
    def test_hand_calculation_ties_zero_removal_and_isolated(self):
        a = torch.tensor([[1., 4., 4., 0.], [2., 0., 1., 0.],
                          [0., 0., 0., 0.], [1., 0., 2., 3.]])
        expected = torch.tensor([[.2, .8, 0., 0.], [2/3, 1/3, 0., 0.],
                                 [0., 0., 1., 0.], [0., 0., .4, .6]])
        for inp in (a, a.to_sparse()):
            b, info = BudgetedGraphRouter(2)(inp)
            torch.testing.assert_close(b.to_dense(), expected)
            self.assertEqual(b._nnz(), 7)
            self.assertEqual(info["route_added_self"], 2)
            self.assertTrue((b.values() > 0).all())
        b, _ = BudgetedGraphRouter(1)(a)
        self.assertTrue(torch.equal(b.to_dense(), torch.eye(4)))

    def test_budget_degeneracy_and_no_reciprocal_completion(self):
        a = torch.tensor([[.2, .8, 0.], [0., .5, .5], [0., 0., 1.]])
        b, _ = BudgetedGraphRouter(20)(a)
        torch.testing.assert_close(b.to_dense(), a)
        self.assertEqual(b._nnz(), 5)
        self.assertEqual(float(b.to_dense()[1, 0]), 0)

    def test_invalid_input_and_fixed_checkpoint_rejection(self):
        for budget in (0, -1, True, 2.0):
            with self.assertRaises(ValueError): BudgetedGraphRouter(budget)
        for a in (torch.ones(2, 3), torch.zeros(0, 0), torch.eye(2).long(),
                  torch.tensor([[float('nan')]]), torch.tensor([[-1.]])):
            with self.assertRaises(ValueError): BudgetedGraphRouter()(a)
        bad = torch.sparse_coo_tensor(torch.tensor([[0, 0], [1, 1]]),
                                     torch.tensor([2., -1.]), (2, 2))
        with self.assertRaises(ValueError): BudgetedGraphRouter()(bad)
        bad = torch.sparse_coo_tensor(torch.tensor([[0], [3]]), torch.ones(1), (2, 2))
        with self.assertRaises(ValueError): BudgetedGraphRouter()(bad)
        r = BudgetedGraphRouter(2)
        r.freeze_support(torch.ones(3, 3))
        with self.assertRaises(ValueError): r(torch.eye(3))
        state = copy.deepcopy(r.state_dict())
        with self.assertRaises(ValueError): BudgetedGraphRouter(3).load_state_dict(state)
        state['_extra_state']['fixed']['selected'] = [0, 0]
        r.load_state_dict(state)
        with self.assertRaises(ValueError): r(torch.ones(3, 3))

    def test_random_determinism_rng_and_equal_degree(self):
        a = torch.ones(11, 11)
        torch.manual_seed(71)
        rng = torch.random.get_rng_state().clone()
        r = BudgetedGraphRouter(4, "random", seed=2048)
        b, _ = r(a)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(torch.equal(b.indices(), r(a)[0].indices()))
        r.set_step(1)
        c, _ = r(a)
        self.assertFalse(torch.equal(b.indices(), c.indices()))
        restored = BudgetedGraphRouter(4, "random", seed=2048)
        restored.load_state_dict(r.state_dict())
        self.assertTrue(torch.equal(c.indices(), restored(a)[0].indices()))
        self.assertTrue(torch.equal(torch.bincount(b.indices()[0]), torch.full((11,), 4)))

    def test_cosine_graph_joint_and_detached_selection(self):
        a = torch.ones(4, 4, requires_grad=True)
        f = torch.tensor([[1., 0.], [.9, .1], [-1., 0.], [0., 1.]], requires_grad=True)
        b, _ = BudgetedGraphRouter(2, "graph")(a, f)
        self.assertEqual(b.to_dense()[0].nonzero().flatten().tolist(), [0, 1])
        p = torch.flip(f.detach(), [0]).requires_grad_()
        c, _ = BudgetedGraphRouter(2, "joint")(a, f, p)
        c.to_dense().square().sum().backward()
        self.assertIsNone(f.grad)
        self.assertIsNone(p.grad)
        self.assertIsNotNone(a.grad)
        with self.assertRaises(ValueError): BudgetedGraphRouter(2, "joint")(a, f)

    def test_retained_value_gradient_and_m7_mix_not_reintroduced(self):
        a = torch.tensor([[1., 3., 2.], [3., 1., 2.], [3., 2., 1.]], requires_grad=True)
        b, _ = BudgetedGraphRouter(2)(a)
        weights = torch.arange(9.).reshape(3, 3)
        (b.to_dense() * weights).sum().backward()
        self.assertGreater(float(a.grad.abs().sum()), 0)
        self.assertTrue((a.grad[b.to_dense() == 0] == 0).all())
        for mode in ("graph", "joint"):
            m7 = GraphEdgeUpdater(4, mode=mode)
            f, p = torch.randn(3, 4), torch.randn(3, 4)
            mixed, _ = m7(f, torch.ones(3, 3), p)
            routed, _ = BudgetedGraphRouter(2)(mixed)
            self.assertEqual(routed._nnz(), 6)
            (routed.to_dense() * weights).sum().backward()
            self.assertGreater(float(m7.query.weight.grad.abs().sum()), 0)

    def test_fixed_support_retrains_values_and_roundtrips(self):
        a = torch.rand(5, 5) + .1
        r = BudgetedGraphRouter(3)
        r.freeze_support(a)
        b, _ = r(a)
        other = BudgetedGraphRouter(3)
        other.load_state_dict(copy.deepcopy(r.state_dict()))
        changed = a.flip(0).requires_grad_()
        c, _ = other(changed)
        self.assertTrue(torch.equal(b.indices(), c.indices()))
        self.assertFalse(torch.equal(b.values(), c.values()))
        c.values().square().sum().backward()
        self.assertGreater(float(changed.grad.abs().sum()), 0)

    def test_backbone_off_identity_rng_and_checkpoint(self):
        torch.manual_seed(64)
        base = model().eval()
        torch.manual_seed(64)
        explicit_off = model(graph_route=None).eval()
        self.assertEqual(base.export_config(), explicit_off.export_config())
        self.assertNotIn("graph_route", base.export_config()['backbone_kwargs'])
        for name, tensor in base.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit_off.state_dict()[name]))
        data = inputs()
        for a, b in zip(predict(base, data), predict(explicit_off, data)):
            self.assertTrue(torch.equal(a, b))
        for mode in BudgetedGraphRouter.MODES:
            net = model(graph_route=dict(budget=3, mode=mode, seed=64), graph_update_mode="joint").eval()
            result = predict(net, data, True)
            self.assertEqual(result[3]['interaction']['route_output_nnz'], 18)
            net.backbone.graph_router.set_step(7)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'model.pt'
                save_model_checkpoint(net, path)
                restored = load_model_checkpoint(path)
                for a, b in zip(predict(net, data), predict(restored, data)):
                    self.assertTrue(torch.equal(a, b))

    def test_small_learning_and_split_optimizer_resume(self):
        torch.manual_seed(512)
        data = inputs()
        labels = torch.arange(42) % 3
        net = model(graph_route=dict(budget=3, mode="random", seed=64))
        initial = copy.deepcopy(net.state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=.003)
        losses = []
        saved = None
        for step in range(100):
            net.backbone.graph_router.set_step(step)
            opt.zero_grad()
            prediction = predict(net, data)
            loss = sum(torch.nn.functional.nll_loss(p.clamp_min(1e-10).log(), labels) for p in prediction)
            losses.append(float(loss.detach()))
            loss.backward()
            opt.step()
            if step == 39: saved = (copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict()))
        self.assertLess(sum(losses[-10:])/10, sum(losses[:10])/10)
        self.assertFalse(torch.equal(net.state_dict()['backbone.HGNN_Branch.hgc2.weight'],
                                     initial['backbone.HGNN_Branch.hgc2.weight']))
        resumed = model(graph_route=dict(budget=3, mode="random", seed=64))
        resumed.load_state_dict(saved[0])
        opt2 = torch.optim.Adam(resumed.parameters(), lr=.003)
        opt2.load_state_dict(saved[1])
        for step in range(40, 100):
            resumed.backbone.graph_router.set_step(step)
            opt2.zero_grad()
            loss = sum(torch.nn.functional.nll_loss(p.clamp_min(1e-10).log(), labels) for p in predict(resumed, data))
            loss.backward()
            opt2.step()
        for key, value in net.state_dict().items():
            if torch.is_tensor(value): self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            else: self.assertEqual(value, resumed.state_dict()[key])


if __name__ == '__main__':
    unittest.main()
