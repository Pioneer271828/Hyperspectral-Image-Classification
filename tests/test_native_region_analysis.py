"""Sparse affinity geometry and branch-local representation operations."""
from unittest import TestCase

import numpy as np
import torch

from monitoring.native_region_analysis import Plan, branch_update, region_options, stages
from transfer.native_scale_observation import Lattice


class AnalysisTests(TestCase):
    def test_geometry_tie_partial_and_no_donor(self):
        lattice = Lattice.make((0, 3, 0, 3), (4, 4), 2)
        q = torch.tensor([[.6, .4], [.5, .5], [.2, .8], [.4, .6]]).to_sparse().coalesce()
        rows, hard = region_options(q, lattice)
        np.testing.assert_array_equal(hard, [0, 0, 1, 1])
        self.assertEqual(rows["edge"].target, 0)
        self.assertEqual(rows["core"].donor, 1)
        only, _ = region_options(torch.ones(4, 1).to_sparse().coalesce(), lattice)
        self.assertIsNone(only["least"].donor)
        self.assertIsNone(only["edge"].source)

    def test_disjoint_graph_ranges_and_cleanup(self):
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        plan = Plan(0, 1, 2)
        change = branch_update(x, 3, plan, "replace", None)
        self.assertTrue(torch.equal(change[:3], x[:3]))
        self.assertTrue(torch.equal(change[3], x[4]))
        self.assertTrue(torch.equal(branch_update(x, 3, plan, "self", None), x))
        with self.assertRaisesRegex(ValueError, "distinct"):
            branch_update(x, 0, Plan(0, 0, 1), "replace", None)
        var = torch.tensor(0., requires_grad=True)
        changed = branch_update(x, 3, plan, "delta", var)
        grad, = torch.autograd.grad(changed[5, 0], var)
        self.assertEqual(float(grad), 1.)
        class Backbone:
            def graph_stage1(self, x, a, dataset=None):
                return a @ x

            graph_stage2 = graph_stage1

        b = Backbone()
        transform = lambda base, values, a, dataset: base(values, a, dataset)
        with self.assertRaisesRegex(RuntimeError, "interrupt"):
            with stages(b, transform, transform):
                raise RuntimeError("interrupt")
        self.assertFalse(vars(b))
