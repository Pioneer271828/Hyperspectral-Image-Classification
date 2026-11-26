from pathlib import Path
import subprocess
import sys
import unittest

import torch
from research_runtime import tensor_hash

from transfer import wp07_graph_range as graph
from transfer.native_scale_fixture import Fixture
from transfer.native_scale_observation import _materialize, forward
from transfer.native_scale_training import baseline


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        source, _, _ = Fixture(9, "devtrain").materialize(3, 1, "full", "a", "fixture")
        self.actual = _materialize(source.dataset, source.classes, source.view, source.lattice,
                                   source.receipt["source"], "devtrain", "fixture")
        self.base = baseline(9, 64)
        self.model = graph.initialize(self.base, 7385)

    def test_geometry_stops_before_heads_and_distinguishes_real_edges(self):
        called = []
        handles = [head.register_forward_hook(lambda *args: called.append(True))
                   for group in (self.model.heads, self.model.cnn_heads, self.model.hgnn_heads) for head in group.values()]
        try:
            entry = graph.capture_geometry(self.model, self.actual, "fixture", "devtrain", 16)
        finally:
            for handle in handles:
                handle.remove()
        self.assertFalse(called)
        self.assertTrue(graph.compare_supports(entry["near"], entry["wide"], require_distinct=True)["identifiable"])
        self.assertNotIn("_pair", vars(self.model.backbone.learnable_graph))
        self.assertNotIn("forward_stages", vars(self.model.backbone))
        with self.assertRaisesRegex(ValueError, "Equivalent graph supports"):
            graph.compare_supports(entry["near"], entry["near"], require_distinct=True)

    def test_frozen_q_all_arms_same_capacity_and_unhooks_on_failure(self):
        entry = graph.capture_geometry(self.model, self.actual, "fixture", "devtrain", 16)
        counts, weights = [], []
        for arm in graph.ARMS:
            model = graph.initialize(self.base, 7385)
            counts.append(sum(p.numel() for p in model.parameters()))
            with graph.frozen_support(model, entry, arm):
                out, trace = forward(model, self.actual, "fixture", "devtrain", graph_scale=16)
                self.assertEqual(trace["q"], tensor_hash(entry["q"]))
                self.assertEqual(trace["range_nnz"], [entry["near"]._nnz(), entry["wide"]._nnz()])
                weights.append(trace["weights"]["graph"]["weights"])
                out[0].square().mean().backward()
            if arm == "content":
                self.assertGreater(float(model.backbone.range_content.weight.grad.abs().sum()), 0)
            else:
                self.assertIsNone(model.backbone.range_content.weight.grad)
            self.assertNotIn("_build_assignment", vars(model.backbone.learnable_graph))
        self.assertEqual(len(set(counts)), 1)
        self.assertEqual(weights[:3], [[1., 0.], [0., 1.], [.5, .5]])
        before, _ = forward(self.model, self.actual, "fixture", "devtrain", graph_scale=16)
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            with graph.frozen_support(self.model, entry, "near"):
                raise RuntimeError("cleanup")
        self.assertNotIn("_build_graph", vars(self.model.backbone.learnable_graph))
        after, _ = forward(self.model, self.actual, "fixture", "devtrain", graph_scale=16)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before[:3], after[:3])))

    def test_optimized_python_refused(self):
        result = subprocess.run([sys.executable, "-O", "scripts/run_wp07_graph_geometry.py", "--help"],
                                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refuses optimized Python", result.stderr)
