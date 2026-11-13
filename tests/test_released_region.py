"""Independent geometry and differentiable region-replacement oracles."""
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import torch

from monitoring.released_region import RegionAction, narrow_components, observed_boundary, point_geometry, region_action
from transfer.native_scale_observation import Lattice


class RegionTests(TestCase):
    def test_dense_geometry_with_thin_strip_island_and_partial_native_cell(self):
        labels = np.ones((7, 9), dtype=int)
        labels[1:5, 3] = 2
        labels[5, 6] = 3
        labels[5, 8] = 4
        points = [(y, x, int(labels[y, x])) for y, x in np.ndindex(labels.shape)]
        states, bands = observed_boundary(labels.shape, points)
        boundary = set()
        for i, j in np.ndindex(labels.shape):
            ns = [(p, q) for p, q in np.ndindex(labels.shape) if abs(i-p)+abs(j-q) == 1]
            conflict = any(labels[p, q] != labels[i, j] for p, q in ns)
            expected = "observed_boundary" if conflict else "observed_interior" if len(ns) == 4 else "unknown"
            self.assertEqual(states[i, j], expected)
            if conflict:
                boundary.add((i, j))
        for width, observed in bands.items():
            self.assertEqual(observed, {(i, j) for i, j in np.ndindex(labels.shape)
                                       if min(abs(i-p)+abs(j-q) for p, q in boundary) < width})
        pure = [(y, x, 1) for y, x in np.ndindex((5, 5))]
        pure_states, pure_bands = observed_boundary((5, 5), pure)
        self.assertEqual(set(pure_states.values()), {"unknown", "observed_interior"})
        self.assertTrue(all(not band for band in pure_bands.values()))
        narrow = narrow_components(points)
        self.assertEqual(sorted((r["class_id"], r["area"]) for r in narrow["enclosed_components"]), [(2, 4), (3, 1)])
        thin_support = narrow_components([(4, 4, 1), (4, 5, 2)])
        self.assertIsNone(thin_support["width_one_components"])
        lattice = Lattice.make((1, 8, 1, 10), (11, 13), 4)
        ids = tuple(((y+1)*13+x+1, label) for y, x, label in points)
        cells = np.array([(y//4)*3+x//4 for y, x, _ in points])
        histogram = np.zeros((6, 4), dtype=int)
        for cell, (_, _, label) in zip(cells, points, strict=True):
            histogram[cell, label-1] += 1
        geometry = point_geometry(SimpleNamespace(lattice=lattice, classes=4),
                                  SimpleNamespace(points=ids, cells=cells, histogram=histogram))
        last = next(row for row in geometry["coarse_footprints"] if row["cell"] == 5)
        self.assertEqual(last["box"], [5, 8, 9, 10])
        self.assertEqual(last["known_counts"], [2, 0, 0, 1])
        self.assertEqual(last["band_counts"][0], 3)

    def test_action_replaces_graph_stage_only_and_gradients(self):
        class Oracle:
            def graph_stage2(self, feature, adjacency, dataset_name=None):
                return adjacency @ feature

        observed = Oracle()
        x = torch.tensor([[2., 3.], [4., 5.], [6., 7.]], requires_grad=True)
        adjacency = torch.diag(torch.tensor([3., 5., 7.]))
        baseline = observed.graph_stage2(x, adjacency)
        with region_action(observed, RegionAction("replace", donor=1)):
            replaced = observed.graph_stage2(x, adjacency)
        self.assertTrue(torch.equal(replaced, torch.stack([baseline[1], baseline[1], baseline[2]])))
        replaced.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([[0., 0.], [10., 10.], [7., 7.]])))
        with self.assertRaisesRegex(RuntimeError, "fixture exception"):
            with region_action(observed, RegionAction("zero")):
                result = observed.graph_stage2(x, adjacency)
                self.assertTrue(torch.equal(result[0], torch.zeros(2)))
                raise RuntimeError("fixture exception")
        self.assertNotIn("graph_stage2", vars(observed))
        self.assertTrue(torch.equal(observed.graph_stage2(x, adjacency), baseline))
        with self.assertRaisesRegex(ValueError, "No legal donor"):
            RegionAction("replace", donor=1).apply(x[:1])
