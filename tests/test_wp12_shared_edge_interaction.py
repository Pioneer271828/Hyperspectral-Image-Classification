"""Regression checks for the frozen, same-mask edge and spectral interaction."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from evaluation import wp12_shared_edge_interaction as experiment


def grid(shape, *, weights=None):
    gh, gw = shape
    size = gh*gw
    nodes = torch.arange(size)
    points = torch.stack(((nodes//gw+.5)/gh, (nodes%gw+.5)/gw), dim=1)
    ij = torch.cartesian_prod(nodes, nodes).T
    if weights is None:
        weights = torch.full((size,size), 1/size)
    a = torch.sparse_coo_tensor(ij, weights.flatten(), (size,size)).coalesce()
    return SimpleNamespace(A0=a, grid_shape=shape, image_shape=(13,13), region_positions=points,
                           diagnostics={"position_kind":"grid_seed_centers"})


class SharedEdgeBandTests(unittest.TestCase):
    def test_mask_is_exact_and_each_view_reuses_it_with_its_own_weights(self):
        dense = torch.full((4,4), .2)
        dense[torch.arange(4), torch.tensor([1,0,3,2])] = .4
        states = {band:grid((2,2)) for band in experiment.VIEWS}
        states[experiment.ANCHOR] = grid((2,2), weights=dense)
        rule = experiment.registration("a"*40)
        plan, masks = experiment.shared_masks(states, rule)
        self.assertEqual(plan["status"], "paired")
        self.assertEqual(masks, experiment.confirmed_masks(plan))
        self.assertEqual(masks["important"], [1,4,11,14])
        self.assertNotEqual(masks["important"], masks["low_weight"])
        self.assertEqual(plan["distance_squared"]["important"], plan["distance_squared"]["low_weight"])
        self.assertEqual(experiment.shared_masks(states, rule, prescribed=plan), (plan,masks))
        changed = deepcopy(plan)
        changed["distance_squared"]["important"][0] += 1
        with self.assertRaisesRegex(ValueError, "pixel-distance"):
            experiment.confirmed_masks(changed)

    def test_different_grid_or_missing_mask_keeps_the_unavailable_status(self):
        rule = experiment.registration("a"*40)
        states = {band:grid((1,2)) for band in experiment.VIEWS}
        plan, masks = experiment.shared_masks(states, rule)
        self.assertEqual(plan["status"], "no_matched_arms")
        self.assertFalse(masks)
        states["central_gap"] = grid((2,1))
        self.assertEqual(experiment.shared_masks(states, rule), ({"status":"different_native_nodes_or_support"}, {}))

    def test_interaction_counts_same_points_without_imputing_missing_classes(self):
        predictions = {head:[(1,1,2,2,1),(1,2,2,1,1),(2,2,2,2,1)]
                       for head in experiment.HEADS}
        scores = experiment.interaction_counts(predictions, "dispersed", "central", "weak", "strong", 3)["fused"]
        self.assertEqual(scores["classes"]["1"]["interaction_correct"], 2)
        self.assertEqual(scores["classes"]["1"]["per_point_delta"], {0:1,2:1})
        self.assertEqual(scores["classes"]["2"]["interaction_correct"], -1)
        self.assertIsNone(scores["classes"]["3"]["interaction_recall"])
        self.assertFalse(scores["classes"]["1"]["floor_met"])
        self.assertEqual(scores["views"]["central"]["1"]["moves"], [[1,1,1],[2,1,1]])


if __name__ == "__main__":
    unittest.main()
