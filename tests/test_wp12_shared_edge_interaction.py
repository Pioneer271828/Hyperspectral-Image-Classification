"""Regression checks for the frozen, same-mask edge and spectral interaction."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from evaluation import wp12_shared_edge_interaction as experiment
from evaluation.wp12_shared_edge_roster import focal_roster
from loadData.observation_sampling import BandMetadata
from model.research_config import digest


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
    def test_grid_identity_allows_a_float_rounding_step_but_not_a_shift(self):
        original = grid((3,3))
        integer_grid, divisor = experiment.pixel_grid(original)
        original.region_positions = torch.nextafter(original.region_positions, torch.ones_like(original.region_positions))
        rounded_grid, rounded_divisor = experiment.pixel_grid(original)
        self.assertTrue(torch.equal(integer_grid, rounded_grid))
        self.assertEqual(divisor, rounded_divisor)
        original.region_positions[0,0] += 1e-4
        with self.assertRaisesRegex(ValueError, "spatial seed centers"):
            experiment.pixel_grid(original)

    def test_native_views_bind_the_same_original_footprint_not_the_view_namespace(self):
        axis = BandMetadata("fixture", tuple(range(6)), tuple(range(400,406)), "nm", "verified", (True,)*6)
        spectra = np.arange(24, dtype=np.float32).reshape(2,2,6)
        views, receipts = {}, {}
        for band, columns in (("full", (0,1,2,3,4,5)), ("central_gap",(0,4,5)), ("matched_dispersed",(0,2,5))):
            views[band] = SimpleNamespace(bands=axis.subset(columns), observation=SimpleNamespace(
                spectral=spectra[..., columns], plan_hash=band, partition="val",
                global_ids=np.arange(4).reshape(2,2), coordinates=np.zeros((2,2,2))))
            receipts[band] = dict(namespace=band, role="devval", factor=1, box=[0,2,0,2],
                                  original_boxes="physical", membership="membership", coords="coords", local_ids="ids",
                                  source=dict(band_ids=list(columns), statistics=dict(parent_statistics_hash="train_stats")))
        self.assertEqual(experiment.validate_observations(views, receipts)["central_gap"], [0,4,5])
        receipts["central_gap"]["membership"] = "another_grid"
        with self.assertRaisesRegex(ValueError, "point order"):
            experiment.validate_observations(views, receipts)

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

    def test_first_focal_roster_uses_preceding_blocks_and_original_parent(self):
        groups, truth = focal_roster([
            dict(id=2, parent_tile=2, samples=[]),
            dict(id=3, parent_tile=3, samples=[(20, 2)]),
            dict(id=5, parent_tile=3, samples=[(27, 5), (30, 2)]),
            dict(id=8, parent_tile=8, samples=[(35, 5)]),
            dict(id=11, parent_tile=10, samples=[(39, 6)]),
        ], [5, 6])
        self.assertEqual(truth, [2, 5, 2, 5, 6])
        self.assertEqual(groups, [
            dict(ordinal=2, tile=5, parent=3, offset=1, points_sha256=digest([27, 30]), truth=[5, 2], first_focus=[5]),
            dict(ordinal=4, tile=11, parent=10, offset=4, points_sha256=digest([39]), truth=[6], first_focus=[6]),
        ])

    def test_trusted_roster_rejects_rehashed_block_and_point_substitutions(self):
        rows = [dict(id=3, parent_tile=2, samples=[(20, 5), (21, 5)]),
                dict(id=7, parent_tile=7, samples=[(25, 6)])]
        groups, _ = focal_roster(rows, [5, 6])
        identity = dict(scene="xuzhou", focus=[5, 6])
        archive = {view: {"band_ids": [1, 2]} for view in experiment.VIEWS}
        receipt = dict(**identity, groups=[
            dict(**spec, columns={view: [1, 2] for view in experiment.VIEWS}) for spec in groups])
        authority = identity, groups, archive
        experiment.validate_roster(receipt, authority)
        omitted = deepcopy(receipt)
        omitted["groups"].pop()
        duplicate = deepcopy(receipt)
        duplicate["groups"][1] = deepcopy(duplicate["groups"][0])
        relocated = deepcopy(receipt)
        relocated["groups"][0]["parent"] = 6
        exchanged = deepcopy(receipt)
        exchanged["groups"][0]["points_sha256"] = digest([21, 20])
        for modified, reason in ((omitted, "Incomplete"), (duplicate, "Duplicate"),
                                 (relocated, "original parent"), (exchanged, "point identity")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                experiment.validate_roster(modified, authority)

    def test_score_requires_finite_consistent_signed_margins(self):
        score = dict(predicted=2, highest_other=2, fixed_rival=2, highest_margin=-.5, margin=-.5)
        self.assertTrue(experiment.valid_score(score, 1, 2, 3))
        for fields in (dict(highest_margin=float("nan")), dict(margin=float("inf")), dict(margin=-1.1),
                       dict(highest_other=1), dict(predicted=1), dict(fixed_rival=3), dict(margin=-.7)):
            with self.subTest(fields=fields):
                self.assertFalse(experiment.valid_score(score | fields, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
