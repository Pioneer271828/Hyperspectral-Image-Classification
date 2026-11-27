"""Check strict and unmatched dispositions used by the PaviaU intervention."""
import unittest

import torch

from scripts.run_pavia_matched_deletion import ARMS, matching


def directed_graph(heavy):
    weights = torch.ones(4, 4)
    for receiver, sender in enumerate(heavy):
        weights[receiver, sender] = 8
    return (weights / weights.sum(1, keepdim=True)).to_sparse_coo().coalesce()


class GeometryTests(unittest.TestCase):
    def test_exact_alternative_preserves_squared_distances_and_both_degrees(self):
        plan = matching(directed_graph([1, 3, 0, 2]), [2, 2], [28, 28], 1)
        controls = plan["controls"]
        for arm in ARMS[2:]:
            row = controls[arm]
            self.assertEqual(row["status"], "strict")
            self.assertNotEqual(row["removed"], controls["important"]["removed"])
            self.assertEqual(set(row["difference"].values()), {0})
            self.assertEqual(len(row["graph_after"]), 12)
            self.assertTrue({0, 5, 10, 15} <= set(row["graph_after"]))
        self.assertEqual(len(controls["none"]["graph_after"]), 16)

    def test_nonexistent_strict_alternative_reports_actual_geometry_mismatch(self):
        plan = matching(directed_graph([1, 0, 0, 0]), [2, 2], [28, 28], 2)
        for arm in ARMS[2:]:
            row = plan["controls"][arm]
            self.assertEqual(row["solver"], "no_distinct_control")
            self.assertNotEqual(row["status"], "strict")
            self.assertEqual(row["difference"]["count_delta"], 0)
            self.assertEqual(row["difference"]["receiver_l1"], 0)
        self.assertGreater(plan["controls"]["low_weight"]["difference"]["sender_l1"], 0)

    def test_same_only_mask_is_not_marked_as_an_experimental_control(self):
        adjacency = (torch.eye(4) * 0.5 + torch.roll(torch.eye(4), 1, dims=1) * 0.5).to_sparse_coo()
        plan = matching(adjacency, [2, 2], [28, 28], 3)
        for arm in ARMS[2:]:
            row = plan["controls"][arm]
            self.assertEqual(row["status"], "unmatched_same_mask")
            self.assertEqual(set(row["difference"].values()), {0})
