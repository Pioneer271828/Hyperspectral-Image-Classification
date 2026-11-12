"""Analytic checks for registered comparisons, including unmatched tail costs."""

import copy
import unittest

from evaluation.dynamic_cost_pairs import POLICY, matched_cost_pairs
from model.research_config import digest
from tests.test_dynamic_curve import fixture, run


def comparison_fixture(fixed_samples):
    gate, plan, records, results = fixture()
    fixed = copy.deepcopy(next(r for r in results if r["id"] == "r1"))
    fixed.update(id="fixed", policy_hash=digest("fixed"), wall_ms=fixed_samples)
    plan["points"].append(dict(id="fixed", method="fixed", policy_hash=fixed["policy_hash"]))
    curve = run(gate, plan, records, results + [fixed])
    spec = dict(policy=POLICY, curve_protocol_hash=curve["protocol_hash"],
                relative_tolerances=dict(p50_ms=.05, p95_ms=.05),
                pairs=[dict(adaptive=f"a{i}", fixed="fixed") for i in range(3)])
    return curve, spec


def compare(curve, spec):
    return matched_cost_pairs(curve, spec, expected_spec_hash=digest(spec), expected_curve_hash=digest(curve))


class DynamicCostPairsTests(unittest.TestCase):
    def test_paired_quality_and_all_work_points_retained(self):
        curve, spec = comparison_fixture([17., 19., 21., 23.])
        report = compare(curve, spec)
        self.assertEqual([v["cost_matched"] for v in report["comparisons"]], [False, True, False])
        middle = report["comparisons"][1]
        self.assertEqual(middle["delta"]["support"], [2, 2])
        self.assertEqual(middle["delta"]["oa"], .25)
        self.assertEqual(middle["delta"]["recall"], [0., .5])
        self.assertTrue(middle["adaptive_all_groups_protected"])
        self.assertFalse(middle["fixed_all_groups_protected"])
        self.assertFalse(report["selection_performed"])

    def test_equal_medians_do_not_hide_tail_mismatch(self):
        curve, spec = comparison_fixture([17., 19., 21., 35.])
        middle = compare(curve, spec)["comparisons"][1]
        self.assertTrue(middle["cost"]["p50_ms"]["within_tolerance"])
        self.assertFalse(middle["cost"]["p95_ms"]["within_tolerance"])
        self.assertFalse(middle["cost_matched"])

    def test_rejects_incomplete_wrong_method_or_replaced_timing(self):
        curve, spec = comparison_fixture([17., 19., 21., 23.])
        bad = copy.deepcopy(spec)
        bad["pairs"].pop()
        with self.assertRaisesRegex(ValueError, "every adaptive point"):
            compare(curve, bad)
        bad = copy.deepcopy(spec)
        bad["pairs"][0]["fixed"] = "r1"
        with self.assertRaisesRegex(ValueError, "registered fixed path"):
            compare(curve, bad)
        pin = digest(curve)
        curve["points"][-1]["cost"]["p95_ms"] = 1
        with self.assertRaisesRegex(ValueError, "Pinned curve"):
            matched_cost_pairs(curve, spec, expected_spec_hash=digest(spec), expected_curve_hash=pin)
        with self.assertRaisesRegex(ValueError, "raw samples"):
            compare(curve, spec)
