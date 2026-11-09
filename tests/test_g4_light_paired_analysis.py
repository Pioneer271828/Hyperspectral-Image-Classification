"""Synthetic count checks for the two independently decided light hypotheses."""
from copy import deepcopy
import unittest

from evaluation.classification_metrics import metrics_from_confusion
from g4_light_paired_analysis import CONTROLS, HEADS, PLAN_SHA256, analyze_rows, roles
from g4_matched_target_study import case_name, grid
from loadData.dataset_registry import DATASETS


def metric(classes, correct, *, first_class=None, support=100):
    matrix = [[0] * classes for _ in range(classes)]
    for index, row in enumerate(matrix):
        row[index] = first_class if index == 0 and first_class is not None else correct
        row[(index + 1) % classes] = support - row[index]
    return {**metrics_from_confusion(matrix, range(1, classes + 1)), "nll": 0.5}


class LightPairedAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = []
        for target, seed, method, mode in grid():
            score = metric(DATASETS[target][2], 80)
            cls.baseline.append({"target": target, "seed": seed, "method": method, "mode": mode,
                                 "name": case_name(target, seed, method, mode), "checkpoint_sha256": "1" * 64,
                                 "release_sha256": "2" * 64, "metrics": {head: score for head in HEADS}})

    def light(self, first_class):
        rows = []
        for target, seed, method, mode in roles():
            special = target == "paviaU" and mode == "x_residual" and method == CONTROLS[1]
            value = metric(DATASETS[target][2], 85, first_class=first_class if special else None)
            rows.append({"target": target, "seed": seed, "method": method, "mode": mode,
                         "metrics": {head: value for head in HEADS}})
        return rows

    def test_two_primaries_independent_with_positive_macro_and_local_failure(self):
        result = analyze_rows(self.light(77), self.baseline, plan_sha256=PLAN_SHA256)
        a, b = result["primary"]
        self.assertAlmostEqual(a["equal_mode_equal_scene_fused_oa_delta_pp"]["mean"], 5.0)
        self.assertTrue(a["passes_all_gates"])
        self.assertTrue(b["gates"]["all_model_seeds_positive"])
        self.assertFalse(b["passes_all_gates"])
        self.assertEqual(b["failed_class_protections"],
                         [{"target": "paviaU", "mode": "x_residual", "class_id": 1, "delta_pp": -3.0}])
        self.assertEqual(len(result["descriptive"]), 3)
        self.assertTrue(all("gates" not in row for row in result["descriptive"]))

    def test_exact_two_point_drop_passes_protection(self):
        result = analyze_rows(self.light(78), self.baseline, plan_sha256=PLAN_SHA256)
        self.assertTrue(result["primary"][1]["passes_all_gates"])

    def test_unequal_query_support_is_rejected(self):
        light = self.light(78)
        light[0]["metrics"]["graph"] = metric(DATASETS[light[0]["target"]][2], 85, support=101)
        with self.assertRaisesRegex(ValueError, "Matched reference class supports"):
            analyze_rows(light, self.baseline, plan_sha256=PLAN_SHA256)

    def test_changed_reported_accuracy_and_incomplete_rows_are_rejected(self):
        light = self.light(78)
        with self.assertRaisesRegex(ValueError, "All54"):
            analyze_rows(light[:-1], self.baseline, plan_sha256=PLAN_SHA256)
        forged = deepcopy(light)
        forged[0]["metrics"]["fused"]["oa"] = 1.0
        with self.assertRaisesRegex(ValueError, "Count-consistent"):
            analyze_rows(forged, self.baseline, plan_sha256=PLAN_SHA256)

    def test_wrong_preregistration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Exact Issue249"):
            analyze_rows([], [], plan_sha256="wrong")


if __name__ == "__main__":
    unittest.main()
