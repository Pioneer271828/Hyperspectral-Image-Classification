"""Independent 9/22-class oracles for ordered V5 errors and source lineage."""

from copy import deepcopy
import unittest

from evaluation.wp12_low_rank_evidence import verify_guard
from evaluation.wp12_production_evidence import FULL_HEADS, metrics, provenance


class LowRankEvidenceTests(unittest.TestCase):
    def test_recomputed_guards_preserve_new_small_class_errors(self):
        for size in (9, 22):
            with self.subTest(classes=size):
                classes = list(range(1, size+1))
                targets = [c for c in classes for _ in range(5)]
                baseline = [[5 if i == j else 0 for j in range(size)] for i in range(size)]
                endpoints = dict(targets=targets, predictions={h: targets.copy() for h in FULL_HEADS},
                                 metrics={h: dict(confusion_matrix=baseline, count=5*size, oa=1., nll=.1) for h in FULL_HEADS})
                altered = deepcopy(endpoints)
                altered["metrics"]["cnn"]["confusion_matrix"][0][0] = 4
                altered["metrics"]["cnn"]["confusion_matrix"][0][1] = 1
                altered["metrics"]["cnn"]["oa"] = (5*size-1)/(5*size)
                altered["predictions"]["cnn"][0] = 2
                rows = verify_guard(endpoints, altered, classes, classes, dict(oa=.005, aa=.005, recall=.005))
                self.assertTrue(rows["fused"]["passes_metric_guards"])
                self.assertFalse(rows["cnn"]["passes_metric_guards"])
                self.assertEqual(rows["cnn"]["failed_recall_classes"], [1])
                self.assertEqual(rows["cnn"]["new_small_class_errors"]["1"], 1)
                quality = metrics(altered["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix")
                self.assertEqual(quality["cnn"]["per_class"][0]["recall"], .8)
                with self.assertRaisesRegex(ValueError, "score identities"):
                    verify_guard(endpoints, dict(altered, targets=targets[::-1]), classes, classes, {})
                with self.assertRaisesRegex(ValueError, "Original source producer"):
                    provenance("f"*40, "e"*40, source_steps=256,
                               source_evidence={"path": "/tmp/source.json", "sha256": "a"*64},
                               recovery_commit="f"*40, recovery_steps=178)
