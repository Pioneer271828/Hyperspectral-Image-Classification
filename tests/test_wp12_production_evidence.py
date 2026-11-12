"""Check the production adapter boundaries and original reviewed WP10 evidence."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.wp12_production_evidence import WP10_MANIFEST, adapt_wp10, metrics, wp11_cost


class WP12ProductionEvidenceTests(unittest.TestCase):
    def test_refuses_false_model_or_missing_classifier(self):
        head = dict(confusion=[[4, 1], [2, 3]], count=10, oa=.7, nll=.8)
        stats = metrics(dict(light=head), [1, 2], ("light",), matrix_key="confusion")
        self.assertEqual(stats["light"]["per_class"][0]["recall"], .8)
        with self.assertRaisesRegex(ValueError, "Classifier exit"):
            metrics(dict(cnn=head), [1, 2], ("light",), matrix_key="confusion")
        with self.assertRaisesRegex(ValueError, "differs from its confusion"):
            metrics(dict(light=dict(head, oa=1.)), [1, 2], ("light",), matrix_key="confusion")
        with self.assertRaisesRegex(ValueError, "timed outputs"):
            wp11_cost(dict(summary={"light": {"weights": "a"*64}}, startup={"light": []}, resident=[]),
                      "light", "b"*64)

    def test_refuses_unreviewed_archive(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            (root / "SHA256SUMS").write_text("a" * 64 + "  real/audit.json\n")
            with self.assertRaisesRegex(ValueError, "independently reviewed"):
                adapt_wp10(dict(root=str(root), manifest_sha256=WP10_MANIFEST))

    @unittest.skipUnless(os.environ.get("WP10_REVIEWED_PROOF"), "requires the reviewed WP10 evidence archive")
    def test_reviewed_production_records(self):
        rows, comparison = adapt_wp10(dict(root=os.environ["WP10_REVIEWED_PROOF"], manifest_sha256=WP10_MANIFEST))
        self.assertEqual(len(rows), 8)
        self.assertTrue(comparison["fused_guard"])
        self.assertFalse(comparison["cnn_guard"])
        self.assertEqual(comparison["cnn_failed_classes"], [4])
        self.assertGreater(comparison["first_p50_delta_ms"], 0.)
        self.assertEqual(comparison["ternary_fallback"], "fp32")
        self.assertFalse(comparison["overall_deployment_benefit"])
        self.assertEqual({row["identity"]["train_labels"] for row in rows}, {45})
