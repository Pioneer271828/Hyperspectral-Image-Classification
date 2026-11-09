"""Preserve first-max identity and distinguish unsupported classes from failures."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from g4_none_ordered_source import options
from joint_training import validation_epochs
from research.reproduction.m4g4b.none_source_report import derive_case, head_record, plot_case


class NoneSourceReportTests(unittest.TestCase):
    def test_full_curve_first_tie_selection_loss_windows_and_real_png(self):
        settings = options("paviaU", full=True)
        metrics = {}
        for name, count in (("xuzhou", 9), ("whu-honghu", 22)):
            cm = np.eye(count, dtype=np.int64) * 10
            cm[0, 0], cm[0, 1] = 8, 2
            recall = cm.diagonal() / cm.sum(1)
            metric = {"oa": float(cm.trace() / cm.sum()), "aa": float(recall.mean()), "nll": .2,
                      "confusion": cm.tolist(), "recall": recall.tolist()}
            metrics[name] = {phase: {h: copy.deepcopy(metric) for h in ("fused", "graph", "cnn")}
                             for phase in ("train", "val")}
        score = sum(x["val"]["fused"]["oa"] for x in metrics.values()) / 2
        history = [{"epoch": e, "step": (e + 1) * 2, "macro_oa": score, "val_loss": .2, "metrics": metrics}
                   for e in validation_epochs(settings)]
        trace = [{"step": i, "loss": {"total": (1003 - i) / 1002}, "learning_rates": [.001, .001]}
                 for i in range(1, 1003)]
        case = {"name": "paviaU-s64", "held_out": "paviaU", "seed": 64, "settings": settings}
        accepted = {"selected_step": 2, "source_validation_macro_oa": score, "elapsed_seconds": 123.}
        row = derive_case(case, accepted, history, trace)
        self.assertEqual(row["selected_step"], 2)
        self.assertFalse(row["selected_at_budget_end"])
        self.assertEqual(len(row["curve"]), 51)
        self.assertGreater(row["loss_first30"], row["loss_last30"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "curve.png"
            plot_case(row, trace, path)
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            if os.environ.get("NONE_REPORT_PREVIEW"):
                Path(os.environ["NONE_REPORT_PREVIEW"]).write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "first-max"):
            derive_case(case, {**accepted, "selected_step": 22}, history, trace)
        with self.assertRaisesRegex(ValueError, "Complete finite"):
            derive_case(case, accepted, history, trace[:-1])

    def test_unsupported_class_is_na_and_not_a_zero_recall_failure(self):
        metric = {"confusion": [[2, 1, 0], [0, 2, 0], [0, 0, 0]], "recall": [2 / 3, 1., 0.],
                  "oa": 4 / 5, "aa": 5 / 9, "nll": .2}
        row = head_record(metric)
        self.assertEqual(row["recall"], [2 / 3, 1., None])
        self.assertEqual(row["unsupported_classes"], [3])
        self.assertEqual(row["zero_recall_classes"], [])
        self.assertEqual(row["minimum_observed_recall"], 2 / 3)
        self.assertEqual(row["aa"], metric["aa"])
        with self.assertRaises(ValueError): head_record({**metric, "nll": float("nan")})
        with self.assertRaises(ValueError): head_record({**metric, "confusion": [[-1, 0], [0, 1]]})


if __name__ == "__main__":
    unittest.main()
