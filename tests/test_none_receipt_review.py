"""A valid negative must survive review; changed claims must be rejected."""
import copy
import unittest

from research.reproduction.m4g4b.none_receipt_review import event_gate, pure_rates


class ReceiptReviewTests(unittest.TestCase):
    def test_scheduler_arithmetic_detects_near_threshold_change_without_optimizer(self):
        options = {"config": {"training": {"lr": .01, "head_lr": .1}},
                   "scheduler": {"factor": .5, "patience": 20, "threshold": .0001}}
        old = [{"step": s, "val_loss": 1. if s == 1 else .9999001} for s in range(1, 26)]
        new = copy.deepcopy(old)
        new[1]["val_loss"] = .9998999
        trace = [{"step": s} for s in range(1, 26)]
        a, b = (pure_rates(options, v, trace) for v in (old, new))
        self.assertEqual(a[21], [.01, .1])
        self.assertEqual(a[22], [.005, .05])
        self.assertEqual([i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y], [23])
        with self.assertRaises(ValueError):
            pure_rates(options, old + old[:1], trace)

    def test_off_diagonal_count_change_is_valid_negative_but_forged_pass_is_rejected(self):
        metric = {"confusion": [[1, 1, 0], [0, 2, 0], [0, 0, 2]],
                  "recall": [.5, 1., 1.], "oa": 5 / 6, "aa": 2.5 / 3, "nll": .5}
        old = {"step": 1, "macro_oa": .8, "val_loss": .5, "metrics": {"task": {
            phase: {head: copy.deepcopy(metric) for head in ("fused", "cnn", "graph")}
            for phase in ("train", "val")}}}
        event = {"step": 1, "original_macro_oa": .8, "original_val_loss": .5, "ordered": copy.deepcopy(old),
                 "passes_metrics_gate": True, "failures": []}
        self.assertEqual(event_gate(old, event), (True, 6, 18))
        event["ordered"]["metrics"]["task"]["train"]["fused"]["confusion"][0] = [1, 0, 1]
        event["passes_metrics_gate"] = False
        event["failures"] = ["task/train/fused/classification_counts"]
        self.assertEqual(event_gate(old, event), (False, 6, 18))
        event["passes_metrics_gate"] = True
        with self.assertRaises(ValueError):
            event_gate(old, event)
        event["passes_metrics_gate"] = False
        event["ordered"]["metrics"]["task"]["val"]["cnn"]["oa"] = 1.
        with self.assertRaises(ValueError):
            event_gate(old, event)


if __name__ == "__main__":
    unittest.main()
