"""Selection ties, scheduler thresholds and immutable real synthetic trajectories."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from joint_training import run_training
from research.reproduction.m4g4b.none_trajectory_compatibility import (
    audit_trajectory, close_scalar, first_selected, metrics_compatible, replay_lr, trajectory_decision)
from source_training import SourceRuntime, source_settings
from target_training import TargetRuntime, target_settings
from target_initialization import METHODS
from test_calibrated_selection import items as source_items
from test_joint_training import PROVENANCE, SOURCE, hashes
from test_target_training import fixture

DEVICE = os.environ.get("M4G4_DEVICE", "cpu")


class NoneTrajectoryTests(unittest.TestCase):
    def test_first_ties_and_tiny_nll_difference_cannot_change_selection(self):
        events = [{"step": s, "macro_oa": oa, "val_loss": loss}
                  for s, oa, loss in ((1, .5, 1.), (2, .6, .9), (3, .6, .9))]
        self.assertEqual(first_selected(events, "source"), 2)
        self.assertEqual(first_selected(events, "target"), 2)
        other = copy.deepcopy(events)
        other[1]["val_loss"] += 1e-7
        self.assertTrue(close_scalar(events[1]["val_loss"], other[1]["val_loss"]))
        self.assertEqual(first_selected(other, "target"), 3)
        for bad in ([], [events[0], events[0]], [{**events[0], "val_loss": float("nan")}]):
            with self.assertRaises(ValueError):
                first_selected(bad, "target")

    def test_scheduler_replay_uses_next_update_and_near_threshold_difference_is_rejected(self):
        opts = {"config": {"training": {"lr": .01, "head_lr": .1}},
                "scheduler": {"factor": .5, "patience": 20, "threshold": .0001}}
        old = [{"step": s, "macro_oa": .5, "val_loss": 1. if s == 1 else .9999001}
               for s in range(1, 26)]
        trace = [{"step": s} for s in range(1, 26)]
        with patch.object(torch.optim.SGD, "step", side_effect=AssertionError("Optimizer update forbidden")):
            rates = replay_lr(opts, old, trace)
            for row, lr in zip(trace, rates):
                row["learning_rates"] = lr
            self.assertEqual(rates[21], [.01, .1])
            self.assertEqual(rates[22], [.005, .05])
            self.assertTrue(trajectory_decision(opts, old, old, trace, "target", 2)["learning_rates_unchanged"])
            new = copy.deepcopy(old)
            new[1]["val_loss"] = .9998999
            self.assertTrue(close_scalar(old[1]["val_loss"], new[1]["val_loss"]))
            row = trajectory_decision(opts, old, new, trace, "target", 2)
            self.assertTrue(row["selected_step_unchanged"])
            self.assertEqual(row["lr_mismatch_update_steps"], [23])
            with self.assertRaises(ValueError):
                trajectory_decision(opts, old, new[:-1], trace, "target", 2)
            bad = copy.deepcopy(trace)
            bad[0]["learning_rates"] = [.5, .5]
            with self.assertRaisesRegex(ValueError, "Original scheduler"):
                trajectory_decision(opts, old, old, bad, "target", 2)

    def test_per_head_classification_counts_and_scalar_metrics_are_both_protected(self):
        metric = {"oa": 1., "aa": 1., "nll": .4, "recall": [1., 1.], "confusion": [[2, 0], [0, 2]]}
        old = {"macro_oa": 1., "val_loss": .4, "metrics": {"task": {
            p: {h: copy.deepcopy(metric) for h in ("fused", "cnn", "graph")} for p in ("train", "val")}}}
        self.assertTrue(metrics_compatible(old, old)["passes_metrics_gate"])
        new = copy.deepcopy(old)
        new["metrics"]["task"]["val"]["cnn"]["confusion"] = [[1, 1], [0, 2]]
        self.assertFalse(metrics_compatible(old, new)["passes_metrics_gate"])
        new = copy.deepcopy(old)
        new["metrics"]["task"]["val"]["graph"]["nll"] += .001
        self.assertFalse(metrics_compatible(old, new)["passes_metrics_gate"])

    def test_actual_synthetic_raw_source_target_trajectories_do_not_update_or_mutate_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (target_data, provenance, _), _ = fixture(root)
            target_opts = target_settings("xuzhou", METHODS[0], device=DEVICE)
            source_opts = source_settings("xuzhou")
            source_data = {n: i for n, i in source_items().items() if n in source_opts["source_tasks"]}
            for role, opts, data, prov, constructor in (
                ("source", source_opts, source_data, PROVENANCE, lambda directory: SourceRuntime()),
                ("target", target_opts, target_data, provenance,
                 lambda directory: TargetRuntime(target_opts, target_data, directory, source_spec={}))):
                with self.subTest(role=role):
                    folder = root / role
                    runtime = constructor(folder)
                    result = run_training(opts, data, folder, prov, SOURCE, device=DEVICE,
                                          runtime_adapter=runtime)["result"]
                    moved = {n: {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in item.items()}
                             for n, item in data.items()}
                    model = runtime.load_selected(folder / "primary.pt", moved,
                        folder / f'raw-validation-{result["selected_step"]:06d}.pt', device=DEVICE)
                    before = hashes(folder)
                    owned = root / (role + "-audit")
                    owned.mkdir()
                    with patch.object(torch.optim.SGD, "step", side_effect=AssertionError("No audit update")):
                        row = audit_trajectory(model, runtime, moved, folder, result, owned, role)
                    self.assertEqual(before, hashes(folder))
                    count = len(json.loads((folder / "history.json").read_text()))
                    self.assertEqual(row["validation_events"], count)
                    self.assertEqual(row["head_metrics"], count * 6 * len(data))
                    self.assertTrue(row["original_trace_reproduced"])
                    self.assertTrue(row["learning_rates_unchanged"])
                    self.assertEqual(len(json.loads((owned / "events.json").read_text())), count)


if __name__ == "__main__":
    unittest.main()
