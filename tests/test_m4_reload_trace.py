"""Trace validation on the registered model with synthetic source-only inputs."""
import os
import unittest

import torch

from joint_baseline import forward, readout
from m4_source_study import full_settings
from model.research_config import ResearchConfig
from research.reproduction.m4g4b.trace_reload import csr_intervention, differences, study_case
from test_calibrated_selection import items


class ReloadTraceTests(unittest.TestCase):
    def test_trace_preserves_outputs_and_exercises_sparse_replays(self):
        device = os.environ.get("M4G4B_DEVICE", "cpu")
        config = ResearchConfig.from_dict(full_settings("paviaU", 64, "x_residual")["config"])
        data = {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
                for n, item in items().items() if n in config.data.datasets}
        model = config.build_model({n: item["class_num"] for n, item in data.items()}).to(device).eval()
        saved = {}
        with torch.no_grad():
            for name, item in data.items():
                outputs = forward(model, name, item)
                saved[name] = {p: readout(outputs, item, p)[1] for p in ("train", "val")}
        rows = study_case(model, data, saved)
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(len(row["probabilities"]), 6)
            if row["phase"] == "traced":
                ops = [x["op"].split(".")[0] for x in row["same_input_sparse_replays"]]
                self.assertEqual(ops, ["pool_regions", "broadcast_regions", "_graph_conv",
                                       "_graph_conv", "broadcast_regions"])
            if device == "cpu":
                self.assertTrue(all(p["unequal"] == 0 for p in row["probabilities"].values()))
        self.assertFalse(any(layer._forward_hooks for layer in model.modules()))
        interventions = csr_intervention(model, data, saved)
        self.assertEqual(len(interventions), 4 * len(data) * 3)
        for row in interventions:
            self.assertEqual(len(row["vs_first"]), 6)
            self.assertTrue(all(p["over_tolerance"] == 0 for p in row["vs_saved"].values()))

    def test_original_gate_and_nonfinite_rejection(self):
        original = torch.tensor([0.1, 0.9])
        result = differences(original, original + 3e-6)
        self.assertEqual(result["over_tolerance"], 1)
        self.assertEqual(result["unequal"], 2)
        with self.assertRaises(ValueError):
            differences(original, torch.tensor([float("nan"), 0.9]))

