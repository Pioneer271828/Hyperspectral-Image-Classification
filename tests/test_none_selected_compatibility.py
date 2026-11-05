"""Strict compatibility gates and real synthetic calibration paths, never query."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch
from joint_training import run_training
from research.reproduction.m4g4b.none_selected_compatibility import audit_selected, compare, selected_conclusion
from source_training import SourceRuntime, source_settings
from target_training import TargetRuntime, target_settings
from target_initialization import METHODS, state_id
from test_calibrated_selection import items as source_items
from test_joint_training import PROVENANCE, SOURCE, hashes
from test_target_training import fixture

DEVICE = os.environ.get("M4G4_DEVICE", "cpu")


class NoneCompatibilityTests(unittest.TestCase):
    def test_elementwise_tolerance_and_decisions_are_separate_gates(self):
        old = torch.tensor([[.1, .9]])
        new = old + torch.tensor([[3e-6, -3e-6]])
        check = compare(old, new)
        self.assertLess(check["max_abs"], 1e-5)
        self.assertGreater(check["over_tolerance"], 0)
        self.assertEqual(check["decision_changes"], 0)
        check = compare(torch.tensor([[.5000001, .4999999]]), torch.tensor([[.4999999, .5000001]]))
        self.assertEqual(check["over_tolerance"], 0)
        self.assertEqual(check["decision_changes"], 1)
        for invalid in (torch.tensor([[float("nan"), 0.]]), torch.ones(2, 2)):
            with self.assertRaises(ValueError):
                compare(old, invalid)

    def test_complete_roles_and_selected_success_never_authorize_reuse(self):
        rows = []
        for target in ("paviaU", "xuzhou", "whu-honghu"):
            for seed in (64, 512, 2048):
                rows.append({"role": "source", "target": target, "seed": seed, "passes_selected_gate": True})
                rows.extend({"role": "target", "target": target, "seed": seed, "method": method,
                             "passes_selected_gate": True} for method in METHODS[:2])
        self.assertEqual(selected_conclusion(rows), "selected_gate_passed_trajectory_audit_required")
        rows[-1]["passes_selected_gate"] = False
        self.assertEqual(selected_conclusion(rows), "historical_none_reuse_rejected")
        with self.assertRaises(ValueError):
            selected_conclusion(rows[:-1])
        rows[-1] = copy.deepcopy(rows[-2])
        with self.assertRaises(ValueError):
            selected_conclusion(rows)

    def test_synthetic_source_target_recalibration_reload_and_artifacts_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (target_data, provenance, _), _ = fixture(root)
            opts = target_settings("xuzhou", METHODS[0], device=DEVICE)
            source_opts = source_settings("xuzhou")
            data = {n: i for n, i in source_items().items() if n in source_opts["source_tasks"]}
            for name, settings, inputs, prov, constructor in (
                ("source", source_opts, data, PROVENANCE, lambda directory: SourceRuntime()),
                ("target", opts, target_data, provenance,
                 lambda directory: TargetRuntime(opts, target_data, directory, source_spec={}))):
                with self.subTest(role=name):
                    folder = root / name
                    runtime = constructor(folder)
                    result = run_training(settings, inputs, folder, prov, SOURCE,
                        device=DEVICE, runtime_adapter=runtime)["result"]
                    step = result["selected_step"]
                    moved = {n: {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in item.items()}
                             for n, item in inputs.items()}
                    model = runtime.load_selected(folder / "primary.pt", moved,
                        folder / f"raw-validation-{step:06d}.pt", device=DEVICE)
                    before, files = state_id(model), hashes(folder)
                    out = root / (name + "-audit")
                    out.mkdir()
                    row = audit_selected(model, runtime, moved, folder, result, out)
                    self.assertEqual(state_id(model), before)
                    self.assertEqual(hashes(folder), files)
                    self.assertEqual(len(row["comparisons"]), 6)
                    self.assertTrue(all(c["vs_first"]["unequal"] == 0 for r in row["comparisons"]
                                        for c in r["comparisons"]))
                    self.assertEqual(row["head_metrics"], 6 * len(inputs))
                    self.assertGreater(row["class_recalls"], 0)


if __name__ == "__main__":
    unittest.main()
