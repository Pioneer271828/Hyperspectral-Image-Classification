"""Version isolation, strict finalization and continuation for ordered M4 sources."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import m4_ordered_source as ordered
from m4_source_study import M4SourceFullRuntime
from m4_transfer import graph_binding
from joint_training import run_training, _tree_fingerprint
from test_calibrated_selection import items
from test_joint_training import SOURCE, checkpoints, hashes
from test_m4_source_study import M4SourceStudyTests

DEVICE = os.environ.get("M4G4B_DEVICE", "cpu")


class OrderedSourceTests(unittest.TestCase):
    def test_v1_and_v2_launch_settings_are_disjoint(self):
        launch = M4SourceStudyTests().fixture()
        with self.assertRaises(ValueError):
            ordered.select_case(launch, SOURCE, "paviaU", 64, "x_residual")
        launch["study"] = copy.deepcopy(ordered.EXPECTED)
        for case in launch["cases"]:
            case["settings"] = ordered.full_settings(case["held_out"], case["seed"], case["mode"])
            with self.assertRaises(ValueError):
                M4SourceFullRuntime().validate(case["settings"])
        ordered.select_case(launch, SOURCE, "paviaU", 64, "x_residual")
        launch["cases"][0]["settings"]["m4_source_study"]["validation_arithmetic"]["atol"] = 0.1
        with self.assertRaises(ValueError):
            ordered.select_case(launch, SOURCE, "paviaU", 64, "x_residual")

    def test_training_resume_metadata_reload_and_unchanged_completed_recovery(self):
        for mode in ordered.MODES:
            tiny = ordered.full_settings("paviaU", 64, mode)
            tiny.update(epochs=3, val_interval=1, optimizer_steps=8)
            tiny["config"]["training"].update(epochs=3, val_interval=1)
            data = {n: i for n, i in items().items() if n in tiny["source_tasks"]}
            provenance = {"held_out": "paviaU", "sources": tiny["source_tasks"],
                          "target_payload_accessed": False, "m4_mode": mode,
                          "m4_graph_binding": graph_binding(data)}
            with self.assertRaises(ValueError):
                ordered.M4SourceOrderedRuntime().validate(tiny)
            with patch("m4_ordered_source.full_settings", return_value=tiny), tempfile.TemporaryDirectory() as temp:
                runtime = ordered.M4SourceOrderedRuntime()
                complete, resumed = Path(temp) / "complete", Path(temp) / "resumed"
                run_training(tiny, data, complete, provenance, SOURCE, device=DEVICE, runtime_adapter=runtime)
                run_training(tiny, data, resumed, provenance, SOURCE, device=DEVICE,
                             runtime_adapter=runtime, stop_after=2)
                result = run_training(tiny, data, resumed, provenance, SOURCE, device=DEVICE,
                                      runtime_adapter=runtime, resume=True)
                self.assertEqual(result["updates_this_call"], 6)
                if DEVICE == "cpu":
                    for field in ("model", "optimizer", "scheduler", "history", "best_probabilities"):
                        self.assertEqual(_tree_fingerprint(checkpoints(complete)[field]),
                                         _tree_fingerprint(checkpoints(resumed)[field]))
                prior = hashes(resumed)
                self.assertEqual(run_training(tiny, data, resumed, provenance, SOURCE, device=DEVICE,
                    runtime_adapter=runtime, resume=True)["updates_this_call"], 0)
                self.assertEqual(prior, hashes(resumed))
                step = result["result"]["selected_step"]
                raw = resumed / f"raw-validation-{step:06d}.pt"
                gpu_data = {n: {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in item.items()}
                            for n, item in data.items()}
                snapshots = []
                for _ in range(3):
                    model = runtime.load_selected(resumed / "primary.pt", gpu_data, raw, device=DEVICE)
                    self.assertEqual(model.training_config["source_validation_arithmetic"], ordered.POLICY)
                    snapshots.append(_tree_fingerprint(runtime.validation_snapshot(model, gpu_data)[1]))
                self.assertEqual(len(set(snapshots)), 1)

