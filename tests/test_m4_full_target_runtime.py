"""Synthetic training verifies the full-source adapter through the normal trainer."""
import os
from pathlib import Path
import tempfile
import unittest
import copy

import torch

from joint_training import run_training, _tree_fingerprint
from m4_full_target_runtime import settings, FullSourceTargetRuntime
from model.research_config import ResearchConfig
from target_initialization import METHODS
from test_joint_training import SOURCE, checkpoints, hashes
from test_m4_full_transfer import fixture as full_source_fixture
from test_target_training import fixture as target_fixture


class FullSourceRuntimeTests(unittest.TestCase):
    def test_ordered_calibration_snapshot_resume_and_three_exact_reloads(self):
        device = os.environ.get("M4G4_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target_root = root / "target"
            target_root.mkdir()
            (data, provenance, guard), _ = target_fixture(target_root)
            gpu = {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
                   for n, item in data.items()}
            for mode in ("capacity_control", "x_residual"):
                _, spec = full_source_fixture(root / ("source-" + mode), mode)
                for method in METHODS[:2]:
                    options = settings("xuzhou", method, mode, device=device, ordered=True)
                    kwargs = {} if method == METHODS[0] else {"source_spec": spec}
                    full, resumed = root / (mode + method + "-full"), root / (mode + method + "-resumed")
                    run_training(options, data, full, provenance, SOURCE, device=device,
                        runtime_adapter=FullSourceTargetRuntime(options, data, full, **kwargs))
                    runtime = FullSourceTargetRuntime(options, data, resumed, **kwargs)
                    run_training(options, data, resumed, provenance, SOURCE, device=device,
                        runtime_adapter=runtime, stop_after=2)
                    result = run_training(options, data, resumed, provenance, SOURCE, device=device,
                        runtime_adapter=runtime, resume=True)
                    self.assertEqual(result["updates_this_call"], 10)
                    if device == "cpu":
                        self.assertEqual(_tree_fingerprint(checkpoints(full)["best_probabilities"]),
                                         _tree_fingerprint(checkpoints(resumed)["best_probabilities"]))
                    expected = _tree_fingerprint(torch.load(resumed / "selected-probabilities.pt", weights_only=True))
                    for _ in range(3):
                        model = runtime.load_selected(resumed / "primary.pt", gpu,
                            resumed / f'raw-validation-{result["result"]["selected_step"]:06d}.pt', device=device)
                        self.assertEqual(_tree_fingerprint(runtime.validation_snapshot(model, gpu)[1]), expected)
                    before = hashes(resumed)
                    self.assertEqual(run_training(options, data, resumed, provenance, SOURCE, device=device, resume=True,
                        runtime_adapter=runtime)["updates_this_call"], 0)
                    self.assertEqual(before, hashes(resumed))
                    bad = copy.deepcopy(options)
                    bad["target_validation_arithmetic"]["atol"] = 0.1
                    with self.assertRaises(ValueError):
                        FullSourceTargetRuntime(bad, data, root / "unused", **kwargs)

    def test_native_target_training_resume_and_completed_recovery(self):
        device = os.environ.get("M4G4_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, spec = full_source_fixture(root / "sources")
            (data, provenance, guard), _ = target_fixture(root)
            for method in METHODS[:2]:
                options = settings("xuzhou", method, "x_residual", device=device)
                kwargs = {} if method == METHODS[0] else {"source_spec": spec}
                invalid = FullSourceTargetRuntime(options, data, root / "unused",
                    initial_state_sha256="0" * 64, **kwargs)
                with self.assertRaises(ValueError):
                    invalid.build_model(ResearchConfig.from_dict(options["config"]), {"xuzhou": 9})
                full, resumed = root / (method + "-full"), root / (method + "-resume")
                result = run_training(options, data, full, provenance, SOURCE, device=device,
                    runtime_adapter=FullSourceTargetRuntime(options, data, full, **kwargs))
                run_training(options, data, resumed, provenance, SOURCE, device=device, stop_after=2,
                    runtime_adapter=FullSourceTargetRuntime(options, data, resumed, **kwargs))
                continued = run_training(options, data, resumed, provenance, SOURCE, device=device, resume=True,
                    runtime_adapter=FullSourceTargetRuntime(options, data, resumed, **kwargs))
                self.assertEqual(continued["updates_this_call"], 10)
                if device == "cpu":
                    for field in ("model", "optimizer", "scheduler", "history", "best_probabilities"):
                        self.assertEqual(_tree_fingerprint(checkpoints(full)[field]),
                                         _tree_fingerprint(checkpoints(resumed)[field]))
                self.assertEqual(result["result"]["optimizer_steps"], 12)
                before = hashes(resumed)
                self.assertEqual(run_training(options, data, resumed, provenance, SOURCE, device=device, resume=True,
                    runtime_adapter=FullSourceTargetRuntime(options, data, resumed, **kwargs))["updates_this_call"], 0)
                self.assertEqual(before, hashes(resumed))

