"""Synthetic training verifies the full-source adapter through the normal trainer."""
import os
from pathlib import Path
import tempfile
import unittest

from joint_training import run_training, _tree_fingerprint
from m4_full_target_runtime import settings, FullSourceTargetRuntime
from target_initialization import METHODS
from test_joint_training import SOURCE, checkpoints, hashes
from test_m4_full_transfer import fixture as full_source_fixture
from test_target_training import fixture as target_fixture


class FullSourceRuntimeTests(unittest.TestCase):
    def test_native_target_training_resume_and_completed_recovery(self):
        device = os.environ.get("M4G4_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, spec = full_source_fixture(root / "sources")
            (data, provenance, guard), _ = target_fixture(root)
            for method in METHODS[:2]:
                options = settings("xuzhou", method, "x_residual", device=device)
                kwargs = {} if method == METHODS[0] else {"source_spec": spec}
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

