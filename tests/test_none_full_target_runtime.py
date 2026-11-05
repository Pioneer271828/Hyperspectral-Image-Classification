"""Actual synthetic target updates/resume under the explicit ordered none protocol."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch
from joint_training import _tree_fingerprint, run_training
from model.research_config import ResearchConfig
from none_full_target_runtime import NoneFullSourceTargetRuntime, settings
from target_initialization import METHODS
from test_joint_training import SOURCE, checkpoints, hashes
from test_none_full_transfer import fixture as source_fixture
from test_target_training import fixture as target_fixture


class NoneFullTargetRuntimeTests(unittest.TestCase):
    def test_protocol_full_budget_and_unchanged_none_training_options(self):
        for target in ("paviaU", "xuzhou", "whu-honghu"):
            for method in METHODS[:2]:
                short, full = settings(target, method), settings(target, method, full=True)
                self.assertEqual((short["optimizer_steps"], full["optimizer_steps"]), (12, 1500))
                self.assertEqual(full["config"]["model"]["region_residual_mode"], "none")
                self.assertEqual(full["support_seed"], 64)
                self.assertEqual((full["support_per_class"], full["validation_per_class"]), (10, 5))
        with self.assertRaises(ValueError): settings("xuzhou", METHODS[2])

    def test_both_methods_update_resume_readonly_and_exact_ordered_reloads(self):
        device = os.environ.get("NONE_TARGET_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, spec = source_fixture(root / "source")
            target = root / "target"
            target.mkdir()
            (data, provenance, _), _ = target_fixture(target)
            moved = {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()} for n, item in data.items()}
            for method in METHODS[:2]:
                opts = settings("xuzhou", method, device=device)
                kwargs = {} if method == METHODS[0] else {"source_spec": spec}
                full, split = root / (method + "-full"), root / (method + "-split")
                runtime = NoneFullSourceTargetRuntime(opts, data, full, **kwargs)
                result = run_training(opts, data, full, provenance, SOURCE, device=device, runtime_adapter=runtime)["result"]
                resumed_runtime = NoneFullSourceTargetRuntime(opts, data, split, **kwargs)
                run_training(opts, data, split, provenance, SOURCE, device=device, runtime_adapter=resumed_runtime, stop_after=2)
                resumed = run_training(opts, data, split, provenance, SOURCE, device=device, runtime_adapter=resumed_runtime, resume=True)
                self.assertEqual(resumed["updates_this_call"], 10)
                if device == "cpu":
                    for key in ("model", "optimizer", "scheduler", "history", "best_probabilities"):
                        self.assertEqual(_tree_fingerprint(checkpoints(full)[key]), _tree_fingerprint(checkpoints(split)[key]), key)
                expected = _tree_fingerprint(torch.load(full / "selected-probabilities.pt", map_location="cpu", weights_only=True))
                for _ in range(3):
                    model = runtime.load_selected(full / "primary.pt", moved, full / f'raw-validation-{result["selected_step"]:06d}.pt', device=device)
                    self.assertEqual(_tree_fingerprint(runtime.validation_snapshot(model, moved)[1]), expected)
                before = hashes(split)
                self.assertEqual(run_training(opts, data, split, provenance, SOURCE, device=device,
                    runtime_adapter=resumed_runtime, resume=True)["updates_this_call"], 0)
                self.assertEqual(before, hashes(split))
                self.assertTrue(all(r["gradient_norms"]["graph"] > 0 and r["graph_changed_elements"] > 0 for r in checkpoints(split)["trace"]))
                bad = copy.deepcopy(opts)
                bad["target_validation_arithmetic"]["atol"] = .1
                with self.assertRaises(ValueError): NoneFullSourceTargetRuntime(bad, data, root / "bad", **kwargs)
                invalid = NoneFullSourceTargetRuntime(opts, data, root / "invalid", initial_state_sha256="0" * 64, **kwargs)
                with self.assertRaises(ValueError): invalid.build_model(ResearchConfig.from_dict(opts["config"]), {"xuzhou": 9})


if __name__ == "__main__":
    unittest.main()
