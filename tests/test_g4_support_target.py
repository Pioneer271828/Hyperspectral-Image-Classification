"""Real synthetic updates/recovery with explicit support/K and frozen source roles."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from g4_support_target import MODES, METHODS, main, prepare, settings
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.research_config import ResearchConfig
from research_runtime import code_snapshot
from test_g4_support_context import prepared
from test_joint_training import checkpoints, hashes
from test_m4_full_transfer import fixture as active_source
from test_none_full_transfer import fixture as none_source


def bound(root, manifest, out, *, mode="none", method=METHODS[0], support_seed=512, k=1,
          model_seed=64, spec=None, device="cpu"):
    return prepare(manifest, file_hash(manifest), root / out, target="xuzhou", method=method,
        mode=mode, model_seed=model_seed, support_seed=support_seed, per_class=k,
        source_spec=spec, device=device)


class SupportTargetTests(unittest.TestCase):
    def test_explicit_support_model_axes_and_bounded_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, catalog, _ = prepared(root)
            for support, k, model in ((64, 10, 2048), (512, 1, 64), (2048, 5, 512)):
                opts, items, provenance, _, runtime, guard = bound(root, catalog / "manifest.json", "unused",
                    support_seed=support, k=k, model_seed=model)
                self.assertEqual((opts["support_seed"], opts["model_seed"], opts["head_seed"]), (support, model, model + 1000))
                self.assertEqual((len(items["xuzhou"]["train_labels"]), len(items["xuzhou"]["val_labels"])), (9*k, 45))
                self.assertEqual((opts["optimizer_steps"], opts["epochs"], opts["val_interval"]), (12, 11, 3))
                self.assertFalse(provenance["support_context"]["execution_enabled"])
                model_object = runtime.build_model(ResearchConfig.from_dict(opts["config"]), {"xuzhou": 9})
                self.assertEqual(set(model_object.heads), {"xuzhou"})
                guard()

    def test_all_six_roles_actual_updates_exact_resume_reload_and_readonly(self):
        device = os.environ.get("SUPPORT_TARGET_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, catalog, _ = prepared(root)
            manifest = catalog / "manifest.json"
            for mode in MODES:
                _, spec = (none_source(root / mode) if mode == "none" else active_source(root / mode, mode))
                for method in METHODS[:2]:
                    role = mode + "-" + method
                    source = spec if method == METHODS[1] else None
                    args = dict(mode=mode, method=method, spec=source, device=device)
                    opts, items, provenance, code, runtime, guard = bound(root, manifest, role + "-full", **args)
                    full = root / (role + "-full")
                    run_training(opts, items, full, provenance, code, runtime_adapter=runtime, finish_guard=guard, device=device)
                    opts, items, provenance, code, runtime, guard = bound(root, manifest, role + "-split", **args)
                    split = root / (role + "-split")
                    partial = run_training(opts, items, split, provenance, code, runtime_adapter=runtime,
                        finish_guard=guard, device=device, stop_after=2)
                    self.assertEqual(partial["updates_this_call"], 2)
                    result = run_training(opts, items, split, provenance, code, runtime_adapter=runtime,
                        finish_guard=guard, device=device, resume=True)
                    self.assertEqual(result["updates_this_call"], 10)
                    if device == "cpu":
                        for key in ("model", "optimizer", "scheduler", "history", "best_probabilities"):
                            self.assertEqual(_tree_fingerprint(checkpoints(full)[key]), _tree_fingerprint(checkpoints(split)[key]), role + key)
                    moved = {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()} for n, item in items.items()}
                    raw = split / f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
                    expected = _tree_fingerprint(torch.load(split / "selected-probabilities.pt", weights_only=True))
                    for _ in range(3):
                        model = runtime.load_selected(split / "primary.pt", moved, raw, device=device)
                        self.assertEqual(_tree_fingerprint(runtime.validation_snapshot(model, moved)[1]), expected)
                    trace = checkpoints(split)["trace"]
                    self.assertEqual(len(trace), 12)
                    self.assertTrue(all(r["graph_changed_elements"] > 0 and r["gradient_norms"]["adapter"] > 0 for r in trace))
                    if mode != "none":
                        self.assertTrue(all(r["m4_changed_elements"] > 0 and r["m4_parameter_gradient_norm"] > 0 for r in trace))
                    before = hashes(split)
                    completed = run_training(opts, items, split, provenance, code, runtime_adapter=runtime,
                        finish_guard=guard, device=device, resume=True)
                    self.assertEqual(completed["updates_this_call"], 0)
                    self.assertEqual(before, hashes(split))
                    wrong = bound(root, manifest, role + "-split", support_seed=2048, **args)
                    with self.assertRaisesRegex(ValueError, "Resume identity"):
                        run_training(wrong[0], wrong[1], split, wrong[2], wrong[3], runtime_adapter=wrong[4], resume=True, device=device)
                    self.assertEqual(before, hashes(split))
                    model = load_model_checkpoint(split / "primary.pt", device="cpu")
                    model.training_config["support_context_binding"]["context"]["family_hash"] = "f" * 64
                    save_model_checkpoint(model, root / (role + "-tampered.pt"))
                    with self.assertRaisesRegex(ValueError, "Selected support/K"):
                        runtime.load_selected(root / (role + "-tampered.pt"), moved, raw, device=device)

    def test_reject_budget_arithmetic_labels_query_and_source_role(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, catalog, _ = prepared(root)
            opts, items, _, _, runtime, _ = bound(root, catalog / "manifest.json", "unused")
            for key, value in (("optimizer_steps", 1500), ("profile", "full"), ("support_seed", 64), ("model_seed", 512)):
                bad = copy.deepcopy(opts); bad[key] = value
                with self.assertRaises(ValueError): runtime.validate(bad)
            bad = copy.deepcopy(opts); bad["target_validation_arithmetic"]["atol"] = .1
            with self.assertRaises(ValueError): runtime.validate(bad)
            bad = copy.deepcopy(items); bad["xuzhou"]["train_labels"] = bad["xuzhou"]["train_labels"].roll(1)
            with self.assertRaises(ValueError): runtime.validate_inputs(bad)
            bad = copy.deepcopy(items); bad["xuzhou"]["query_labels"] = torch.zeros(2)
            with self.assertRaises(ValueError): runtime.validate_inputs(bad)
            with self.assertRaises(ValueError): bound(root, catalog / "manifest.json", "unused", method=METHODS[1])
            with self.assertRaises(ValueError): bound(root, catalog / "manifest.json", "unused", method=METHODS[2])
            with self.assertRaises(ValueError): bound(root, catalog / "manifest.json", "unused", spec={"source_path": "missing"})
            with self.assertRaises(ValueError):
                settings("xuzhou", METHODS[0], "none", model_seed=True, support_seed=512, per_class=1,
                         binding=opts["support_context_binding"])

    def test_manifest_input_output_and_guard_barriers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, spec, catalog, _ = prepared(root)
            manifest = catalog / "manifest.json"
            for out in (catalog / "run", root / "inputs" / "run", catalog.parent):
                with self.assertRaises(ValueError): bound(root, manifest, out)
            with self.assertRaises(ValueError):
                prepare(manifest, "0"*64, root / "bad", target="xuzhou", method=METHODS[0], mode="none",
                        model_seed=64, support_seed=512, per_class=1)
            *_, guard = bound(root, manifest, "unused")
            original = manifest.read_bytes()
            manifest.write_bytes(original + b" ")
            with self.assertRaises(ValueError): guard()
            manifest.write_bytes(original)
            path = Path(spec["observations"]["xuzhou"]["record"])
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError): guard()

    def test_cli_two_ten_zero_expected_sha_and_no_full_option(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, catalog, _ = prepared(root)
            manifest, out = catalog / "manifest.json", root / "cli"
            device = os.environ.get("SUPPORT_TARGET_DEVICE", "cpu")
            command = [sys.executable, "scripts/train_g4_support_target_short.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out),
                "--target", "xuzhou", "--method", METHODS[0], "--mode", "none", "--model-seed", "64",
                "--support-seed", "2048", "--per-class", "5", "--device", device]
            def invoke(extra):
                return json.loads(subprocess.check_output(command + extra, text=True))
            self.assertEqual(invoke(["--stop-after", "2"])["updates_this_call"], 2)
            self.assertEqual(invoke(["--resume"])["updates_this_call"], 10)
            before = hashes(out)
            self.assertEqual(invoke(["--resume"])["updates_this_call"], 0)
            self.assertEqual(before, hashes(out))
            for extra in (["--full"], ["--stop-after", "1500"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)
            wrong = command.copy(); wrong[wrong.index("--expected-commit") + 1] = "0"*40
            self.assertNotEqual(subprocess.run(wrong + ["--resume"], capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
