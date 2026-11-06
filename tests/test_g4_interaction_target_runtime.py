"""Bounded M6 updates/recovery; synthetic data, CPU exact trajectory checks."""
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from g4_interaction_target import CALIBRATION_KEY, CONDITIONS, PREFIX
from g4_interaction_target_runtime import KEY, prepare
from g4_support_target import prepare as old_prepare
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.research_config import ResearchConfig
from research_runtime import code_snapshot
from target_initialization import METHODS
from test_g4_coverage_target import fixture
from test_g4_interaction_target import COUNTS
from test_g4_observation_target import FixedGraphReference
from test_joint_training import checkpoints, hashes
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("INTERACTION_RUNTIME_DEVICE", "cpu")


def bound(root, manifest, folder, *, condition="gated", method=METHODS[0], spec=None,
          model_seed=64, support_seed=512, per_class=1):
    return prepare(manifest, file_hash(manifest), root / folder, target="xuzhou", method=method,
        condition=condition, model_seed=model_seed, support_seed=support_seed, per_class=per_class,
        source_spec=spec, device=DEVICE)


def run(b, folder, **kwargs):
    options, items, provenance, code, runtime, guard = b
    return run_training(options, items, folder, provenance, code, runtime_adapter=runtime,
                        finish_guard=guard, device=DEVICE, **kwargs)


class InteractionRuntimeTests(unittest.TestCase):
    def test_conditions_roles_original_base_and_unique_full_parameter_enrollment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root)
            _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                for condition in CONDITIONS:
                    b = bound(root, manifest, "unused", condition=condition, method=method,
                              spec=spec if method == METHODS[1] else None)
                    options, _, _, _, runtime, guard = b
                    cfg = ResearchConfig.from_dict(options["config"])
                    model = runtime.build_model(cfg, {"xuzhou": 9}); optimizer = runtime.optimizer(model, cfg)
                    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids))); self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                    self.assertEqual(sum(p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX)), COUNTS[condition])
                    self.assertEqual(options["optimizer_steps"], 12); self.assertFalse(options[KEY]["observation_augmentation"])
                    self.assertEqual(cfg.interaction.graph_update_mode, "none"); self.assertEqual(cfg.model.coverage_mode, "none")
                    guard()
            for support, k, model in ((64, 10, 2048), (512, 1, 64), (2048, 5, 512)):
                options, items, *_ = bound(root, manifest, "unused", support_seed=support, per_class=k, model_seed=model)
                self.assertEqual((options["support_seed"], options["model_seed"], options["head_seed"]), (support, model, model + 1000))
                self.assertEqual((len(items["xuzhou"]["train_labels"]), len(items["xuzhou"]["val_labels"])), (9 * k, 45))

    def test_disabled_exact_old190_control_cpu_or_declared_gpu_graph_reference(self):
        with tempfile.TemporaryDirectory() as temp, (FixedGraphReference() if DEVICE == "cuda" else nullcontext()):
            root = Path(temp); manifest, _, _ = fixture(root); _, spec = source_fixture(root / "source")
            for index, method in enumerate(METHODS[:2]):
                source = spec if method == METHODS[1] else None
                old = root / (str(index) + "-old"); new = root / (str(index) + "-off")
                a = old_prepare(manifest, file_hash(manifest), old, target="xuzhou", method=method, mode="none",
                    model_seed=64, support_seed=512, per_class=1, source_spec=source, device=DEVICE)
                run(a, old); run(bound(root, manifest, new.name, condition="none", method=method, spec=source), new)
                first, second = checkpoints(old), checkpoints(new)
                for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities"):
                    self.assertEqual(_tree_fingerprint(first[key]), _tree_fingerprint(second[key]), key)
                self.assertEqual(first["trace"], [{k: v for k, v in row.items() if k != "interaction"} for row in second["trace"]])

    def test_twelve_roles_updates_fresh_resume_participation_reload_and_completed_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                for condition in CONDITIONS:
                    label = method + "-" + condition
                    args = dict(condition=condition, method=method, spec=spec if method == METHODS[1] else None)
                    whole, split = root / (label + "-whole"), root / (label + "-split")
                    run(bound(root, manifest, whole.name, **args), whole)
                    self.assertEqual(run(bound(root, manifest, split.name, **args), split, stop_after=2)["updates_this_call"], 2)
                    b = bound(root, manifest, split.name, **args)
                    result = run(b, split, resume=True); self.assertEqual(result["updates_this_call"], 10)
                    a, state = checkpoints(whole), checkpoints(split)
                    if DEVICE == "cpu":
                        for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities", "trace"):
                            self.assertEqual(_tree_fingerprint(a[key]), _tree_fingerprint(state[key]), label + key)
                    for row in state["trace"]:
                        record = row["interaction"]
                        self.assertEqual(record["parameters"], COUNTS[condition])
                        if condition != "none":
                            self.assertGreater(record["gradient_norm"], 0.); self.assertGreater(record["changed_elements"], 0)
                        for direction, stats in record["directions"].items():
                            disabled = condition == "none" or condition == ("graph_to_cnn" if direction == "cnn_to_graph" else "cnn_to_graph")
                            self.assertEqual(stats is None, disabled)
                            if stats:
                                self.assertGreater(stats["gradient_norm"], 0.); self.assertGreater(stats["changed_elements"], 0)
                    moved = {n: {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in item.items()} for n, item in b[1].items()}
                    raw = split / f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
                    expected = _tree_fingerprint(torch.load(split / "selected-probabilities.pt", weights_only=True))
                    for _ in range(3):
                        loaded = b[4].load_selected(split / "primary.pt", moved, raw, device=DEVICE)
                        self.assertEqual(_tree_fingerprint(b[4].validation_snapshot(loaded, moved)[1]), expected)
                    before = hashes(split)
                    self.assertEqual(run(bound(root, manifest, split.name, **args), split, resume=True)["updates_this_call"], 0)
                    self.assertEqual(before, hashes(split))
                    wrong = bound(root, manifest, split.name, **{**args, "condition": "none" if condition != "none" else "gated"})
                    with self.assertRaisesRegex(ValueError, "Resume identity"): run(wrong, split, resume=True)
                    self.assertEqual(before, hashes(split))
                    model = load_model_checkpoint(split / "primary.pt", device="cpu")
                    model.training_config[CALIBRATION_KEY]["source_state_sha256"] = "0" * 64
                    bad = root / (label + "-bad.pt"); save_model_checkpoint(model, bad)
                    with self.assertRaises(ValueError): b[4].load_selected(bad, moved, raw, device=DEVICE)
                    if condition == "gated" and method == METHODS[0]:
                        nested = load_model_checkpoint(split / "primary.pt", device="cpu")
                        nested.training_config[CALIBRATION_KEY]["legacy_calibration_receipt"]["state_sha256"] = "0" * 64
                        path = root / "bad-nested-receipt.pt"; save_model_checkpoint(nested, path)
                        with self.assertRaisesRegex(ValueError, "Nested original calibration receipt"):
                            b[4].load_selected(path, moved, raw, device=DEVICE)

    def test_policy_operator_query_and_nonfinite_message_preupdate_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); b = bound(root, manifest, "unused")
            options, items, _, _, runtime, guard = b
            for key, value in (("optimizer_steps", 1500), ("profile", "full"), ("support_seed", 2048)):
                bad = copy.deepcopy(options); bad[key] = value
                with self.assertRaises(ValueError): runtime.validate(bad)
            bad = copy.deepcopy(options); bad[KEY]["condition"] = "add"
            with self.assertRaises(ValueError): runtime.validate(bad)
            bad_items = copy.deepcopy(items); bad_items["xuzhou"]["query_labels"] = torch.zeros(1)
            with self.assertRaises(ValueError): runtime.validate_inputs(bad_items)
            cfg = ResearchConfig.from_dict(options["config"])
            model = runtime.build_model(cfg, {"xuzhou": 9}); optimizer = runtime.optimizer(model, cfg)
            model.backbone.interaction.cnn_to_graph.strength = .2
            with self.assertRaises(ValueError): runtime.train_step_at(0, model, optimizer, "xuzhou", items["xuzhou"], cfg)
            model.backbone.interaction.cnn_to_graph.strength = .1
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            parameter = next(model.backbone.interaction.parameters())
            handle = parameter.register_hook(lambda g: torch.full_like(g, float("nan")))
            try:
                with self.assertRaisesRegex(ValueError, "before optimizer update"):
                    runtime.train_step_at(0, model, optimizer, "xuzhou", items["xuzhou"], cfg)
            finally: handle.remove()
            for n, p in model.named_parameters(): torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            self.assertFalse(optimizer.state); self.assertFalse(optimizer._optimizer_step_pre_hooks)
            self.assertFalse((root / "unused").exists()); guard()

    def test_actual_cli_two_ten_zero_and_no_other_mechanisms_or_full_query(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); out = root / "cli"
            command = [sys.executable, "scripts/train_g4_interaction_target_short.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out), "--target", "xuzhou",
                "--method", METHODS[0], "--condition", "gated", "--model-seed", "64", "--support-seed", "2048", "--per-class", "5", "--device", DEVICE]
            def call(extra): return json.loads(subprocess.check_output(command + extra, text=True))
            self.assertEqual(call(["--stop-after", "2"])["updates_this_call"], 2)
            self.assertEqual(call(["--resume"])["updates_this_call"], 10)
            before = hashes(out); self.assertEqual(call(["--resume"])["updates_this_call"], 0); self.assertEqual(before, hashes(out))
            for extra in (["--full"], ["--query"], ["--mode", "x_residual"], ["--stop-after", "1500"], ["--preset", "alternate75"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
