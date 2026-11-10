"""Residual/capacity and final gate composition; synthetic engineering only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from g4_fusion_target import initialize as fixed_initialize, residual_mode
from g4_position_fusion_target import initialize, PREFIX, verify_initialization_binding
from g4_position_fusion_target_runtime import prepare, KEY
from g4_support_target import prepare as plain_prepare
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig
from research_runtime import code_snapshot
from target_initialization import METHODS, domain_keys
from test_g4_coverage_target import fixture
from test_g4_position_fusion_target_runtime import copy_pair, moved
from test_joint_training import checkpoints, hashes
from test_m4_full_transfer import fixture as source_fixture

torch.set_num_threads(1)
ACTIVE = ("capacity_control", "x_residual")


def bound(root, manifest, folder, mode, condition, method=METHODS[0], spec=None):
    return prepare(manifest, file_hash(manifest), root / folder, target="xuzhou", method=method,
        mode=mode, condition=condition, model_seed=64, support_seed=512, per_class=1,
        source_spec=spec, device="cpu")


def run(b, folder, **kwargs):
    options, items, provenance, code, runtime, guard = b
    return run_training(options, items, folder, provenance, code, runtime_adapter=runtime,
                        finish_guard=guard, device="cpu", **kwargs)


class ResidualPositionTests(unittest.TestCase):
    def test_eight_roles_shared_source_fresh_heads_domains_rng_and_mode_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for mode in ACTIVE:
                source, spec = source_fixture(root / mode, source_mode=mode)
                for method in METHODS[:2]:
                    args = dict(model_seed=64, head_seed=1064, mode=mode,
                                source_spec=spec if method == METHODS[1] else None)
                    base, bb = fixed_initialize("xuzhou", method, "default", **args)
                    rng = torch.get_rng_state().clone()
                    for condition in ("fixed", "position_gate"):
                        model, binding = initialize("xuzhou", method, condition, **args)
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        self.assertEqual(binding["base_binding"], bb)
                        self.assertEqual(residual_mode(bb), mode)
                        for n, v in base.state_dict().items():
                            self.assertTrue(torch.equal(v, model.state_dict()[n]), n)
                        self.assertEqual(binding["added_parameter_count"], 129 if condition == "position_gate" else 0)
                        if method == METHODS[1]:
                            for n, v in source.backbone.state_dict().items():
                                if n not in domain_keys(source.backbone):
                                    self.assertTrue(torch.equal(v, model.backbone.state_dict()[n]), n)
                            scratch, _ = initialize("xuzhou", METHODS[0], condition,
                                model_seed=64, head_seed=1064, mode=mode)
                            fresh = domain_keys(model.backbone)
                            for n, v in model.backbone.state_dict().items():
                                if n in fresh or n.startswith("position_fusion."):
                                    self.assertTrue(torch.equal(v, scratch.backbone.state_dict()[n]), n)
                            for family in ("heads", "hgnn_heads", "cnn_heads"):
                                for n, v in getattr(model, family).state_dict().items():
                                    self.assertTrue(torch.equal(v, getattr(scratch, family).state_dict()[n]), n)
                        bad = copy.deepcopy(binding)
                        bad["base_binding"]["residual_mode"] = "none"
                        with self.assertRaises(ValueError): verify_initialization_binding(model, bad)
                with self.assertRaises(ValueError):
                    initialize("xuzhou", METHODS[1], "position_gate", model_seed=64, head_seed=1064,
                               source_spec=spec, mode="x_residual" if mode == "capacity_control" else "capacity_control")
            for seed in (512, 2048):
                for mode in ACTIVE:
                    model, binding = initialize("whu-honghu", METHODS[0], "position_gate",
                        model_seed=seed, head_seed=seed+1000, mode=mode)
                    self.assertEqual(model.heads["whu-honghu"].out_features, 22)
                    verify_initialization_binding(model, binding)

    def test_fixed_training_matches_original_active_consumer_including_state_and_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root)
            for mode in ACTIVE:
                _, spec = source_fixture(root / (mode+"-source"), source_mode=mode)
                for method in METHODS[:2]:
                    source = spec if method == METHODS[1] else None
                    label = mode+"-"+method
                    a, b = root / (label+"-old"), root / (label+"-new")
                    old = plain_prepare(manifest, file_hash(manifest), a, target="xuzhou", method=method,
                        mode=mode, model_seed=64, support_seed=512, per_class=1, source_spec=source, device="cpu")
                    run(old, a); run(bound(root, manifest, b.name, mode, "fixed", method, source), b)
                    first, second = checkpoints(a), checkpoints(b)
                    for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities"):
                        self.assertEqual(_tree_fingerprint(first[key]), _tree_fingerprint(second[key]), label+key)
                    self.assertEqual(first["trace"], [{k:v for k,v in row.items() if k != "position_fusion"}
                                                     for row in second["trace"]])

    def test_eight_roles_two_ten_equals_twelve_selected_reload_and_completed_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root)
            for mode in ACTIVE:
                _, spec = source_fixture(root / (mode+"-source"), source_mode=mode)
                for method in METHODS[:2]:
                    for condition in ("fixed", "position_gate"):
                        label = mode+method+condition; source = spec if method == METHODS[1] else None
                        args = (mode, condition, method, source)
                        whole, split = root / (label+"-whole"), root / (label+"-split")
                        run(bound(root, manifest, whole.name, *args), whole)
                        self.assertEqual(run(bound(root, manifest, split.name, *args), split, stop_after=2)["updates_this_call"], 2)
                        b = bound(root, manifest, split.name, *args)
                        result = run(b, split, resume=True)
                        self.assertEqual(result["updates_this_call"], 10)
                        first, second = checkpoints(whole), checkpoints(split)
                        for key in ("model", "optimizer", "scheduler", "history", "trace", "rng", "best_probabilities"):
                            self.assertEqual(_tree_fingerprint(first[key]), _tree_fingerprint(second[key]), label+key)
                        for row in second["trace"]:
                            self.assertGreater(row["m4_parameter_gradient_norm"], 0.)
                            self.assertGreater(row["m4_changed_elements"], 0)
                            if condition == "position_gate":
                                self.assertGreater(row["position_fusion"]["gradient_norm"], 0.)
                                self.assertGreater(row["position_fusion"]["changed_elements"], 0)
                        raw = split / f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
                        expected = _tree_fingerprint(torch.load(split / "selected-probabilities.pt", weights_only=True))
                        for _ in range(3):
                            model = b[4].load_selected(split / "primary.pt", moved(b[1]), raw)
                            self.assertEqual(_tree_fingerprint(b[4].validation_snapshot(model, moved(b[1]))[1]), expected)
                        before = hashes(split)
                        self.assertEqual(run(bound(root, manifest, split.name, *args), split, resume=True)["updates_this_call"], 0)
                        self.assertEqual(before, hashes(split))
                        wrong = bound(root, manifest, split.name, mode,
                            "fixed" if condition == "position_gate" else "position_gate", method, source)
                        with self.assertRaisesRegex(ValueError, "Resume identity"): run(wrong, split, resume=True)
                        self.assertEqual(before, hashes(split))

    def test_residual_nonfinite_before_adam_and_nested_mode_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root)
            for condition in ("fixed", "position_gate"):
                b = bound(root, manifest, condition, "x_residual", condition)
                cfg = ResearchConfig.from_dict(b[0]["config"])
                model = b[4].build_model(cfg, {"xuzhou":9}); optimizer = b[4].optimizer(model, cfg)
                before = {n:p.detach().clone() for n,p in model.named_parameters()}
                hook = model.backbone.region_residual.projection.weight.register_hook(lambda g: torch.full_like(g, float("nan")))
                try:
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        b[4].train_step_at(0, model, optimizer, "xuzhou", b[1]["xuzhou"], cfg)
                finally: hook.remove()
                self.assertFalse(optimizer.state); self.assertFalse(optimizer._optimizer_step_pre_hooks)
                for n,p in model.named_parameters(): self.assertTrue(torch.equal(p, before[n]), n)
            folder = root / "trained"; b = bound(root, manifest, folder.name, "x_residual", "position_gate")
            result = run(b, folder); raw = folder / f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
            bad = copy_pair(folder, root / "bad.pt", lambda meta: meta[KEY].__setitem__("residual_mode", "capacity_control"))
            with self.assertRaises(ValueError): b[4].load_selected(bad, moved(b[1]), raw)

    def test_active_cli_two_ten_zero_no_full_or_query(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); folder = root / "cli"
            cmd = [sys.executable, "scripts/train_g4_position_fusion_target_short.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(folder),
                "--target", "xuzhou", "--method", METHODS[0], "--mode", "x_residual", "--condition", "position_gate",
                "--model-seed", "64", "--support-seed", "512", "--per-class", "1", "--device", "cpu"]
            for extra, expected in ((["--stop-after","2"],2), (["--resume"],10), (["--resume"],0)):
                self.assertEqual(json.loads(subprocess.check_output(cmd+extra, text=True))["updates_this_call"], expected)
            for extra in (["--query"], ["--full"], ["--mode", "residual_only"]):
                self.assertNotEqual(subprocess.run(cmd+extra, capture_output=True).returncode, 0)


if __name__ == "__main__": unittest.main()
