"""Four bounded position-fusion roles, closed path and strict recovery."""
import copy
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from g4_fusion_target import KEY as BASE_KEY
from g4_fusion_target_runtime import prepare as old_prepare
from g4_position_fusion_target import CALIBRATION_KEY, CONDITIONS, KEY as BINDING_KEY, PREFIX
from g4_position_fusion_target_runtime import KEY, prepare
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig
from research_runtime import code_snapshot, write_json
from target_initialization import METHODS
from test_g4_coverage_target import fixture
from test_g4_observation_target import FixedGraphReference
from test_joint_training import checkpoints, hashes
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("POSITION_RUNTIME_DEVICE", "cpu")


def bound(root, manifest, folder, *, condition="position_gate", method=METHODS[0], spec=None,
          model_seed=64, support_seed=512, per_class=1):
    return prepare(manifest, file_hash(manifest), root / folder, target="xuzhou", method=method,
        condition=condition, model_seed=model_seed, support_seed=support_seed, per_class=per_class,
        source_spec=spec, device=DEVICE)


def run(b, folder, **kwargs):
    options, items, provenance, code, runtime, guard = b
    return run_training(options, items, folder, provenance, code, runtime_adapter=runtime,
                        finish_guard=guard, device=DEVICE, **kwargs)


def moved(items, device=DEVICE):
    return {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
            for n, item in items.items()}


def copy_pair(folder, destination, mutate=None, tensor_mutate=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if tensor_mutate:
        state = torch.load(folder / "primary.pt", map_location="cpu", weights_only=True)
        tensor_mutate(state)
        torch.save(state, destination)
    else:
        shutil.copyfile(folder / "primary.pt", destination)
    metadata = json.loads((folder / "primary.pt.json").read_text())
    if mutate:
        mutate(metadata["training_config"])
    write_json(str(destination) + ".json", metadata)
    return destination


class PositionRuntimeTests(unittest.TestCase):
    def test_four_roles_original_configuration_enrollment_and_independent_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                for condition in CONDITIONS:
                    b = bound(root, manifest, "unused", condition=condition, method=method,
                              spec=spec if method == METHODS[1] else None)
                    options, _, _, _, runtime, guard = b
                    cfg = ResearchConfig.from_dict(options["config"])
                    model = runtime.build_model(cfg, {"xuzhou": 9}); optimizer = runtime.optimizer(model, cfg)
                    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                    self.assertEqual((cfg.model.alph, cfg.loss.cnn_loss_weight, cfg.loss.graph_branch_loss_weight, cfg.loss.balanced),
                                     (.8, 1., 1., True))
                    self.assertEqual(runtime.position_binding["added_parameter_count"], 129 if condition == "position_gate" else 0)
                    self.assertEqual(options["optimizer_steps"], 12)
                    self.assertFalse(options[KEY]["observation_augmentation"])
                    self.assertFalse(cfg.interaction.active)
                    self.assertEqual(cfg.model.coverage_mode, "none")
                    self.assertEqual(cfg.model.region_residual_mode, "none")
                    guard()
            for support, k, model in ((64, 10, 2048), (512, 1, 64), (2048, 5, 512)):
                options, items, *_ = bound(root, manifest, "unused", support_seed=support, per_class=k, model_seed=model)
                self.assertEqual((options["support_seed"], options["model_seed"], options["head_seed"]), (support, model, model + 1000))
                self.assertEqual((len(items["xuzhou"]["train_labels"]), len(items["xuzhou"]["val_labels"])), (9*k, 45))

    def test_fixed_exact216_default_cpu_or_predeclared_gpu_graph_control(self):
        with tempfile.TemporaryDirectory() as temp, (FixedGraphReference() if DEVICE == "cuda" else nullcontext()):
            root = Path(temp); manifest, _, _ = fixture(root); _, spec = source_fixture(root / "source")
            for index, method in enumerate(METHODS[:2]):
                source = spec if method == METHODS[1] else None
                old, new = root / (str(index) + "-old"), root / (str(index) + "-fixed")
                a = old_prepare(manifest, file_hash(manifest), old, target="xuzhou", method=method,
                    condition="default", model_seed=64, support_seed=512, per_class=1, source_spec=source, device=DEVICE)
                run(a, old)
                run(bound(root, manifest, new.name, condition="fixed", method=method, spec=source), new)
                first, second = checkpoints(old), checkpoints(new)
                for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities"):
                    self.assertEqual(_tree_fingerprint(first[key]), _tree_fingerprint(second[key]), key)
                strip = lambda rows, key: [{k: v for k, v in row.items() if k != key} for row in rows]
                self.assertEqual(strip(first["trace"], "fusion_loss"), strip(second["trace"], "position_fusion"))

    def test_four_roles_twelve_fresh_two_ten_zero_trace_and_twelve_exact_reloads(self):
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
                    result = run(b, split, resume=True)
                    self.assertEqual(result["updates_this_call"], 10)
                    first, state = checkpoints(whole), checkpoints(split)
                    if DEVICE == "cpu":
                        for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities", "trace"):
                            self.assertEqual(_tree_fingerprint(first[key]), _tree_fingerprint(state[key]), label + key)
                    for row in state["trace"]:
                        self.assertEqual(len(row["loss"]), 13)
                        record = row["position_fusion"]
                        self.assertEqual(record["condition"], condition)
                        self.assertEqual(record["parameters"], 129 if condition == "position_gate" else 0)
                        self.assertEqual(row["loss"]["weighted_graph_supervised"], row["loss"]["graph_supervised"])
                        self.assertEqual(row["loss"]["weighted_cnn_supervised"], row["loss"]["cnn_supervised"])
                        for head in ("fused_head", "graph_head", "cnn_head"):
                            self.assertGreater(row["gradient_norms"][head], 0.)
                        if condition == "position_gate":
                            self.assertGreater(record["gradient_norm"], 0.)
                            self.assertGreater(record["changed_elements"], 0)
                            self.assertEqual({g: v["parameters"] for g, v in record["groups"].items()},
                                             {"gate_weight": 128, "gate_bias": 1})
                        else:
                            self.assertEqual((record["groups"], record["gradient_norm"], record["changed_elements"]), ({}, 0., 0))
                    data = moved(b[1])
                    raw = split / f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
                    expected = _tree_fingerprint(torch.load(split / "selected-probabilities.pt", weights_only=True))
                    for _ in range(3):
                        loaded = b[4].load_selected(split / "primary.pt", data, raw, device=DEVICE)
                        self.assertEqual(_tree_fingerprint(b[4].validation_snapshot(loaded, data)[1]), expected)
                        self.assertEqual(loaded.training_config[BASE_KEY], b[4].position_binding["base_binding"])
                        self.assertEqual(loaded.training_config[CALIBRATION_KEY]["legacy_receipt"],
                                         loaded.training_config["target_bn_calibration"])
                    if DEVICE == "cuda" and condition == "position_gate" and method == METHODS[0]:
                        cpu = b[4].load_selected(split / "primary.pt", moved(b[1], "cpu"), raw, device="cpu")
                        self.assertEqual(next(cpu.parameters()).device.type, "cpu")
                        self.assertEqual(cpu.training_config["research_config"]["device"], "cuda")
                    before = hashes(split)
                    self.assertEqual(run(bound(root, manifest, split.name, **args), split, resume=True)["updates_this_call"], 0)
                    self.assertEqual(before, hashes(split))
                    wrong = bound(root, manifest, split.name, **{**args, "condition": "fixed" if condition != "fixed" else "position_gate"})
                    with self.assertRaisesRegex(ValueError, "Resume identity"):
                        run(wrong, split, resume=True)
                    self.assertEqual(before, hashes(split))

    def test_selected_metadata_nested_alias_raw_and_version_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); folder = root / "trained"
            b = bound(root, manifest, folder.name); result = run(b, folder)
            raw = folder / f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
            changes = [lambda m: m.pop(BASE_KEY),
                       lambda m: m[BINDING_KEY].__setitem__("condition", "fixed"),
                       lambda m: m[KEY].__setitem__("condition", "fixed"),
                       lambda m: m["support_context_binding"].__setitem__("support_seed", 2048),
                       lambda m: m[CALIBRATION_KEY]["legacy_calibration_receipt"].__setitem__("state_sha256", "0"*64),
                       lambda m: m[CALIBRATION_KEY]["legacy_receipt"].__setitem__("state_sha256", "0"*64)]
            for i, mutate in enumerate(changes):
                path = copy_pair(folder, root / ("bad-" + str(i) + ".pt"), mutate)
                with self.assertRaises(ValueError):
                    b[4].load_selected(path, moved(b[1]), raw, device=DEVICE)
            path = copy_pair(folder, root / "bad-version.pt",
                tensor_mutate=lambda state: state[PREFIX + "protocol_version"].add_(1))
            with self.assertRaises(ValueError):
                b[4].load_selected(path, moved(b[1]), raw, device=DEVICE)
            changed_raw = root / "different-raw.pt"
            shutil.copyfile(raw, changed_raw)
            with changed_raw.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "raw source file changed"):
                b[4].load_selected(folder / "primary.pt", moved(b[1]), changed_raw, device=DEVICE)

    def test_nonfinite_gate_and_head_rejected_before_adam_and_hooks_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root)
            for name in (PREFIX + "gate_weight", "heads.xuzhou.weight"):
                b = bound(root, manifest, "unused")
                options, items, _, _, runtime, guard = b
                cfg = ResearchConfig.from_dict(options["config"])
                model = runtime.build_model(cfg, {"xuzhou": 9}); optimizer = runtime.optimizer(model, cfg)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                handle = dict(model.named_parameters())[name].register_hook(lambda g: torch.full_like(g, float("nan")))
                try:
                    with self.assertRaisesRegex(ValueError, "Missing/nonfinite"):
                        runtime.train_step_at(0, model, optimizer, "xuzhou", items["xuzhou"], cfg)
                finally:
                    handle.remove()
                for n, p in model.named_parameters():
                    torch.testing.assert_close(p, before[n], rtol=0, atol=0)
                self.assertFalse(optimizer.state)
                self.assertFalse(optimizer._optimizer_step_pre_hooks)
                guard()

    def test_zero_gate_gradient_legal_and_policy_config_query_input_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); b = bound(root, manifest, "unused")
            options, items, _, _, runtime, guard = b
            for key, value in (("optimizer_steps", 1500), ("profile", "full"), ("support_seed", 2048)):
                bad = copy.deepcopy(options); bad[key] = value
                with self.assertRaises(ValueError):
                    runtime.validate(bad)
            bad = copy.deepcopy(options); bad[KEY]["condition"] = "unknown"
            with self.assertRaises(ValueError):
                runtime.validate(bad)
            bad_items = copy.deepcopy(items); bad_items["xuzhou"]["query_labels"] = torch.zeros(1)
            with self.assertRaises(ValueError):
                runtime.validate_inputs(bad_items)
            cfg = ResearchConfig.from_dict(options["config"])
            model = runtime.build_model(cfg, {"xuzhou": 9}); optimizer = runtime.optimizer(model, cfg)
            bad_config = replace(cfg, loss=replace(cfg.loss, cnn_loss_weight=0.))
            with self.assertRaises(ValueError):
                runtime.train_step_at(0, model, optimizer, "xuzhou", items["xuzhou"], bad_config)
            hooks = [p.register_hook(torch.zeros_like) for n, p in model.named_parameters() if n.startswith(PREFIX)]
            try:
                row = runtime.train_step_at(0, model, optimizer, "xuzhou", items["xuzhou"], cfg)
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertEqual(row["position_fusion"]["gradient_norm"], 0.)
            self.assertFalse(optimizer._optimizer_step_pre_hooks)
            self.assertTrue(optimizer.state)
            guard()

    def test_actual_cli_two_ten_zero_and_full_query_other_mechanism_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); out = root / "cli"
            command = [sys.executable, "scripts/train_g4_position_fusion_target_short.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out), "--target", "xuzhou",
                "--method", METHODS[0], "--condition", "position_gate", "--model-seed", "64", "--support-seed", "2048",
                "--per-class", "5", "--device", DEVICE]
            call = lambda extra: json.loads(subprocess.check_output(command + extra, text=True))
            self.assertEqual(call(["--stop-after", "2"])["updates_this_call"], 2)
            self.assertEqual(call(["--resume"])["updates_this_call"], 10)
            before = hashes(out)
            self.assertEqual(call(["--resume"])["updates_this_call"], 0)
            self.assertEqual(before, hashes(out))
            for extra in (["--full"], ["--query"], ["--mode", "x_residual"], ["--stop-after", "1500"], ["--preset", "alternate75"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()

