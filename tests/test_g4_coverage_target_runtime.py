"""Bounded E1 updates: matched views, old off control, exact CPU recovery/reload."""
import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from g4_coverage_target import CALIBRATION_KEY, CONTROLS, PREFIX
from g4_coverage_target_runtime import KEY, prepare
from g4_observation_target import KEY as OBSERVATION_KEY, prepare as old_prepare
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from loadData.observation_sampling import BandMetadata
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.research_config import ResearchConfig
from research_runtime import code_snapshot, write_json
from target_initialization import METHODS
from test_g4_coverage_target import fixture
from test_g4_observation_target import FixedGraphReference
from test_joint_training import checkpoints, hashes
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("COVERAGE_RUNTIME_DEVICE", "cpu")


def bound(root, manifest, metadata, folder, *, condition="continuous", method=METHODS[0], spec=None,
          preset="alternate75", support_seed=512, per_class=1, model_seed=64, sampling_seed=8128):
    return prepare(manifest, file_hash(manifest), root / folder, target="xuzhou", method=method,
        condition=condition, model_seed=model_seed, support_seed=support_seed, per_class=per_class,
        source_spec=spec, metadata=metadata, preset=preset, sampling_seed=sampling_seed, device=DEVICE)


def run(bound, folder, **kwargs):
    options, items, provenance, code, runtime, guard = bound
    return run_training(options, items, folder, provenance, code, runtime_adapter=runtime,
        finish_guard=guard, device=DEVICE, **kwargs)


class CoverageRuntimeTests(unittest.TestCase):
    def test_all_conditions_same_frozen_views_and_registered_parameter_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, meta = fixture(root)
            _, spec = source_fixture(root / "source")
            records = None
            for method in METHODS[:2]:
                for condition in CONTROLS:
                    b = bound(root, manifest, meta, "unused", condition=condition, method=method,
                        spec=spec if method == METHODS[1] else None)
                    options, _, _, _, runtime, guard = b
                    if records is None: records = options[OBSERVATION_KEY]["records"]
                    self.assertEqual(records, options[OBSERVATION_KEY]["records"])
                    model = runtime.build_model(ResearchConfig.from_dict(options["config"]), {"xuzhou": 9})
                    optimizer = runtime.optimizer(model, ResearchConfig.from_dict(options["config"]))
                    allowed = [p for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len({id(p) for p in allowed}), len(allowed))
                    self.assertEqual({id(p) for p in allowed}, {id(p) for p in model.parameters()})
                    self.assertEqual(sum(p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX)),
                                     0 if condition == "none" else 194)
                    self.assertEqual(options["optimizer_steps"], 12)
                    self.assertFalse(options[KEY]["full_experiment"])
                    guard()
            for support, k, model in ((64, 10, 2048), (512, 1, 64), (2048, 5, 512)):
                options, items, *_ = bound(root, manifest, meta, "unused", support_seed=support,
                                           per_class=k, model_seed=model)
                self.assertEqual((options["support_seed"], options["model_seed"], options["head_seed"]), (support, model, model+1000))
                self.assertEqual((len(items["xuzhou"]["train_labels"]), len(items["xuzhou"]["val_labels"])), (9*k, 45))

    def test_off_control_exact_state_history_rng_vs196(self):
        # CUDA reference deliberately fixes small-graph reduction order;
        # native independent optimization trajectories remain unaccepted.
        with tempfile.TemporaryDirectory() as temp, (FixedGraphReference() if DEVICE == "cuda" else nullcontext()):
            root = Path(temp); manifest, _, meta = fixture(root)
            _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                for preset in ("full", "alternate75"):
                    args = dict(target="xuzhou", method=method, model_seed=64, support_seed=512, per_class=1,
                        source_spec=spec if method == METHODS[1] else None, metadata=meta, preset=preset,
                        sampling_seed=8128, device=DEVICE)
                    label = method + preset
                    old = old_prepare(manifest, file_hash(manifest), root / (label + "-old"), **args)
                    new = bound(root, manifest, meta, label + "-off", condition="none", method=method,
                        spec=args["source_spec"], preset=preset)
                    run(old, root / (label + "-old")); run(new, root / (label + "-off"))
                    a, b = checkpoints(root / (label + "-old")), checkpoints(root / (label + "-off"))
                    for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities"):
                        self.assertEqual(_tree_fingerprint(a[key]), _tree_fingerprint(b[key]), label+key)
                    self.assertEqual(a["trace"], [{k: v for k, v in row.items() if k != "coverage"} for row in b["trace"]])

    def test_all_ten_roles_real_updates_fresh_resume_three_reload_and_readonly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, meta = fixture(root)
            _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                for condition in CONTROLS:
                    label = method + "-" + condition
                    args = dict(condition=condition, method=method, spec=spec if method == METHODS[1] else None)
                    whole, split = root / (label + "-whole"), root / (label + "-split")
                    run(bound(root, manifest, meta, whole.name, **args), whole)
                    self.assertEqual(run(bound(root, manifest, meta, split.name, **args), split, stop_after=2)["updates_this_call"], 2)
                    b = bound(root, manifest, meta, split.name, **args)
                    result = run(b, split, resume=True)
                    self.assertEqual(result["updates_this_call"], 10)
                    a, state = checkpoints(whole), checkpoints(split)
                    if DEVICE == "cpu":
                        for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities"):
                            self.assertEqual(_tree_fingerprint(a[key]), _tree_fingerprint(state[key]), label+key)
                    self.assertEqual([r["observation"]["bands"] for r in state["trace"]], [7, 6]*6)
                    if condition != "none":
                        self.assertTrue(all(r["coverage"]["parameters"] == 194 and r["coverage"]["gradient_norm"] > 0
                            and r["coverage"]["changed_elements"] > 0 and r["graph_changed_elements"] > 0 for r in state["trace"]))
                    moved = {n: {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in i.items()} for n, i in b[1].items()}
                    raw = split / f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
                    expected = _tree_fingerprint(torch.load(split / "selected-probabilities.pt", weights_only=True))
                    for _ in range(3):
                        model = b[4].load_selected(split / "primary.pt", moved, raw, device=DEVICE)
                        self.assertEqual(_tree_fingerprint(b[4].validation_snapshot(model, moved)[1]), expected)
                    before = hashes(split)
                    self.assertEqual(run(bound(root, manifest, meta, split.name, **args), split, resume=True)["updates_this_call"], 0)
                    self.assertEqual(before, hashes(split))
                    wrong = bound(root, manifest, meta, split.name, **{**args, "condition": "binary" if condition != "binary" else "shuffled"})
                    with self.assertRaisesRegex(ValueError, "Resume identity"): run(wrong, split, resume=True)
                    self.assertEqual(before, hashes(split))
                    model = load_model_checkpoint(split / "primary.pt", device="cpu")
                    model.training_config[CALIBRATION_KEY]["source_state_sha256"] = "a"*64
                    save_model_checkpoint(model, root / (label + "-bad.pt"))
                    with self.assertRaises(ValueError): b[4].load_selected(root / (label + "-bad.pt"), moved, raw, device=DEVICE)

    def test_index_fallback_explicit_zero_grad_and_changed_policy_input_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, meta = fixture(root)
            b = bound(root, manifest, meta, "unused")
            options, items, _, _, runtime, guard = b
            for key, value in (("optimizer_steps", 1500), ("profile", "full"), ("support_seed", 2048)):
                bad = copy.deepcopy(options); bad[key] = value
                with self.assertRaises(ValueError): runtime.validate(bad)
            bad = copy.deepcopy(options); bad[KEY]["condition"] = "binary"
            with self.assertRaises(ValueError): runtime.validate(bad)
            bad = copy.deepcopy(options); bad[OBSERVATION_KEY]["records"][1]["selected_positions"].reverse()
            with self.assertRaises(ValueError): runtime.validate(bad)
            bad = copy.deepcopy(items); bad["xuzhou"]["query_labels"] = torch.zeros(1)
            with self.assertRaises(ValueError): runtime.validate_inputs(bad)
            self.assertFalse((root / "unused").exists())
            guard()
            # An authenticated alternate index-only context is a new fixture;
            # don't forge a physical context's immutable manifest in-place.
            from g4_coverage_target import initialize
            from g4_coverage_target_runtime import CoverageTargetRuntime
            from joint_baseline import _item_fingerprint
            from model.research_config import digest
            from g4_support_target import settings as base_settings
            from g4_observation_target import freeze_settings
            from g4_coverage_target_runtime import settings
            indexed = copy.deepcopy(items)
            indexed["xuzhou"]["wavelengths"] = torch.linspace(0, 1, 7)
            metadata = BandMetadata("synthetic-index", tuple(range(7)), tuple(indexed["xuzhou"]["wavelengths"].tolist()),
                                    "normalized", "index_only", (True,)*7)
            binding = copy.deepcopy(options["support_context_binding"])
            binding["item_sha256"] = digest(_item_fingerprint(indexed["xuzhou"]))
            base = base_settings("xuzhou", METHODS[0], "none", model_seed=64, support_seed=512,
                per_class=1, binding=binding, device=DEVICE)
            opts = settings(freeze_settings(base, indexed, metadata, preset="full", sampling_seed=8128), "continuous")
            runtime = CoverageTargetRuntime(opts, indexed, root / "unused")
            cfg = ResearchConfig.from_dict(opts["config"]); model = runtime.build_model(cfg, {"xuzhou": 9})
            row = runtime.train_step_at(0, model, runtime.optimizer(model, cfg), "xuzhou", indexed["xuzhou"], cfg)
            self.assertEqual(row["coverage"]["fallback_reason"], "index_only_metadata")
            self.assertEqual((row["coverage"]["gradient_norm"], row["coverage"]["changed_elements"]), (0., 0))

    def test_cli_two_ten_zero_metadata_source_pins_and_no_full_query(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, metadata = fixture(root)
            folder = root / "metadata"; folder.mkdir(); path = folder / "bands.json"; write_json(path, asdict(metadata))
            out = root / "cli"
            command = [sys.executable, "scripts/train_g4_coverage_target_short.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--metadata", str(path),
                "--metadata-sha256", file_hash(path), "--out", str(out), "--target", "xuzhou", "--method", METHODS[0],
                "--condition", "continuous", "--model-seed", "64", "--support-seed", "2048", "--per-class", "5",
                "--preset", "alternate75", "--sampling-seed", "8128", "--device", DEVICE]
            def invoke(extra): return json.loads(subprocess.check_output(command+extra, text=True))
            self.assertEqual(invoke(["--stop-after", "2"])["updates_this_call"], 2)
            self.assertEqual(invoke(["--resume"])["updates_this_call"], 10)
            before = hashes(out)
            self.assertEqual(invoke(["--resume"])["updates_this_call"], 0); self.assertEqual(before, hashes(out))
            for extra in (["--full"], ["--query"], ["--stop-after", "1500"], ["--mode", "x_residual"]):
                self.assertNotEqual(subprocess.run(command+extra, capture_output=True).returncode, 0)
            args = dict(target="xuzhou", method=METHODS[0], condition="continuous", model_seed=64,
                support_seed=512, per_class=1, metadata=metadata, metadata_files={str(path): file_hash(path)},
                preset="alternate75", sampling_seed=8128)
            with self.assertRaises(ValueError): prepare(manifest, file_hash(manifest), folder / "bad", **args)
            *_, guard = prepare(manifest, file_hash(manifest), root / "guard", **args)
            path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaises(ValueError): guard()


if __name__ == "__main__":
    unittest.main()
