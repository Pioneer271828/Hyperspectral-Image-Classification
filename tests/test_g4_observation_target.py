"""Target M1: actual changed B, matched full control, immutable replay/reload."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from g4_observation_target import KEY, ObservationTargetRuntime, freeze_settings, prepare
from g4_support_target import prepare as old_prepare
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from loadData.observation_sampling import BandMetadata
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, write_json
from target_initialization import METHODS
from test_g4_support_context import prepared
from test_joint_training import checkpoints, hashes
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("OBS_TARGET_DEVICE", "cpu")


def metadata(item):
    return BandMetadata("synthetic:xuzhou", tuple(range(item["bands"])),
        tuple(item["wavelengths"].tolist()), "nm", "approximate", (True,) * item["bands"],
        None if item["fwhm"] is None else tuple(item["fwhm"].tolist()),
        None if item["fwhm"] is None else "nm")


def setup(root):
    _, _, catalog, _ = prepared(root)
    manifest = catalog / "manifest.json"
    base = old_prepare(manifest, file_hash(manifest), root / "unused", target="xuzhou", method=METHODS[0],
        mode="none", model_seed=64, support_seed=512, per_class=1, device=DEVICE)
    return manifest, base, metadata(base[1]["xuzhou"])


def bound(root, manifest, meta, folder, *, method=METHODS[0], spec=None, preset="alternate75", seed=8128):
    return prepare(manifest, file_hash(manifest), root / folder, target="xuzhou", method=method,
        model_seed=64, support_seed=512, per_class=1, source_spec=spec, device=DEVICE,
        metadata=meta, preset=preset, sampling_seed=seed)


def run(bound, folder, **kwargs):
    opts, items, provenance, code, runtime, guard = bound
    return run_training(opts, items, folder, provenance, code, runtime_adapter=runtime,
        finish_guard=guard, device=DEVICE, **kwargs)


class ObservationTargetTests(unittest.TestCase):
    def test_rng_independent_freeze_and_exact_whole_scene_views(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, base, meta = setup(root)
            opts, items = base[:2]
            before = _tree_fingerprint((torch.get_rng_state(), np.random.get_state()[1].tolist(), random.getstate()))
            frozen = freeze_settings(opts, items, meta, preset="alternate75", sampling_seed=8128)
            self.assertEqual(before, _tree_fingerprint((torch.get_rng_state(), np.random.get_state()[1].tolist(), random.getstate())))
            self.assertEqual(frozen, freeze_settings(opts, items, meta, preset="alternate75", sampling_seed=8128))
            policy = frozen[KEY]
            changed = copy.deepcopy(items); changed["xuzhou"]["train_labels"] = changed["xuzhou"]["train_labels"].roll(1)
            other = freeze_settings(opts, changed, meta, preset="alternate75", sampling_seed=8128)
            self.assertEqual([r["selected_positions"] for r in policy["records"]],
                [r["selected_positions"] for r in other[KEY]["records"]])
            self.assertNotEqual(policy["policy_sha256"], other[KEY]["policy_sha256"])
            other = freeze_settings(opts, items, meta, preset="alternate75", sampling_seed=8129)
            self.assertNotEqual(policy["records"], other[KEY]["records"])
            # Exercise synchronized supplied widths (current real inputs have none).
            items["xuzhou"]["fwhm"] = torch.full((7,), 20.)
            opts["support_context_binding"]["item_sha256"] = digest(__import__("joint_baseline")._item_fingerprint(items["xuzhou"]))
            meta = metadata(items["xuzhou"])
            frozen = freeze_settings(opts, items, meta, preset="alternate75", sampling_seed=8128)
            runtime = ObservationTargetRuntime(frozen, items, root / "unused")
            item = items["xuzhou"]; original = _tree_fingerprint(item)
            for index in range(12):
                view, selected, record = runtime.view(index, "xuzhou", item)
                positions = record["selected_positions"]
                self.assertEqual(view["bands"], 7 if index % 2 == 0 else 6)
                self.assertEqual(tuple(view["wavelengths"].tolist()), selected.wavelengths)
                self.assertEqual(tuple(view["fwhm"].tolist()), selected.fwhm)
                self.assertEqual(selected.band_ids, tuple(meta.band_ids[i] for i in positions))
                torch.testing.assert_close(view["net_input"][..., :-2], item["net_input"][..., positions], rtol=0, atol=0)
                torch.testing.assert_close(view["net_input"][..., -2:], item["net_input"][..., -2:], rtol=0, atol=0)
                for key in ("Q", "A", "train_labels", "train_indices", "val_labels", "val_indices"):
                    self.assertIs(view[key], item[key])
            self.assertEqual(original, _tree_fingerprint(item))

    def test_full_control_matches_accepted_target_arithmetic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, meta = setup(root)
            _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                spec_arg = spec if method == METHODS[1] else None
                old = old_prepare(manifest, file_hash(manifest), root / (method + "-old"), target="xuzhou",
                    method=method, mode="none", model_seed=64, support_seed=512, per_class=1, source_spec=spec_arg, device=DEVICE)
                new = bound(root, manifest, meta, method + "-full", method=method, spec=spec_arg, preset="full")
                run(old, root / (method + "-old")); run(new, root / (method + "-full"))
                a, b = checkpoints(root / (method + "-old")), checkpoints(root / (method + "-full"))
                for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities"):
                    self.assertEqual(_tree_fingerprint(a[key]), _tree_fingerprint(b[key]), method + key)
                self.assertEqual(a["trace"], [{k: v for k, v in row.items() if k != "observation"} for row in b["trace"]])

    def test_subsets_reach_forward_gradients_resume_reload_and_readonly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, meta = setup(root)
            _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                spec_arg = spec if method == METHODS[1] else None
                probe = bound(root, manifest, meta, "probe", method=method, spec=spec_arg)
                runtime, opts, items = probe[4], probe[0], probe[1]
                cfg = ResearchConfig.from_dict(opts["config"])
                model = runtime.build_model(cfg, {"xuzhou": 9}); optimizer = runtime.optimizer(model, cfg)
                seen = []
                hook = model.backbone.adapter.register_forward_pre_hook(
                    lambda module, args: seen.append(args[0].shape[-1]))
                row = runtime.train_step_at(1, model, optimizer, "xuzhou", items["xuzhou"], cfg)
                hook.remove()
                self.assertEqual(seen, [6]); self.assertGreater(row["gradient_norms"]["adapter"], 0)
                full, split = root / (method + "-whole"), root / (method + "-split")
                run(bound(root, manifest, meta, full.name, method=method, spec=spec_arg), full)
                self.assertEqual(run(bound(root, manifest, meta, split.name, method=method, spec=spec_arg), split,
                    stop_after=2)["updates_this_call"], 2)
                resumed = bound(root, manifest, meta, split.name, method=method, spec=spec_arg)
                result = run(resumed, split, resume=True)
                self.assertEqual(result["updates_this_call"], 10)
                a, b = checkpoints(full), checkpoints(split)
                if DEVICE == "cpu":
                    for key in ("model", "optimizer", "scheduler", "history", "rng", "best_probabilities"):
                        self.assertEqual(_tree_fingerprint(a[key]), _tree_fingerprint(b[key]), method + key)
                self.assertEqual([r["observation"]["bands"] for r in b["trace"]], [7, 6]*6)
                self.assertTrue(all(r["graph_changed_elements"] > 0 and r["gradient_norms"]["adapter"] > 0 for r in b["trace"]))
                moved = {n: {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in i.items()} for n, i in resumed[1].items()}
                raw = split / f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
                expected = _tree_fingerprint(torch.load(split / "selected-probabilities.pt", weights_only=True))
                for _ in range(3):
                    model = resumed[4].load_selected(split / "primary.pt", moved, raw, device=DEVICE)
                    self.assertEqual(_tree_fingerprint(resumed[4].validation_snapshot(model, moved)[1]), expected)
                before = hashes(split)
                self.assertEqual(run(bound(root, manifest, meta, split.name, method=method, spec=spec_arg), split,
                    resume=True)["updates_this_call"], 0)
                self.assertEqual(before, hashes(split))
                wrong = bound(root, manifest, meta, split.name, method=method, spec=spec_arg, seed=8129)
                with self.assertRaisesRegex(ValueError, "Resume identity"): run(wrong, split, resume=True)
                self.assertEqual(before, hashes(split))
                model = load_model_checkpoint(split / "primary.pt", device="cpu")
                model.training_config[KEY]["sampling_seed"] += 1
                save_model_checkpoint(model, root / (method + "-bad.pt"))
                with self.assertRaisesRegex(ValueError, "Selected target observation"):
                    resumed[4].load_selected(root / (method + "-bad.pt"), moved, raw, device=DEVICE)
                self.assertNotEqual(_tree_fingerprint(a["model"]), _tree_fingerprint(checkpoints(root / (method + "-split"))["best_model"]))

    def test_profile_record_metadata_and_input_negatives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, meta = setup(root)
            opts, items, _, _, runtime, _ = bound(root, manifest, meta, "unused")
            for key, value in (("profile", "full"), ("optimizer_steps", 1500), ("support_seed", 2048)):
                bad = copy.deepcopy(opts); bad[key] = value
                with self.assertRaises(ValueError): runtime.validate(bad)
            bad = copy.deepcopy(opts); bad[KEY]["records"][1]["selected_positions"].reverse()
            with self.assertRaises(ValueError): runtime.validate(bad)
            bad = copy.deepcopy(items); bad["xuzhou"]["Q"] = bad["xuzhou"]["Q"].to_dense().add(1)
            with self.assertRaises(ValueError): runtime.validate_inputs(bad)
            bad = copy.deepcopy(items); bad["xuzhou"]["query_labels"] = torch.zeros(2)
            with self.assertRaises(ValueError): runtime.validate_inputs(bad)
            for cursor in (-1, 12, True):
                with self.assertRaises(ValueError): runtime.view(cursor, "xuzhou", items["xuzhou"])
            for seed in (-1, True):
                with self.assertRaises(ValueError): bound(root, manifest, meta, "unused", seed=seed)
            values = asdict(meta); values["wavelengths"] = tuple(reversed(meta.wavelengths))
            with self.assertRaises(ValueError): bound(root, manifest, BandMetadata(**values), "unused")
            bad = copy.deepcopy(items); bad["xuzhou"]["fwhm"] = torch.full((7,), 2.)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                freeze_settings({k: v for k, v in opts.items() if k != KEY}, bad, metadata(bad["xuzhou"]), preset="full", sampling_seed=1)
            self.assertFalse((root / "unused").exists())

    def test_cli_metadata_source_guard_two_ten_zero_and_rejected_full(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _, meta = setup(root)
            side = root / "metadata"; side.mkdir(); path = side / "bands.json"
            write_json(path, asdict(meta))
            out = root / "cli"
            command = [sys.executable, "scripts/train_g4_observation_target_short.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out),
                "--metadata", str(path), "--metadata-sha256", file_hash(path), "--target", "xuzhou", "--method", METHODS[0],
                "--model-seed", "64", "--support-seed", "512", "--per-class", "1", "--preset", "alternate75",
                "--sampling-seed", "8128", "--device", DEVICE]
            def invoke(extra): return json.loads(subprocess.check_output(command + extra, text=True))
            self.assertEqual(invoke(["--stop-after", "2"])["updates_this_call"], 2)
            self.assertEqual(invoke(["--resume"])["updates_this_call"], 10)
            before = hashes(out)
            self.assertEqual(invoke(["--resume"])["updates_this_call"], 0)
            self.assertEqual(before, hashes(out))
            for extra in (["--full"], ["--stop-after", "1500"], ["--mode", "x_residual"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)
            wrong = command.copy(); wrong[wrong.index("--expected-commit") + 1] = "0"*40
            self.assertNotEqual(subprocess.run(wrong + ["--resume"], capture_output=True).returncode, 0)
            args = dict(target="xuzhou", method=METHODS[0], model_seed=64, support_seed=512, per_class=1,
                metadata=meta, preset="full", sampling_seed=8128, metadata_files={str(path): file_hash(path)})
            with self.assertRaises(ValueError): prepare(manifest, file_hash(manifest), side / "out", **args)
            *_, guard = prepare(manifest, file_hash(manifest), root / "guard", **args)
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError): guard()


if __name__ == "__main__":
    unittest.main()
