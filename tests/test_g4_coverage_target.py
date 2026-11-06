"""E1 target factory, explicit view forward and immutable population calibration."""
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from g4_coverage_target import (CALIBRATION_KEY, CONTROLS, KEY, PREFIX, calibrate, forward_observation,
    initialize, load_calibrated, verify_calibrated, verify_initialization_binding)
from g4_support_target import prepare
from joint_baseline import forward as old_forward
from loadData.dataset_registry import file_hash
from loadData.observation_sampling import BandMetadata, sample
from joint_observation import _stub
from m4_ordered_source import OrderedValidation
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from none_full_transfer import initialize_none_full_target
from research_runtime import code_snapshot, write_json
from target_bn_calibration import OBS_KEYS, POLICY, CONTEXT, GRAPH_POLICY, calibrate_target
from target_initialization import METHODS, state_id
from target_training import loss_terms
from test_g4_support_context import prepared
from test_g4_observation_target import metadata as make_metadata
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("COVERAGE_TARGET_DEVICE", "cpu")
torch.set_num_threads(1)


def fixture(root):
    _, _, catalog, _ = prepared(root)
    manifest = catalog / "manifest.json"
    opts, items, *_ = prepare(manifest, file_hash(manifest), root / "unused", target="xuzhou",
        method=METHODS[0], mode="none", model_seed=64, support_seed=512, per_class=1)
    item = items["xuzhou"]
    return manifest, item, make_metadata(item)


def create(meta, condition="continuous", method=METHODS[0], spec=None, device=DEVICE):
    return initialize("xuzhou", method, condition, metadata=meta, model_seed=64,
        head_seed=1064, source_spec=spec, device=device)


def obs(item, device=DEVICE):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items() if k in OBS_KEYS}


def view(observation, meta, config):
    selected, record = sample(_stub(meta, "synthetic-target"), meta, config,
                              __import__("numpy").random.default_rng(8128))
    positions = record["selected_positions"]
    value = observation["net_input"]
    return {**observation, "net_input": torch.cat((value[..., positions], value[..., -2:]), -1),
        "wavelengths": observation["wavelengths"][positions],
        "fwhm": None if observation["fwhm"] is None else observation["fwhm"][positions]}, selected.bands


class CoverageTargetTests(unittest.TestCase):
    def test_all_controls_shared_fresh_state_rng_capacity_and_source_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, meta = fixture(root)
            _, source = source_fixture(root / "source")
            for method in METHODS[:2]:
                spec = source if method == METHODS[1] else None
                before = torch.get_rng_state().clone()
                base, _ = initialize_none_full_target("xuzhou", method, model_seed=64,
                    head_seed=1064, source_spec=spec, device=DEVICE)
                expected_rng = torch.get_rng_state().clone()
                for condition in CONTROLS:
                    torch.set_rng_state(before)
                    model, binding = create(meta, condition, method, spec)
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                    self.assertEqual(binding["base_initial_state_sha256"], state_id(base))
                    self.assertEqual(binding["coverage_parameter_count"], 0 if condition == "none" else 194)
                    self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in base.state_dict().items()))
                    self.assertEqual(set(model.heads), {"xuzhou"})
                    self.assertFalse(any(m.training for m in model.modules()))
                    verify_initialization_binding(model, binding)
                for seed, head in ((True, 1064), (64, 1065)):
                    with self.assertRaises(ValueError):
                        initialize("xuzhou", method, "continuous", metadata=meta, model_seed=seed,
                            head_seed=head, source_spec=spec)
            with self.assertRaises(ValueError): create(meta, method=METHODS[1])
            with self.assertRaises(ValueError): create(meta, method=METHODS[2], spec=source)

    def test_zero_initialization_actual_views_metadata_and_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            _, item, meta = fixture(Path(temp)); data = obs(item)
            base, _ = create(meta, "none")
            configs = ({"mode": "none"}, {"mode": "random_subset", "keep_fraction": .75},
                {"mode": "contiguous_gap", "lower": 600., "upper": 800., "unit": "nm"},
                {"mode": "range_crop", "lower": 450., "upper": 700., "unit": "nm"})
            for condition in CONTROLS[1:]:
                model, _ = create(meta, condition)
                for config in configs:
                    current, selected = view(data, meta, config)
                    with torch.no_grad(), OrderedValidation():
                        expected = forward_observation(base, current, selected)
                        actual = forward_observation(model, current, selected, diagnostics=True)
                    for x, y in zip(expected, actual[:3]): torch.testing.assert_close(x, y, rtol=0, atol=0)
                    info = actual[3]["coverage"]
                    self.assertEqual(info["metadata_hash"], selected.fingerprint)
                    self.assertEqual(info["features"].shape, (64, 9))
                    torch.testing.assert_close(current["net_input"][..., -2:], data["net_input"][..., -2:], rtol=0, atol=0)
                    with self.assertRaises(ValueError): forward_observation(model, current, replace(selected, wavelength_quality="verified"))
            # Index-only is an explicit fallback, even with nonzero biases.
            index = BandMetadata("index-fixture", tuple(range(7)), tuple(torch.linspace(0, 1, 7).tolist()),
                "normalized", "index_only", (True,)*7)
            indexed = {**data, "wavelengths": torch.tensor(index.wavelengths, dtype=data["net_input"].dtype, device=DEVICE)}
            model, _ = create(index)
            with torch.no_grad(), OrderedValidation():
                model.backbone.coverage_modulation.generator[2].bias.fill_(.7)
                actual = forward_observation(model, indexed, index, diagnostics=True)
            self.assertEqual(actual[3]["coverage"]["fallback_reason"], "index_only_metadata")
            self.assertEqual(actual[3]["coverage"]["usable_anchors"], 0)

    def test_all_equal_capacity_conditions_have_gradients_and_actual_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, meta = fixture(root); data = obs(item)
            options, *_ = prepare(root / "contexts/manifest.json", file_hash(root / "contexts/manifest.json"),
                root / "unused", target="xuzhou", method=METHODS[0], mode="none", model_seed=64,
                support_seed=512, per_class=1, device=DEVICE)
            from model.research_config import ResearchConfig
            config = ResearchConfig.from_dict(options["config"])
            for condition in CONTROLS[1:]:
                model, binding = create(meta, condition); model.train()
                before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(PREFIX)}
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    outputs = forward_observation(model, data, meta)
                    selected = tuple(x[item["train_indices"].to(DEVICE)] for x in outputs)
                    loss = loss_terms(selected, item["train_labels"].to(DEVICE), config)["total"]
                    loss.backward(); optimizer.step()
                params = [p for n, p in model.named_parameters() if n.startswith(PREFIX)]
                self.assertEqual(sum(p.numel() for p in params), 194)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0)
                self.assertGreater(sum(int(torch.count_nonzero(p.detach()-before[n])) for n, p in model.named_parameters() if n in before), 0)
                model.eval(); source = state_id(model)
                candidate, receipt = calibrate(model, data, meta, expected_binding=binding, source_state_sha256=source)
                verify_calibrated(candidate, data, meta, expected_binding=binding, source_state_sha256=source)
                self.assertEqual(state_id(model), source)
                self.assertEqual(receipt["optimizer_updates"], 0)

    def test_population_oracle_off_equivalence_and_protected_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            _, item, meta = fixture(Path(temp)); data = obs(item)
            model, binding = create(meta, "none"); source = state_id(model)
            with OrderedValidation():
                old, _ = calibrate_target(model, data, mode=POLICY, context=CONTEXT,
                    graph_policy=GRAPH_POLICY, source_state_sha256=source)
            new, receipt = calibrate(model, data, meta, expected_binding=binding, source_state_sha256=source)
            self.assertEqual(state_id(old), state_id(new)); self.assertEqual(len(receipt["layers"]), 5)
            for row in receipt["layers"]:
                self.assertEqual(row["samples_per_channel"], data["A"].shape[0] if row["population_units"] == "regions" else 400)
            # Independent CPUFP64 functional population oracle; no checkpoint
            # dtype roundtrip claim for the legacy FP32 checkpoint constructor.
            model, binding = create(meta, device="cpu"); model.double()
            double = {k: v.cpu().double() if torch.is_tensor(v) else v for k, v in data.items()}
            oracle = copy.deepcopy(model)
            for module in oracle.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    def independent(self, x): return F.batch_norm(x, None, None, self.weight, self.bias, True, 0., self.eps)
                    module.forward = types.MethodType(independent, module)
            with torch.no_grad(): expected = forward_observation(oracle, double, meta)
            candidate, receipt = calibrate(model, double, meta, expected_binding=binding, source_state_sha256=state_id(model))
            with torch.no_grad(): actual = forward_observation(candidate, double, meta)
            for x, y in zip(expected, actual): torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-11)
            for name, value in model.state_dict().items():
                if name not in receipt["calibrated_buffers"]:
                    torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)

    def test_checkpoint_three_reload_and_binding_source_metadata_label_negatives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, meta = fixture(root); data = obs(item)
            for condition in CONTROLS:
                model, binding = create(meta, condition); source = state_id(model)
                candidate, receipt = calibrate(model, data, meta, expected_binding=binding, source_state_sha256=source)
                path = root / (condition + ".pt"); save_model_checkpoint(candidate, path)
                with torch.no_grad(), OrderedValidation(): expected = forward_observation(candidate, data, meta)
                for _ in range(3):
                    loaded = load_calibrated(path, file_hash(path), file_hash(str(path)+".json"), data, meta,
                        expected_binding=binding, source_state_sha256=source, device=DEVICE)
                    with torch.no_grad(), OrderedValidation(): actual = forward_observation(loaded, data, meta)
                    for x, y in zip(expected, actual): torch.testing.assert_close(x, y, rtol=0, atol=0)
                wrong = copy.deepcopy(binding); wrong["metadata"]["source_id"] += "-wrong"
                with self.assertRaises(ValueError): verify_calibrated(candidate, data, meta, expected_binding=wrong, source_state_sha256=source)
                with self.assertRaises(ValueError): verify_calibrated(candidate, data, meta, expected_binding=binding, source_state_sha256="a"*64)
                with self.assertRaises(ValueError): calibrate(model, {**data, "labels": torch.zeros(1)}, meta,
                    expected_binding=binding, source_state_sha256=source)
                current, selected = view(data, meta, {"mode": "random_subset", "keep_fraction": .75})
                with self.assertRaises(ValueError): calibrate(model, current, selected, expected_binding=binding, source_state_sha256=source)
                with self.assertRaises(ValueError): load_calibrated(path, "0"*64, file_hash(str(path)+".json"), data, meta,
                    expected_binding=binding, source_state_sha256=source)
                if condition != "none":
                    candidate.backbone.coverage_modulation.strength = .2
                    with self.assertRaises(ValueError): verify_calibrated(candidate, data, meta, expected_binding=binding, source_state_sha256=source)
            # Old frozen calibrator still refuses an E1-enabled target.
            model, _ = create(meta)
            with self.assertRaisesRegex(ValueError, "Accepted target initialization"):
                calibrate_target(model, data, mode=POLICY, context=CONTEXT, graph_policy=GRAPH_POLICY, source_state_sha256=state_id(model))

    def test_cli_is_cpu_zero_forward_and_has_no_training_or_query_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, meta = fixture(root)
            folder = root / "metadata"; folder.mkdir(); path = folder / "bands.json"; write_json(path, asdict(meta))
            out = root / "cli"
            command = [sys.executable, "scripts/audit_g4_coverage_target_initialization.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--metadata", str(path),
                "--metadata-sha256", file_hash(path), "--out", str(out), "--target", "xuzhou", "--method", METHODS[0],
                "--condition", "continuous", "--model-seed", "64", "--support-seed", "512", "--per-class", "1"]
            self.assertEqual(json.loads(subprocess.check_output(command, text=True))["coverage_parameters"], 194)
            receipt = json.loads((out/"receipt.json").read_text())
            self.assertFalse(receipt["bn_calibration_executed"] or receipt["query_labels_accessed"] or receipt["gpu_context_initialized"])
            self.assertEqual((receipt["optimizer_updates"], receipt["model_forwards"]), (0, 0))
            for extra in ([], ["--full"], ["--query"], ["--device", "cuda"]):
                self.assertNotEqual(subprocess.run(command+extra, capture_output=True).returncode, 0)
            wrong = command.copy(); wrong[wrong.index("--expected-commit")+1] = "0"*40
            self.assertNotEqual(subprocess.run(wrong, capture_output=True).returncode, 0)
            # Factory itself never executes a network or optimizer.
            with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("forward forbidden")), patch("torch.optim.Adam", side_effect=AssertionError("optimizer forbidden")):
                create(meta, device="cpu")


if __name__ == "__main__":
    unittest.main()
