"""Existing M6 factory/BN integration; synthetic learning only, no real effect."""
import copy
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

from g4_interaction_target import (CALIBRATION_KEY, CONDITIONS, KEY, PREFIX, calibrate, configuration,
    forward_observation, initialize, load_calibrated, verify_calibrated, verify_initialization_binding)
from joint_baseline import forward as old_forward
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.checkpoint_io import save_model_checkpoint
from model.research_config import ResearchConfig, digest
from none_full_transfer import initialize_none_full_target
from research_runtime import code_snapshot, write_json
from target_bn_calibration import CONTEXT, GRAPH_POLICY, POLICY as BN_POLICY, calibrate_target
from target_initialization import METHODS, state_id
from target_training import loss_terms
from test_g4_coverage_target import fixture, obs
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("INTERACTION_TARGET_DEVICE", "cpu")
torch.set_num_threads(1)
COUNTS = {"none": 0, "add": 33280, "gated": 58498, "cnn_to_graph": 29249, "graph_to_cnn": 29249, "global": 58498}


def create(condition="gated", method=METHODS[0], spec=None, device=DEVICE):
    return initialize("xuzhou", method, condition, model_seed=64, head_seed=1064, source_spec=spec, device=device)


def data_for(item, device=DEVICE):
    # Use the old observation-only field extractor with an explicit device.
    return obs(item, device=device)


class InteractionTargetTests(unittest.TestCase):
    def test_all_twelve_roles_shared_state_rng_fresh_heads_and_parameter_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            _, spec = source_fixture(Path(temp) / "source")
            for method in METHODS[:2]:
                source = spec if method == METHODS[1] else None
                for condition in CONDITIONS:
                    torch.manual_seed(918)
                    base, record = initialize_none_full_target("xuzhou", method, model_seed=64, head_seed=1064,
                        source_spec=source, device=DEVICE)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(918)
                    model, binding = create(condition, method, source)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertEqual(binding["base_initialization"], record)
                    self.assertEqual(binding["base_state_sha256"], state_id(base))
                    for name, tensor in base.state_dict().items():
                        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
                    self.assertEqual(binding["message_parameter_count"], COUNTS[condition])
                    self.assertTrue(all(p.requires_grad for p in model.parameters()))
                    self.assertEqual(tuple(model.heads), ("xuzhou",))
                    self.assertIsNone(model.backbone.edge_updater); self.assertIsNone(model.backbone.coverage_modulation)
                    self.assertEqual(configuration(condition)["interaction_strength"], .1)
                    verify_initialization_binding(model, binding)
            for seed in (512, 2048):
                model, b = initialize("paviaU", METHODS[0], "add", model_seed=seed, head_seed=seed + 1000)
                self.assertEqual(b["head_seed"], seed + 1000)
                self.assertEqual(tuple(model.heads), ("paviaU",))

    def test_off_exact_original_forward_and_ordered_bn_for_both_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item)
            _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                source = spec if method == METHODS[1] else None
                base, _ = initialize_none_full_target("xuzhou", method, model_seed=64, head_seed=1064, source_spec=source, device=DEVICE)
                base.eval(); model, binding = create("none", method, source)
                with torch.no_grad(), OrderedValidation():
                    expected = old_forward(base, "xuzhou", data)
                    actual = forward_observation(model, data, expected_binding=binding)
                for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
                with OrderedValidation():
                    old, old_receipt = calibrate_target(base, data, mode=BN_POLICY, context=CONTEXT,
                        graph_policy=GRAPH_POLICY, source_state_sha256=state_id(base))
                new, _ = calibrate(model, data, expected_binding=binding, source_state_sha256=state_id(model))
                self.assertEqual(state_id(old), state_id(new))
                self.assertEqual(old_receipt, new.training_config["target_bn_calibration"])

    def test_actual_directions_alignment_calls_and_unmodified_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            _, item, _ = fixture(Path(temp)); data = data_for(item)
            from target_bn_calibration import observation_id
            before = observation_id(data)
            for condition in CONDITIONS[1:]:
                model, binding = create(condition); alignment = model.backbone.alignment
                with patch.object(alignment, "to_regions", wraps=alignment.to_regions) as regions, patch.object(alignment, "to_pixels", wraps=alignment.to_pixels) as pixels:
                    with torch.no_grad(), OrderedValidation():
                        output = forward_observation(model, data, expected_binding=binding, diagnostics=True)
                self.assertEqual(regions.call_count, 0 if condition == "graph_to_cnn" else 1)
                self.assertEqual(pixels.call_count, 0 if condition == "cnn_to_graph" else 1)
                self.assertEqual(alignment.mode, "global" if condition == "global" else "region")
                self.assertEqual(len(output), 4)
                self.assertTrue(all(torch.isfinite(v).all() for v in output[:3]))
                info = output[3]["interaction"]
                if condition != "graph_to_cnn": self.assertIn("cnn_to_graph_gate", info)
                if condition != "cnn_to_graph": self.assertIn("graph_to_cnn_gate", info)
                self.assertEqual(before, observation_id(data))

    def test_enabled_conditions_three_synthetic_updates_gradients_and_enrollment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item)
            _, spec = source_fixture(root / "source")
            from joint_baseline import load_config
            config = load_config("configs/experiments/base97_joint.json")
            for method in METHODS[:2]:
                for condition in CONDITIONS[1:]:
                    model, binding = create(condition, method, spec if method == METHODS[1] else None); model.train()
                    before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(PREFIX)}
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        outputs = forward_observation(model, data, expected_binding=binding)
                        selected = tuple(v[item["train_indices"].to(DEVICE)] for v in outputs)
                        loss_terms(selected, item["train_labels"].to(DEVICE), config)["total"].backward()
                        params = [p for n, p in model.named_parameters() if n.startswith(PREFIX)]
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0.)
                        optimizer.step()
                    self.assertGreater(sum(int(torch.count_nonzero(p.detach() - before[n])) for n, p in model.named_parameters() if n in before), 0)

    def test_all_roles_immutable_population_bn_and_three_exact_checkpoint_reloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item)
            _, spec = source_fixture(root / "source")
            from target_bn_calibration import observation_id
            observation_before = observation_id(data)
            for method in METHODS[:2]:
                for condition in CONDITIONS:
                    model, binding = create(condition, method, spec if method == METHODS[1] else None)
                    source = state_id(model)
                    candidate, receipt = calibrate(model, data, expected_binding=binding, source_state_sha256=source)
                    self.assertEqual(source, state_id(model)); self.assertEqual(observation_before, observation_id(data))
                    bn = candidate.training_config["target_bn_calibration"]
                    self.assertEqual((len(bn["layers"]), len(bn["calibrated_buffers"])), (5, 10))
                    for name, value in model.state_dict().items():
                        if name not in bn["calibrated_buffers"]:
                            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
                    path = root / (condition + method + ".pt"); save_model_checkpoint(candidate, path)
                    with torch.no_grad(), OrderedValidation(): expected = forward_observation(candidate, data, expected_binding=binding)
                    for _ in range(3):
                        loaded = load_calibrated(path, file_hash(path), file_hash(str(path) + ".json"), data,
                            expected_binding=binding, source_state_sha256=source, device=DEVICE)
                        with torch.no_grad(), OrderedValidation(): actual = forward_observation(loaded, data, expected_binding=binding)
                        for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(receipt, verify_calibrated(candidate, data, expected_binding=binding, source_state_sha256=source))
                    with self.assertRaises(ValueError): load_calibrated(path, "0" * 64, file_hash(str(path) + ".json"), data,
                        expected_binding=binding, source_state_sha256=source, device=DEVICE)

    def test_independent_cpu_double_moments_reference_and_attribute_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            _, item, _ = fixture(Path(temp)); data = data_for(item, "cpu")
            model, binding = create(device="cpu"); oracle = copy.deepcopy(model); moments = {}
            for name, layer in oracle.named_modules():
                if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    def independent(self, x, name=name):
                        axes = (0, *range(2, x.ndim)); value = x.double()
                        mean = value.mean(dim=axes); shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
                        var = (value - mean.reshape(shape)).square().mean(dim=axes)
                        moments[name] = (mean, var)
                        return F.batch_norm(x, mean.to(x.dtype), var.to(x.dtype), self.weight, self.bias, False, 0., self.eps)
                    layer.forward = types.MethodType(independent, layer)
            with torch.no_grad(), OrderedValidation(): reference = forward_observation(oracle, data, expected_binding=binding)
            candidate, _ = calibrate(model, data, expected_binding=binding, source_state_sha256=state_id(model))
            with torch.no_grad(), OrderedValidation(): actual = forward_observation(candidate, data, expected_binding=binding)
            for a, b in zip(reference, actual): torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            for name, (mean, var) in moments.items():
                layer = dict(candidate.named_modules())[name]
                torch.testing.assert_close(layer.running_mean.double(), mean, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(layer.running_var.double(), var, rtol=1e-5, atol=1e-6)
            for field, value in (("strength", .2), ("chunk_size", 2)):
                bad = copy.deepcopy(model); setattr(bad.backbone.interaction.cnn_to_graph, field, value)
                with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            bad = copy.deepcopy(model); bad.backbone.alignment.mode = "global"
            with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            with self.assertRaises(ValueError): calibrate(model, {**data, "labels": torch.zeros(1)}, expected_binding=binding, source_state_sha256=state_id(model))
            with self.assertRaises(ValueError): verify_calibrated(candidate, data, expected_binding=binding, source_state_sha256="0" * 64)

    def test_cpu_cli_pins_output_and_nonexecuting_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); out = root / "cli"
            command = [sys.executable, "scripts/audit_g4_interaction_target_initialization.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out), "--target", "xuzhou",
                "--method", METHODS[0], "--condition", "gated", "--model-seed", "64", "--support-seed", "2048", "--per-class", "5"]
            result = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(result["message_parameters"], 58498)
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual((receipt["optimizer_updates"], receipt["model_forwards"]), (0, 0))
            self.assertFalse(receipt["bn_calibration_executed"]); self.assertFalse(receipt["gpu_context_initialized"])
            before = {p.name: file_hash(p) for p in out.iterdir()}
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, {p.name: file_hash(p) for p in out.iterdir()})
            for extra in (["--query"], ["--full"], ["--condition", "joint"], ["--model-seed", "128"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)
            for kwargs in ({"model_seed": 64, "head_seed": 64}, {"model_seed": True, "head_seed": 1001}):
                with self.assertRaises(ValueError): initialize("xuzhou", METHODS[0], "gated", **kwargs)


if __name__ == "__main__":
    unittest.main()
