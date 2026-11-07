"""Position-fusion target integration: synthetic-only, original protocols unchanged."""
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

from g4_fusion_target import initialize as fixed_initialize, calibrate as fixed_calibrate
from g4_position_fusion_target import (CALIBRATION_KEY, CONDITIONS, KEY, PREFIX, calibrate, configuration,
    forward_observation, initialize, load_calibrated, loss_terms, verify_calibrated, verify_initialization_binding)
from joint_baseline import forward as old_forward
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from research_runtime import code_snapshot
from target_bn_calibration import observation_id
from target_initialization import METHODS, state_id
from target_training import loss_terms as old_loss_terms
from test_g4_coverage_target import fixture, obs
from test_g4_fusion_target import config_for
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("POSITION_TARGET_DEVICE", "cpu")
torch.set_num_threads(1)


def create(condition="position_gate", method=METHODS[0], spec=None, device=DEVICE):
    return initialize("xuzhou", method, condition, model_seed=64, head_seed=1064, source_spec=spec, device=device)


def data_for(item, device=DEVICE):
    data = obs(item, device=device); n = data["A"].shape[0]
    data["A"] = 2 * torch.eye(n, device=device) + .25 * torch.eye(n, device=device).roll(1, dims=1)
    return data


class PositionFusionTargetTests(unittest.TestCase):
    def test_four_roles_original_states_heads_domains_rng_permissions_and_129_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            _, spec = source_fixture(Path(temp) / "source")
            for method in METHODS[:2]:
                source = spec if method == METHODS[1] else None
                for condition in CONDITIONS:
                    torch.manual_seed(918)
                    base, record = fixed_initialize("xuzhou", method, "default", model_seed=64, head_seed=1064, source_spec=source, device=DEVICE)
                    rng = torch.get_rng_state().clone()
                    cuda = torch.cuda.get_rng_state(0).clone() if DEVICE == "cuda" else None
                    model, binding = create(condition, method, source)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    if cuda is not None: self.assertTrue(torch.equal(cuda, torch.cuda.get_rng_state(0)))
                    self.assertEqual(binding["base_binding"], record); self.assertTrue(all(p.requires_grad for p in model.parameters()))
                    self.assertTrue(all(torch.is_tensor(v) for v in model.state_dict().values()))
                    for n, v in base.state_dict().items(): torch.testing.assert_close(v, model.state_dict()[n], rtol=0, atol=0)
                    extra = set(model.state_dict()) - set(base.state_dict())
                    self.assertEqual(extra, {PREFIX + n for n in ("gate_weight", "gate_bias", "protocol_version")} if condition == "position_gate" else set())
                    self.assertEqual(binding["added_parameter_count"], 129 if condition == "position_gate" else 0)
                    self.assertEqual(binding["additional_state_bytes"], 8 if condition == "position_gate" else 0)
                    self.assertEqual(tuple(model.heads), ("xuzhou",)); verify_initialization_binding(model, binding)
            for seed in (512, 2048):
                _, binding = initialize("paviaU", METHODS[0], "position_gate", model_seed=seed, head_seed=seed + 1000)
                self.assertEqual(binding["head_seed"], seed + 1000)

    def test_fixed_original214_forward_and_ordered_bn_exact_both_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item); _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                source = spec if method == METHODS[1] else None
                base, bb = fixed_initialize("xuzhou", method, "default", model_seed=64, head_seed=1064, source_spec=source, device=DEVICE)
                model, binding = create("fixed", method, source)
                with torch.no_grad(), OrderedValidation():
                    a = old_forward(base, "xuzhou", data); b = forward_observation(model, data, expected_binding=binding)
                for x, y in zip(a, b): torch.testing.assert_close(x, y, rtol=0, atol=0)
                first, _ = fixed_calibrate(base, data, expected_binding=bb, source_state_sha256=state_id(base))
                second, _ = calibrate(model, data, expected_binding=binding, source_state_sha256=state_id(model))
                self.assertEqual(state_id(first), state_id(second))
                self.assertEqual(first.training_config["target_bn_calibration"], second.training_config["target_bn_calibration"])

    def test_enabled_only_final_fusion_manual_features_logits_and_original13_losses(self):
        with tempfile.TemporaryDirectory() as temp:
            _, item, _ = fixture(Path(temp)); data = data_for(item)
            base, bb = create("fixed"); model, binding = create()
            with torch.no_grad():
                model.backbone.position_fusion.gate_weight.fill_(.01)
                model.backbone.position_fusion.gate_bias.fill_(-.4)
            with torch.no_grad(), OrderedValidation():
                before = base.backbone.forward_features(data["net_input"], Q=data["Q"], A=data["A"], dataset_name="xuzhou",
                    wavelengths=data["wavelengths"], fwhm=data["fwhm"], superpixel_scale=data["superpixel_scale"])
                actual = model.backbone.forward_features(data["net_input"], Q=data["Q"], A=data["A"], dataset_name="xuzhou",
                    wavelengths=data["wavelengths"], fwhm=data["fwhm"], superpixel_scale=data["superpixel_scale"])
                c, g = actual[2], actual[1]; operator = model.backbone.position_fusion
                z = torch.sigmoid(torch.cat((c, g), dim=1) @ operator.gate_weight.T + operator.gate_bias)
                torch.testing.assert_close(actual[0], z * c + (1 - z) * g, rtol=1e-6, atol=1e-7)
                for x, y in zip(before[1:], actual[1:]): torch.testing.assert_close(x, y, rtol=0, atol=0)
                probs = forward_observation(model, data, expected_binding=binding)
                logits = forward_observation(model, data, expected_binding=binding, logits=True)
                old = forward_observation(base, data, expected_binding=bb)
                for p, raw in zip(probs, logits): torch.testing.assert_close(p, raw.softmax(-1), rtol=1e-5, atol=1e-6)
                for x, y in zip(probs[1:], old[1:]): torch.testing.assert_close(x, y, rtol=0, atol=0)
                self.assertGreater(float((probs[0] - old[0]).abs().max()), 0.)
            labels = torch.arange(9, device=DEVICE)
            selected = tuple(v[:9] for v in probs); config = config_for("default", DEVICE)
            terms = loss_terms(model, selected, labels, config, expected_binding=binding); ref = old_loss_terms(selected, labels, config)
            self.assertEqual(len(terms), 13)
            for n in terms: torch.testing.assert_close(terms[n], ref[n], rtol=0, atol=0)
            with self.assertRaises(ValueError): loss_terms(model, selected, labels, config_for("aux_off", DEVICE), expected_binding=binding)

    def test_four_roles_three_synthetic_updates_bn_protection_12_reloads_and_generic_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item); _, spec = source_fixture(root / "source")
            identity = observation_id(data)
            for method in METHODS[:2]:
                for condition in CONDITIONS:
                    model, binding = create(condition, method, spec if method == METHODS[1] else None)
                    before = {n: p.detach().clone() for n, p in model.named_parameters()}
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001); config = config_for("default", DEVICE)
                    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids))); self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                    prefixes = ("backbone.CNN_Branch.", "backbone.HGNN_Branch.", PREFIX, "heads.", "hgnn_heads.", "cnn_heads.")
                    model.train()
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True); out = forward_observation(model, data, expected_binding=binding)
                        selected = tuple(v[item["train_indices"].to(DEVICE)] for v in out)
                        terms = loss_terms(model, selected, item["train_labels"].to(DEVICE), config, expected_binding=binding)
                        self.assertTrue(torch.isfinite(terms["total"])); terms["total"].backward()
                        for prefix in prefixes:
                            params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
                            if params:
                                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                                self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0.)
                        optimizer.step()
                    for prefix in prefixes:
                        names = [n for n in before if n.startswith(prefix)]
                        if names: self.assertGreater(sum(int(torch.count_nonzero(model.state_dict()[n] - before[n])) for n in names), 0)
                    model.eval(); source = state_id(model)
                    candidate, receipt = calibrate(model, data, expected_binding=binding, source_state_sha256=source)
                    self.assertEqual(source, state_id(model)); self.assertEqual(identity, observation_id(data))
                    legacy = candidate.training_config["target_bn_calibration"]
                    self.assertEqual((len(legacy["layers"]), len(legacy["calibrated_buffers"])), (5, 10))
                    self.assertEqual(receipt["legacy_receipt"], legacy)
                    for n, v in model.state_dict().items():
                        if n not in legacy["calibrated_buffers"]: torch.testing.assert_close(v, candidate.state_dict()[n], rtol=0, atol=0)
                    path = root / (condition + method + ".pt"); save_model_checkpoint(candidate, path)
                    if condition == "position_gate":
                        with self.assertRaisesRegex(RuntimeError, "position_fusion"): load_model_checkpoint(path, device=DEVICE)
                    with torch.no_grad(), OrderedValidation(): expected = forward_observation(candidate, data, expected_binding=binding)
                    for _ in range(3):
                        loaded = load_calibrated(path, file_hash(path), file_hash(str(path) + ".json"), data,
                            expected_binding=binding, source_state_sha256=source, device=DEVICE)
                        with torch.no_grad(), OrderedValidation(): actual = forward_observation(loaded, data, expected_binding=binding)
                        for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
                    verify_calibrated(candidate, data, expected_binding=binding, source_state_sha256=source)

    def test_independent_cpu_fp64_population_moments_preserves_gate_and_original_bn(self):
        with tempfile.TemporaryDirectory() as temp:
            _, item, _ = fixture(Path(temp)); data = data_for(item, "cpu"); model, binding = create(device="cpu")
            reference = copy.deepcopy(model); moments = {}
            for n, layer in reference.named_modules():
                if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    def independent(self, x, n=n):
                        axes = (0, *range(2, x.ndim)); d = x.double(); shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
                        mean = d.mean(dim=axes); var = (d - mean.reshape(shape)).square().mean(dim=axes); moments[n] = (mean, var)
                        return F.batch_norm(x, mean.to(x.dtype), var.to(x.dtype), self.weight, self.bias, False, 0., self.eps)
                    layer.forward = types.MethodType(independent, layer)
            with torch.no_grad(), OrderedValidation(): expected = forward_observation(reference, data, expected_binding=binding)
            candidate, _ = calibrate(model, data, expected_binding=binding, source_state_sha256=state_id(model))
            with torch.no_grad(), OrderedValidation(): actual = forward_observation(candidate, data, expected_binding=binding)
            for x, y in zip(expected, actual): torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
            for n, (mean, var) in moments.items():
                layer = dict(candidate.named_modules())[n]
                torch.testing.assert_close(layer.running_mean.double(), mean, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(layer.running_var.double(), var, rtol=1e-5, atol=1e-6)

    def test_bound_method_version_permissions_labels_source_and_checkpoint_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item); _, spec = source_fixture(root / "source")
            model, binding = create()
            for change in ("version", "method", "config", "permission"):
                bad = copy.deepcopy(model)
                if change == "version": bad.backbone.position_fusion.protocol_version.fill_(0)
                if change == "method": bad.backbone.position_fusion.forward = types.MethodType(lambda self, c, g: c, bad.backbone.position_fusion)
                if change == "config": bad.backbone.position_fusion.alph = .35
                if change == "permission": bad.backbone.position_fusion.gate_weight.requires_grad_(False)
                with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            with self.assertRaises(ValueError): create(method=METHODS[1], spec={**spec, "source_weight_sha256": "0" * 64})
            with self.assertRaises(ValueError): forward_observation(model, {**data, "labels": torch.zeros(1)}, expected_binding=binding)
            candidate, _ = calibrate(model, data, expected_binding=binding, source_state_sha256=state_id(model))
            path = root / "selected.pt"; save_model_checkpoint(candidate, path)
            with patch("torch.load", side_effect=AssertionError("Pair pins before tensor decode")):
                with self.assertRaises(ValueError): load_calibrated(path, "0" * 64, file_hash(str(path) + ".json"), data,
                    expected_binding=binding, source_state_sha256=state_id(model), device=DEVICE)
            bad = copy.deepcopy(candidate); bad.training_config[CALIBRATION_KEY]["legacy_receipt"]["target"] = "wrong"
            with self.assertRaises(ValueError): verify_calibrated(bad, data, expected_binding=binding, source_state_sha256=state_id(model))

    def test_actual_cpu_zero_forward_cli_and_no_full_query_wrong_condition_or_output_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); out = root / "cli"
            cmd = [sys.executable, "scripts/audit_g4_position_fusion_initialization.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out), "--target", "xuzhou",
                "--method", METHODS[0], "--condition", "position_gate", "--model-seed", "64", "--support-seed", "2048", "--per-class", "5"]
            result = json.loads(subprocess.check_output(cmd, text=True)); self.assertEqual(result["added_parameters"], 129)
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual((receipt["model_forwards"], receipt["optimizer_updates"]), (0, 0)); self.assertFalse(receipt["gpu_context_initialized"])
            for extra in (["--full"], ["--query"], ["--condition", "alpha035"], ["--model-seed", "128"]):
                self.assertNotEqual(subprocess.run(cmd + extra, capture_output=True).returncode, 0)
            before = {p.name: file_hash(p) for p in out.iterdir()}
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(before, {p.name: file_hash(p) for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
