"""Original-prior control, matching M6 initialization and explicit restore semantics."""
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

from g4_interaction_target import initialize as base_initialize
from g4_normalized_edge_target import (CONDITIONS, PREFIX, CandidateRowNormalization, calibrate, configuration,
    forward_observation, initialize, load_calibrated, verify_calibrated, verify_initialization_binding)
from joint_baseline import forward as old_forward
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.graph_edge_update import GraphEdgeUpdater
from model.research_config import ResearchConfig
from research_runtime import code_snapshot
from target_bn_calibration import observation_id
from target_initialization import METHODS, state_id
from target_training import loss_terms
from test_g4_coverage_target import fixture, obs
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("NORMALIZATION_TARGET_DEVICE", "cpu")
torch.set_num_threads(1)


def create(condition="gated_row_normalized", method=METHODS[0], spec=None, device=DEVICE):
    return initialize("xuzhou", method, condition, model_seed=64, head_seed=1064, source_spec=spec, device=device)


def data_for(item, device=DEVICE):
    value = obs(item, device=device); count = value["A"].shape[0]
    value["A"] = 2. * torch.eye(count, device=device) + .25 * torch.eye(count, device=device).roll(1, dims=1)
    return value


class NormalizationTargetTests(unittest.TestCase):
    def test_independent_double_prior_gradient_support_isolation_and_original_mix_zero(self):
        adjacency = torch.tensor([[.2, .8, 0.], [0., 0., 0.], [1e-15, 2e-15, 0.]], dtype=torch.float64, requires_grad=True)
        regions = torch.tensor([[1., 2., 0.], [2., 0., 1.], [0., 1., 2.]], dtype=torch.float64)
        operator = CandidateRowNormalization(3).double()
        actual, info = operator(regions, adjacency, collect_stats=True, return_diagnostics=True)
        positive = torch.where(adjacency > 0, adjacency, 0.)
        isolated = (positive.sum(1) == 0).double()
        prior = positive + torch.diag(isolated)
        expected = prior / prior.sum(1, keepdim=True).clamp_min(1e-12)
        torch.testing.assert_close(actual.to_dense(), expected, rtol=0, atol=0)
        coefficients = torch.arange(1., 10., dtype=torch.float64).reshape(3, 3)
        a = torch.autograd.grad((actual.to_dense() * coefficients).sum(), adjacency, retain_graph=True)[0]
        b = torch.autograd.grad((expected * coefficients).sum(), adjacency)[0]
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        with torch.random.fork_rng(devices=[]):
            legacy = GraphEdgeUpdater(channels=3, mode="graph", embed_dim=2, mix=0.).double()
        reference, _ = legacy(regions, adjacency.detach())
        torch.testing.assert_close(actual.to_dense(), reference.to_dense(), rtol=0, atol=0)
        self.assertAlmostEqual(float(actual.to_dense()[2].sum()), .003)
        self.assertEqual(info["edge_count"], 5); self.assertEqual(float(info["edge_update_l1"]), 0.)
        self.assertEqual(list(operator.parameters()), [])

    def test_all_roles_matching_M6_states_rng_permissions_and_only_version_buffer(self):
        with tempfile.TemporaryDirectory() as temp:
            _, spec = source_fixture(Path(temp) / "source")
            for method in METHODS[:2]:
                source = spec if method == METHODS[1] else None
                for condition in CONDITIONS:
                    config = configuration(condition); torch.manual_seed(918)
                    base, record = base_initialize("xuzhou", method, config["m6_condition"], model_seed=64, head_seed=1064, source_spec=source, device=DEVICE)
                    rng = torch.get_rng_state().clone(); torch.manual_seed(918)
                    model, binding = create(condition, method, source)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state())); self.assertEqual(binding["base_binding"], record)
                    self.assertEqual(binding["normalization_parameter_count"], 0)
                    self.assertEqual(binding["matching_M6_parameter_count"], 58498 if config["m6_condition"] == "gated" else 0)
                    for n, value in base.state_dict().items(): torch.testing.assert_close(value, model.state_dict()[n], rtol=0, atol=0)
                    extra = set(model.state_dict()) - set(base.state_dict())
                    self.assertEqual(extra, {PREFIX + "protocol_version"} if config["normalize"] else set())
                    self.assertEqual(binding["additional_state_bytes"], 8 if config["normalize"] else 0)
                    self.assertTrue(all(p.requires_grad for p in model.parameters())); verify_initialization_binding(model, binding)
            for seed in (512, 2048):
                _, binding = initialize("paviaU", METHODS[0], "row_normalized", model_seed=seed, head_seed=seed + 1000)
                self.assertEqual(binding["base_binding"]["head_seed"], seed + 1000)

    def test_disabled_exact206_and_identity_enabled_equivalence_and_actual_stage2_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item); _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                source = spec if method == METHODS[1] else None
                for condition in ("none", "gated"):
                    base, _ = base_initialize("xuzhou", method, condition, model_seed=64, head_seed=1064, source_spec=source, device=DEVICE)
                    model, binding = create(condition, method, source)
                    with torch.no_grad(), OrderedValidation():
                        expected = old_forward(base, "xuzhou", data); actual = forward_observation(model, data, expected_binding=binding)
                    for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
                    enabled = "row_normalized" if condition == "none" else "gated_row_normalized"
                    normalized, proof = create(enabled, method, source)
                    identity = {**data, "A": torch.eye(data["A"].shape[0], device=DEVICE)}
                    with torch.no_grad(), OrderedValidation():
                        expected = old_forward(base, "xuzhou", identity); actual = forward_observation(normalized, identity, expected_binding=proof)
                    for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
            model, binding = create(); seen = {}; original = model.backbone.HGNN_Branch._graph_conv
            def logged(conv, x, adjacency):
                seen["stage1" if conv is model.backbone.HGNN_Branch.hgc1 else "stage2"] = adjacency.detach().to_dense()
                return original(conv, x, adjacency)
            with patch.object(model.backbone.HGNN_Branch, "_graph_conv", side_effect=logged), torch.no_grad(), OrderedValidation():
                forward_observation(model, data, expected_binding=binding)
            torch.testing.assert_close(seen["stage1"], data["A"], rtol=0, atol=0)
            torch.testing.assert_close(seen["stage2"], data["A"] / data["A"].sum(1, keepdim=True), rtol=0, atol=0)

    def test_eight_roles_synthetic_three_updates_population_bn_24_reloads_and_generic_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item); _, spec = source_fixture(root / "source")
            identity = observation_id(data)
            for method in METHODS[:2]:
                for condition in CONDITIONS:
                    model, binding = create(condition, method, spec if method == METHODS[1] else None)
                    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001); config = ResearchConfig(device=DEVICE)
                    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
                    self.assertEqual(len(ids), len(set(ids))); self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                    model.train()
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True); outputs = forward_observation(model, data, expected_binding=binding)
                        selected = tuple(v[item["train_indices"].to(DEVICE)] for v in outputs)
                        terms = loss_terms(selected, item["train_labels"].to(DEVICE), config)
                        self.assertTrue(torch.isfinite(terms["total"])); terms["total"].backward()
                        for prefix in ("backbone.HGNN_Branch.", "backbone.interaction."):
                            params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
                            if params:
                                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                                self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0.)
                        optimizer.step()
                    for prefix in ("backbone.HGNN_Branch.", "backbone.interaction."):
                        names = [n for n in initial if n.startswith(prefix)]
                        if names: self.assertGreater(sum(int(torch.count_nonzero(model.state_dict()[n] - initial[n])) for n in names), 0)
                    model.eval(); source = state_id(model)
                    candidate, receipt = calibrate(model, data, expected_binding=binding, source_state_sha256=source)
                    self.assertEqual(source, state_id(model)); self.assertEqual(identity, observation_id(data))
                    legacy = candidate.training_config["target_bn_calibration"]
                    self.assertEqual((len(legacy["layers"]), len(legacy["calibrated_buffers"])), (5, 10))
                    for n, value in model.state_dict().items():
                        if n not in legacy["calibrated_buffers"]: torch.testing.assert_close(value, candidate.state_dict()[n], rtol=0, atol=0)
                    path = root / (condition + method + ".pt"); save_model_checkpoint(candidate, path)
                    if configuration(condition)["normalize"]:
                        with self.assertRaisesRegex(RuntimeError, "protocol_version"): load_model_checkpoint(path, device=DEVICE)
                    with torch.no_grad(), OrderedValidation(): expected = forward_observation(candidate, data, expected_binding=binding)
                    for _ in range(3):
                        loaded = load_calibrated(path, file_hash(path), file_hash(str(path) + ".json"), data,
                            expected_binding=binding, source_state_sha256=source, device=DEVICE)
                        with torch.no_grad(), OrderedValidation(): actual = forward_observation(loaded, data, expected_binding=binding)
                        for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(receipt, verify_calibrated(candidate, data, expected_binding=binding, source_state_sha256=source))

    def test_independent_cpu_fp64_population_moments_and_bound_attributes(self):
        with tempfile.TemporaryDirectory() as temp:
            _, item, _ = fixture(Path(temp)); data = data_for(item, "cpu"); model, binding = create(device="cpu")
            oracle = copy.deepcopy(model); moments = {}
            for name, layer in oracle.named_modules():
                if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    def independent(self, x, name=name):
                        axes = (0, *range(2, x.ndim)); value = x.double()
                        mean = value.mean(dim=axes); shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
                        var = (value - mean.reshape(shape)).square().mean(dim=axes); moments[name] = (mean, var)
                        return F.batch_norm(x, mean.to(x.dtype), var.to(x.dtype), self.weight, self.bias, False, 0., self.eps)
                    layer.forward = types.MethodType(independent, layer)
            with torch.no_grad(), OrderedValidation(): expected = forward_observation(oracle, data, expected_binding=binding)
            candidate, _ = calibrate(model, data, expected_binding=binding, source_state_sha256=state_id(model))
            with torch.no_grad(), OrderedValidation(): actual = forward_observation(candidate, data, expected_binding=binding)
            for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            for name, (mean, var) in moments.items():
                layer = dict(candidate.named_modules())[name]
                torch.testing.assert_close(layer.running_mean.double(), mean, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(layer.running_var.double(), var, rtol=1e-5, atol=1e-6)
            for attr, value in (("denominator_floor", 1e-8), ("mode", "joint")):
                bad = copy.deepcopy(model); setattr(bad.backbone.edge_updater, attr, value)
                with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            bad = copy.deepcopy(model); bad.backbone.edge_updater.protocol_version.fill_(0)
            with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            bad = copy.deepcopy(model); bad.heads["xuzhou"].weight.requires_grad_(False)
            with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            with self.assertRaises(ValueError): calibrate(model, {**data, "labels": torch.zeros(1)}, expected_binding=binding, source_state_sha256=state_id(model))
            with self.assertRaises(ValueError): verify_calibrated(candidate, data, expected_binding=binding, source_state_sha256="0" * 64)

    def test_invalid_operator_interfaces_source_and_checkpoint_pins_rejected(self):
        operator = CandidateRowNormalization(3); regions = torch.ones(3, 3)
        for adjacency in (-torch.eye(3), torch.full((3, 3), float("nan")), torch.eye(2), torch.eye(3).long()):
            with self.assertRaises(ValueError): operator(regions, adjacency)
        for bad in (torch.ones(3), torch.ones(0, 3), torch.full((3, 3), float("inf")), regions.half()):
            with self.assertRaises(ValueError): operator(bad, torch.eye(3))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item); _, spec = source_fixture(root / "source")
            wrong = {**spec, "source_weight_sha256": "0" * 64}
            with self.assertRaises(ValueError): create(method=METHODS[1], spec=wrong)
            model, binding = create(); source = state_id(model)
            candidate, _ = calibrate(model, data, expected_binding=binding, source_state_sha256=source)
            path = root / "checkpoint.pt"; save_model_checkpoint(candidate, path)
            with patch("torch.load", side_effect=AssertionError("Pins before tensor load")):
                with self.assertRaises(ValueError): load_calibrated(path, "0" * 64, file_hash(str(path) + ".json"), data,
                    expected_binding=binding, source_state_sha256=source, device=DEVICE)

    def test_actual_cpu_zero_forward_cli_and_no_full_query_or_wrong_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); out = root / "cli"
            command = [sys.executable, "scripts/audit_g4_normalized_edge_initialization.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out), "--target", "xuzhou",
                "--method", METHODS[0], "--condition", "gated_row_normalized", "--model-seed", "64", "--support-seed", "2048", "--per-class", "5"]
            result = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual((result["normalization_added_parameters"], result["matching_m6_parameters"]), (0, 58498))
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual((receipt["optimizer_updates"], receipt["model_forwards"]), (0, 0)); self.assertFalse(receipt["gpu_context_initialized"])
            for extra in (["--full"], ["--query"], ["--condition", "graph"], ["--model-seed", "128"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)
            before = {p.name: file_hash(p) for p in out.iterdir()}
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, {p.name: file_hash(p) for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
