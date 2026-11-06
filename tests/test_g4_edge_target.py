"""Existing M7 core-factor factory/BN integration; synthetic learning only."""
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

from g4_edge_target import (CALIBRATION_KEY, CONDITIONS, KEY, PREFIXES, calibrate, configuration,
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

DEVICE = os.environ.get("EDGE_TARGET_DEVICE", "cpu")
torch.set_num_threads(1)
COUNTS = {"none": 0, "gated": 58498, "graph": 8448, "gated_graph": 66946, "joint": 16896, "gated_joint": 75394}


def create(condition="gated_joint", method=METHODS[0], spec=None, device=DEVICE):
    return initialize("xuzhou", method, condition, model_seed=64, head_seed=1064, source_spec=spec, device=device)


def data_for(item, device=DEVICE):
    # A synthetic multi-candidate observation exercises edge-conditioned BN
    # and reloads as well as forward/gradients. The on-disk catalogue is intact;
    # the separate CLI test only initializes from its original authenticated A.
    value = obs(item, device=device)
    count = value["A"].shape[0]
    value["A"] = .5 * torch.eye(count, device=device) + .5 * torch.eye(count, device=device).roll(1, dims=1)
    return value


class EdgeTargetTests(unittest.TestCase):
    def test_edge_formula_double_dense_reference_gradients_support_and_input_roles(self):
        from model.graph_edge_update import GraphEdgeUpdater
        torch.manual_seed(73)
        adjacency = torch.tensor([[.2, .8, 0., 0.], [.4, .1, .5, 0.],
                                  [0., .6, .4, 0.], [0., 0., 0., 0.]], dtype=torch.float64)
        def reference(module, regions, pixels):
            # Independent dense candidate formula, including explicit isolated fallback.
            prior = adjacency.clone(); prior[3, 3] = 1.
            prior = prior / prior.sum(1, keepdim=True)
            x = torch.cat((regions, pixels), 1) if module.mode == "joint" else regions
            x = F.layer_norm(x, module.norm.normalized_shape, module.norm.weight, module.norm.bias, module.norm.eps)
            q = F.linear(x, module.query.weight); k = F.linear(x, module.key.weight)
            logits = q @ k.T / (module.embed_dim ** .5)
            logits = (logits + prior.clamp_min(1e-12).log()).masked_fill(prior == 0, -float("inf"))
            return (1. - module.mix) * prior + module.mix * logits.softmax(1)
        for mode in ("graph", "joint"):
            module = GraphEdgeUpdater(channels=5, mode=mode, embed_dim=3, mix=.5).double()
            other = copy.deepcopy(module)
            g = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
            c = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
            gg, cc = g.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
            identity, _ = module(g, torch.eye(4, dtype=torch.float64), c)
            torch.testing.assert_close(identity.to_dense(), torch.eye(4, dtype=torch.float64), rtol=0, atol=0)
            frozen = torch.autograd.grad(identity.to_dense().square().sum(), tuple(module.parameters()))
            self.assertTrue(all(int(torch.count_nonzero(v)) == 0 for v in frozen))
            actual, info = module(g, adjacency.to_sparse(), c, collect_stats=True, return_diagnostics=True)
            expected = reference(other, gg, cc)
            torch.testing.assert_close(actual.to_dense(), expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual.to_dense().sum(1), torch.ones(4, dtype=torch.float64), rtol=1e-5, atol=1e-6)
            support = adjacency > 0; support[3, 3] = True
            self.assertTrue(torch.equal(actual.to_dense() > 0, support))
            weight = torch.arange(16, dtype=torch.float64).reshape(4, 4).square()
            inputs = (g, c, *module.parameters()); refs = (gg, cc, *other.parameters())
            a = torch.autograd.grad((actual.to_dense() * weight).sum(), inputs, allow_unused=True)
            b = torch.autograd.grad((expected * weight).sum(), refs, allow_unused=True)
            for first, second in zip(a, b):
                self.assertEqual(first is None, second is None)
                if first is not None: torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
            self.assertEqual(a[1] is None, mode == "graph")
            shifted, _ = module(g, adjacency, c.flip(0))
            if mode == "graph": torch.testing.assert_close(actual.to_dense(), shifted.to_dense(), rtol=0, atol=0)
            else: self.assertGreater(float((actual.to_dense() - shifted.to_dense()).abs().max()), 0.)
            self.assertEqual(info["edge_count"], int(support.sum()))
            self.assertGreater(float(info["edge_update_l1"]), 0.)

    def test_stage_order_post_message_edge_inputs_and_old206_edge_off_equivalence(self):
        from g4_interaction_target import initialize as old_initialize, forward_observation as old_forward_bound
        with tempfile.TemporaryDirectory() as temp:
            _, item, _ = fixture(Path(temp)); data = data_for(item)
            for condition in ("none", "gated"):
                model, binding = create(condition)
                old, old_binding = old_initialize("xuzhou", METHODS[0], condition, model_seed=64, head_seed=1064, device=DEVICE)
                self.assertEqual(state_id(model), state_id(old))
                with torch.no_grad(), OrderedValidation():
                    a = forward_observation(model, data, expected_binding=binding)
                    b = old_forward_bound(old, data, expected_binding=old_binding)
                for first, second in zip(a, b): torch.testing.assert_close(first, second, rtol=0, atol=0)
            for condition in ("gated_graph", "gated_joint"):
                model, binding = create(condition); backbone = model.backbone; order = []; saved = {}
                original_message = backbone.interaction.forward_aligned
                original_edge = backbone.edge_updater.forward
                def message(*args, **kwargs):
                    values = original_message(*args, **kwargs); order.append("message")
                    saved.update(pixels=values[0], regions=values[1], Q=args[2]); return values
                def edge(regions, adjacency, aligned_pixels, **kwargs):
                    order.append("edge"); torch.testing.assert_close(regions, saved["regions"], rtol=0, atol=0)
                    if condition == "gated_joint":
                        torch.testing.assert_close(aligned_pixels, backbone.alignment.to_regions(saved["pixels"], saved["Q"]), rtol=0, atol=0)
                    else: self.assertIsNone(aligned_pixels)
                    return original_edge(regions, adjacency, aligned_pixels, **kwargs)
                with patch.object(backbone.interaction, "forward_aligned", side_effect=message), patch.object(backbone.edge_updater, "forward", side_effect=edge):
                    with torch.no_grad(), OrderedValidation(): forward_observation(model, data, expected_binding=binding, diagnostics=True)
                self.assertEqual(order, ["message", "edge"])
                for field, value in (("mix", .25), ("embed_dim", 31), ("mode", "graph" if condition.endswith("joint") else "joint")):
                    bad = copy.deepcopy(model); setattr(bad.backbone.edge_updater, field, value)
                    with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)

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
                    self.assertEqual(binding["added_parameter_count"], COUNTS[condition])
                    self.assertTrue(all(p.requires_grad for p in model.parameters()))
                    self.assertEqual(tuple(model.heads), ("xuzhou",))
                    self.assertEqual(model.backbone.edge_updater is None, condition in ("none", "gated")); self.assertIsNone(model.backbone.coverage_modulation)
                    self.assertEqual(configuration(condition)["interaction_strength"], .1)
                    verify_initialization_binding(model, binding)
            for seed in (512, 2048):
                model, b = initialize("paviaU", METHODS[0], "graph", model_seed=seed, head_seed=seed + 1000)
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

    def test_enabled_conditions_three_synthetic_updates_gradients_and_enrollment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = data_for(item)
            # The inherited context fixture has A=I: one candidate per row makes
            # softmax reweighting constant and correctly gives zero gradients.
            # Only this synthetic participation test uses two legal candidates.
            count = data["A"].shape[0]
            data["A"] = .5 * torch.eye(count, device=DEVICE) + .5 * torch.eye(count, device=DEVICE).roll(1, dims=1)
            _, spec = source_fixture(root / "source")
            from joint_baseline import load_config
            config = load_config("configs/experiments/base97_joint.json")
            for method in METHODS[:2]:
                for condition in CONDITIONS[1:]:
                    model, binding = create(condition, method, spec if method == METHODS[1] else None); model.train()
                    before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(PREFIXES)}
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        outputs = forward_observation(model, data, expected_binding=binding)
                        selected = tuple(v[item["train_indices"].to(DEVICE)] for v in outputs)
                        loss_terms(selected, item["train_labels"].to(DEVICE), config)["total"].backward()
                        for prefix in PREFIXES:
                            params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
                            if params:
                                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                                self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0.)
                        optimizer.step()
                    self.assertGreater(sum(int(torch.count_nonzero(p.detach() - before[n])) for n, p in model.named_parameters() if n in before), 0)
                    for prefix in PREFIXES:
                        names = [n for n in before if n.startswith(prefix)]
                        if names:
                            self.assertGreater(sum(int(torch.count_nonzero(model.state_dict()[n] - before[n])) for n in names), 0)

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
            command = [sys.executable, "scripts/audit_g4_edge_target_initialization.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out), "--target", "xuzhou",
                "--method", METHODS[0], "--condition", "gated", "--model-seed", "64", "--support-seed", "2048", "--per-class", "5"]
            result = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(result["added_parameters"], 58498)
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual((receipt["optimizer_updates"], receipt["model_forwards"]), (0, 0))
            self.assertFalse(receipt["bn_calibration_executed"]); self.assertFalse(receipt["gpu_context_initialized"])
            before = {p.name: file_hash(p) for p in out.iterdir()}
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, {p.name: file_hash(p) for p in out.iterdir()})
            for extra in (["--query"], ["--full"], ["--condition", "global"], ["--model-seed", "128"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)
            for kwargs in ({"model_seed": 64, "head_seed": 64}, {"model_seed": True, "head_seed": 1001}):
                with self.assertRaises(ValueError): initialize("xuzhou", METHODS[0], "gated", **kwargs)


if __name__ == "__main__":
    unittest.main()
