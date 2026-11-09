"""Existing fixed M8 contracts on full-NONE target initialization; synthetic only."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import torch
import torch.nn.functional as F

from g4_fusion_target import (CALIBRATION_KEY, CONDITIONS, KEY, calibrate, configuration,
    forward_observation, initialize, load_calibrated, loss_terms, verify_calibrated, verify_initialization_binding)
from joint_baseline import forward as old_forward, load_config
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.branch_fusion import fixed_fusion
from model.checkpoint_io import save_model_checkpoint
from none_full_transfer import initialize_none_full_target
from research_runtime import code_snapshot
from target_bn_calibration import CONTEXT, GRAPH_POLICY, POLICY as BN_POLICY, calibrate_target, observation_id
from target_initialization import METHODS, state_id
from target_training import loss_terms as old_loss_terms
from test_g4_coverage_target import fixture, obs
from test_none_full_transfer import fixture as source_fixture

DEVICE = os.environ.get("FUSION_TARGET_DEVICE", "cpu")
torch.set_num_threads(1)


def create(condition="default", method=METHODS[0], spec=None, device=DEVICE):
    return initialize("xuzhou", method, condition, model_seed=64, head_seed=1064, source_spec=spec, device=device)


def config_for(condition, device=DEVICE):
    base = load_config("configs/experiments/base97_joint.json"); value = configuration(condition)
    return replace(base, device=device, model=replace(base.model, alph=value["alph"]),
        loss=replace(base.loss, **{k: value[k] for k in ("cnn_loss_weight", "graph_branch_loss_weight", "balanced")}))


class FusionTargetTests(unittest.TestCase):
    def test_six_roles_shared_fresh_state_rng_architecture_and_no_new_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            _, spec = source_fixture(Path(temp) / "source")
            for method in METHODS[:2]:
                source = spec if method == METHODS[1] else None
                for condition in CONDITIONS:
                    torch.manual_seed(918)
                    base, record = initialize_none_full_target("xuzhou", method, model_seed=64, head_seed=1064,
                        source_spec=source, device=DEVICE)
                    rng = torch.get_rng_state().clone(); torch.manual_seed(918)
                    model, binding = create(condition, method, source)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertEqual(binding["base_initialization"], record)
                    self.assertEqual(state_id(base), state_id(model)); self.assertEqual(binding["added_parameter_count"], 0)
                    self.assertEqual(model.export_config()["backbone_kwargs"]["alph"], configuration(condition)["alph"])
                    self.assertTrue(all(p.requires_grad for p in model.parameters()))
                    self.assertEqual(tuple(model.heads), ("xuzhou",)); verify_initialization_binding(model, binding)
            for seed in (512, 2048):
                model, binding = initialize("paviaU", METHODS[0], "alpha035", model_seed=seed, head_seed=seed + 1000)
                self.assertEqual(binding["head_seed"], seed + 1000); self.assertEqual(tuple(model.heads), ("paviaU",))

    def test_default_exact_original_forward_and_population_bn_both_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = obs(item, device=DEVICE)
            _, spec = source_fixture(root / "source")
            for method in METHODS[:2]:
                source = spec if method == METHODS[1] else None
                base, _ = initialize_none_full_target("xuzhou", method, model_seed=64, head_seed=1064,
                    source_spec=source, device=DEVICE)
                base.eval(); model, binding = create(method=method, spec=source)
                with torch.no_grad(), OrderedValidation():
                    first = old_forward(base, "xuzhou", data)
                    second = forward_observation(model, data, expected_binding=binding)
                for a, b in zip(first, second): torch.testing.assert_close(a, b, rtol=0, atol=0)
                with OrderedValidation():
                    original, receipt = calibrate_target(base, data, mode=BN_POLICY, context=CONTEXT,
                        graph_policy=GRAPH_POLICY, source_state_sha256=state_id(base))
                new, _ = calibrate(model, data, expected_binding=binding, source_state_sha256=state_id(model))
                self.assertEqual(state_id(original), state_id(new))
                self.assertEqual(receipt, new.training_config["target_bn_calibration"])

    def test_manual_features_logits_legacy_loss_auxiliary_zero_and_decay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = obs(item, device=DEVICE)
            model, binding = create("alpha035")
            cnn, graph = torch.randn(7, 64, device=DEVICE), torch.randn(7, 64, device=DEVICE)
            fused = .35 * cnn + .65 * graph
            expected = (model.heads["xuzhou"](fused), model.hgnn_heads["xuzhou"](graph), model.cnn_heads["xuzhou"](cnn))
            actual = model.fuse_and_predict(cnn, graph, "xuzhou", return_logits=True)
            for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(fixed_fusion(cnn, graph, 1.), cnn, rtol=0, atol=0)
            torch.testing.assert_close(fixed_fusion(cnn, graph, 0.), graph, rtol=0, atol=0)
            with torch.no_grad(), OrderedValidation():
                logits = forward_observation(model, data, expected_binding=binding, logits=True)
                probabilities = forward_observation(model, data, expected_binding=binding)
            for a, b in zip(logits, probabilities): torch.testing.assert_close(a.softmax(-1), b, rtol=1e-5, atol=1e-6)
            labels = torch.arange(9, device=DEVICE)
            for condition in CONDITIONS:
                current, current_binding = create(condition); config = config_for(condition)
                outputs = []
                for offset in (1, 2, 3):
                    p = torch.arange(offset, offset + 81, device=DEVICE, dtype=torch.float64).reshape(9, 9)
                    outputs.append((p / p.sum(-1, keepdim=True)).detach().requires_grad_())
                terms = loss_terms(current, outputs, labels, config, expected_binding=current_binding)
                old = old_loss_terms(outputs, labels, config); self.assertEqual(len(terms), 13)
                for name in old: torch.testing.assert_close(terms[name], old[name], rtol=0, atol=0)
                for value, name in zip(outputs, ("fused_supervised", "graph_supervised", "cnn_supervised")):
                    reference = -value.gather(1, labels[:, None]).log().mean()
                    torch.testing.assert_close(terms[name], reference, rtol=1e-5, atol=1e-6)
                gradients = torch.autograd.grad(terms["total"], outputs)
                self.assertGreater(float(gradients[0].abs().sum()), 0.)
                for value in gradients[1:]:
                    self.assertEqual(float(value.abs().sum()) == 0., condition == "aux_off")
                if condition == "aux_off":
                    self.assertEqual(float(terms["weighted_graph_supervised"]), 0.)
                    self.assertEqual(float(terms["weighted_cnn_supervised"]), 0.)
                with self.assertRaises(ValueError): loss_terms(current, [torch.zeros_like(v) for v in outputs], labels, config, expected_binding=current_binding)
                with self.assertRaises(ValueError): loss_terms(current, [outputs[0], outputs[1][:, :1], outputs[2]], labels, config, expected_binding=current_binding)
            # Preserve original Adam weight decay: zero auxiliary loss gradients
            # do not imply frozen independent head parameters.
            current, current_binding = create("aux_off"); current.train(); config = config_for("aux_off")
            optimizer = torch.optim.Adam(current.parameters(), lr=.001, weight_decay=config.training.weight_decay)
            selected = tuple(v[item["train_indices"].to(DEVICE)] for v in forward_observation(current, data, expected_binding=current_binding))
            loss_terms(current, selected, item["train_labels"].to(DEVICE), config, expected_binding=current_binding)["total"].backward()
            heads = [*current.hgnn_heads.parameters(), *current.cnn_heads.parameters()]
            self.assertTrue(all(p.grad is not None and int(torch.count_nonzero(p.grad)) == 0 for p in heads))
            before = [p.detach().clone() for p in heads]; optimizer.step()
            self.assertGreater(sum(int(torch.count_nonzero(p.detach() - v)) for p, v in zip(heads, before)), 0)

    def test_all_roles_three_updates_population_bn_protection_and_eighteen_reloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, item, _ = fixture(root); data = obs(item, device=DEVICE)
            _, spec = source_fixture(root / "source"); observation_before = observation_id(data)
            for method in METHODS[:2]:
                for condition in CONDITIONS:
                    model, binding = create(condition, method, spec if method == METHODS[1] else None)
                    config = config_for(condition); model.train()
                    before = {n: p.detach().clone() for n, p in model.named_parameters()}
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=config.training.weight_decay)
                    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids))); self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        outputs = forward_observation(model, data, expected_binding=binding)
                        selected = tuple(v[item["train_indices"].to(DEVICE)] for v in outputs)
                        loss_terms(model, selected, item["train_labels"].to(DEVICE), config, expected_binding=binding)["total"].backward()
                        gradients = [p.grad for p in model.parameters() if p.grad is not None]
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.heads.parameters()), 0.)
                        optimizer.step()
                    self.assertGreater(sum(int(torch.count_nonzero(p.detach() - before[n])) for n, p in model.named_parameters()), 0)
                    model.eval(); source = state_id(model)
                    candidate, receipt = calibrate(model, data, expected_binding=binding, source_state_sha256=source)
                    self.assertEqual(source, state_id(model)); self.assertEqual(observation_before, observation_id(data))
                    bn = candidate.training_config["target_bn_calibration"]
                    self.assertEqual((len(bn["layers"]), len(bn["calibrated_buffers"])), (5, 10))
                    for name, tensor in model.state_dict().items():
                        if name not in bn["calibrated_buffers"]:
                            torch.testing.assert_close(tensor, candidate.state_dict()[name], rtol=0, atol=0)
                    path = root / (method + condition + ".pt"); save_model_checkpoint(candidate, path)
                    with torch.no_grad(), OrderedValidation(): expected = forward_observation(candidate, data, expected_binding=binding)
                    for _ in range(3):
                        loaded = load_calibrated(path, file_hash(path), file_hash(str(path) + ".json"), data,
                            expected_binding=binding, source_state_sha256=source, device=DEVICE)
                        with torch.no_grad(), OrderedValidation(): actual = forward_observation(loaded, data, expected_binding=binding)
                        for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(receipt, verify_calibrated(candidate, data, expected_binding=binding, source_state_sha256=source))
                    with self.assertRaises(ValueError): load_calibrated(path, "0" * 64, file_hash(str(path) + ".json"), data,
                        expected_binding=binding, source_state_sha256=source, device=DEVICE)

    def test_cpu_double_population_reference_and_actual_operator_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            _, item, _ = fixture(Path(temp)); data = obs(item, device="cpu")
            model, binding = create("alpha035", device="cpu"); oracle = copy.deepcopy(model); moments = {}
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
            bad = copy.deepcopy(model); bad.backbone.alph = .8
            with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            bad = copy.deepcopy(model); bad.backbone.fuse_features = types.MethodType(lambda self, c, g: c, bad.backbone)
            with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            bad = copy.deepcopy(model); next(bad.parameters()).requires_grad_(False)
            with self.assertRaises(ValueError): verify_initialization_binding(bad, binding)
            with self.assertRaises(ValueError): calibrate(model, {**data, "labels": torch.zeros(1)}, expected_binding=binding, source_state_sha256=state_id(model))
            with self.assertRaises(ValueError): verify_calibrated(candidate, data, expected_binding=binding, source_state_sha256="0" * 64)
            wrong = config_for("default", "cpu")
            with self.assertRaises(ValueError): loss_terms(model, [torch.ones(9, 9) / 9] * 3, torch.arange(9), wrong, expected_binding=binding)

    def test_cpu_cli_source_seed_output_and_nonexecuting_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _, _ = fixture(root); out = root / "cli"
            command = [sys.executable, "scripts/audit_g4_fusion_target_initialization.py", "--expected-commit", code_snapshot()["commit"],
                "--manifest", str(manifest), "--manifest-sha256", file_hash(manifest), "--out", str(out), "--target", "xuzhou",
                "--method", METHODS[0], "--condition", "alpha035", "--model-seed", "64", "--support-seed", "2048", "--per-class", "5"]
            result = json.loads(subprocess.check_output(command, text=True)); self.assertEqual(result["added_parameters"], 0)
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual(receipt["binding"]["fusion_loss_contract"]["alph"], .35)
            self.assertEqual((receipt["optimizer_updates"], receipt["model_forwards"]), (0, 0))
            self.assertFalse(receipt["bn_calibration_executed"]); self.assertFalse(receipt["gpu_context_initialized"])
            before = {p.name: file_hash(p) for p in out.iterdir()}
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, {p.name: file_hash(p) for p in out.iterdir()})
            for extra in (["--query"], ["--full"], ["--condition", "gated"], ["--model-seed", "128"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)
            for kwargs in ({"model_seed": 64, "head_seed": 64}, {"model_seed": True, "head_seed": 1001}):
                with self.assertRaises(ValueError): initialize("xuzhou", METHODS[0], "default", **kwargs)


if __name__ == "__main__":
    unittest.main()
