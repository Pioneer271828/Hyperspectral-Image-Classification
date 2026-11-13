"""Acceptance checks for the review of released model boundaries."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import torch

from inference.native_support_checkpoint import load_endpoint
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from research_runtime import tensor_hash
from transfer import registered_development as dev
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import state_hash
from transfer.native_support import NativeSupportBackbone, NativeSupportModel, bank_digest


class NativePublicationReviewTests(unittest.TestCase):
    def setUp(self):
        self.options = dict(graph_mode="learned", norm_type="group_norm", latent_channels=8, adapter_embed_dim=8)
        self.baseline = MultiDatasetVCASAHGCN({"synthetic": 9}, **self.options).eval()
        self.baseline.training_config = {"evidence": {"branch": "source"}}
        self.baseline.backbone.adapter.cache_attention = True
        self.baseline.heads["synthetic"].train()
        self.registration = {"path": "/registration.json", "sha256": "a" * 64}
        self.case = dict(development={"path": "/development.json", "sha256": "b" * 64}, budget=48, scorer_seed=7321)

    def package(self, root, model, endpoint, amount=0., config=None):
        clone = deepcopy(model)
        with torch.no_grad():
            clone.heads["synthetic"].weight.add_(amount)
        path = root / (endpoint + ".pt")
        CurriculumRun._write(dict(version=2, registration=self.registration, development=self.case["development"],
            config=clone.export_config() if config is None else config, work_package="WP08",
            endpoint=endpoint, step=7 if endpoint == "selected" else 48, state=clone.state_dict(),
            state_hash=state_hash(clone), training_config={"evidence": {"branch": endpoint}}, bank=bank_digest({})), path)
        return dev._pin(path)

    def test_off_loads_are_independent_in_both_orders(self):
        baseline = self.baseline
        tensors = {key: tensor_hash(t) for key, t in baseline.state_dict().items()}
        flags = {key: module.training for key, module in baseline.named_modules()}
        config = deepcopy(baseline.export_config())
        rng = torch.get_rng_state().clone()
        with TemporaryDirectory() as name:
            pins = {key: self.package(Path(name), baseline, key, amount)
                    for key, amount in (("selected", .25), ("last", -.5))}
            for order in (("selected", "last"), ("last", "selected")):
                results = [load_endpoint(pins[key], self.registration, self.case, (baseline, None, None, None, None))
                           for key in order]
                for (model, packet), endpoint in zip(results, order, strict=True):
                    self.assertIsNot(model, baseline)
                    self.assertEqual(state_hash(model), packet["state_hash"])
                    self.assertEqual(model.training_config["evidence"]["branch"], endpoint)
                    self.assertFalse(model.backbone.adapter.cache_attention)
                    self.assertTrue(all(not module.training for module in model.modules()))
                results[0][0].training_config["evidence"]["branch"] = "mutated"
                self.assertNotEqual(results[1][0].training_config["evidence"]["branch"], "mutated")
                self.assertEqual(tensors, {k: tensor_hash(t) for k, t in baseline.state_dict().items()})
                self.assertEqual(flags, {key: module.training for key, module in baseline.named_modules()})
                self.assertEqual(config, baseline.export_config())
                self.assertEqual(baseline.training_config, {"evidence": {"branch": "source"}})
                self.assertTrue(baseline.backbone.adapter.cache_attention)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_direct_constructor_rejects_incompatible_graph_and_observation_modes(self):
        for overrides in (dict(graph_mode="static"), dict(learned_graph_radius=2),
                          dict(graph_assignment_mode="fixed_grid"), dict(graph_adjacency_mode="fixed_local"),
                          dict(interaction_mode="gated"), dict(graph_update_mode="graph"),
                          dict(graph_route={"mode": "joint"}), dict(region_residual_mode="x_residual"),
                          dict(coverage_mode="anchor_only"), dict(extra_channels=0), dict(norm_type="domain_bn")):
            for constructor in (lambda **kw: NativeSupportModel({"synthetic": 9}, scorer_seed=7321, **kw),
                                NativeSupportBackbone):
                with self.subTest(overrides=overrides, constructor=constructor):
                    with self.assertRaisesRegex(ValueError, "Native support requires"):
                        constructor(**{**self.options, **overrides})

    def test_published_load_rejects_incompatible_configuration(self):
        model = NativeSupportModel({"synthetic": 9}, scorer_seed=7321, **self.options).eval()
        config = model.export_config()
        config["backbone_kwargs"]["interaction_mode"] = "gated"
        coupled = MultiDatasetVCASAHGCN({"synthetic": 9}, **self.options, interaction_mode="gated").eval()
        coupled.backbone.spectrum_score = deepcopy(model.backbone.spectrum_score)
        coupled.backbone.graph_score = deepcopy(model.backbone.graph_score)
        with TemporaryDirectory() as name:
            pin = self.package(Path(name), coupled, "selected", config=config)
            with patch.object(NativeSupportModel, "load_state_dict", side_effect=AssertionError("Loaded coupled state")):
                with self.assertRaisesRegex(ValueError, "Native support requires"):
                    load_endpoint(pin, self.registration, self.case, (self.baseline, None, None, None, {}))


if __name__ == "__main__":
    unittest.main()
