"""WP07 integration through the original optimizer, sampler and validation."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan
from tests.test_curriculum_exposure import freeze
from tests.test_int8_model import fixture
from transfer.development_training import prepare_scene
from transfer.g3_training import state_hash
from transfer.multiscale_run import MultiscaleRun, selector_group
from transfer.multiscale_training import MultiscaleTrial
from transfer import xuzhou_multiscale as wp07
from transfer.xuzhou_multiscale_audit import _exact


class XuzhouMultiscaleTests(unittest.TestCase):
    def setUp(self):
        self.base, prior, _ = fixture()
        spatial = make_tile_plan((84, 84), 28, 1, 2026)
        self.record = make_development(np.resize(np.arange(1, 10), (84, 84)), spatial, list(range(1, 10)))
        cube = np.random.default_rng(12).normal(size=(84, 84, 8)).astype(np.float32)
        self.scene = prepare_scene(cube, prior.bands, spatial, self.record)
        self.base.training_config = {"synthetic_seed": 512}
        self.plan = freeze(self.scene, self.record)

    def run_trial(self, trial):
        return wp07.new_run(self.base, self.scene, self.record, self.plan, trial)

    def test_all_modes_update_original_network_and_route_selectors(self):
        self.scene.bands = replace(self.scene.bands, wavelength_quality="verified",
                                   fwhm=(8.,) * 8, fwhm_unit="nm")
        self.plan = freeze(self.scene, self.record)
        before = state_hash(self.base)
        trials = [*wp07.ARMS.values(), MultiscaleTrial(graph_multiscale="fixed"),
                  MultiscaleTrial(graph_multiscale="fixed", graph_radius=2)]
        for mode in ("metadata", "joint"):
            trials.extend([MultiscaleTrial(cnn_multiscale=mode), MultiscaleTrial(graph_multiscale=mode),
                           MultiscaleTrial(cnn_multiscale=mode, graph_multiscale=mode)])
        trials.append(MultiscaleTrial(graph_multiscale="content"))
        outputs = {}
        for trial in trials:
            with self.subTest(trial=trial):
                run = self.run_trial(trial)
                for name, value in self.base.state_dict().items():
                    self.assertTrue(torch.equal(value, run.model.state_dict()[name]))
                row = run.step()
                run._validate_history_extensions(row)
                self.assertGreater(row["changed_parameters"]["cnn"], 0)
                self.assertGreater(row["changed_parameters"]["graph"], 0)
                self.assertGreater(row["gradient_l1"]["adapter"], 0)
                self.assertEqual(row["validation"]["count"], 45)
                self.assertEqual(len(run.timings["train_graph_build"]), 1)
                self.assertEqual(len(run.timings["validation_graph_build"]),
                                 sum(r["role"] == "devval" for r in self.record["roles"]))
                groups = {selector_group(name) for name in row["selectors"]}
                for group in groups:
                    subset = [item for name, item in row["selectors"].items() if selector_group(name) == group]
                    self.assertTrue(any(item["gradient_l1"] > 0 and item["changed"] > 0 for item in subset), group)
                outputs[trial] = row
        self.assertEqual(outputs[None], outputs[MultiscaleTrial()])
        self.assertEqual(outputs[None], outputs[MultiscaleTrial(graph_multiscale="fixed")])
        self.assertGreater(outputs[MultiscaleTrial(graph_multiscale="fixed", graph_radius=2)]["scales"]["graph_nnz"][0],
                           outputs[None]["scales"]["graph_nnz"][0])
        self.assertEqual(before, state_hash(self.base))

    def test_approximate_metadata_fallback_repeats_optimizer_and_predictions(self):
        for first, second in (("mean", "metadata"), ("content", "joint")):
            plain, conditioned = [self.run_trial(MultiscaleTrial(cnn_multiscale=mode, graph_multiscale=mode))
                                  for mode in (first, second)]
            for _ in range(2):
                a, b = plain.step(), conditioned.step()
                for key in ("loss", "validation", "gradient_l1", "changed_parameters", "scales"):
                    self.assertEqual(a[key], b[key])
                for name, evidence in b["selectors"].items():
                    if selector_group(name).endswith("_metadata"):
                        self.assertEqual(evidence, {"gradient_l1": None, "changed": 0})
                self.assertTrue(all(torch.equal(t, conditioned.model.state_dict()[k])
                                    for k, t in plain.model.state_dict().items()))
                for run in (plain, conditioned):
                    self.assertTrue(run.spec["multiscale"]["metadata"]["metadata_fallback"] == "mean")
                self.assertEqual(len(plain.optimizer.state), len(conditioned.optimizer.state))
                for name, param in plain.model.named_parameters():
                    other = dict(conditioned.model.named_parameters())[name]
                    self.assertTrue(_exact(plain.optimizer.state[param], conditioned.optimizer.state[other]))

    def test_strict_restore_rejects_wrong_arm_hash_source_and_budget(self):
        trial = wp07.ARMS["both_content"]
        run = self.run_trial(trial)
        run.advance(3)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "prefix.pt"
            options = dict(plan=self.plan, trial=trial, selector_seed=64, expected_sha256=run.save(path),
                           expected_contract_hash=run.contract_hash)
            restored = MultiscaleRun.load(self.base, self.scene, self.record, path, **options)
            self.assertEqual(run.step(), restored.step())
            self.assertTrue(_exact(run.optimizer.state_dict(), restored.optimizer.state_dict()))
            self.assertTrue(torch.equal(run.torch_rng, restored.torch_rng))
            for wrong in (dict(trial=wp07.ARMS["cnn_content"]), dict(expected_sha256="0" * 64),
                          dict(plan=freeze(self.scene, self.record, updates=25))):
                with self.assertRaises(ValueError):
                    MultiscaleRun.load(self.base, self.scene, self.record, path, **{**options, **wrong})
            with torch.no_grad():
                next(self.base.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError, "source state"):
                MultiscaleRun.load(self.base, self.scene, self.record, path, **options)

    def test_single_use_stage_and_cross_arm_budget_admission(self):
        run = self.run_trial(wp07.ARMS["off"])
        with TemporaryDirectory() as folder, patch.object(wp07, "BUDGET", self.plan.updates):
            root = Path(folder).resolve()
            registration = {"path": str(root / "registration.json"), "sha256": "a" * 64}
            settings = dict(value=dict(budget=self.plan.updates, contracts={"off": run.contract_hash}),
                            baseline=self.base, scene=self.scene, record=self.record, plan=self.plan)
            prefix = wp07.execute_stage(root, registration, "off", "prefix", None, **settings)
            with self.assertRaisesRegex(FileExistsError, "consumed"):
                wp07.execute_stage(root, registration, "off", "prefix", None, **settings)
            with self.assertRaisesRegex(ValueError, "Cross-arm"):
                wp07.execute_stage(root, registration, "cnn_fixed1", "resume", prefix, **settings)
            with self.assertRaisesRegex(ValueError, "budget"):
                wp07.execute_stage(root, registration, "off", "resume", prefix,
                                   **{**settings, "value": {"budget": self.plan.updates + 1}})


if __name__ == "__main__":
    unittest.main()
