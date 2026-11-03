"""C0 contract, ownership and real synthetic training tests; execute remotely only."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model.research_config import ResearchConfig
from model.checkpoint_io import load_model_checkpoint
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from research_ablation_presets import build_research_experiments
from research_runtime import RunRecord, code_snapshot, hash_file, parameter_counts, prepared_fingerprints
from run_research_ablation import execute_one, main, plan_experiments
from test_interaction_modules import synthetic_item
import train_vca_multidataset as train


torch.set_num_threads(1)


def tiny_config():
    return ResearchConfig.from_dict({"name": "synthetic_c0", "device": "cpu",
        "model": {"latent_channels": 8, "adapter_embed_dim": 8, "norm_type": "group_norm", "graph_mode": "learned"},
        "interaction": {"interaction_mode": "gated", "graph_update_mode": "joint"},
        "data": {"datasets": ["demo"]}, "training": {"epochs": 0, "steps_per_epoch": 1, "val_interval": 1}})


class ConfigTests(unittest.TestCase):
    def test_defaults_match_legacy_and_round_trip(self):
        legacy = ResearchConfig.from_legacy(train.build_parser().parse_args(["--device", "cpu"]))
        self.assertEqual(legacy.to_dict(), replace(ResearchConfig(), name="legacy").to_dict())
        config = tiny_config()
        self.assertEqual(config.to_dict(), ResearchConfig.from_dict(config.to_dict()).to_dict())
        restored = ResearchConfig.from_legacy(config.trainer_args("/tmp/model.pt", "/tmp/metrics.json"))
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(config.budget["optimizer_steps"], 1)
        self.assertEqual(ResearchConfig().budget["optimizer_steps"], 1503)

    def test_reject_unknown_invalid_nonfinite_and_unimplemented(self):
        cases = [{"unknown": True}, {"model": {"residual": True}}, {"schema_version": 2},
                 {"model": {"latent_channels": 0}}, {"model": {"adapter_embed_dim": True}},
                 {"model": {"graph_mode": "future"}}, {"model": {"wavelength_max": 300}},
                 {"interaction": {"interaction_direction": "wrong"}},
                 {"interaction": {"interaction_hidden_dim": 1.2}},
                 {"interaction": {"graph_update_mix": float("nan")}},
                 {"loss": {"cnn_loss_weight": float("inf")}}, {"training": {"lr": 0}},
                 {"training": {"epochs": -1}}, {"training": {"seed": -1}},
                 {"training": {"eval_checkpoint": "test"}}, {"data": {"datasets": ["demo", "demo"]}},
                 {"data": {"protocol": "spatial_disjoint"}}, {"data": {"train_ratio": 0.999}},
                 {"augmentation": {"train_drop_band_ratio": 1.0}}]
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                ResearchConfig.from_dict(item)
        with patch.object(train, "prepare_dataset") as prepare, self.assertRaises(ValueError):
            train.main(["--device", "cpu", "--epochs", "-1"])
        prepare.assert_not_called()

    def test_hashes_change_with_effective_fields(self):
        config = tiny_config()
        self.assertNotEqual(config.config_hash, replace(config, training=replace(config.training, seed=1)).config_hash)
        self.assertNotEqual(config.config_hash, replace(config, loss=replace(config.loss, cnn_loss_weight=0.0)).config_hash)
        self.assertEqual(config.config_hash, ResearchConfig.from_dict(json.loads(json.dumps(config.to_dict()))).config_hash)

    def test_presets_reset_switches_and_reject_duplicates(self):
        configs = build_research_experiments(tiny_config(), ["all"], [0, 2026])
        self.assertEqual(len(configs), 20)
        self.assertEqual(len({c.config_hash for c in configs}), 20)
        baseline = configs[0]
        self.assertEqual(baseline.interaction.interaction_mode, "none")
        self.assertEqual(baseline.interaction.graph_update_mode, "none")
        for names, seeds in [(["missing"], [0]), (["all", "configured"], [0]), (["configured"], [0, 0]), ([], [0])]:
            with self.assertRaises(ValueError):
                build_research_experiments(tiny_config(), names, seeds)


class FactoryTests(unittest.TestCase):
    def test_factory_matches_original_constructor_and_checkpoint_shape(self):
        config = tiny_config()
        torch.manual_seed(31)
        direct = MultiDatasetVCASAHGCN({"demo": 3}, **config.model_kwargs())
        torch.manual_seed(31)
        factory = config.build_model({"demo": 3})
        for key, value in direct.state_dict().items():
            self.assertTrue(torch.equal(value, factory.state_dict()[key]), key)
        self.assertEqual(parameter_counts(factory)["total"], sum(p.numel() for p in factory.parameters()))

    def test_disabled_plugins_have_no_parameters_or_common_initialization_change(self):
        baseline, full = build_research_experiments(tiny_config(), ["interaction_baseline", "interaction_full"], [7])
        torch.manual_seed(7)
        base_model = baseline.build_model({"demo": 3})
        torch.manual_seed(7)
        full_model = full.build_model({"demo": 3})
        self.assertIsNone(base_model.backbone.interaction)
        self.assertIsNone(base_model.backbone.edge_updater)
        self.assertEqual(parameter_counts(base_model)["research_parameters"], {"M6": 0, "M7": 0})
        for key, value in base_model.state_dict().items():
            self.assertTrue(torch.equal(value, full_model.state_dict()[key]), key)
        self.assertTrue(all(v > 0 for v in parameter_counts(full_model)["research_parameters"].values()))


class ProvenanceTests(unittest.TestCase):
    def test_data_and_supervision_fingerprints_are_separate(self):
        item = synthetic_item()
        initial = prepared_fingerprints({"demo": item})["demo"]
        item["net_input"][0, 0, 0] += 1
        changed = prepared_fingerprints({"demo": item})["demo"]
        self.assertNotEqual(initial["data_hash"], changed["data_hash"])
        self.assertEqual(initial["split_hash"], changed["split_hash"])
        item["train_label_mask"][0] = 0
        changed_again = prepared_fingerprints({"demo": item})["demo"]
        self.assertNotEqual(changed["split_hash"], changed_again["split_hash"])

    def test_plan_is_data_free_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = Path(root) / "config.json"
            config_path.write_text(json.dumps(tiny_config().to_dict()), encoding="utf-8")
            output = Path(root) / "plan"
            argv = ["--config", str(config_path), "--out-root", str(output), "--experiments", "all",
                    "--seeds", "0,2026", "--dry-run"]
            with patch.object(train, "prepare_dataset") as prepare, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
                self.assertEqual(main(argv + ["--resume"]), 0)
            prepare.assert_not_called()
            plan = json.loads((output / "plan.json").read_text())
            self.assertEqual(len(plan["entries"]), 20)
            self.assertEqual(len(list(output.glob("*/attempts"))), 0)
            with self.assertRaises(ValueError):
                plan_experiments([tiny_config()], output, code_snapshot(), resume=True)

    def test_run_claim_and_retry_preserve_failure(self):
        config, code = tiny_config(), code_snapshot()
        with tempfile.TemporaryDirectory() as root:
            record = RunRecord(root, config, code)
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with record:
                    with self.assertRaises(FileExistsError):
                        with RunRecord(root, config, code, resume=True):
                            pass
                    raise RuntimeError("intentional")
            old = json.loads((record.attempt / "manifest.json").read_text())
            self.assertEqual(old["status"], "failed")
            with self.assertRaises(FileExistsError):
                with RunRecord(root, config, code):
                    pass
            self.assertFalse(record.lock.exists())


class IntegrationTests(unittest.TestCase):
    def test_train_record_strict_reload_and_verified_skip(self):
        config, code, item = tiny_config(), code_snapshot(), synthetic_item()
        with tempfile.TemporaryDirectory() as root, patch.object(train, "prepare_dataset", return_value=item):
            result = execute_one(config, root, code)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["budget"]["optimizer_steps"], 1)
            directory = Path(root) / result["run_id"] / "attempts" / result["attempt"]
            loaded = load_model_checkpoint(directory / "model.pt")
            self.assertEqual(loaded.training_config["research_config"], config.to_dict())
            self.assertEqual(loaded.training_config["research_run"]["run_id"], result["run_id"])
            expected = config.build_model({"demo": 3})
            expected.load_state_dict(loaded.state_dict(), strict=True)
            expected.eval()
            with torch.no_grad():
                for a, b in zip(loaded("demo", item["net_input"], wavelengths=item["wavelengths"]),
                                expected("demo", item["net_input"], wavelengths=item["wavelengths"])):
                    self.assertTrue(torch.allclose(a, b))
            old_hash = hash_file(directory / "model.pt")
            with patch.object(torch.optim, "Adam", side_effect=AssertionError("must not train")):
                resumed = execute_one(config, root, code, resume=True)
            self.assertEqual(resumed["status"], "skipped")
            self.assertEqual(old_hash, hash_file(directory / "model.pt"))
            self.assertNotEqual(result["attempt"], resumed["attempt"])

    def test_changed_data_and_corrupt_artifact_cannot_be_silently_skipped(self):
        config, code, item = tiny_config(), code_snapshot(), synthetic_item()
        with tempfile.TemporaryDirectory() as root, patch.object(train, "prepare_dataset", return_value=item):
            result = execute_one(config, root, code)
            original = item["net_input"].clone()
            item["net_input"][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "Data or split changed"):
                execute_one(config, root, code, resume=True)
            item["net_input"] = original
            directory = Path(root) / result["run_id"] / "attempts" / result["attempt"]
            (directory / "metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
                execute_one(config, root, code, resume=True)
            states = [json.loads(p.read_text())["status"] for p in sorted((directory.parent).glob("*/manifest.json"))]
            self.assertEqual(states, ["completed", "failed", "failed"])

    def test_failure_retry_creates_new_attempt_and_retains_log(self):
        config, code, item = tiny_config(), code_snapshot(), synthetic_item()
        with tempfile.TemporaryDirectory() as root:
            with patch.object(train, "prepare_dataset", side_effect=FileNotFoundError("intentional missing data")):
                with self.assertRaises(FileNotFoundError):
                    execute_one(config, root, code)
            first = next(Path(root).glob("*/attempts/0001"))
            old_log = (first / "training.log").read_text()
            self.assertIn("intentional missing data", old_log)
            with patch.object(train, "prepare_dataset", return_value=item):
                result = execute_one(config, root, code, resume=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["attempt"], "0002")
            self.assertEqual((first / "training.log").read_text(), old_log)


if __name__ == "__main__":
    unittest.main()
