"""Registered CPU segments use only the isolated development package."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.wp11_confirmation import HISTORY_POLICY, unused_tiles
from model.checkpoint_io import save_model_checkpoint
from research_runtime import code_snapshot, hash_file, write_json
from scripts.run_wp11_light import execute, plan
from tests.test_int8_model import fixture
from transfer.g3_training import state_hash
from transfer.light_head_training import LightHeadRun
from transfer.observation_policy import scene_identity
from transfer.registered_development import export_development


def pin(path):
    return dict(path=str(path), sha256=hash_file(path))


def save(root, name, value):
    path = root / f"{name}.json"
    write_json(path, value)
    return pin(path)


def inputs(root):
    full, scene, record = fixture()
    model = root / "full.pt"
    save_model_checkpoint(full, model)
    development = export_development(scene, record, root / "development")
    evidence = save(root, "evidence", dict(scope="synthetic"))
    train = sorted(t["id"] for t in record["roles"] if t["role"] == "devtrain")
    selection = sorted(t["id"] for t in record["roles"] if t["role"] == "devval")
    history = dict(policy=HISTORY_POLICY, scope="all_same_scene_ancestry", full_state=state_hash(full),
                   scene_identity=scene_identity(scene, record), history_evidence_sha256=evidence["sha256"],
                   tile_usage=dict(fit=train, training=train, selection=selection))
    spec = dict(development=development, full_weights=pin(model), full_metadata=pin(root / "full.pt.json"),
                full_history=save(root, "history", history), history_evidence=evidence,
                training=dict(updates=3, seed=64, lr=0.0001, class_balance=True))
    return spec, unused_tiles(scene.plan, record)


class RegisteredLightTests(unittest.TestCase):
    def test_plan_resume_publish_and_completed_zero_updates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = inputs(root)
            code = code_snapshot()
            with patch.object(LightHeadRun, "step", side_effect=AssertionError("planning stepped")):
                registered = plan(save(root, "spec", spec), root / "plan", code)
            self.assertEqual((registered["optimizer_updates"], registered["query"]), (0, 0))
            reg = pin(root / "plan/registration.json")
            direct = execute(reg, root / "direct", code, to_step=3)
            partial = execute(reg, root / "partial", code, to_step=1)
            self.assertEqual((partial["status"], partial["optimizer_updates"]), ("partial", 1))
            self.assertFalse((root / "partial/light.pt").exists())
            resume = pin(root / "partial/training.pt")
            split = execute(reg, root / "split", code, to_step=3, resume=resume)
            done = execute(reg, root / "done", code, to_step=3, resume=pin(root / "split/training.pt"))
            self.assertEqual([v["optimizer_updates"] for v in (direct, split, done)], [3, 2, 0])
            for other in (split, done):
                for key in ("source_state", "head_state", "history_hash", "selected_step", "contract_hash"):
                    self.assertEqual(direct[key], other[key])
                self.assertEqual(direct["selected"]["light_state"], other["selected"]["light_state"])
                self.assertEqual(other["query"], 0)
            with self.assertRaisesRegex(ValueError, "absolute step"):
                execute(reg, root / "over-budget", code, to_step=4, resume=resume)
            with self.assertRaisesRegex(ValueError, "New external output"):
                execute(reg, root / "partial", code, to_step=1, resume=resume)

    def test_exposure_and_bad_evidence_rejected_before_updates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec, held_aside = inputs(root)
            code = code_snapshot()
            history = json.loads(Path(spec["full_history"]["path"]).read_text())
            history["tile_usage"]["training"] = sorted(history["tile_usage"]["training"] + held_aside[:1])
            altered = dict(spec, full_history=save(root, "altered-history", history))
            with self.assertRaisesRegex(ValueError, "footprint escapes"):
                plan(save(root, "altered", altered), root / "bad", code)
            other = dict(spec, history_evidence=save(root, "other-evidence", {}))
            with self.assertRaisesRegex(ValueError, "evidence differs"):
                plan(save(root, "other", other), root / "bad", code)
            self.assertFalse((root / "bad").exists())
