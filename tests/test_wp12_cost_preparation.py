import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from evaluation.wp12_cost_preparation import prepare, runtime_admission
from joint_baseline import _item_fingerprint
from joint_training import _tree_fingerprint
from model.research_config import digest
from research_runtime import hash_file, write_json
from target_bn_calibration import GRAPH_POLICY, OBS_KEYS
from tests.test_joint_baseline import tiny_items
from tests.test_wp12_evidence import frozen_fixture


def fixture(root):
    analysis, receipt, freeze = frozen_fixture(root)
    launch_path = Path(freeze["training_root"]) / "launch.json"
    launch = json.loads(launch_path.read_text())
    source = dict(commit="s" * 40, dirty=False, files={"model.py": "1" * 64},
                  source_hash=digest({"model.py": "1" * 64}))
    freeze["producer"] = launch["code"] = source
    observations = root / "observations"
    observations.mkdir()
    fixtures = tiny_items()
    for scene, item in fixtures.items():
        payload = {k: item[k] for k in OBS_KEYS | {"name", "height", "width", "bands", "class_num"}}
        path = observations / (scene + ".pt")
        torch.save(payload, path)
        record = dict(policy=GRAPH_POLICY, target=scene, graph_fit_labels=0,
                      observation_sha256=hash_file(path), prepared_fields=_item_fingerprint(payload))
        rec_path = observations / (scene + ".json")
        write_json(rec_path, record)
        freeze["protected_files"][str(path)] = hash_file(path)
        freeze["boundary_files"][str(rec_path)] = hash_file(rec_path)
    for case, model in zip(launch["cases"], freeze["models"]):
        case["out"] = str(Path(freeze["training_root"]) / "cases" / case["name"])
        case["inputs"].update(observation=str(observations / (case["target"] + ".pt")),
                              record=str(observations / (case["target"] + ".json")),
                              record_sha256=hash_file(observations / (case["target"] + ".json")))
        case["target_context"] = dict(family_hash=model["family_hash"])
        model["case_sha256"] = digest(case)
    write_json(launch_path, launch)
    freeze["launch_sha256"] = hash_file(launch_path)
    freeze["boundary_files"][str(launch_path)] = hash_file(launch_path)
    freeze["protected_files"].update(freeze["boundary_files"])
    return analysis, receipt, freeze, launch, copy.deepcopy(source)


def invoke(values, out, **kwargs):
    return prepare(*values, out, analysis_sha256="a" * 64, freeze_sha256="f" * 64,
                   launch_sha256=values[2]["launch_sha256"], **kwargs)


class PreparationTests(unittest.TestCase):
    def test_full54_observation_only_and_source_difference_admission(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = fixture(root)
            values[4]["files"]["model.py"] = "2" * 64
            values[4]["source_hash"] = digest(values[4]["files"])
            old_load = torch.load
            loaded = []
            def load(path, **kwargs):
                loaded.append(str(path))
                self.assertTrue("observations" in str(path) or "prepared" in str(path))
                return old_load(path, **kwargs)
            with patch("torch.load", load), patch("torch.nn.Module._call_impl", side_effect=AssertionError("No inference")):
                result = invoke(values, root / "prepared")
            self.assertEqual(len(result["cases"]), 54)
            self.assertEqual(len(loaded), 6)
            self.assertEqual(result["runtime_admission"]["status"], "compatibility_pending")
            self.assertEqual(len(result["runtime_admission"]["differences"]), 1)
            self.assertFalse(result["executable_cost_jobs_created"])
            self.assertFalse(result["hardware_plan_registered"])
            for work in result["workloads"].values():
                payload = torch.load(work["file"]["path"], weights_only=True)
                self.assertEqual(set(payload), OBS_KEYS)
                self.assertEqual(_tree_fingerprint(payload), work["tensor_sha256"])
            with self.assertRaisesRegex(ValueError, "New external"):
                invoke(values, root / "prepared")

    def test_wrong_case_checkpoint_and_observation_pin_rejected(self):
        for defect in ("case", "checkpoint", "observation"):
            with self.subTest(defect=defect), TemporaryDirectory() as directory:
                root = Path(directory)
                values = fixture(root)
                if defect == "case":
                    values[3]["cases"][0]["seed"] = 999
                elif defect == "checkpoint":
                    values[0]["models"][0]["checkpoint_sha256"] = "0" * 64
                else:
                    path = Path(values[3]["cases"][0]["inputs"]["observation"])
                    with path.open("ab") as stream:
                        stream.write(b"changed")
                with self.assertRaises(ValueError):
                    invoke(values, root / "rejected")
                self.assertFalse((root / "rejected/preparation.json").exists())

    def test_label_payload_rejected_even_with_rebound_synthetic_pins(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = fixture(root)
            freeze, launch = values[2:4]
            inputs = launch["cases"][0]["inputs"]
            path, record_path = Path(inputs["observation"]), Path(inputs["record"])
            payload = torch.load(path, weights_only=True)
            payload["query_labels"] = torch.zeros(36)
            torch.save(payload, path)
            record = json.loads(record_path.read_text())
            record.update(observation_sha256=hash_file(path), prepared_fields=_item_fingerprint(payload))
            write_json(record_path, record)
            for case, model in zip(launch["cases"], freeze["models"]):
                if case["target"] == "paviaU":
                    case["inputs"]["record_sha256"] = hash_file(record_path)
                    model["case_sha256"] = digest(case)
            freeze["protected_files"][str(path)] = hash_file(path)
            freeze["boundary_files"][str(record_path)] = hash_file(record_path)
            freeze["protected_files"].update(freeze["boundary_files"])
            with self.assertRaisesRegex(ValueError, "include labels"):
                invoke(values, root / "rejected")
            self.assertFalse((root / "rejected/preparation.json").exists())

    def test_no_success_manifest_if_final_guard_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def fail():
                raise ValueError("Input changed")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                invoke(fixture(root), root / "failed", finish=fail)
            self.assertTrue((root / "failed/paviaU-observation.pt").exists())
            self.assertFalse((root / "failed/preparation.json").exists())

    def test_additions_allowed_changes_missing_files_and_dirty_refused(self):
        original = dict(commit="1" * 40, dirty=False, files={"model.py": "a" * 64},
                        source_hash=digest({"model.py": "a" * 64}))
        current = copy.deepcopy(original)
        current["files"]["collector.py"] = "b" * 64
        current["source_hash"] = digest(current["files"])
        self.assertEqual(runtime_admission(original, current)["status"], "original_files_preserved")
        current["files"].pop("model.py")
        current["source_hash"] = digest(current["files"])
        self.assertEqual(runtime_admission(original, current)["status"], "compatibility_pending")
        current["dirty"] = True
        with self.assertRaises(ValueError):
            runtime_admission(original, current)


if __name__ == "__main__":
    unittest.main()
