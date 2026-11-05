"""Full launch requires immutable real short artifacts and explicit resource gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g4_none_source_launch import REFERENCE_LAUNCH, STUDY_PATH, grid
from g4_none_ordered_source import load_study, options
from joint_training import load_plan
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research.reproduction.m4g4b.none_source_campaign import accepted_short, wait_gpu
from research_runtime import write_json


class NoneSourceCampaignTests(unittest.TestCase):
    def test_short_acceptance_additive_producer_and_artifact_barrier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            producer = {"dirty": False, "commit": "fixture", "files": {}, "source_hash": digest({})}
            launch = {"out_dir": str(root), "code": producer, "study": load_study(STUDY_PATH),
                "study_file_sha256": file_hash(STUDY_PATH), "profile": "short", "expected_updates": 24,
                "execution_device": "cuda", "reference_launch_sha256": REFERENCE_LAUNCH,
                "prepared_manifest_sha256": load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"],
                "query_labels_accessed": False, "test_scoring": False, "cases": []}
            rows = []
            for held, seed in grid(False):
                settings, name = options(held, seed), held + "-s" + str(seed)
                folder = root / "cases" / name
                folder.mkdir(parents=True)
                artifact = folder / "primary.pt"
                artifact.write_bytes(b"fixture")
                result = {"status": "completed", "optimizer_steps": 8, "start_code": producer,
                          "settings": settings, "artifacts": {"primary.pt": file_hash(artifact)}}
                write_json(folder / "result.json", result)
                launch["cases"].append({"held_out": held, "seed": seed, "name": name, "settings": settings,
                    "out": str(folder), "source_context": {"held_out": held, "sources": settings["source_tasks"],
                        "target_payload_accessed": False}, "graph_binding": dict.fromkeys(settings["source_tasks"], {}),
                    "shared_initialization_equal": True, "common_state_tensors": 1, "initial_state_sha256": "a" * 64,
                    "common_initial_sha256": dict.fromkeys(("capacity_control", "x_residual"), "a" * 64)})
                rows.append({"name": name, "updates": 8, "metrics": 12, "recalls": {"paviaU": 186,
                    "xuzhou": 186, "whu-honghu": 108}[held], "exact_reload_repeats": 3,
                    "result_sha256": file_hash(folder / "result.json")})
            write_json(root / "launch.json", launch)
            receipt = {"status": "passed", "code": producer, "launch_sha256": file_hash(root / "launch.json"),
                       "optimizer_updates": 24, "query_labels_accessed": False, "test_scoring": False, "cases": rows}
            rp = root / "receipt.json"
            write_json(rp, receipt)
            sha = file_hash(rp)
            self.assertEqual(accepted_short(root, sha, producer), receipt)
            bad = copy.deepcopy(producer)
            bad["dirty"] = True
            with self.assertRaises(ValueError): accepted_short(root, sha, bad)
            artifact.write_bytes(b"changed")
            with self.assertRaises(ValueError): accepted_short(root, sha, producer)
            with self.assertRaises(ValueError): accepted_short(root, "0" * 64, producer)

    def test_single_gpu_queue_waits_at_fixed_memory_and_utilization_gates(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
            with self.assertRaises(ValueError): wait_gpu(2, Path(temp))
            with patch("subprocess.check_output", side_effect=["12287, 0", "24000, 61", "12288, 60"]), patch("time.sleep") as sleep:
                wait_gpu(1, Path(temp))
                self.assertEqual(sleep.call_count, 2)
                self.assertEqual(sleep.call_args.args, (30,))
                self.assertEqual(json.loads((Path(temp) / "status.json").read_text())["status"], "waiting_for_resources")


if __name__ == "__main__":
    unittest.main()
