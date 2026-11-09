"""GPU-only full worker CLI exercise on bounded synthetic labels, never real data."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from g4_matched_target_audit import audit_case
from g4_matched_target_study import METHODS
from loadData.dataset_registry import file_hash
from research_runtime import ROOT, write_json
from test_g4_matched_target_audit import actual_case
from test_g4_matched_target_execution import registration_fixture
from test_g4_matched_target_study import launch_fixture


class MatchedWorkerCLITests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MATCHED_TARGET_DEVICE") == "cuda", "Explicit GPU fixture run only")
    def test_cli_registered_2_plus1498_readonly0_and_independent_full_case_oracle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items, actual, _ = actual_case(root, device="cuda")
            code, registration, rp = registration_fixture(root, out=root / "campaign")
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            self.assertTrue(visible.isdigit(), "One physical GPU must be explicitly selected for this fixture")
            registration["gpu"] = int(visible)
            write_json(rp, registration)
            out = Path(registration["out_dir"])
            launch = launch_fixture(out)
            launch.update(code=code, phase="full_campaign", execution_enabled=True, optimizer_updates=81000,
                registration_path=str(rp.resolve()), registration_sha256=file_hash(rp))
            for case in launch["cases"]:
                if case["target"] == "xuzhou":
                    case.update(inputs=copy.deepcopy(actual["inputs"]), target_context=copy.deepcopy(actual["target_context"]),
                                graph_binding=copy.deepcopy(actual["graph_binding"]))
                if (case["target"], case["seed"], case["method"], case["mode"]) == ("xuzhou", 64, METHODS[0], "none"):
                    actual["out"] = case["out"]
                    case.update(actual)
                    selected = case
            out.mkdir()
            lp = out / "launch.json"
            write_json(lp, launch)
            command = [sys.executable, str(ROOT / "scripts/train_g4_matched_target_full.py"),
                "--expected-commit", code["commit"], "--registration", str(rp), "--registration-sha256", file_hash(rp),
                "--launch-plan", str(lp), "--launch-sha256", file_hash(lp), "--target", "xuzhou", "--seed", "64",
                "--method", METHODS[0], "--mode", "none"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible, PYTHONDONTWRITEBYTECODE="1")
            folder = Path(selected["out"])
            for extra, expected in ((["--stop-after", "2"], ("interrupted", 2)),
                                    (["--resume"], ("completed", 1498)), (["--resume"], ("verified_completed", 0))):
                before = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()} if expected[1] == 0 else None
                result = subprocess.run(command + extra, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 0, result.stdout[-7000:])
                row = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual((row["operation"], row["updates_this_call"]), expected)
                if before is not None:
                    self.assertEqual(before, {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()})
            verified = audit_case(selected, code, file_hash(lp), file_hash(rp), device="cuda")
            self.assertEqual((verified["updates"], verified["head_metrics"], verified["class_recalls"], verified["exact_reload_repeats"]),
                             (1500, 6, 54, 3))
            del items
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
