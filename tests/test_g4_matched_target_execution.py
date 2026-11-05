"""Full execution boundary tests; synthetic acceptances never authorize real training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from g4_matched_target_execution import (REGISTRATION_POLICY, freeze_campaign, read_full_case, read_registration)
from g4_matched_target_study import METHODS, STUDY_PATH
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research_runtime import code_snapshot, write_json
from test_g4_matched_target_study import launch_fixture


def registration_fixture(root, *, preflight=None, out=None):
    code = code_snapshot()
    preflight = preflight or {"root": str(root / "old"), "receipt_sha256": "a" * 64, "protected_sha256": "b" * 64}
    acceptance = root / "synthetic-engineering.json"
    plan = root / "synthetic-analysis-plan.md"
    plan.write_text("Synthetic fixed metrics/protection plan fixture, never a real experiment registration.")
    write_json(acceptance, {"status": "engineering_passed", "issue": 182, "code": code,
        "cpu_passed": True, "gpu_passed": True, "full_regression_passed": True,
        "worker_campaign_audit_contracts_passed": True, "query_labels_accessed": False,
        "full_training_completed": False, "preflight_receipt_sha256": preflight["receipt_sha256"]})
    value = {"policy": REGISTRATION_POLICY, "experiment_issue": 184, "producer_commit": code["commit"],
        "study_file_sha256": file_hash(STUDY_PATH), "planned_optimizer_updates": 81000, "gpu": 0,
        "estimated_seconds": 1000, "preflight": preflight, "out_dir": str(out or root / "new"),
        "engineering_acceptance": {"path": str(acceptance), "sha256": file_hash(acceptance)},
        "analysis_plan": {"path": str(plan), "sha256": file_hash(plan)},
        "query_labels_during_training": False, "test_scoring_during_training": False}
    path = root / "synthetic-registration.json"
    write_json(path, value)
    return code, value, path


class MatchedExecutionTests(unittest.TestCase):
    def test_separate_registration_requires_complete_engineering_budget_permissions_and_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code, original, path = registration_fixture(root)
            self.assertEqual(read_registration(path, file_hash(path), code)["planned_optimizer_updates"], 81000)
            for key, changed in (("planned_optimizer_updates", 1500), ("experiment_issue", 182), ("gpu", True),
                                 ("estimated_seconds", -1), ("query_labels_during_training", True),
                                 ("producer_commit", "0" * 40), ("extra_switch", True)):
                value = copy.deepcopy(original)
                value[key] = changed
                write_json(path, value)
                with self.assertRaises(ValueError): read_registration(path, file_hash(path), code)
            write_json(path, original)
            bad = copy.deepcopy(code)
            bad["files"][next(iter(bad["files"]))] = "0" * 64
            bad["source_hash"] = digest(bad["files"])
            with self.assertRaises(ValueError): read_registration(path, file_hash(path), bad)
            acceptance = Path(original["engineering_acceptance"]["path"])
            record = json.loads(acceptance.read_text())
            record["status"] = "engineering_zero_update_passed"
            write_json(acceptance, record)
            original["engineering_acceptance"]["sha256"] = file_hash(acceptance)
            write_json(path, original)
            with self.assertRaises(ValueError): read_registration(path, file_hash(path), code)

    def test_worker_boundary_rejects_engineering_phase_and_registration_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code, registration, path = registration_fixture(root)
            out = Path(registration["out_dir"])
            launch = launch_fixture(out)
            launch.update(code=code, registration_path=str(path.resolve()), registration_sha256=file_hash(path))
            out.mkdir()
            lp = out / "launch.json"
            write_json(lp, launch)
            key = ("xuzhou", 64, METHODS[0], "none")
            with self.assertRaises(ValueError): read_full_case(lp, file_hash(lp), path, file_hash(path), code, key)
            launch.update(phase="full_campaign", execution_enabled=True, optimizer_updates=81000)
            write_json(lp, launch)
            case, _ = read_full_case(lp, file_hash(lp), path, file_hash(path), code, key)
            self.assertEqual(case["settings"]["optimizer_steps"], 1500)
            with self.assertRaises(ValueError): read_full_case(lp, "0" * 64, path, file_hash(path), code, key)
            launch["registration_sha256"] = "0" * 64
            write_json(lp, launch)
            with self.assertRaises(ValueError): read_full_case(lp, file_hash(lp), path, file_hash(path), code, key)

    def test_fresh_campaign_must_not_write_under_preflight_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "old"
            old.mkdir()
            launch = launch_fixture(old)
            code = code_snapshot()
            launch["code"] = code
            launch["short_acceptances"] = {g: {"root": str(root / (g + "-short"))} for g in ("none", "active")}
            for case in launch["cases"]:
                case["inputs"] = {"observation": str(root / "observations" / case["target"] / "observation.pt"),
                                  "family": str(root / "families" / case["target"] / "family.json")}
            write_json(old / "launch.json", launch)
            write_json(old / "protected-files.json", {})
            write_json(old / "receipt.json", {"status": "engineering_zero_update_passed", "optimizer_updates": 0,
                "execution_enabled": False, "full_training_completed": False, "cases": 54, "common_initial_proofs": 9,
                "launch_sha256": file_hash(old / "launch.json"), "code": code,
                "query_labels_accessed": False, "test_scoring": False})
            preflight = {"root": str(old), "receipt_sha256": file_hash(old / "receipt.json"),
                         "protected_sha256": file_hash(old / "protected-files.json")}
            _, _, path = registration_fixture(root, preflight=preflight, out=old / "forbidden-run")
            with self.assertRaises(ValueError): freeze_campaign(path, file_hash(path), code)
            self.assertFalse((old / "forbidden-run").exists())


if __name__ == "__main__":
    unittest.main()
