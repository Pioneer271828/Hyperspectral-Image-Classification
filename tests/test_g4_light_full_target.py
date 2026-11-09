"""Registered full-budget workers using synthetic inputs and real model updates."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import g4_light_full_target as full
from g4_light_target import ADAPTER_HEADS, CONTROLS, HEAD_ONLY, LightTargetRuntime
from g4_matched_target_study import METHODS, initialize
from joint_training import _tree_fingerprint, run_training, validation_epochs
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, write_json
from scripts.train_joint_m4 import graph_binding
from test_g4_matched_target_study import launch_fixture
from test_joint_training import checkpoints, hashes
from test_none_full_transfer import fixture as none_source
from test_target_training import fixture as target_fixture


def registration_fixture(root):
    code = code_snapshot()
    old = root / "reference"
    old.mkdir()
    input_dir = root / "input"
    input_dir.mkdir()
    (items, context, _), (archive, record, family) = target_fixture(input_dir)
    _, spec = none_source(root / "source")
    _, initial = initialize("xuzhou", 64, METHODS[1], "none", source_spec=spec)
    launch = launch_fixture(old)
    launch.update(code=code, phase="full_campaign", execution_enabled=True, optimizer_updates=81000,
                  short_acceptances={"none": {"root": str(root / "short-none")},
                                     "active": {"root": str(root / "short-active")}})
    for case in launch["cases"]:
        if case["target"] == "xuzhou":
            case.update(inputs={"observation": str(archive), "record": str(record), "record_sha256": file_hash(record),
                                "family": str(family), "family_sha256": file_hash(family)},
                        target_context=context, graph_binding=graph_binding(items))
        else:
            case["inputs"] = {"observation": str(input_dir / case["target"] / "observation.pt"),
                              "family": str(input_dir / case["target"] / "family.json")}
        if (case["target"], case["seed"], case["method"], case["mode"]) == ("xuzhou", 64, METHODS[1], "none"):
            case.update(initialization=initial, initial_state_sha256=initial["initial_state_sha256"], source_spec=spec)
    reference = old / "launch.json"
    write_json(reference, launch)
    reference_sha = file_hash(reference)
    engineering = root / "engineering.json"
    write_json(engineering, {"issue": 245, "status": "engineering_passed", "code": code,
        "reference_launch_sha256": reference_sha, "cpu_passed": True, "gpu_passed": True,
        "full_budget_recovery_passed": True, "real_training_updates": 0, "query_labels_accessed": False})
    plan = root / "synthetic-plan.md"
    plan.write_text("Synthetic fixture only")
    registration = {"policy": full.POLICY, "experiment_issue": 246, "producer_commit": code["commit"],
        "planned_optimizer_updates": 81000, "estimated_seconds": 60, "gpu_ids": [1, 3],
        "out_dir": str(root / "light"), "query_labels_during_training": False, "test_scoring_during_training": False,
        "reference_launch": {"path": str(reference), "sha256": reference_sha},
        "engineering_acceptance": {"path": str(engineering), "sha256": file_hash(engineering)},
        "analysis_plan": {"path": str(plan), "sha256": file_hash(plan)}}
    path = root / "registration.json"
    write_json(path, registration)
    return path, registration, code


class FullLightTargetTests(unittest.TestCase):
    def test_registration_budget_resources_owner_and_reference_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, registration, code = registration_fixture(root)
            reference_sha = registration["reference_launch"]["sha256"]
            with patch.object(full, "REFERENCE_LAUNCH_SHA256", reference_sha):
                self.assertEqual(full.read_registration(path, file_hash(path), code)[0], registration)
                for field, value in (("planned_optimizer_updates", 1500), ("gpu_ids", [1]),
                                     ("gpu_ids", [1, 1]), ("gpu_ids", [True, 3]),
                                     ("test_scoring_during_training", True), ("producer_commit", "bad")):
                    write_json(path, {**registration, field: value})
                    with self.assertRaises(ValueError):
                        full.read_registration(path, file_hash(path), code)
                write_json(path, registration)
                with self.assertRaises(ValueError):
                    full.read_registration(path, "0" * 64, code)
                with self.assertRaises(ValueError):
                    full.prepare_case(path, file_hash(path), code, ("xuzhou", 64, METHODS[1], "none"), device="cpu")
            self.assertFalse((root / "light").exists())
            out = root / "claimed"
            full.claim_output(out, "a" * 64)
            full.claim_output(out, "a" * 64)
            with self.assertRaises(ValueError):
                full.claim_output(out, "b" * 64)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                with patch.object(full.subprocess, "check_output", return_value="12288, 60\n"):
                    full.require_gpu(1)
                    with self.assertRaises(ValueError):
                        full.require_gpu(3)
                with patch.object(full.subprocess, "check_output", return_value="12287, 60\n"):
                    with self.assertRaises(RuntimeError):
                        full.require_gpu(1)

    def test_both_full_cases_resume_exact_budget_select_first_minimum_and_readonly_completion(self):
        device = os.environ.get("LIGHT_FULL_TEST_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, registration, code = registration_fixture(root)
            with patch.object(full, "REFERENCE_LAUNCH_SHA256", registration["reference_launch"]["sha256"]):
                for method in CONTROLS:
                    key = ("xuzhou", 64, method, "none")

                    def prepare():
                        return full.prepare_case(path, file_hash(path), code, key, device=device)

                    items, provenance, runtime, guard = prepare()
                    settings = runtime.settings
                    self.assertEqual(settings["optimizer_steps"], 1500)
                    self.assertEqual(validation_epochs(settings), list(range(0, 1499, 30)) + [1499])
                    self.assertEqual(settings["light_target_parameter_policy"]["optimizer_updates"], 1500)
                    with self.assertRaises(ValueError):
                        LightTargetRuntime(settings, items, runtime.directory, source_spec=runtime.source_spec)
                    bad = copy.deepcopy(settings)
                    bad["config"]["training"]["val_interval"] = 31
                    with self.assertRaises(ValueError):
                        runtime.validate(bad)
                    directory = runtime.directory
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    self.assertEqual(run_training(settings, items, directory, provenance, code, device=device,
                        stop_after=2, runtime_adapter=runtime, finish_guard=guard)["updates_this_call"], 2)
                    items, provenance, runtime, guard = prepare()
                    result = run_training(settings, items, directory, provenance, code, device=device, resume=True,
                        runtime_adapter=runtime, finish_guard=guard)
                    self.assertEqual(result["updates_this_call"], 1498)
                    state = checkpoints(directory)
                    self.assertEqual(len(state["trace"]), 1500)
                    self.assertEqual(len(state["history"]), 51)
                    self.assertTrue(all(r["frozen_parameters_unchanged"] and r["frozen_buffers_unchanged"]
                                        for r in state["trace"]))
                    selected = min(state["history"], key=lambda row: row["val_loss"])["step"]
                    self.assertEqual(selected, result["result"]["selected_step"])
                    self.assertEqual((directory / "feature-bank.pt").exists(), method == HEAD_ONLY)
                    self.assertTrue(all((r["adapter_changed_elements"] > 0) == (method == ADAPTER_HEADS)
                                        for r in state["trace"]))
                    raw = directory / f"raw-validation-{selected:06d}.pt"
                    before = hashes(directory)
                    model = runtime.load_selected(directory / "primary.pt", items, raw, device=device)
                    self.assertEqual(model.training_config["light_target_contract"]["policy"]["optimizer_updates"], 1500)
                    expected = torch.load(directory / "selected-probabilities.pt", weights_only=True)
                    self.assertEqual(_tree_fingerprint(runtime.validation_snapshot(model, items)[1]), _tree_fingerprint(expected))
                    self.assertEqual(run_training(settings, items, directory, provenance, code, device=device, resume=True,
                        runtime_adapter=runtime, finish_guard=guard)["updates_this_call"], 0)
                    self.assertEqual(before, hashes(directory))


if __name__ == "__main__":
    unittest.main()
