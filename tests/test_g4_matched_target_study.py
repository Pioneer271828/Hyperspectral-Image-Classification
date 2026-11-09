"""Matched54 semantics/identities and a bounded real-network full1500 execution."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from g4_matched_target_launch import accepted_short
from g4_matched_target_study import (EXPECTED, METHODS, MODES, SEEDS, STUDY_PATH, assert_matched_profiles,
    case_name, common_fresh_initials, grid, initialize, load_study, runtime, select_case, settings, training_contract,
    verify_fresh_target)
from joint_baseline import NAMES
from joint_training import _tree_fingerprint, run_training, schedule_for, validation_epochs
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research.reproduction.m4g4b.none_trajectory_compatibility import replay_lr
from research_runtime import code_snapshot, write_json
from test_joint_training import SOURCE, checkpoints, hashes
from test_target_training import fixture


def launch_fixture(root):
    proofs = [common_fresh_initials(t, s) for t in NAMES for s in SEEDS]
    cases = []
    for t, s, m, mode in grid():
        proof = next(p for p in proofs if (p["target"], p["seed"]) == (t, s))
        name = case_name(t, s, m, mode)
        initial = {"initial_state_sha256": proof["mode_initial_states"][mode],
                   "source": None if m == METHODS[0] else {"target": t, "seed": s, "mode": mode}}
        cases.append({"name": name, "target": t, "seed": s, "method": m, "mode": mode,
            "settings": settings(t, s, m, mode), "out": str(root / "cases" / name),
            "initialization": initial, "initial_state_sha256": initial["initial_state_sha256"],
            "fresh_state_sha256": proof["fresh_state_sha256"], "training_contract_sha256": assert_matched_profiles(t, s, m),
            "source_spec": {} if m == METHODS[0] else {"fixture": True}, "inputs": {"paired": t},
            "target_context": {"query_labels_accessed": False}, "graph_binding": {t: {"Q": "a" * 64, "A": "b" * 64}}})
    return {"study": load_study(), "study_file_sha256": file_hash(STUDY_PATH), "code": {**SOURCE, "dirty": False},
        "out_dir": str(root), "phase": "engineering_zero_update", "execution_enabled": False, "optimizer_updates": 0,
        "planned_full_optimizer_updates": 81000, "cases": cases, "common_initials": proofs,
        "query_labels_accessed": False, "test_scoring": False}


class MatchedTargetStudyTests(unittest.TestCase):
    def test_full54_common_semantics_initialization_and_manifest(self):
        self.assertEqual(load_study(), EXPECTED)
        self.assertEqual((len(grid()), len(set(grid()))), (54, 54))
        self.assertEqual(sum(settings(*key)["optimizer_steps"] for key in grid()), 81000)
        for t in NAMES:
            for s in SEEDS:
                proof = common_fresh_initials(t, s)
                for m in METHODS[:2]:
                    assert_matched_profiles(t, s, m)
                for mode in MODES:
                    model, _ = initialize(t, s, METHODS[0], mode)
                    verify_fresh_target(model, proof)
        value = settings("xuzhou", 64, METHODS[0], "none")
        for field in ("loss", "training"):
            bad = copy.deepcopy(value)
            bad["config"][field][next(iter(bad["config"][field]))] = "different"
            self.assertNotEqual(training_contract(bad), training_contract(value))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            bad = copy.deepcopy(EXPECTED)
            bad["planned_full_optimizer_updates"] += 1
            write_json(path, bad)
            with self.assertRaises(ValueError): load_study(path)

    def test_zero_update_preflight_cannot_be_executed_and_all54_pair_barriers(self):
        with tempfile.TemporaryDirectory() as temp:
            launch = launch_fixture(Path(temp))
            code = launch["code"]
            key = ("xuzhou", 64, METHODS[0], "none")
            self.assertEqual(select_case(launch, code, *key, phase="engineering_zero_update")["target"], "xuzhou")
            with self.assertRaises(ValueError): select_case(launch, code, *key, phase="full_campaign")
            for mutate in (lambda x: x["cases"].pop(), lambda x: x["cases"].__setitem__(-1, x["cases"][0]),
                lambda x: x.__setitem__("execution_enabled", True), lambda x: x.__setitem__("optimizer_updates", 81000),
                lambda x: x.__setitem__("query_labels_accessed", True),
                lambda x: x["cases"][0]["settings"].__setitem__("optimizer_steps", 12),
                lambda x: x["cases"][0].__setitem__("source_spec", {"forbidden": True}),
                lambda x: x["cases"][1].__setitem__("inputs", {"different": True}),
                lambda x: x["cases"][3]["initialization"]["source"].__setitem__("seed", 512)):
                bad = copy.deepcopy(launch)
                mutate(bad)
                with self.assertRaises(ValueError): select_case(bad, code, *key, phase="engineering_zero_update")

    def test_accepted_short_requires_complete_artifacts_and_unchanged_producer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = code_snapshot()
            rows, cases = [], []
            required = ("identity.json", "progress.json", "primary.pt", "primary.pt.json", "last.pt", "last.pt.json",
                        "selected-probabilities.pt", "trace.json", "history.json")
            for i in range(6):
                name = "fixture" + str(i)
                folder = root / "cases" / name
                folder.mkdir(parents=True)
                for f in required: (folder / f).write_text("synthetic hash-only short gate fixture")
                options = {"fixture": i}
                write_json(folder / "result.json", {"status": "completed", "optimizer_steps": 12,
                    "start_code": code, "settings": options, "artifacts": {n: file_hash(folder / n) for n in required}})
                cases.append({"name": name, "settings": options})
                rows.append({"name": name, "updates": 12, "head_metrics": 6, "class_recalls": 80, "exact_reload_repeats": 3})
            write_json(root / "launch.json", {"code": code, "cases": cases})
            write_json(root / "receipt.json", {"code": code, "status": "passed", "updates": 72, "cases": rows,
                "query_labels_accessed": False, "test_scoring": False})
            with patch.dict(EXPECTED, {"none_short_receipt_sha256": file_hash(root / "receipt.json")}):
                self.assertEqual(accepted_short(root, "none", code)["updates"], 72)
                bad = copy.deepcopy(code)
                bad["files"][next(iter(bad["files"]))] = "0" * 64
                bad["source_hash"] = digest(bad["files"])
                with self.assertRaises(ValueError): accepted_short(root, "none", bad)
                (root / "cases" / "fixture0" / "primary.pt").write_text("changed")
                with self.assertRaises(ValueError): accepted_short(root, "none", code)

    def test_full1500_actual_updates_51_events_resume_selection_lr_and_readonly(self):
        device = os.environ.get("MATCHED_TARGET_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (items, provenance, _), _ = fixture(root)
            opts = settings("xuzhou", 64, METHODS[0], "none", device=device)
            initial, receipt = initialize("xuzhou", 64, METHODS[0], "none")
            folder = root / "run"
            case = {"mode": "none", "settings": opts, "out": str(folder), "source_spec": {},
                    "initial_state_sha256": receipt["initial_state_sha256"]}
            adapter = runtime(case, items)
            first = run_training(opts, items, folder, provenance, SOURCE, device=device, runtime_adapter=adapter, stop_after=2)
            self.assertEqual(first["updates_this_call"], 2)
            result = run_training(opts, items, folder, provenance, SOURCE, device=device, runtime_adapter=adapter, resume=True)
            self.assertEqual(result["updates_this_call"], 1498)
            state = checkpoints(folder)
            self.assertEqual([h["step"] for h in state["history"]], list(range(1, 1472, 30)) + [1500])
            self.assertEqual(len(state["trace"]), len(schedule_for(opts, task_names=("xuzhou",))))
            self.assertEqual(state["best_step"], min(state["history"], key=lambda h: h["val_loss"])["step"])
            self.assertEqual(replay_lr(opts, state["history"], state["trace"]), [r["learning_rates"] for r in state["trace"]])
            before = hashes(folder)
            self.assertEqual(run_training(opts, items, folder, provenance, SOURCE, device=device,
                runtime_adapter=adapter, resume=True)["updates_this_call"], 0)
            self.assertEqual(before, hashes(folder))
            self.assertEqual(len(validation_epochs(opts)), 51)


if __name__ == "__main__":
    unittest.main()
