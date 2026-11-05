"""Independent complete-case oracle and full54 boundary tests on synthetic inputs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import savemat
from g4_matched_target_audit import ACCEPTANCE_POLICY, audit_case, full_acceptance, inspect_case
from g4_matched_target_evaluation import QUERY_POLICY, new_output, paired_summary, read_query_labels, require_frozen_boundary
from g4_matched_target_study import EXPECTED, METHODS, case_name, grid, initialize, runtime, settings
from joint_training import run_training
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research.reproduction.m4g4b.matched_target_campaign import run_queue
from research_runtime import code_snapshot, write_json
from scripts.train_joint_m4 import graph_binding
from test_joint_training import hashes
from test_target_training import fixture
from test_g4_matched_target_study import launch_fixture


def actual_case(root, *, device="cpu"):
    target = root / "target"
    target.mkdir()
    (items, context, _), paths = fixture(target)
    initial, receipt = initialize("xuzhou", 64, METHODS[0], "none")
    case = {"name": case_name("xuzhou", 64, METHODS[0], "none"), "target": "xuzhou", "seed": 64,
        "method": METHODS[0], "mode": "none", "out": str(root / "run"), "source_spec": {},
        "settings": settings("xuzhou", 64, METHODS[0], "none", device=device), "target_context": context,
        "initialization": receipt, "initial_state_sha256": receipt["initial_state_sha256"], "graph_binding": graph_binding(items),
        "inputs": {"observation": str(paths[0]), "record": str(paths[1]), "record_sha256": file_hash(paths[1]),
                   "family": str(paths[2]), "family_sha256": file_hash(paths[2])}}
    provenance = copy.deepcopy(context)
    provenance.update(source_spec={}, m4_mode="none", matched_full_target_launch={"protocol": EXPECTED["protocol"],
        "launch_sha256": "a" * 64, "registration_sha256": "b" * 64, "case": case["name"],
        "initial_state_sha256": receipt["initial_state_sha256"]})
    del initial
    return items, case, provenance


def boundary_fixture(root):
    models, checks, pins = [], [], {}
    for t, s, method, mode in grid():
        name = case_name(t, s, method, mode)
        folder = root / "cases" / name
        folder.mkdir(parents=True)
        pair = {}
        for n, f in (("result.json", "result_sha256"), ("primary.pt", "primary_sha256"), ("primary.pt.json", "sidecar_sha256")):
            path = folder / n
            path.write_text("Synthetic frozen-boundary hash fixture, not trained weights.")
            pins[str(path)] = pair[f] = file_hash(path)
        models.append({"name": name, "target": t, "seed": s, "method": method, "mode": mode, **pair})
        checks.append({"name": name, "status": "passed", "updates": 1500, "validation_events": 51,
            "head_metrics": 6, "class_recalls": 6 * DATASETS[t][2], "exact_reload_repeats": 3, **pair})
    return {"policy": QUERY_POLICY, "status": "all54_frozen_before_query", "study": EXPECTED,
        "models": models, "independent_checks": checks, "training_root": str(root),
        "training_optimizer_updates": 81000, "new_optimizer_updates": 0, "query_labels_accessed": False,
        "protected_files": pins, "boundary_files": dict(pins)}


class MatchedAuditTests(unittest.TestCase):
    def test_final_synthetic_label_release_requires_frozen_family_and_actual_gt_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, case, _ = actual_case(root)
            frozen = boundary_fixture(root)
            case["out"] = str(root / "cases" / case["name"])
            row = next(m for m in frozen["models"] if m["name"] == case["name"])
            row.update(case_sha256=digest(case), family_hash=case["target_context"]["family_hash"])
            data = root / "data"
            data.mkdir()
            labels = (np.arange(18 * 18).reshape(18, 18) % 9 + 1).astype(np.int32)
            path = data / "fixture_gt.mat"
            savemat(path, {DATASETS["xuzhou"][1] + "_gt": labels})
            manifest = root / "prepared.json"
            write_json(manifest, {"protocol": "acc1p_joint_prepared_v1", "datasets": {"xuzhou": {"raw": {
                "class_count": 9, "label_path": path.name, "label_sha256": file_hash(path)}}}})
            with patch("g4_matched_target_evaluation.PREPARATION_HASH", file_hash(manifest)):
                release, pins = read_query_labels(frozen, case, data, manifest, file_hash(manifest))
                self.assertEqual(len(release["indices"]), 18 * 18 - 9 * 15)
                self.assertEqual(release["checkpoint_sha256"], row["primary_sha256"])
                self.assertEqual(pins[str(path)], file_hash(path))
                path.write_bytes(path.read_bytes() + b"changed")
                with patch("scipy.io.loadmat", side_effect=AssertionError("Changed GT must be rejected before load")) as load:
                    with self.assertRaises(ValueError): read_query_labels(frozen, case, data, manifest, file_hash(manifest))
                    load.assert_not_called()

    def test_query_output_must_be_new_sibling_outside_input_trees(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launch = {"out_dir": str(root / "training"), "short_acceptances": {"none": {"root": str(root / "short")}},
                "cases": [{"inputs": {"observation": str(root / "data" / "observation.pt"),
                    "family": str(root / "labels" / "family.json")}, "source_spec": {}}]}
            self.assertEqual(new_output(root / "query", launch), root / "query")
            for path in (root / "training" / "new", root / "data" / "new", root / "short", root):
                with self.assertRaises(ValueError): new_output(path, launch)

    def test_all54_training_acceptance_rejects_per_case_coverage_hidden_by_totals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launch = launch_fixture(root)
            code = code_snapshot()
            launch.update(code=code, phase="full_campaign", execution_enabled=True, optimizer_updates=81000,
                          registration_sha256="b" * 64)
            write_json(root / "launch.json", launch)
            write_json(root / "status.json", {"status": "completed"})
            write_json(root / "protected-files.json", {str(root / "launch.json"): file_hash(root / "launch.json")})
            frozen = boundary_fixture(root)
            rows = [{**r, "completed_resume_updates": 0} for r in frozen["independent_checks"]]
            accepted = {"policy": ACCEPTANCE_POLICY, "status": "passed", "code": code,
                "launch_sha256": file_hash(root / "launch.json"), "registration_sha256": "b" * 64,
                "optimizer_updates": 81000, "models": rows, "query_labels_accessed": False, "test_scoring": False,
                "protected_manifest_sha256": file_hash(root / "protected-files.json")}
            ap = root / "training-acceptance.json"
            write_json(ap, accepted)
            # The complete-state oracle is exercised by the actual1500 test;
            # this fixture isolates global completeness and per-role count gates.
            with patch("g4_matched_target_audit.inspect_case", return_value=(None, None, None, None, None, {})):
                self.assertEqual(len(full_acceptance(root, file_hash(root / "launch.json"), file_hash(ap), code)[1]["models"]), 54)
                bad = copy.deepcopy(accepted)
                first = next(r for r in bad["models"] if r["class_recalls"] == 54)
                second = next(r for r in bad["models"] if r["class_recalls"] == 132)
                first["class_recalls"], second["class_recalls"] = 132, 54
                write_json(ap, bad)
                with self.assertRaises(ValueError): full_acceptance(root, file_hash(root / "launch.json"), file_hash(ap), code)
                write_json(ap, accepted)
                Path(next(p for p in frozen["boundary_files"] if p.endswith("primary.pt"))).write_bytes(b"changed")
                with self.assertRaises(ValueError): full_acceptance(root, file_hash(root / "launch.json"), file_hash(ap), code)

    def test_full1500_case_independent_metrics_exact_reloads_provenance_and_artifact_barriers(self):
        device = os.environ.get("MATCHED_TARGET_DEVICE", "cpu")
        with tempfile.TemporaryDirectory() as temp:
            items, case, provenance = actual_case(Path(temp), device=device)
            code = code_snapshot()
            run_training(case["settings"], items, case["out"], provenance, code, device=device, runtime_adapter=runtime(case, items))
            before = hashes(Path(case["out"]))
            row = audit_case(case, code, "a" * 64, "b" * 64, device=device)
            self.assertEqual((row["updates"], row["validation_events"], row["head_metrics"], row["class_recalls"], row["exact_reload_repeats"]),
                             (1500, 51, 6, 54, 3))
            self.assertTrue(all(v > 0 for v in row["changed_parameter_elements"].values()))
            self.assertEqual(hashes(Path(case["out"])), before)
            with self.assertRaises(ValueError): inspect_case(case, code, "c" * 64, "b" * 64)
            raw = Path(case["out"]) / "raw-validation-000001.pt"
            raw.write_bytes(raw.read_bytes() + b"changed")
            with self.assertRaises(ValueError): inspect_case(case, code, "a" * 64, "b" * 64)

    def test_all54_boundary_and_bad_partial_manifest_precede_any_query_gt_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            frozen = boundary_fixture(Path(temp))
            self.assertTrue(require_frozen_boundary(frozen))
            self.assertTrue(require_frozen_boundary(frozen, full_integrity=False))
            changes = (lambda x: x["models"].pop(), lambda x: x["independent_checks"].pop(),
                lambda x: x["independent_checks"][0].__setitem__("status", "failed"),
                lambda x: x.__setitem__("training_optimizer_updates", 72),
                lambda x: x.__setitem__("policy", "g4_all27_frozen_query_v1"),
                lambda x: x.__setitem__("query_labels_accessed", True),
                lambda x: x["boundary_files"].pop(next(iter(x["boundary_files"]))))
            with patch("scipy.io.loadmat", side_effect=AssertionError("Premature query GT read")) as loader:
                for change in changes:
                    bad = copy.deepcopy(frozen)
                    change(bad)
                    with self.assertRaises(ValueError): read_query_labels(bad, {}, "missing", "missing", "0" * 64)
                loader.assert_not_called()
            Path(next(p for p in frozen["boundary_files"] if p.endswith("primary.pt"))).write_bytes(b"changed")
            with self.assertRaises(ValueError): require_frozen_boundary(frozen, full_integrity=False)

    def test_paired_seed_statistics_do_not_mix_models_with_support_repetitions(self):
        rows = []
        for t, s, method, mode in grid():
            mode_effect = {"none": 0, "capacity_control": .01, "x_residual": .02}[mode]
            effect = mode_effect + (.03 if method == METHODS[1] else 0)
            classes = list(range(1, DATASETS[t][2] + 1))
            rows.append({"target": t, "seed": s, "method": method, "mode": mode,
                "metrics": {"fused": {"oa": .5 + effect, "aa": .6 + effect, "nll": .7,
                    "class_ids": classes, "per_class": [{"class_id": c, "recall": .6 + effect} for c in classes]}}})
        result = paired_summary(rows)
        self.assertEqual(result["support_seed_repetitions"], 1)
        self.assertFalse(result["accuracy_hypothesis_decided_here"])
        key = "x_residual/" + METHODS[1] + " - capacity_control/" + METHODS[1]
        self.assertAlmostEqual(result["paired_macro_scene_deltas_by_model_seed"][key]["oa"]["mean"], .01)
        with self.assertRaises(ValueError): paired_summary(rows[:-1])
        rows[0]["metrics"]["fused"]["per_class"][0]["recall"] = None
        with self.assertRaises(ValueError): paired_summary(rows)


class MatchedQueueFailureTests(unittest.TestCase):
    def test_subprocess_failure_stops_queue_preserves_log_and_never_accepts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "logs").mkdir()
            write_json(root / "launch.json", {"fixture": True})
            write_json(root / "protected-files.json", {})
            case = {"name": "first", "target": "xuzhou", "seed": 64, "method": METHODS[0], "mode": "none", "out": str(root / "cases" / "first")}
            launch = {"out_dir": str(root), "cases": [case, {**case, "name": "second"}]}
            with patch("research.reproduction.m4g4b.matched_target_campaign.select_case"), \
                 patch("research.reproduction.m4g4b.matched_target_campaign.wait_gpu"), \
                 patch("research.reproduction.m4g4b.matched_target_campaign.subprocess.run", side_effect=subprocess.CalledProcessError(7, "fixture")) as run:
                with self.assertRaises(subprocess.CalledProcessError): run_queue(launch, {"gpu": 0}, root / "registration", "a" * 64, {"commit": "fixture"})
            self.assertEqual(run.call_count, 1)
            self.assertTrue((root / "logs" / "first-first2.log").exists())
            self.assertFalse((root / "training-acceptance.json").exists())
            status = json.loads((root / "status.json").read_text())
            self.assertEqual((status["status"], status["completed_cases"]), ("failed", 0))

    def test_zero_update_exit_with_mutated_completed_files_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "logs").mkdir()
            write_json(root / "launch.json", {"fixture": True})
            write_json(root / "protected-files.json", {})
            folder = root / "cases" / "first"
            case = {"name": "first", "target": "xuzhou", "seed": 64, "method": METHODS[0], "mode": "none", "out": str(folder)}
            count = []
            def fake(command, *, stdout, **kwargs):
                folder.mkdir(parents=True, exist_ok=True)
                n = len(count)
                count.append(n)
                value = (("interrupted", 2), ("completed", 1498), ("verified_completed", 0))[n]
                (folder / "checkpoint.pt").write_text("state" + str(n))
                stdout.write(json.dumps({"operation": value[0], "updates_this_call": value[1]}) + "\n")
            with patch("research.reproduction.m4g4b.matched_target_campaign.select_case"), \
                 patch("research.reproduction.m4g4b.matched_target_campaign.wait_gpu"), \
                 patch("research.reproduction.m4g4b.matched_target_campaign.subprocess.run", side_effect=fake):
                with self.assertRaisesRegex(ValueError, "mutated"):
                    run_queue({"out_dir": str(root), "cases": [case]}, {"gpu": 0}, root / "registration", "a" * 64, {"commit": "fixture"})
            self.assertEqual(len(count), 3)
            self.assertFalse((root / "training-acceptance.json").exists())
