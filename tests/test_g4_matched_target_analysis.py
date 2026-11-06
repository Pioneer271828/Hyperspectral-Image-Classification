"""Hand-counted synthetic54 only; no real query/model/GT consumption."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluation.classification_metrics import metrics_from_confusion
from g4_matched_target_analysis import (FT, SCRATCH, PLAN_SHA256, PRODUCER, PRODUCER_SOURCE, REVIEW_PRODUCER,
    HEADS, analyze_rows, export_analysis, report_markdown)
from g4_matched_target_evaluation import QUERY_POLICY, paired_summary
from g4_matched_target_execution import REGISTRATION_POLICY
from g4_matched_target_review import POLICY as REVIEW_POLICY
from g4_matched_target_study import EXPECTED, SEEDS, STUDY_PATH, case_name, grid
from joint_baseline import NAMES
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research_runtime import ROOT, code_snapshot, write_json


def rows_fixture(support=10000):
    rows = []
    for target, seed, method, mode in grid():
        classes = DATASETS[target][2]
        supports = support[target] if isinstance(support, dict) else [support] * classes
        matrix = [[0] * classes for _ in range(classes)]
        for i, count in enumerate(supports):
            matrix[i][i] = count * 4 // 5
            matrix[i][(i + 1) % classes] = count - matrix[i][i]
        metric = {**metrics_from_confusion(matrix, list(range(1, classes + 1))), "nll": .5}
        rows.append({"target": target, "seed": seed, "method": method, "mode": mode,
            "name": case_name(target, seed, method, mode), "metrics": {h: copy.deepcopy(metric) for h in HEADS},
            "checkpoint_sha256": "a" * 64, "release_sha256": "b" * 64, "branches": {"synthetic": True}})
    return rows


def improve(rows, amount, *, mode="x_residual", method=FT, target=None, seed=None, heads=HEADS):
    for row in rows:
        if (row["mode"] != mode or row["method"] != method or target and row["target"] != target
                or seed and row["seed"] != seed):
            continue
        for head in heads:
            matrix = copy.deepcopy(row["metrics"][head]["confusion_matrix"])
            for i, line in enumerate(matrix):
                delta = amount[i] if isinstance(amount, list) else amount
                line[i] += delta; line[(i + 1) % len(matrix)] -= delta
            row["metrics"][head] = {**metrics_from_confusion(matrix, list(range(1, len(matrix) + 1))), "nll": .4}


def producer_fixture():
    current = code_snapshot()
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", PRODUCER], text=True).splitlines()
    files = {n: current["files"][n] for n in names if n.endswith(".py") or n.startswith("configs/") and n.endswith(".json")}
    assert digest(files) == PRODUCER_SOURCE
    return {"commit": PRODUCER, "branch": "synthetic-original331-schema", "dirty": False,
            "files": files, "source_hash": digest(files)}


def files_fixture(root, producer):
    """Complete original-schema boundary with text stand-ins; no trained tensors."""
    train, query, frozen, review = (root / name for name in ("train", "query", "frozen", "review"))
    for folder in (train, query, frozen, review): folder.mkdir()
    plan = root / "analysis-plan.md"
    plan.write_bytes((ROOT / "research/experiments/G4MatchedFullTarget_完整54目标配对预登记_20261106.md").read_bytes())
    assert file_hash(plan) == PLAN_SHA256
    eng = root / "engineering.json"
    write_json(eng, {"status": "engineering_passed", "issue": 182, "code": producer, "cpu_passed": True,
        "gpu_passed": True, "full_regression_passed": True, "worker_campaign_audit_contracts_passed": True,
        "query_labels_accessed": False, "full_training_completed": False, "preflight_receipt_sha256": "d" * 64})
    registration = root / "registration.json"
    write_json(registration, {"policy": REGISTRATION_POLICY, "experiment_issue": 184, "producer_commit": PRODUCER,
        "study_file_sha256": file_hash(STUDY_PATH), "planned_optimizer_updates": 81000, "gpu": 1,
        "estimated_seconds": 86400, "preflight": {"receipt_sha256": "d" * 64}, "out_dir": str(train),
        "engineering_acceptance": {"path": str(eng), "sha256": file_hash(eng)},
        "analysis_plan": {"path": str(plan), "sha256": PLAN_SHA256},
        "query_labels_during_training": False, "test_scoring_during_training": False})
    rows = rows_fixture(); improve(rows, 100)
    models, checks, cases, pins = [], [], [], {}
    for row in rows:
        folder = train / "cases" / row["name"]; folder.mkdir(parents=True)
        fields = {}
        for name, key in (("result.json", "result_sha256"), ("primary.pt", "primary_sha256"), ("primary.pt.json", "sidecar_sha256")):
            path = folder / name; path.write_text("SYNTHETIC_TEXT_NOT_A_CHECKPOINT " + row["name"] + name)
            fields[key] = file_hash(path); pins[str(path)] = fields[key]
        identity = {k: row[k] for k in ("name", "target", "seed", "method", "mode")}
        models.append({**identity, **fields, "selected_step": 1500})
        checks.append({"name": row["name"], "status": "passed", "updates": 1500, "validation_events": 51,
            "head_metrics": 6, "class_recalls": 6 * DATASETS[row["target"]][2], "exact_reload_repeats": 3, **fields})
        cases.append({**identity, "source_spec": {}, "inputs": {"observation": str(root / "obs" / "input.pt"),
                      "family": str(root / "families" / "family.json")}})
        row["checkpoint_sha256"] = fields["primary_sha256"]
        write_json(query / (row["name"] + ".json"), row)
    launch = train / "launch.json"
    write_json(launch, {"code": producer, "out_dir": str(train), "cases": cases,
        "short_acceptances": {"none": {"root": str(root / "short-none")}, "active": {"root": str(root / "short-active")}},
        "registration_path": str(registration), "registration_sha256": file_hash(registration)})
    for name in ("training-acceptance.json", "independent.json"):
        path = train / name; write_json(path, {"synthetic": True}); pins[str(path)] = file_hash(path)
    pins[str(launch)] = file_hash(launch)
    boundary = {"policy": QUERY_POLICY, "status": "all54_frozen_before_query", "study": EXPECTED,
        "code": producer, "producer": producer, "training_root": str(train), "launch_sha256": file_hash(launch),
        "acceptance_sha256": file_hash(train / "training-acceptance.json"),
        "independent_audit_sha256": file_hash(train / "independent.json"), "models": models,
        "independent_checks": checks, "training_optimizer_updates": 81000, "new_optimizer_updates": 0,
        "query_labels_accessed": False, "protected_files": pins, "boundary_files": pins}
    freeze_path = frozen / "freeze.json"; write_json(freeze_path, boundary)
    query_path = query / "receipt.json"
    write_json(query_path, {"status": "passed", "policy": QUERY_POLICY, "code": producer,
        "freeze_sha256": file_hash(freeze_path), "models": rows, "paired_summary": paired_summary(rows),
        "query_labels_accessed": True, "optimizer_updates": 0, "elapsed_seconds": 1.})
    exports = {}
    for row in rows:
        for suffix in (".png", "-curve.csv"):
            path = review / (row["name"] + suffix); path.write_text("SYNTHETIC_EXPORT_NO_REAL_IMAGE")
            exports[path.name] = file_hash(path)
    review_code = {**copy.deepcopy(producer), "commit": REVIEW_PRODUCER}
    review_path = review / "summary.json"
    write_json(review_path, {"policy": REVIEW_POLICY, "status": "passed", "code": review_code, "producer": producer,
        "cases": [{"name": r["name"], "updates": 1500, "validation_events": 51} for r in rows],
        "launch_sha256": boundary["launch_sha256"], "acceptance_sha256": boundary["acceptance_sha256"],
        "audit_sha256": boundary["independent_audit_sha256"], "optimizer_updates_reviewed": 81000,
        "new_optimizer_updates": 0, "query_labels_accessed": False, "test_scoring": False, "exported_files": exports,
        "controller_seconds_excluding_initial_freeze": 10., "independent_audit_seconds": 2.,
        "sum_case_training_resume_wait_and_acceptance_seconds": 12., "report_elapsed_seconds": 3.})
    return query_path, freeze_path, review_path


def invoke(paths, out):
    return export_analysis(paths[0], file_hash(paths[0]), paths[1], file_hash(paths[1]),
                           paths[2], file_hash(paths[2]), out, code_snapshot())


class MatchedAnalysisTests(unittest.TestCase):
    def test_resource_review_profile_hash_cost_and_audit_binding(self):
        from g4_matched_target_analysis import RESOURCE_REVIEW_PRODUCER, RESOURCE_REVIEW_POLICY
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = files_fixture(root, self.producer)
            freeze = json.loads(paths[1].read_text())
            review = json.loads(paths[2].read_text())
            proof_folder = root / "resource-audit"
            proof_folder.mkdir()
            proof_path = proof_folder / "receipt.json"
            proof = {"policy": "g4_issue184_resource_final_independent_v1", "status": "passed",
                "consumer": {"commit": "5dcc6964a25858f6793995126ec4606dfb8ad533"},
                "launch_sha256": freeze["launch_sha256"], "acceptance_sha256": freeze["acceptance_sha256"],
                "full_audit_sha256": freeze["independent_audit_sha256"], "models": 54, "preserved_rows": 51,
                "new_optimizer_updates": 0, "query_labels_accessed": False, "test_scoring": False}
            write_json(proof_path, proof)
            review["code"]["commit"] = RESOURCE_REVIEW_PRODUCER
            review.update(policy=RESOURCE_REVIEW_POLICY, resource_audit=str(proof_path),
                resource_audit_sha256=file_hash(proof_path), acceptance_elapsed_seconds=review.pop("controller_seconds_excluding_initial_freeze"),
                acceptance_elapsed_scope="resource_controller_segment_only")
            write_json(paths[2], review)
            with patch("torch.load", side_effect=AssertionError("No tensor decoding")), \
                    patch("torch.nn.Module._call_impl", side_effect=AssertionError("No inference")):
                receipt = invoke(paths, root / "resource-analysis")
            analysis = json.loads((root / "resource-analysis/analysis.json").read_text())
            self.assertEqual(analysis["recorded_cost_scopes"]["acceptance_elapsed"], "resource_controller_segment_only")
            self.assertNotIn("controller_seconds_excluding_initial_freeze", analysis["recorded_cost_seconds"])
            self.assertTrue(receipt["primary_benefit_supported"])
            self.assertEqual(analysis["primary"], analyze_rows(json.loads(paths[0].read_text())["models"])["primary"])
            for key, value in (("query_labels_accessed", True), ("new_optimizer_updates", 1),
                    ("acceptance_sha256", "f" * 64), ("status", "failed"), ("preserved_rows", 50)):
                write_json(proof_path, {**proof, key: value})
                bad_review = {**review, "resource_audit_sha256": file_hash(proof_path)}
                write_json(paths[2], bad_review)
                with self.assertRaises(ValueError):
                    invoke(paths, root / "bad-resource-analysis")
                self.assertFalse((root / "bad-resource-analysis").exists())
            write_json(proof_path, proof)
            write_json(paths[2], {**review, "resource_audit_sha256": "0" * 64})
            with self.assertRaises(ValueError):
                invoke(paths, root / "wrong-proof-pin")

    @classmethod
    def setUpClass(cls):
        cls.producer = producer_fixture()

    def test_equal_scene_weighting_reverses_pixel_pool_and_matches_legacy_summary(self):
        supports = {t: [100 if t != "whu-honghu" else 10000] * DATASETS[t][2] for t in NAMES}
        rows = rows_fixture(supports)
        for target in NAMES: improve(rows, 10 if target != "whu-honghu" else -100, target=target)
        result = analyze_rows(rows); primary = result["primary"]
        self.assertTrue(primary["passes_all_gates"])
        self.assertAlmostEqual(primary["equal_scene_macro_fused_oa_delta_pp"]["mean"], float(Fraction(19, 300) * 100), places=12)
        pooled = (2 * 9 * 10 - 22 * 100) / (2 * 9 * 100 + 22 * 10000)
        self.assertLess(pooled, 0)
        old = paired_summary(rows)["paired_macro_scene_deltas_by_model_seed"]["x_residual/" + FT + " - capacity_control/" + FT]["oa"]
        self.assertAlmostEqual(old["mean"] * 100, primary["equal_scene_macro_fused_oa_delta_pp"]["mean"], places=12)

    def test_model_seed_pairing_sample_sd_and_strict_positive_gate(self):
        rows = rows_fixture()
        for seed, gain in zip(SEEDS, (100, 200, 300)): improve(rows, gain, seed=seed)
        r = analyze_rows(rows)["primary"]
        self.assertEqual(r["equal_scene_macro_fused_oa_delta_pp"]["seed_values"], [1., 2., 3.])
        self.assertEqual(r["equal_scene_macro_fused_oa_delta_pp"]["sample_sd"], statistics.stdev([1., 2., 3.]))
        self.assertTrue(r["passes_all_gates"])
        improve(rows, -100, seed=64)
        r = analyze_rows(rows)["primary"]
        self.assertTrue(r["gates"]["mean_equal_scene_macro_oa_positive"])
        self.assertFalse(r["gates"]["all_three_model_seeds_positive"])
        r = analyze_rows(rows_fixture())["primary"]
        self.assertFalse(r["gates"]["mean_equal_scene_macro_oa_positive"])

    def test_exact_inclusive_class_two_pp_and_one_count_violation(self):
        rows = rows_fixture(); improve(rows, 100)
        improve(rows, [-300] + [0] * 8, target="paviaU")
        self.assertTrue(analyze_rows(rows)["primary"]["passes_all_gates"])
        improve(rows, [-1] + [0] * 8, target="paviaU")
        r = analyze_rows(rows)["primary"]
        self.assertFalse(r["gates"]["every_class_mean_recall_drop_at_most_2pp"])
        self.assertEqual(r["scenes"][0]["failed_class_ids"], [1])

    def test_honghu_and_scene_aa_exact_one_pp_boundaries(self):
        rows = rows_fixture(); improve(rows, 100)
        improve(rows, -200, target="whu-honghu")
        self.assertTrue(analyze_rows(rows)["primary"]["passes_all_gates"])
        improve(rows, [-1] + [0] * 21, target="whu-honghu")
        self.assertFalse(analyze_rows(rows)["primary"]["gates"]["honghu_mean_oa_drop_at_most_1pp"])
        supports = {t: [400] * DATASETS[t][2] for t in NAMES}; supports["paviaU"][0] = 40000
        rows = rows_fixture(supports)
        improve(rows, [400] + [-5] * 8, target="paviaU")
        improve(rows, 4, target="xuzhou"); improve(rows, 4, target="whu-honghu")
        self.assertTrue(analyze_rows(rows)["primary"]["passes_all_gates"])
        improve(rows, [0, -1] + [0] * 7, target="paviaU")
        r = analyze_rows(rows)["primary"]
        self.assertFalse(r["gates"]["every_scene_mean_aa_drop_at_most_1pp"])
        self.assertTrue(r["gates"]["every_class_mean_recall_drop_at_most_2pp"])

    def test_primary_failure_not_replaced_by_four_successful_secondary_claims(self):
        rows = rows_fixture(); improve(rows, 100); improve(rows, 200, mode="capacity_control")
        for mode in ("none", "capacity_control", "x_residual"): improve(rows, -100, mode=mode, method=SCRATCH)
        result = analyze_rows(rows)
        self.assertEqual(result["benefit_conclusion"], "primary_benefit_not_supported")
        self.assertTrue(all(v["passes_all_gates"] for v in result["secondary"].values()))
        self.assertFalse(result["secondary_replaces_primary"]); self.assertFalse(result["significance_claim"])
        self.assertIn("primary_benefit_not_supported", report_markdown(result))

    def test_three_heads_descriptive_did_groups_and_input_immutability(self):
        rows = rows_fixture(); improve(rows, 600); improve(rows, 300, mode="capacity_control")
        improve(rows, 200, method=SCRATCH); improve(rows, 100, method=SCRATCH, mode="capacity_control")
        improve(rows, -100, heads=("graph",)); before = digest(rows)
        result = analyze_rows(rows)
        self.assertEqual(before, digest(rows)); self.assertEqual(len(result["models"]), 54)
        self.assertEqual(len(result["groups"]), 18)
        self.assertAlmostEqual(result["difference_in_differences_fraction_units"]["fused"]["oa"]["mean"], .02)
        self.assertTrue(all("passes_all_gates" not in v for v in result["descriptive"].values()))
        self.assertNotEqual(result["groups"][-1]["heads"]["fused"]["oa"], result["groups"][-1]["heads"]["graph"]["oa"])

    def test_incomplete_duplicate_noninteger_class_cohort_and_metric_negatives(self):
        original = rows_fixture()
        bads = [original[:-1], original + [original[0]], original[:-1] + [original[0]]]
        for mutate in (lambda r: r[0].update(seed=True), lambda r: r[0].update(name="wrong"),
            lambda r: r[0]["metrics"].pop("cnn"), lambda r: r[0]["metrics"]["fused"].update(nll=float("nan")),
            lambda r: r[0]["metrics"]["fused"].update(oa=.1),
            lambda r: r[0]["metrics"]["fused"]["confusion_matrix"][0].__setitem__(0, True),
            lambda r: r[0]["metrics"]["fused"]["confusion_matrix"][0].__setitem__(0, 1.5),
            lambda r: r[0]["metrics"]["fused"]["class_ids"].reverse()):
            value = copy.deepcopy(original); mutate(value); bads.append(value)
        value = copy.deepcopy(original); m = value[0]["metrics"]["fused"]["confusion_matrix"]; m[0][0] += 1
        value[0]["metrics"]["fused"] = {**metrics_from_confusion(m, list(range(1, 10))), "nll": .5}; bads.append(value)
        value = copy.deepcopy(original); m = value[0]["metrics"]["fused"]["confusion_matrix"]; m[0] = [0] * 9
        value[0]["metrics"]["fused"] = {**metrics_from_confusion(m, list(range(1, 10))), "nll": .5}; bads.append(value)
        for index, value in enumerate(bads):
            with self.subTest(index=index):
                with self.assertRaises(ValueError): analyze_rows(value)

    def test_full_synthetic_boundary_export_cli_and_no_tensor_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = files_fixture(root, self.producer)
            before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
            with patch("torch.load", side_effect=AssertionError("No tensor loads")), patch("torch.nn.Module._call_impl", side_effect=AssertionError("No inference")):
                receipt = invoke(paths, root / "analysis")
            self.assertEqual((receipt["models_analyzed"], receipt["heads_analyzed"], receipt["new_optimizer_updates"]), (54, 162, 0))
            self.assertEqual(before, {p: file_hash(p) for p in before})
            with self.assertRaises(ValueError): invoke(paths, root / "analysis")
            command = [sys.executable, "scripts/analyze_g4_matched_target_full.py", "--expected-commit", code_snapshot()["commit"]]
            for name, path in zip(("query", "freeze", "review"), paths): command += ["--" + name, str(path), "--" + name + "-sha256", file_hash(path)]
            cli = json.loads(subprocess.check_output(command + ["--out", str(root / "cli-analysis")], text=True))
            self.assertTrue(cli["primary_benefit_supported"])
            self.assertEqual(json.loads((root / "analysis/analysis.json").read_text()), json.loads((root / "cli-analysis/analysis.json").read_text()))

    def test_pinned_completed_query_review_freeze_and_output_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = files_fixture(root, self.producer)
            with self.assertRaises(ValueError): export_analysis(paths[0], "0" * 64, paths[1], file_hash(paths[1]), paths[2], file_hash(paths[2]), root / "bad", code_snapshot())
            for path, key, value in ((paths[0], "query_labels_accessed", False), (paths[0], "policy", "native27"),
                    (paths[0], "optimizer_updates", 1), (paths[1], "status", "running"), (paths[2], "new_optimizer_updates", 1),
                    (paths[2], "optimizer_updates_reviewed", 1500), (paths[2], "audit_sha256", "0" * 64)):
                saved = path.read_bytes(); data = json.loads(saved); data[key] = value; write_json(path, data)
                with self.subTest(key=key):
                    with self.assertRaises(ValueError): invoke(paths, root / "bad")
                    self.assertFalse((root / "bad").exists())
                path.write_bytes(saved)
            with self.assertRaises(ValueError): invoke(paths, paths[2].parent / "nested")
            self.assertFalse((paths[2].parent / "nested").exists())


if __name__ == "__main__":
    unittest.main()
