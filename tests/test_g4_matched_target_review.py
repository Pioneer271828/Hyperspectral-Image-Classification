"""Review gates, first-min selection, real training schema, PNG/CSV and immutability."""
import copy
import csv
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from g4_matched_target_audit import AUDIT_POLICY, audit_case
from g4_matched_target_review import HEADS, derive_case, export_review, metric_record, plot_case, validate_audit
from g4_matched_target_study import METHODS, case_name, grid
from joint_training import run_training
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research_runtime import code_snapshot, write_json
from test_g4_matched_target_audit import actual_case


def fixture(root, target="xuzhou", seed=64, method=METHODS[0], mode="none"):
    name = case_name(target, seed, method, mode)
    case = {"name": name, "target": target, "seed": seed, "method": method, "mode": mode,
        "out": str(root / "cases" / name), "inputs": {"observation": str(root.parent / "data" / "observation.pt"),
        "family": str(root.parent / "labels" / "family.json")}, "source_spec": {}}
    count = DATASETS[target][2]
    cm = np.eye(count, dtype=np.int64) * 5
    cm[0, 0], cm[0, 1], cm[-1, -1] = 0, 5, 0
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(count, dtype=float), where=support > 0)
    base = {"oa": float(cm.trace() / cm.sum()), "aa": float(recall.mean()), "nll": .2,
        "confusion": cm.tolist(), "recall": recall.tolist()}
    history = []
    for step in list(range(1, 1472, 30)) + [1500]:
        nll = .4 if step == 1 else .1
        metrics = {target: {phase: {h: {**copy.deepcopy(base), "nll": nll if phase == "val" else nll / 2}
                     for h in HEADS} for phase in ("train", "val")}}
        history.append({"step": step, "epoch": step - 1, "val_loss": nll, "macro_oa": base["oa"],
            "selection_score": -nll, "metrics": metrics})
    trace = [{"step": i, "dataset": target, "loss": {"total": (1501 - i) / 1500},
        "learning_rates": [.001 if i < 100 else .0005, .001], "gradient_norms": {"adapter": 1., "graph": 1.},
        **({"m4_parameter_gradient_norm": .2} if mode != "none" else {})} for i in range(1, 1501)]
    accepted = {"name": name, "status": "passed", "updates": 1500, "validation_events": 51,
        "selected_step": 31, "selected_validation": {k: v for k, v in history[1].items() if k not in ("step", "epoch", "selection_score")},
        "last_validation": history[-1], "completed_resume_updates": 0, "head_metrics": 6,
        "class_recalls": 6 * count, "exact_reload_repeats": 3,
        "changed_parameter_elements": {"adapter": 20, "cnn": 30, "graph": 40, **({"m4": 10} if mode != "none" else {})},
        "positive_gradient_steps": {"adapter": 1500, "graph": 1500, **({"m4": 1500} if mode != "none" else {})},
        "loss_first30": sum(r["loss"]["total"] for r in trace[:30]) / 30,
        "loss_last30": sum(r["loss"]["total"] for r in trace[-30:]) / 30,
        "lr_change_steps": [100], "elapsed_seconds": 2., "query_labels_accessed": False,
        "result_sha256": "r", "primary_sha256": "p", "sidecar_sha256": "s"}
    # Same floating-point reduction as the authoritative audit.
    import statistics
    accepted["loss_first30"] = statistics.mean(r["loss"]["total"] for r in trace[:30])
    accepted["loss_last30"] = statistics.mean(r["loss"]["total"] for r in trace[-30:])
    return case, accepted, history, trace


def campaign_fixture(root):
    code = {"commit": "synthetic", "dirty": False, "files": {}, "source_hash": digest({})}
    cases, rows = [], []
    for key in grid():
        case, accepted, history, trace = fixture(root, *key)
        folder = Path(case["out"])
        folder.mkdir(parents=True)
        write_json(folder / "history.json", history)
        write_json(folder / "trace.json", trace)
        cases.append(case)
        rows.append(accepted)
    launch = {"out_dir": str(root), "code": code, "cases": cases, "short_acceptances": {}}
    accepted = {"models": rows, "protected_manifest_sha256": "m", "elapsed_seconds": 110.}
    audit = {"policy": AUDIT_POLICY, "status": "passed", "code": code, "producer": code,
        "launch_sha256": "l", "acceptance_sha256": "a", "protected_manifest_sha256": "m",
        "new_optimizer_updates": 0, "query_labels_accessed": False, "test_scoring": False, "elapsed_seconds": 12.,
        "cases": [{k: v for k, v in row.items() if k not in ("completed_resume_updates", "elapsed_seconds")} for row in rows]}
    return launch, accepted, audit, code


class MatchedReviewTests(unittest.TestCase):
    def resource_fixture(self, root):
        launch, accepted, audit, code = campaign_fixture(root)
        accepted["resource_execution"] = {"sha256": "resource-amendment", "receipt_sha256": "resource-receipt"}
        accepted["elapsed_scope"] = "resource_controller_segment_only"
        for row in accepted["models"][-3:]:
            row.update(elapsed_scope="resource_case_wall_with_waits", resource_amendment_sha256="resource-amendment")
        proof = {"policy": "g4_issue184_resource_final_independent_v1", "status": "passed",
            "consumer": {"commit": "5dcc6964a25858f6793995126ec4606dfb8ad533"},
            "amendment_sha256": "resource-amendment", "resource_receipt_sha256": "resource-receipt",
            "launch_sha256": "l", "acceptance_sha256": "a", "models": 54, "preserved_rows": 51,
            "preserved_completed_files": 5763, "new_optimizer_updates": 0,
            "query_labels_accessed": False, "test_scoring": False,
            "totals": {"updates": 81000, "head_metrics": 324, "class_recalls": 4320, "exact_reload_repeats": 162}}
        return launch, accepted, audit, code, proof

    def test_resource_cost_metadata_requires_bound_proof_and_preserves_numeric_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            launch, accepted, audit, code, proof = self.resource_fixture(Path(temp) / "run")
            with self.assertRaises(ValueError):
                validate_audit(launch, accepted, audit, code, "l", "a")
            validate_audit(launch, accepted, audit, code, "l", "a", proof)
            for key, value in (("status", "failed"), ("query_labels_accessed", True),
                    ("acceptance_sha256", "changed"), ("amendment_sha256", "changed"),
                    ("preserved_rows", 50), ("new_optimizer_updates", 1)):
                bad = {**proof, key: value}
                with self.assertRaises(ValueError):
                    validate_audit(launch, accepted, audit, code, "l", "a", bad)
            for field, value in (("selected_step", 1500), ("resource_amendment_sha256", "wrong"),
                    ("elapsed_scope", "whole_batch"), ("unrecognized_metadata", True),
                    ("positive_gradient_steps", {"adapter": 0, "graph": 1500})):
                bad = copy.deepcopy(accepted)
                bad["models"][-1][field] = value
                with self.assertRaises(ValueError):
                    validate_audit(launch, bad, audit, code, "l", "a", proof)

    def test_resource_export_hash_binding_and_cost_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, out = base / "run", base / "review"
            launch, accepted, audit, code, proof = self.resource_fixture(root)
            audit_path = base / "audit" / "receipt.json"
            proof_path = base / "resource-audit" / "receipt.json"
            audit_path.parent.mkdir()
            proof_path.parent.mkdir()
            write_json(audit_path, audit)
            proof["full_audit_sha256"] = file_hash(audit_path)
            write_json(proof_path, proof)
            with patch("g4_matched_target_review.full_acceptance", return_value=(launch, accepted, {})), \
                    patch("g4_matched_target_review.require_frozen_code"), \
                    patch("g4_matched_target_review.plot_case"):
                for bad in ("wrong", None):
                    with self.assertRaises(ValueError):
                        export_review(root, "l", "a", audit_path, file_hash(audit_path), out, code,
                            resource_audit_path=proof_path, resource_audit_sha256=bad)
                    self.assertFalse(out.exists())
                result = export_review(root, "l", "a", audit_path, file_hash(audit_path), out, code,
                    resource_audit_path=proof_path, resource_audit_sha256=file_hash(proof_path))
            self.assertEqual(result["acceptance_elapsed_scope"], "resource_controller_segment_only")
            self.assertEqual(result["resource_audit_sha256"], file_hash(proof_path))
            self.assertNotIn("controller_seconds_excluding_initial_freeze", result)
            self.assertEqual(sum(r["case_elapsed_scope"] == "resource_case_wall_with_waits" for r in result["cases"]), 3)
            self.assertEqual((result["new_optimizer_updates"], result["query_labels_accessed"]), (0, False))

    def test_first_min_tie_supported_zero_unsupported_na_and_png(self):
        with tempfile.TemporaryDirectory() as temp:
            case, accepted, history, trace = fixture(Path(temp), mode="x_residual")
            row = derive_case(case, accepted, history, trace)
            self.assertEqual(row["selected_step"], 31)
            self.assertFalse(row["selected_at_budget_end"])
            self.assertEqual(row["heads"]["val"]["fused"]["zero_recall_classes"], [1])
            self.assertEqual(row["heads"]["val"]["fused"]["unsupported_classes"], [9])
            self.assertIsNone(row["heads"]["val"]["fused"]["recall"][-1])
            self.assertEqual(row["positive_gradient_steps"]["m4"], 1500)
            self.assertEqual(len(row["curve"]), 51)
            path = Path(temp) / "preview.png"
            plot_case(row, trace, path)
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            if os.environ.get("MATCHED_REVIEW_PREVIEW"):
                Path(os.environ["MATCHED_REVIEW_PREVIEW"]).write_bytes(path.read_bytes())

    def test_partial_nonfinite_selection_gradient_lr_and_cost_rejected(self):
        for field, value in (("selected_step", 61), ("elapsed_seconds", float("nan")),
                ("completed_resume_updates", 1), ("positive_gradient_steps", {"adapter": 0, "graph": 1500}),
                ("lr_change_steps", []), ("changed_parameter_elements", {"adapter": 1, "cnn": 0, "graph": 1})):
            with self.subTest(field=field):
                case, accepted, history, trace = fixture(Path("/synthetic"))
                accepted[field] = value
                with self.assertRaises(ValueError): derive_case(case, accepted, history, trace)
        case, accepted, history, trace = fixture(Path("/synthetic"))
        for bad_history, bad_trace in ((history[:-1], trace), (history, trace[:-1]),
                (history, [{**r, "learning_rates": [float("inf")]} for r in trace])):
            with self.assertRaises(ValueError): derive_case(case, accepted, bad_history, bad_trace)
        metric = history[0]["metrics"]["xuzhou"]["val"]["fused"]
        with self.assertRaises(ValueError): metric_record({**metric, "recall": [1.] * 9}, 9)
        with self.assertRaises(ValueError): metric_record({**metric, "oa": float("nan")}, 9)

    def test_actual_full1500_network_artifact_schema_zero_update_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items, case, provenance = actual_case(root, device="cpu")
            code = code_snapshot()
            run_training(case["settings"], items, case["out"], provenance, code,
                device="cpu", runtime_adapter=__import__("g4_matched_target_study").runtime(case, items))
            accepted = audit_case(case, code, "a" * 64, "b" * 64, device="cpu")
            accepted.update(completed_resume_updates=0, elapsed_seconds=10.)
            folder = Path(case["out"])
            pins = {str(p): file_hash(p) for p in folder.iterdir() if p.is_file()}
            import json
            row = derive_case(case, accepted, json.loads((folder / "history.json").read_text()),
                json.loads((folder / "trace.json").read_text()))
            self.assertEqual(row["selected_step"], accepted["selected_step"])
            self.assertEqual(pins, {str(p): file_hash(p) for p in folder.iterdir() if p.is_file()})

    def test_full54_audit_rejects_missing_duplicate_changed_and_query_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            launch, accepted, audit, code = campaign_fixture(Path(temp) / "run")
            validate_audit(launch, accepted, audit, code, "l", "a")
            mutations = (lambda a: a["cases"].pop(), lambda a: a["cases"].__setitem__(-1, a["cases"][0]),
                lambda a: a.__setitem__("query_labels_accessed", True), lambda a: a.__setitem__("new_optimizer_updates", 1),
                lambda a: a.__setitem__("acceptance_sha256", "changed"),
                lambda a: a["cases"][0].__setitem__("selected_step", 1500),
                lambda a: a["cases"][0].pop("selected_validation"))
            for mutate in mutations:
                bad = copy.deepcopy(audit)
                mutate(bad)
                with self.assertRaises(ValueError): validate_audit(launch, accepted, bad, code, "l", "a")

    def test_export54_csv_groups_input_hashes_and_no_query_or_training(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, out = base / "run", base / "review"
            launch, accepted, audit, code = campaign_fixture(root)
            audit_path = base / "audit" / "receipt.json"
            audit_path.parent.mkdir()
            write_json(audit_path, audit)
            from g4_matched_target_audit import files
            before = files(root)
            # Only the already separately tested full-acceptance model oracle is replaced.
            # Derivation, independent-audit equality, hashing and all54 exports are real.
            with patch("g4_matched_target_review.full_acceptance", return_value=(launch, accepted, {})), \
                    patch("g4_matched_target_review.require_frozen_code"), \
                    patch("g4_matched_target_review.plot_case") as plots:
                result = export_review(root, "l", "a", audit_path, file_hash(audit_path), out, code)
            self.assertEqual(plots.call_count, 54)
            self.assertEqual((len(result["cases"]), len(result["groups"]), result["optimizer_updates_reviewed"]), (54, 18, 81000))
            self.assertEqual((result["new_optimizer_updates"], result["query_labels_accessed"]), (0, False))
            self.assertEqual(len(list(out.glob("*-curve.csv"))), 54)
            with next(out.glob("*-curve.csv")).open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 51)
            self.assertEqual(before, files(root))
            self.assertTrue((out / "summary.json").is_file())

    def test_export_rejects_unaccepted_batch_bad_pin_input_output_and_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, out = base / "run", base / "review"
            launch, accepted, audit, code = campaign_fixture(root)
            audit_path = base / "audit" / "receipt.json"
            audit_path.parent.mkdir()
            write_json(audit_path, audit)
            with patch("g4_matched_target_review.full_acceptance", side_effect=ValueError("Unaccepted batch")):
                with self.assertRaisesRegex(ValueError, "Unaccepted"):
                    export_review(root, "l", "a", audit_path, file_hash(audit_path), out, code)
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError): export_review(root, "l", "a", audit_path, "changed", out, code)
            existing = next(root.rglob("history.json"))
            with patch("g4_matched_target_review.full_acceptance", return_value=(launch, accepted, {str(existing): "stale"})):
                with self.assertRaises(ValueError): export_review(root, "l", "a", audit_path, file_hash(audit_path), out, code)
            self.assertFalse(out.exists())
            with patch("g4_matched_target_review.full_acceptance", return_value=(launch, accepted, {})):
                for bad in (root, root / "nested", audit_path.parent, base / "labels" / "output"):
                    with self.assertRaises(ValueError): export_review(root, "l", "a", audit_path, file_hash(audit_path), bad, code)
                def mutate(*args):
                    audit_path.write_bytes(audit_path.read_bytes() + b" ")
                with patch("g4_matched_target_review.plot_case", side_effect=mutate), patch("g4_matched_target_review.require_frozen_code"):
                    with self.assertRaises(ValueError): export_review(root, "l", "a", audit_path, file_hash(audit_path), out, code)
                self.assertFalse((out / "summary.json").exists())

    def test_cli_rejects_wrong_sha_before_accessing_training_or_query(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "not-created"
            command = [sys.executable, "research/reproduction/m4g4b/matched_target_report.py",
                "--expected-commit", "0" * 40]
            for field in ("run", "launch-sha256", "acceptance-sha256", "audit", "audit-sha256", "out"):
                command.extend(["--" + field, str(missing)])
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Clean expected remote review exporter required", result.stderr)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
