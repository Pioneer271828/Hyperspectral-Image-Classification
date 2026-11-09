"""Synthetic model captures and structural54 boundary; no real training/query."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation.frozen_representation_capture import ANALYSIS_COMMIT, STAGES, capture, export_batch, export_case
from g4_matched_target_analysis import POLICY as ANALYSIS_POLICY, PLAN_SHA256
from g4_matched_target_study import METHODS, MODES, case_name, grid, initialize
from joint_baseline import NAMES, _item_fingerprint, forward
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import OrderedValidation
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot, write_json
from target_bn_calibration import OBS_KEYS
from target_initialization import state_id
from test_g4_matched_target_analysis import producer_fixture
from test_g4_matched_target_audit import boundary_fixture
from test_m4_full_transfer import fixture as active_source
from test_none_full_transfer import fixture as none_source
from test_target_training import fixture

DEVICE = os.environ.get("REPRESENTATION_CAPTURE_DEVICE", "cpu")


def observation(root):
    root.mkdir(); (items, _, _), paths = fixture(root)
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in items["xuzhou"].items() if k in OBS_KEYS}, paths


def synthetic_boundary(root, all_cases=False):
    # Batch fixture repeats one synthetic model per scene, with structural metadata only.
    frozen = boundary_fixture(root / "train"); producer = producer_fixture()
    obs, paths = observation(root / "inputs")
    model, _ = initialize("xuzhou", 64, METHODS[0], "none")
    chosen = next(m for m in frozen["models"] if (m["target"], m["seed"], m["method"], m["mode"]) == ("xuzhou", 64, METHODS[0], "none"))
    primary = root / "train" / "cases" / chosen["name"] / "primary.pt"
    save_model_checkpoint(model, primary)
    check = next(r for r in frozen["independent_checks"] if r["name"] == chosen["name"])
    for path, key in ((primary, "primary_sha256"), (Path(str(primary) + ".json"), "sidecar_sha256")):
        chosen[key] = check[key] = file_hash(path)
        frozen["protected_files"][str(path)] = frozen["boundary_files"][str(path)] = file_hash(path)
    cases = []
    observations = {"xuzhou": paths}
    if all_cases:
        template = torch.load(paths[0], map_location="cpu", weights_only=True)
        for target in NAMES:
            if target == "xuzhou":
                continue
            folder = root / target
            folder.mkdir()
            item = {**template, "name": target, "class_num": DATASETS[target][2]}
            archive, record = folder / "observation.pt", folder / "record.json"
            torch.save(item, archive)
            write_json(record, {**json.loads(paths[1].read_text()), "target": target,
                "observation_sha256": file_hash(archive), "prepared_fields": _item_fingerprint(item)})
            observations[target] = (archive, record, paths[2])
            frozen["protected_files"].update({str(p): file_hash(p) for p in (archive, record)})
        models = {target: initialize(target, 64, METHODS[0], "none")[0] for target in NAMES}
    for row in frozen["models"]:
        archive, record, family = observations[row["target"]] if all_cases else paths
        case = {**{k: row[k] for k in ("name", "target", "seed", "method", "mode")},
            "out": str(root / "train" / "cases" / row["name"]), "source_spec": {},
            "inputs": {"observation": str(archive), "record": str(record), "record_sha256": file_hash(record),
                       "family": str(family), "family_sha256": file_hash(family)}}
        if all_cases:
            checkpoint = Path(case["out"]) / "primary.pt"
            save_model_checkpoint(models[row["target"]], checkpoint)
            audit_row = next(r for r in frozen["independent_checks"] if r["name"] == row["name"])
            for path, key in ((checkpoint, "primary_sha256"), (Path(str(checkpoint) + ".json"), "sidecar_sha256")):
                row[key] = audit_row[key] = file_hash(path)
                frozen["protected_files"][str(path)] = frozen["boundary_files"][str(path)] = row[key]
        row["case_sha256"] = digest(case); cases.append(case)
    launch = root / "train" / "launch.json"
    write_json(launch, {"code": producer, "cases": cases, "out_dir": str(root / "train"),
        "short_acceptances": {"none": {"root": str(root / "short-none")}, "active": {"root": str(root / "short-active")}}})
    frozen.update(producer=producer, code=producer, launch_sha256=file_hash(launch))
    frozen["protected_files"].update({str(p): file_hash(p) for p in (*paths, launch)})
    (root / "frozen").mkdir(); (root / "analysis").mkdir()
    freeze = root / "frozen" / "freeze.json"; write_json(freeze, frozen)
    current = code_snapshot(); names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", ANALYSIS_COMMIT], text=True).splitlines()
    files = {n: current["files"][n] for n in names if n in current["files"]}
    analysis_code = {**current, "commit": ANALYSIS_COMMIT, "files": files, "source_hash": digest(files)}
    outputs = {}
    for n in ("analysis.json", "report.md"):
        p = root / "analysis" / n; p.write_text("SYNTHETIC_STRUCTURAL_METADATA_NOT_REAL54_ANALYSIS"); outputs[n] = file_hash(p)
    analysis = {"status": "passed", "policy": ANALYSIS_POLICY, "code": analysis_code, "plan_sha256": PLAN_SHA256,
        "freeze_sha256": file_hash(freeze), "models_analyzed": 54, "heads_analyzed": 162, "groups": 18,
        "new_optimizer_updates": 0, "raw_labels_deserialized": False, "model_inference_executed": False,
        "new_query_release": False, "input_query_metrics_consumed": True, "primary_benefit_supported": False,
        "protected_files": frozen["protected_files"], "outputs": outputs}
    receipt = root / "analysis" / "receipt.json"; write_json(receipt, analysis)
    return freeze, receipt, chosen["name"], paths


class FrozenCaptureTests(unittest.TestCase):
    def test_full_batch_actual_synthetic_forward_outputs_and_late_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze, analysis, _, _ = synthetic_boundary(root, all_cases=True)
            code = code_snapshot()
            inputs = (freeze, file_hash(freeze), analysis, file_hash(analysis))
            out = root / "batch"
            with patch("g4_matched_target_evaluation.read_query_labels", side_effect=AssertionError("No query reader")), patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")):
                result = export_batch(*inputs, out, code, device=DEVICE, maximum_regions=4)
            self.assertEqual(result["model_forwards"], 54)
            self.assertEqual([r["name"] for r in result["cases"]], [case_name(*role) for role in grid()])
            self.assertEqual(len(list(out.iterdir())), 55)
            samples = {}
            for row in result["cases"]:
                path = out / row["summary"]
                self.assertEqual(file_hash(path), row["summary_sha256"])
                for stage, value in json.loads(path.read_text())["stages"].items():
                    key = (row["target"], stage)
                    self.assertEqual(samples.setdefault(key, value["sampled_region_indices"]), value["sampled_region_indices"])
            for path, sha in result["protected_files"].items():
                self.assertEqual(file_hash(path), sha)
            with patch("torch.load", side_effect=AssertionError("Existing output must fail before decode")):
                with self.assertRaises(FileExistsError):
                    export_batch(*inputs, out, code, device=DEVICE, maximum_regions=4)
            last = result["cases"][-1]
            sidecar = root / "train" / "cases" / last["name"] / "primary.pt.json"
            sidecar.write_text("changed")
            with patch("torch.load", side_effect=AssertionError("Any late pin mismatch must fail before decode")):
                with self.assertRaises(ValueError):
                    export_batch(*inputs, root / "blocked", code, device=DEVICE, maximum_regions=4)
            self.assertFalse((root / "blocked").exists())

    def test_actual_six_synthetic_roles_original_stages_predictions_state_rng_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); obs, _ = observation(root / "obs")
            _, none = none_source(root / "source-none")
            sources = {"none": none}
            for mode in MODES:
                if mode != "none": _, sources[mode] = active_source(root / ("source-" + mode), source_mode=mode)
            for mode in MODES:
                for method in METHODS[:2]:
                    model, _ = initialize("xuzhou", 64, method, mode, source_spec=sources[mode] if method == METHODS[1] else None, device=DEVICE)
                    model.eval()
                    before = state_id(model); rng = torch.get_rng_state().clone()
                    with torch.no_grad(), OrderedValidation(): expected = tuple(v.detach().cpu().clone() for v in forward(model, "xuzhou", obs))
                    stages, predictions, proof = capture(model, "xuzhou", obs)
                    self.assertEqual(state_id(model), before); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertEqual(proof["stage_calls"], {n: 1 for n in STAGES})
                    self.assertFalse(set(STAGES) & model.backbone.__dict__.keys())
                    for a, b in zip(expected, predictions): torch.testing.assert_close(a, b, rtol=0, atol=0)
                    for stage, width in (("stage1", 128), ("stage2", 64)):
                        self.assertEqual(stages[stage]["cnn"].shape, (obs["Q"].shape[0], width))
                        self.assertEqual(stages[stage]["graph"].shape, (obs["Q"].shape[1], width))
                        torch.testing.assert_close(stages[stage]["Q"], obs["Q"].cpu(), rtol=0, atol=0)

    def test_observer_exception_and_policy_refusals_remove_temporary_attributes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); obs, _ = observation(root / "obs"); model, _ = initialize("xuzhou", 64, METHODS[0], "none", device=DEVICE)
            model.train()
            with self.assertRaises(ValueError): capture(model, "xuzhou", obs)
            model.eval(); bad = {**obs, "query_labels": torch.zeros(1)}
            with self.assertRaises(ValueError): capture(model, "xuzhou", bad)
            before, rng = state_id(model), torch.get_rng_state().clone()
            with patch.object(model.backbone.HGNN_Branch, "_graph_conv", side_effect=RuntimeError("synthetic injected stage failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic injected"):
                    capture(model, "xuzhou", obs)
            self.assertEqual(state_id(model), before); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(set(STAGES) & model.backbone.__dict__.keys())
            capture(model, "xuzhou", obs)

    def test_structural54_gate_before_load_negative_benefit_allowed_and_files_readonly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); freeze, analysis, name, paths = synthetic_boundary(root)
            pins = {p: file_hash(p) for p in root.rglob("*") if p.is_file()}
            def call(out): return export_case(freeze, file_hash(freeze), analysis, file_hash(analysis), name, out,
                code_snapshot(), device=DEVICE, maximum_regions=4, seed=64)
            original = json.loads(analysis.read_text())
            previous_analysis_code = {**original["code"],
                "commit": "73234283581c536f6e42c49bf0b3326ce3067d60",
                "source_hash": "a7dfd3e784d0463c45ed5dcbe0b9bbd91cdbeaf88d39d5a9cd13f7dcf3995cdf"}
            for key, value in (("models_analyzed", 53), ("new_optimizer_updates", 1), ("new_query_release", True),
                               ("code", previous_analysis_code),
                               ("code", {**original["code"], "source_hash": "0" * 64})):
                write_json(analysis, {**original, key: value})
                with patch("torch.load", side_effect=AssertionError("No archive before gate")):
                    with self.assertRaises(ValueError): call(root / "blocked")
                self.assertFalse((root / "blocked").exists())
            write_json(analysis, original)
            def safe_read(path):
                if Path(path) == paths[2]: raise AssertionError("Family labels must not be deserialized")
                return json.loads(Path(path).read_text())
            with patch("evaluation.frozen_representation_capture.read", side_effect=safe_read), patch("g4_matched_target_evaluation.read_query_labels", side_effect=AssertionError("No query reader")), patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")):
                receipt = call(root / "diagnostic")
            self.assertEqual(receipt["model_forwards"], 1); self.assertFalse(receipt["raw_labels_deserialized"])
            self.assertEqual(set(receipt["outputs"]), {s + suffix for s in ("stage1", "stage2") for suffix in ("-features.pt", "-summary.json")})
            self.assertIn("full_target_scene_transductive", receipt["scope"])
            for p, sha in pins.items(): self.assertEqual(file_hash(p), sha)
            with self.assertRaises(ValueError): call(root / "diagnostic")
            broken = json.loads(freeze.read_text()); broken["models"].pop(); write_json(freeze, broken)
            with patch("torch.load", side_effect=AssertionError("No old/incomplete boundary load")):
                with self.assertRaises(ValueError): call(root / "blocked")

    def test_actual_synthetic_cli_requires_boundary_and_no_query_training_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); freeze, analysis, name, _ = synthetic_boundary(root)
            out = root / "cli"
            command = [sys.executable, "scripts/export_frozen_representation_diagnostics.py", "--expected-commit", code_snapshot()["commit"],
                "--freeze", str(freeze), "--freeze-sha256", file_hash(freeze), "--analysis", str(analysis), "--analysis-sha256", file_hash(analysis),
                "--case", name, "--out", str(out), "--device", DEVICE, "--maximum-regions", "4", "--seed", "64", "--variance-floor", "1e-24"]
            result = json.loads(subprocess.check_output(command, text=True)); self.assertEqual(result["status"], "passed")
            for extra in (["--query"], ["--train"], ["--width", ".5"]):
                self.assertNotEqual(subprocess.run(command + extra, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
