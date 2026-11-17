import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.wp12_evidence import COST_POLICY, accuracy_records, join_hardware, legacy_cpu_cost
from g4_matched_target_analysis import PLAN_SHA256, POLICY
from research_runtime import hash_file
from tests.test_g4_matched_target_analysis import files_fixture


def frozen_fixture(root):
    query_path, freeze_path, _ = files_fixture(root, {"commit": "synthetic"})
    query = json.loads(query_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    for model in freeze["models"]:
        model["family_hash"] = "1" * 64
    for row in query["models"]:
        row["release_sha256"] = row["checkpoint_sha256"]
    analysis = dict(policy=POLICY, plan_sha256=PLAN_SHA256, models=query["models"],
                    benefit_conclusion="fixture_no_effect")
    receipt = dict(policy=POLICY, status="passed", plan_sha256=PLAN_SHA256,
                   freeze_sha256="f" * 64, outputs={"analysis.json": "a" * 64},
                   models_analyzed=54, heads_analyzed=162, new_optimizer_updates=0,
                   new_query_release=False, model_inference_executed=False, raw_labels_deserialized=False,
                   input_query_metrics_consumed=True)
    return analysis, receipt, freeze


def load_fixture(values):
    return accuracy_records(*values, analysis_sha="a" * 64, freeze_sha="f" * 64)


def hardware_fixture(accuracy):
    timing = dict(environment_sha256="e" * 64, scope="end_to_end", device="cuda", warmups=2, repeats=4)
    workloads, registrations, measured = {}, [], []
    for row in accuracy["cases"]:
        scene = row["scene"]
        workloads[scene] = {"workload_sha256": "b" * 64, "work_items": 100}
        key = {k: row[k] for k in ("scene", "seed", "arm")}
        registrations.append(dict(key, checkpoint_sha256=row["checkpoint_sha256"],
                                  evaluation_sha256=row["evaluation_sha256"]))
        measured.append(dict(key, cost=dict(checkpoint_sha256=row["checkpoint_sha256"],
                             workload_sha256="b" * 64, wall_ms=[1, 2, 3, 4], peak_bytes=200,
                             parameters=12, package_bytes=140)))
    plan = dict(policy=COST_POLICY, protocol_sha256=accuracy["protocol_sha256"], timing=timing,
                workloads=workloads, cases=registrations)
    receipt = dict(policy=COST_POLICY, status="passed", timing=copy.deepcopy(timing),
                   plan_sha256="c" * 64, cases=measured)
    return plan, receipt


class WP12EvidenceTests(unittest.TestCase):
    def test_full_accuracy_and_hardware_join(self):
        with TemporaryDirectory() as directory:
            with patch("torch.load", side_effect=AssertionError("No tensor decoding")), \
                    patch("torch.nn.Module._call_impl", side_effect=AssertionError("No inference")):
                accuracy = load_fixture(frozen_fixture(Path(directory)))
            rows = accuracy["cases"]
            self.assertEqual(len(rows), 54)
            self.assertEqual(len({row["evaluation_sha256"] for row in rows[:18]}), 1)
            self.assertEqual(len({row["release_sha256"] for row in rows[:18]}), 18)
            plan, receipt = hardware_fixture(accuracy)
            report = join_hardware(accuracy, plan, receipt, cost_plan_sha="c" * 64, baseline=accuracy["arms"][0])
            self.assertEqual(len(report["rows"]), 54)
            self.assertEqual(len(report["summaries"]), 18)
            self.assertEqual(report["rows"][0]["cost"]["p50_ms"], 2.5)
            self.assertEqual(report["rows"][0]["cost"]["items_per_second"], 40000)

    def test_wrong_accuracy_output_checkpoint_and_family(self):
        with TemporaryDirectory() as directory:
            values = frozen_fixture(Path(directory))
            for error in ("receipt", "checkpoint", "family"):
                analysis, receipt, freeze = copy.deepcopy(values)
                if error == "receipt":
                    receipt["outputs"]["analysis.json"] = "0" * 64
                if error == "checkpoint":
                    analysis["models"][1]["checkpoint_sha256"] = "0" * 64
                if error == "family":
                    freeze["models"][1]["family_hash"] = "0" * 64
                with self.subTest(error=error), self.assertRaises(ValueError):
                    load_fixture((analysis, receipt, freeze))

    def test_hardware_join_rejects_mismatched_and_incomplete_cases(self):
        with TemporaryDirectory() as directory:
            accuracy = load_fixture(frozen_fixture(Path(directory)))
            for error in ("plan", "receipt", "missing", "scope", "checkpoint", "peak"):
                plan, receipt = hardware_fixture(accuracy)
                if error == "plan":
                    plan["cases"][0]["evaluation_sha256"] = "0" * 64
                if error == "receipt":
                    receipt["plan_sha256"] = "0" * 64
                if error == "missing":
                    receipt["cases"].pop()
                if error == "scope":
                    receipt["timing"]["scope"] = "resident_forward"
                if error == "checkpoint":
                    receipt["cases"][0]["cost"]["checkpoint_sha256"] = "0" * 64
                if error == "peak":
                    receipt["cases"][0]["cost"].pop("peak_bytes")
                with self.subTest(error=error), self.assertRaises(ValueError):
                    join_hardware(accuracy, plan, receipt, cost_plan_sha="c" * 64, baseline=accuracy["arms"][0])

    def test_legacy_samples_preserved_without_fabricated_peak(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models, pins = {}, {}
            for role in ("dense", "compact"):
                folder = root / role
                folder.mkdir()
                weight = folder / "selected-state.pt"
                weight.write_text(role)
                models[role] = dict(parameters=12, state_tensor_bytes=55)
                pins[str(weight)] = hash_file(weight)
            plan = dict(target="paviaU", input_sha256="b" * 64, threads=1,
                        scope="resident_cpu_full_scene_native_three_head_forward", models=models, pins=pins,
                        order=[["dense", "compact"], ["compact", "dense"]] * 2)
            receipt = dict(status="passed", plan_sha256="a" * 64, optimizer_updates=0,
                           query_labels_accessed=False, samples=[{r: {"wall_ms": 1} for r in models}] * 4,
                           summary={r: {"wall_p50_ms": 1, "wall_p95_ms": 1} for r in models})
            execution = dict(status="passed", receipt_sha256="c" * 64, cpu="fixture")
            result = legacy_cpu_cost(plan, receipt, execution, plan_sha="a" * 64, receipt_sha="c" * 64)
            self.assertEqual(result["missing"], ["peak_bytes", "package_bytes"])
            self.assertFalse(result["eligible_for_joint_report"])
            (root / "dense/selected-state.pt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checkpoint pin"):
                legacy_cpu_cost(plan, receipt, execution, plan_sha="a" * 64, receipt_sha="c" * 64)
