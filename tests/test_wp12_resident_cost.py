import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from evaluation.wp12_resident_cost import (JOB_POLICY, OBSERVATION, assemble, environment, measure)
from evaluation.wp12_evidence import COST_POLICY, join_hardware
from joint_baseline import forward, load_config
from joint_training import _tree_fingerprint
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from research_runtime import ROOT, hash_file
from tests.test_joint_baseline import tiny_items
from tests.test_wp12_evidence import hardware_fixture, frozen_fixture, load_fixture


def fixture(root, device="cpu"):
    checkpoint, observation = root / "model.pt", root / "observation.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(64)
        model = load_config("configs/experiments/base97_joint.json").build_model({"paviaU": 9}).eval()
    save_model_checkpoint(model, checkpoint)
    item = {k: tiny_items()["paviaU"][k] for k in OBSERVATION.split()}
    torch.save(item, observation)
    def spec(path):
        return dict(path=str(path), sha256=hash_file(path))
    timing = dict(environment_sha256=digest(environment(device, "ordered")), scope="resident_forward",
                  device=device, warmups=1, repeats=4)
    job = dict(policy=JOB_POLICY, case=dict(scene="paviaU", seed=64, arm="fixture",
               evaluation_sha256="e" * 64, checkpoint_sha256=hash_file(checkpoint)), timing=timing,
               arithmetic="ordered", checkpoint=spec(checkpoint), sidecar=spec(Path(str(checkpoint) + ".json")),
               workload=dict(file=spec(observation), tensor_sha256=_tree_fingerprint(item), work_items=36))
    return job


def run_cli(args):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return subprocess.run([sys.executable, str(ROOT / "scripts/measure_wp12_resident_cost.py"),
                           "--expected-commit", commit, *map(str, args)], capture_output=True, text=True)


class ResidentCostTests(unittest.TestCase):
    def test_native_model_measurement_exact_inputs_and_weights(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job = fixture(root)
            pins = {p: hash_file(p) for p in root.iterdir()}
            with patch("torch.optim.Optimizer.step", side_effect=AssertionError("No training")):
                result = measure(job)
            self.assertEqual(len(result["row"]["cost"]["wall_ms"]), 4)
            self.assertTrue(all(v > 0 for v in result["row"]["cost"]["wall_ms"]))
            self.assertEqual(result["row"]["cost"]["package_bytes"],
                             (root / "model.pt").stat().st_size + (root / "model.pt.json").stat().st_size)
            self.assertGreater(result["row"]["cost"]["peak_bytes"], 0)
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertFalse(result["query_labels_accessed"])
            self.assertEqual({p: hash_file(p) for p in pins}, pins)

    def test_reject_changed_pins_scope_budget_and_environment_before_loading(self):
        with TemporaryDirectory() as directory:
            original = fixture(Path(directory))
            for defect in ("checkpoint", "sidecar", "input", "scope", "environment", "budget", "items"):
                job = copy.deepcopy(original)
                if defect in ("checkpoint", "sidecar"):
                    job[defect]["sha256"] = "0" * 64
                elif defect == "input":
                    job["workload"]["file"]["sha256"] = "0" * 64
                elif defect == "scope":
                    job["timing"]["scope"] = "end_to_end"
                elif defect == "environment":
                    job["timing"]["environment_sha256"] = "0" * 64
                elif defect == "budget":
                    job["timing"]["repeats"] = 3
                else:
                    job["workload"]["work_items"] = 0
                with self.subTest(defect=defect), patch("torch.load", side_effect=AssertionError("No decoding")), \
                        self.assertRaises(ValueError):
                    measure(job)

    def test_labels_and_wrong_workload_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = fixture(root)
            path = root / "observation.pt"
            item = torch.load(path, weights_only=True)
            for defect in ("labels", "count", "fingerprint"):
                job = copy.deepcopy(original)
                payload = copy.deepcopy(item)
                if defect == "labels":
                    payload["query_labels"] = torch.ones(36)
                elif defect == "count":
                    job["workload"]["work_items"] = 37
                else:
                    payload["net_input"] += 1
                torch.save(payload, path)
                job["workload"]["file"]["sha256"] = hash_file(path)
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    measure(job)

    def test_model_mutation_and_nonfinite_output_rejected(self):
        with TemporaryDirectory() as directory:
            job = fixture(Path(directory))
            def mutate(model, name, item):
                next(model.parameters()).add_(.01)
                return forward(model, name, item)
            def bad_output(model, name, item):
                return (torch.full((36, 9), float("nan")),) * 3
            for fn in (mutate, bad_output):
                with self.subTest(fn=fn.__name__), patch("evaluation.wp12_resident_cost.forward", fn), \
                        self.assertRaises(ValueError):
                    measure(job)

    def test_fresh_process_cli_output_and_exclusive_write(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job = fixture(root)
            env_path = root / "environment.json"
            env = run_cli(["--out", env_path, "environment", "--device", "cpu", "--arithmetic", "ordered"])
            self.assertEqual(env.returncode, 0, env.stderr)
            job["timing"]["environment_sha256"] = digest(json.loads(env_path.read_text()))
            job_path, output = root / "job.json", root / "result.json"
            job_path.write_text(json.dumps(job))
            args = ["--out", output, "measure", "--job", job_path, "--job-sha256", hash_file(job_path)]
            run = run_cli(args)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(output.read_text())
            self.assertTrue(result["fresh_process"])
            self.assertEqual(result["job_sha256"], hash_file(job_path))
            before = output.read_bytes()
            self.assertNotEqual(run_cli(args).returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            job_path.write_text(job_path.read_text() + " ")
            args[1] = root / "rejected.json"
            self.assertNotEqual(run_cli(args).returncode, 0)
            self.assertFalse(args[1].exists())

    def test_complete54_assembly_into_accuracy_consumer_and_negative_cases(self):
        # Synthetic timings only. Actual native worker execution is tested above.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            accuracy = load_fixture(frozen_fixture(root))
            plan, receipt = hardware_fixture(accuracy)
            native_dir = root / "native"
            native_dir.mkdir()
            job = fixture(native_dir)
            env = environment("cpu", "ordered")
            plan["timing"] = job["timing"]
            results = []
            for registered, row in zip(plan["cases"], receipt["cases"]):
                bound = copy.deepcopy(job)
                bound["case"] = registered
                bound["checkpoint"]["sha256"] = registered["checkpoint_sha256"]
                bound["workload"]["tensor_sha256"] = "b" * 64
                bound["workload"]["work_items"] = 100
                results.append(dict(status="passed", job=bound, environment=env, row=row,
                                    optimizer_updates=0, query_labels_accessed=False))
            assembled = assemble(plan, results, plan_sha256="c" * 64)
            report = join_hardware(accuracy, plan, assembled, cost_plan_sha="c" * 64,
                                   baseline=accuracy["arms"][0])
            self.assertEqual(len(report["rows"]), 54)
            self.assertEqual(len(report["summaries"]), 18)
            for defect in ("missing", "duplicate", "environment", "identity", "checkpoint", "memory", "nan"):
                rows = copy.deepcopy(results)
                if defect == "missing":
                    rows.pop()
                elif defect == "duplicate":
                    rows[-1] = rows[0]
                elif defect == "environment":
                    rows[0]["environment"]["arithmetic"] = "native"
                elif defect == "identity":
                    rows[0]["job"]["case"]["evaluation_sha256"] = "0" * 64
                elif defect == "checkpoint":
                    rows[0]["row"]["cost"]["checkpoint_sha256"] = "0" * 64
                elif defect == "memory":
                    rows[0]["row"]["cost"]["peak_bytes"] = 0
                else:
                    rows[0]["row"]["cost"]["wall_ms"][0] = float("nan")
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    assemble(plan, rows, plan_sha256="c" * 64)

    @unittest.skipUnless(os.environ.get("WP12_COST_DEVICE") == "cuda", "Explicit eligible GPU required")
    def test_cuda_native_worker(self):
        with TemporaryDirectory() as directory:
            result = measure(fixture(Path(directory), "cuda"))
            self.assertGreater(result["row"]["cost"]["peak_bytes"], 0)
            self.assertEqual(result["environment"]["memory"],
                             "measured_forward_peak_allocated_bytes_including_resident_tensors")


if __name__ == "__main__":
    unittest.main()
