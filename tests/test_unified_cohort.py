import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from evaluation.unified_cohort import HEADS, SCHEMA, summarize


ROOT = Path(__file__).resolve().parents[1]


def identity(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    timing = dict(environment_sha256=identity("fixture-host"), scope="end_to_end",
                  device="cuda", warmups=2, repeats=4)
    plan = dict(schema=SCHEMA, protocol_sha256=identity("evaluation-protocol"),
                scenes={"scene": dict(class_ids=[3, 1], workload_sha256=identity("work"), work_items=20)},
                seeds=[64, 512, 2048], arms=["base", "new"], baseline="base", timing=timing, cases=[])
    evidence = dict(timing=copy.deepcopy(timing), cases=[])
    for seed in plan["seeds"]:
        for arm in plan["arms"]:
            key = dict(scene="scene", seed=seed, arm=arm)
            evaluation = identity(f"evaluation-{seed}")
            checkpoint = identity(f"model-{seed}-{arm}")
            plan["cases"].append(dict(key, evaluation_sha256=evaluation, checkpoint_sha256=checkpoint))
            matrix = [[8, 2], [2, 8]] if arm == "base" else [[9, 1], [2, 8]]
            samples = [10, 12, 14, 16] if arm == "base" else [9, 11, 13, 15]
            evidence["cases"].append(dict(key,
                accuracy=dict(checkpoint_sha256=checkpoint, evaluation_sha256=evaluation,
                              heads={head: copy.deepcopy(matrix) for head in HEADS}),
                cost=dict(checkpoint_sha256=checkpoint, workload_sha256=identity("work"),
                          wall_ms=samples, peak_bytes=500, parameters=40, package_bytes=80)))
    return plan, evidence


class UnifiedCohortTests(unittest.TestCase):
    def test_three_seed_metrics_and_descriptive_cost(self):
        plan, evidence = fixture()
        before = copy.deepcopy((plan, evidence))
        evidence["cases"].reverse()
        report = summarize(plan, evidence)
        self.assertEqual((plan, list(reversed(evidence["cases"]))), (before[0], before[1]["cases"]))
        self.assertEqual(len(report["rows"]), 6)
        row = report["rows"][1]
        self.assertEqual(row["heads"]["cnn"]["class_ids"], [3, 1])
        self.assertAlmostEqual(row["heads"]["cnn"]["per_class"][0]["recall"], .9)
        self.assertAlmostEqual(row["heads"]["cnn"]["per_class"][0]["f1"], 18 / 21)
        self.assertAlmostEqual(row["heads"]["graph"]["kappa"], .7)
        self.assertEqual(row["cost"]["p50_ms"], 12)
        self.assertAlmostEqual(row["cost"]["p95_ms"], 14.7)
        self.assertAlmostEqual(row["cost"]["items_per_second"], 20000 / 12)
        summary = report["summaries"][1]
        self.assertAlmostEqual(summary["heads"]["fused"]["oa"]["paired_delta"]["mean"], .05)
        self.assertEqual(summary["cost"]["p50_ms"]["paired_delta"]["values"], [-1, -1, -1])

    def test_missing_and_duplicate_case_rejected(self):
        for replace in (False, True):
            plan, evidence = fixture()
            evidence["cases"].pop()
            if replace:
                evidence["cases"].append(evidence["cases"][0])
            with self.assertRaisesRegex(ValueError, "Incomplete|duplicate"):
                summarize(plan, evidence)

    def test_pairing_checkpoint_and_workload_rejected(self):
        for field in ("evaluation", "checkpoint", "workload"):
            plan, evidence = fixture()
            row = evidence["cases"][1]
            if field == "evaluation":
                # Even a consistently altered arm must use the same samples as its baseline.
                plan["cases"][1]["evaluation_sha256"] = row["accuracy"]["evaluation_sha256"] = identity("other")
            else:
                row["cost"][field + "_sha256"] = identity("other")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity differs"):
                summarize(plan, evidence)

    def test_timing_scope_or_repeats_rejected(self):
        plan, evidence = fixture()
        evidence["timing"]["scope"] = "resident_forward"
        with self.assertRaisesRegex(ValueError, "scope"):
            summarize(plan, evidence)
        evidence["timing"]["scope"] = "end_to_end"
        evidence["cases"][0]["cost"]["wall_ms"].pop()
        with self.assertRaisesRegex(ValueError, "repeats"):
            summarize(plan, evidence)

    def test_class_support_and_counts_rejected(self):
        for matrix, message in (([[0, 0], [2, 8]], "evaluation support"),
                                ([[8, 1], [2, 8]], "class counts"),
                                ([[2**62, 0], [2, 8]], "counts"),
                                ([[True, 0], [2, 8]], "counts")):
            plan, evidence = fixture()
            evidence["cases"][1]["accuracy"]["heads"]["graph"] = matrix
            with self.subTest(matrix=matrix), self.assertRaisesRegex(ValueError, message):
                summarize(plan, evidence)

    def test_cli_pins_output_and_preserves_inputs(self):
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            plan, evidence = fixture()
            command = [sys.executable, str(ROOT / "scripts/summarize_unified_cohort.py"),
                       "--expected-commit", commit]
            original = {}
            for name, value in (("plan", plan), ("evidence", evidence)):
                path = folder / (name + ".json")
                data = json.dumps(value).encode()
                path.write_bytes(data)
                original[path] = data
                command += ["--" + name, str(path), "--" + name + "-sha256", hashlib.sha256(data).hexdigest()]
            out = folder / "report.json"
            command += ["--out", str(out)]
            altered = command.copy()
            altered[altered.index("--evidence-sha256") + 1] = "0" * 64
            result = subprocess.run(altered, cwd=ROOT, capture_output=True, text=True)
            self.assertIn("Input SHA-256 differs", result.stderr)
            self.assertFalse(out.exists())
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = out.read_bytes()
            self.assertEqual(json.loads(report)["consumer_commit"], commit)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(out.read_bytes(), report)
            self.assertEqual({path: path.read_bytes() for path in original}, original)
