"""Exercise the full campaign with fabricated worker records and frozen fixtures."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.wp12_cost_preparation import prepare
from evaluation.wp12_resident_cost import MEMORY
from research_runtime import hash_file, write_json
from scripts.run_wp12_cost_campaign import execute, register, source_accuracy
from tests.test_wp12_cost_preparation import fixture


def spec(path):
    return {"path": str(path), "sha256": hash_file(path)}


def setup(root):
    analysis, receipt, freeze, launch, code = fixture(root)
    paths = {name: root / (name + ".json") for name in ("analysis", "receipt", "freeze")}
    write_json(paths["analysis"], analysis)
    write_json(paths["freeze"], freeze)
    receipt.update(freeze_sha256=hash_file(paths["freeze"]), outputs={"analysis.json": hash_file(paths["analysis"])})
    write_json(paths["receipt"], receipt)
    out = root / "prepared"
    prepare(analysis, receipt, freeze, launch, code, out, analysis_sha256=hash_file(paths["analysis"]),
            freeze_sha256=hash_file(paths["freeze"]), launch_sha256=freeze["launch_sha256"])
    return dict(preparation=spec(out / "preparation.json"), accuracy={k: spec(p) for k, p in paths.items()})


class CostCampaignTests(unittest.TestCase):
    def test_resume_and_join_all54(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = setup(root)
            out = root / "campaign"
            registration = register(inputs, out)
            self.assertEqual(len(registration["cases"]), 54)
            self.assertNotEqual(registration["cases"][0]["arm"], registration["cases"][6]["arm"])
            accuracy, _ = source_accuracy(inputs["accuracy"])
            identity = {"runtime_commit": "1" * 40}
            env = {"arithmetic": "ordered", "memory": MEMORY["cuda"]}
            calls = []

            def worker(job_path, sha, dest):
                job = json.loads(job_path.read_text())
                calls.append(job["case"])
                row = {k: job["case"][k] for k in ("scene", "seed", "arm")}
                row["cost"] = dict(checkpoint_sha256=job["case"]["checkpoint_sha256"],
                                   workload_sha256=job["workload"]["tensor_sha256"],
                                   wall_ms=[2] * job["timing"]["repeats"], peak_bytes=300,
                                   parameters=20, package_bytes=400)
                write_json(dest, dict(status="passed", job=job, job_sha256=sha, environment=env, row=row,
                                      producer_commit=registration["worker_commit"], fresh_process=True,
                                      optimizer_updates=0, query_labels_accessed=False, **identity))

            def run(invoke, limit=None):
                return execute(registration, out, accuracy, env, identity, invoke, max_cases=limit, finish=lambda: None)

            self.assertEqual(run(worker, 1), 1)
            first = hash_file(out / "accepted-01.json")
            self.assertFalse((out / "combined-report.json").exists())

            def wrong_worker(job, sha, dest):
                worker(job, sha, dest)
                result = json.loads(dest.read_text())
                result["job_sha256"] = "0" * 64
                write_json(dest, result)

            with self.assertRaisesRegex(ValueError, "registered job"):
                run(wrong_worker)
            self.assertFalse((out / "accepted-02.json").exists())
            self.assertFalse((out / "combined-report.json").exists())
            self.assertEqual(run(worker), 54)
            self.assertEqual(hash_file(out / "accepted-01.json"), first)
            self.assertEqual(len(calls), 55)
            combined = json.loads((out / "combined-report.json").read_text())["combined"]
            self.assertEqual(len(combined["rows"]), 54)
            self.assertEqual(run(lambda *_: self.fail("accepted cases must not rerun")), 54)
            artifact = json.loads((out / "accepted-01.json").read_text())
            Path(artifact["path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run(worker)

    def test_register_rejects_preparation_with_wrong_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = setup(root)
            path = Path(inputs["preparation"]["path"])
            prepared = json.loads(path.read_text())
            prepared["cases"][0]["checkpoint"] = prepared["cases"][1]["checkpoint"]
            write_json(path, prepared)
            inputs["preparation"] = spec(path)
            with self.assertRaisesRegex(ValueError, "match the freeze"):
                register(inputs, root / "rejected")
            self.assertFalse((root / "rejected").exists())
