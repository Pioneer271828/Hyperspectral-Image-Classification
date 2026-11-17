"""Exercise the real CLI and controller with CPU scenes and controlled GPU probes."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research_runtime import code_snapshot, hash_file
from scripts import audit_oss_c_real as audit
from scripts import run_oss_c_resources as queued
from scripts.run_oss_c_pair import campaign
from test_calibrated_selection import items


GPU0 = "GPU-00000000-0000-0000-0000-000000000000"
GPU1 = "GPU-10000000-0000-0000-0000-000000000001"
GPUS = (GPU0, GPU1)
BASE = dict(model=queued.MODEL, total_mib=24564, free_mib=24079, device_used_mib=5, utilization=0)


def controlled_items(_):
    data = items()
    data["xuzhou"]["wavelengths"][-1] = 2508.
    return data, {name: dict(fixture=True) for name in data}


def inventory(uuid=None):
    rows = {gpu: dict(BASE, index=i) for i, gpu in enumerate(GPUS)}
    if os.environ.get("OSS_C_FIXTURE_LOW") == "1":
        rows[GPU0]["free_mib"] = 4000
    return rows if uuid is None else {uuid: rows[uuid]}


def child_command(original):
    return [sys.executable, str(Path(__file__).resolve()), *original[2:]]


@contextlib.contextmanager
def fixture(runs):
    old_sha = hash_file(runs / queued.OLD_PATH)
    with patch.object(queued, "OLD_SHA", old_sha), patch.object(queued, "inventory", inventory), \
            patch.object(queued, "release", controlled_items), patch("scripts.run_oss_c_pair.release", controlled_items), \
            patch.object(audit, "release", controlled_items), patch.object(queued, "DEVICE", "cpu"), \
            patch.object(queued, "verify_source"):
        yield


class ResourceQueuesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = TemporaryDirectory()
        cls.addClassCleanup(cls.root.cleanup)
        cls.runs = Path(cls.root.name) / "runs"
        cls.runs.mkdir()
        (cls.runs / ".p4_gpu_leases").mkdir()
        old = cls.runs / queued.OLD_PATH
        old.parent.mkdir(parents=True)
        queued.create(old, campaign(cls.runs, {"commit": queued.OLD_COMMIT, "source_hash": queued.OLD_SOURCE},
                                    loaded=controlled_items(cls.runs)))
        cls.source = code_snapshot()
        cls.prefix = ["--expected-commit", cls.source["commit"], "--runs", str(cls.runs)]

    def test_full_cli_and_failure_guards(self):
        with fixture(self.runs), contextlib.redirect_stdout(io.StringIO()):
            with patch.dict(os.environ, {"OSS_C_FIXTURE_LOW": "1"}):
                with self.assertRaisesRegex(ValueError, "individually eligible"):
                    queued.main([*self.prefix, "register", "--gpu", GPU0])
            with queued.lease(self.runs, GPU0), self.assertRaises(BlockingIOError):
                queued.main([*self.prefix, "register", "--gpu", GPU0])
            registration = queued.main([*self.prefix, "register", "--gpu", GPU0, "--gpu", GPU1])
            self.assertEqual(sorted(registration["case_gpu"].values()).count(GPU0), 6)
            sha = registration["registration_sha256"]
            with self.assertRaises(FileExistsError):
                queued.main([*self.prefix, "register", "--gpu", GPU0])
            dispatch = [*self.prefix, "dispatch", "--registration-sha256", sha]
            make = queued.worker_command
            with patch.object(queued, "worker_command", side_effect=lambda *args: child_command(make(*args))):
                self.assertEqual(queued.main(dispatch)["status"], "completed")
            with self.assertRaisesRegex(ValueError, "already exist"):
                queued.main(dispatch)
            root = queued.registration_path(self.runs).parent
            execution = root / "execution"
            key = queued.KEYS[0]
            receipt = hash_file(execution / "receipts" / f"{key}.json")
            worker = [*self.prefix, "run", "--registration-sha256", sha, "--case", key, "--launch-sha256"]
            env = {"CUDA_VISIBLE_DEVICES": GPU0, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
            with patch.dict(os.environ, env):
                with self.assertRaisesRegex(ValueError, "independent launch"):
                    queued.main([*worker, "0" * 64])
                with self.assertRaisesRegex(ValueError, "Case already exists"):
                    queued.main([*worker, receipt])
                with queued.lease(self.runs, GPU0), self.assertRaises(BlockingIOError):
                    queued.main([*worker, receipt, "--resume"])
                with patch.dict(os.environ, {"OSS_C_FIXTURE_LOW": "1"}):
                    with self.assertRaisesRegex(ValueError, "Worker GPU admission"):
                        queued.main([*worker, receipt, "--resume"])
            with patch.dict(os.environ, dict(env, CUDA_VISIBLE_DEVICES=GPU1)):
                with self.assertRaisesRegex(ValueError, "visibility"):
                    queued.main([*worker, receipt, "--resume"])

            filtered = {"scripts/audit_oss_c_real.py", "tests/test_oss_c_real_audit.py", "tests/test_oss_c_resources.py"}
            from model.research_config import digest
            runtime_hash = digest({k: v for k, v in self.source["files"].items() if k not in filtered})
            args = [*self.prefix, "--runtime-commit", self.source["commit"], "--registration-sha256", sha]
            with patch.object(audit, "QUEUED_PRODUCER", self.source["commit"]), \
                    patch.object(audit, "QUEUED_SOURCE", self.source["source_hash"]), \
                    patch.object(audit, "QUEUED_RUNTIME_FILE_HASH", runtime_hash), \
                    patch.object(audit, "QUEUED_REGISTRATION", sha):
                report = audit.main([*args, "--out", str(self.runs / "audit.json")])
                self.assertEqual(report["real_updates"], 108)
                self.assertEqual(report["timing"]["completed_resume_updates"], 0)
                events = execution / "events.jsonl"
                original = events.read_text()
                self.assertEqual({json.loads(line)["uuid"] for line in original.splitlines()[:2]}, set(GPUS))
                lines = original.splitlines()
                for broken in (lines[:-1], lines[:-1] + [lines[-3]]):
                    events.write_text("\n".join(broken))
                    with self.assertRaises(ValueError):
                        audit.main([*args, "--out", str(self.runs / "invalid.json")])
                events.write_text(original)

    def test_worker_failure_stops_future_claims_and_preserves_other_active_queue(self):
        with TemporaryDirectory() as temp:
            runs = Path(temp) / "runs"
            runs.mkdir()
            (runs / ".p4_gpu_leases").mkdir()
            previous = runs / queued.OLD_PATH
            previous.parent.mkdir(parents=True)
            previous.write_bytes((self.runs / queued.OLD_PATH).read_bytes())
            prefix = ["--expected-commit", self.source["commit"], "--runs", str(runs)]
            with fixture(runs), contextlib.redirect_stdout(io.StringIO()):
                sha = queued.main([*prefix, "register", "--gpu", GPU0, "--gpu", GPU1])["registration_sha256"]
                make = queued.worker_command

                def fail_one(*args):
                    if args[3] == queued.KEYS[0]:
                        return [sys.executable, "-c", "import time; time.sleep(2); raise SystemExit(7)"]
                    return child_command(make(*args))

                with patch.object(queued, "worker_command", side_effect=fail_one):
                    with self.assertRaisesRegex(ValueError, "queue failed"):
                        queued.main([*prefix, "dispatch", "--registration-sha256", sha])
                root = queued.registration_path(runs).parent
                events = [json.loads(line) for line in (root / "execution/events.jsonl").read_text().splitlines()]
                started = [row["case"] for row in events if row["action"] == "start"]
                self.assertEqual(set(started), {queued.KEYS[0], queued.KEYS[3]})
                self.assertTrue((root / "cases" / queued.KEYS[3] / "result.json").is_file())
                self.assertEqual(queued.read(root / "execution/controller.json")["status"], "failed")


if __name__ == "__main__":
    args = sys.argv[1:]
    run_root = Path(args[args.index("--runs") + 1])
    with fixture(run_root):
        queued.main(args)
