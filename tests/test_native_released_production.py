"""Run approved fixture releases through the real CLI, transport and training path."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import patch

import torch

from model.research_config import digest
from research_runtime import code_snapshot
from scripts import run_native_released_points as cli
from tests.native_released_bank_fixture import fixture
from transfer import native_released_real as real, p4_runtime_device as p4, registered_development as dev
from transfer.native_scale_campaign import read, packet
from transfer.xuzhou_multiscale_audit import _exact


def release_for(root, preflight, case, target, code, route, gpu=None):
    return dev._json_new(root / (target.name + "-release.json"), {
        "schema": real.ENGINEERING_RELEASE, "decision": "approved", "source_commit": code["commit"],
        "source_hash": code["source_hash"], "preflight": preflight, "case_sha256": digest(case),
        "destination": str(target), "optimizer_updates": case["planned_updates"],
        "physical_uuid": gpu, "protocol": p4.CPU if gpu is None else p4.CUDA_FP64_WEIGHT,
        "raw_gt_authorized": False, "query_authorized": False, "route": route})


class ProductionTests(TestCase):
    def test_authorized_cli_runs_both_routes_and_retains_terminal_intent(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as name:
            root = Path(name)
            code = code_snapshot()
            pins, intake = fixture(root, code)
            # Distinct samples, scopes and approvals, with the actual CLI parser.
            for scene in pins:
                pin = pins[scene]
                roster = read(pin, root)["draft"]["cases"]
                keys = {(row["model_seed"], row["factor"], row["arm"]) for row in roster}
                self.assertEqual(len(keys), 15)
                for i in (0, 1, 7, 13, 14):
                    case = roster[i]
                    destinations = {}
                    for route, phases in (("full", ("full", "completed")), ("split", ("prefix", "resume", "completed"))):
                        out = root / f"{scene}-{i}-{route}"
                        release = release_for(root, pin, case, out, code, route)
                        command = ["--mode", "real-stage", "--runs", str(root), "--out", str(out),
                            "--expected-commit", code["commit"], "--preflight", pin["path"], pin["sha256"],
                            "--release", release["path"], release["sha256"], "--case", case["id"], "--phase"]
                        for phase in phases:
                            cli.main([*command, phase], intake=intake)
                        n = read(dev._pin(out / "completed.json"), root)
                        self.assertEqual(n["updates"], 0)
                        with self.assertRaisesRegex(ValueError, "one training path"):
                            cli.main([*command, "completed"], intake=intake)
                        destinations[route] = packet(n["checkpoint"], root)
                    a, b = destinations.values()
                    self.assertTrue(all(_exact(a[key], b[key]) for key in a if key != "registration"))
                    self.assertEqual(a["selected"]["step"], min(a["validation"], key=lambda v: v["fused_nll"])["step"])
                    self.assertEqual([v["step"] for v in a["validation"]], case["events"])
                    self.assertEqual(sum(v["supervised_area"] for v in a["history"]), sum(case["train_exposures_per_class"]))
            scene = "xuzhou"
            pin = pins[scene]
            case = read(pin, root)["draft"]["cases"][0]
            out = root / "claim-after-failure"
            release = release_for(root, pin, case, out, code, "full")
            entered, proceed = Event(), Event()
            actual = cli.training.Run.update

            def fault(run, event):
                if event["step"] == 2:
                    entered.set()
                    if not proceed.wait(15):
                        raise AssertionError("Competing stage did not return")
                    raise RuntimeError("synthetic mid-update failure")
                return actual(run, event)

            def call():
                return cli.real_stage(root, out, pin, release, case["id"], "full", code, None, intake=intake)

            with self.assertRaisesRegex(ValueError, "unrelated provenance"):
                cli.real_stage(root, out, pin, release, case["id"], "full", code, None)
            with patch.object(cli.training.Run, "update", fault), ThreadPoolExecutor(max_workers=1) as pool:
                job = pool.submit(call)
                self.assertTrue(entered.wait(15))
                try:
                    with self.assertRaises(FileExistsError):
                        call()
                finally:
                    proceed.set()
                with self.assertRaisesRegex(RuntimeError, "mid-update failure"):
                    job.result()
            self.assertTrue((out / "full.intent.json").is_file())
            self.assertFalse((out / "full.json").exists())
            with self.assertRaisesRegex(ValueError, "one training path"):
                call()
