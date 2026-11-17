"""Run both supported fixture class counts through the frozen rank ladder."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from compression.low_rank_recovery import LowRankRecoveryRun, ladder
from compression.low_rank_recovery_audit import audit
from compression.registered_low_rank_recovery import complete, execute, initial_model, load_registration, register
from research_runtime import ROOT, code_snapshot
from tests.test_precision_recovery import fixture
from transfer import matched_route_admission as admission, registered_development as dev


class LowRankRecoveryTests(unittest.TestCase):
    def test_matched_9_and_22_with_independent_factors(self):
        code = code_snapshot()
        persistent = os.environ.get("WP10_LR_SYNTHETIC")
        if persistent:
            root = Path(persistent)
            root.mkdir()
            self.matrix(root, code)
        else:
            with TemporaryDirectory() as directory:
                self.matrix(Path(directory), code)

    def matrix(self, root, code):
        for classes in (9, 22):
            with self.subTest(classes=classes):
                path = root / str(classes)
                path.mkdir()
                spec = admission.read(fixture(path / "inputs", classes))
                del spec["prefix"]
                request = dev._json_new(path / "request.json", spec)
                attempt = path / "run"
                registration = register(request, attempt, code)
                _, value, dense, scene, record, plans = load_registration(registration, code)
                cases = [c["case"] for c in value["cases"]]
                self.assertEqual(ladder(dense)["shape"], [128, 64])
                self.assertEqual(len(cases), 6)
                notice = dev._json_new(attempt / "notice.json", dict(registration=registration, cases=cases,
                                       budget=4, maximum_updates=48, cpu_workers=1, estimate="bounded fixture"))
                previous = None
                for case in cases:
                    execute(registration, code, notice, case, "full", None)
                    prefix = execute(registration, code, notice, case, "prefix", None)
                    if previous is not None:
                        with self.assertRaisesRegex(ValueError, "Cross-arm"):
                            execute(registration, code, notice, case, "resume", previous)
                    if classes == 9 and case.endswith("svd-r16"):
                        argv = [sys.executable, str(ROOT / "scripts/run_low_rank_recovery.py"), "--expected-commit",
                                code["commit"], "stage", "--registration", registration["path"], registration["sha256"],
                                "--notice", notice["path"], notice["sha256"], "--case", case, "--stage", "resume",
                                "--prior", prefix["path"], prefix["sha256"]]
                        process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
                        (path / "resume-process.log").write_text(process.stdout + process.stderr)
                        self.assertEqual(process.returncode, 0, process.stderr)
                        resumed = dev._pin(attempt / "attempts" / case / "resume/status.json")
                    else:
                        resumed = execute(registration, code, notice, case, "resume", prefix)
                    with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("completed optimizer call")):
                        execute(registration, code, notice, case, "completed", resumed)
                    with self.assertRaises(FileExistsError):
                        execute(registration, code, notice, case, "full", None)
                    previous = prefix
                proof = admission.read(audit(registration, complete(registration, code, notice), code))
                self.assertEqual(proof["actual_updates"], 48)
                self.assertEqual(proof["independent_updates"], 0)
                self.assertEqual(proof["report"]["s64-full-rank"]["selected"]["metrics"]["fused"]["count"], 5*classes)
                self.assertLess(max(proof["report"]["s64-full-rank"]["initial_full_rank_probability_max_abs"].values()), 1e-5)
                for case in cases[1:]:
                    row = proof["report"][case]
                    self.assertGreater(row["target_activity"]["left"]["changed"], 0)
                    self.assertGreater(row["target_activity"]["right"]["gradient_nonzero"], 0)
                    self.assertEqual(row["selected"]["factor_calls"]["left"], row["selected"]["forwards"])
                self.assertGreater(proof["report"]["s64-full-rank"]["selected"]["package"]["parameters"],
                                   proof["report"]["s64-dense"]["selected"]["package"]["parameters"])

                chosen = next(c for c in value["cases"] if c["arm"] == "svd-r16")
                seed = initial_model(attempt, value, dense, chosen)
                checkpoint = attempt / "attempts/s64-svd-r16/full/step-4.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                args = dict(plan=plans["64"], trial=chosen["case"], selector_seed=64,
                            expected_contract_hash=value["contracts"][chosen["case"]])
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    LowRankRecoveryRun.load(seed, scene, record, checkpoint, expected_sha256="0"*64, **args)
                saved["selected_step"] = 0
                wrong = path / "bad-selected.pt"
                torch.save(saved, wrong)
                with self.assertRaisesRegex(ValueError, "selection"):
                    LowRankRecoveryRun.load(seed, scene, record, wrong, expected_sha256=dev._pin(wrong)["sha256"], **args)
                wrong_case = dict(chosen, assignment=dict(chosen["assignment"], rank=32))
                with self.assertRaisesRegex(ValueError, "Factor dimensions"):
                    initial_model(attempt, value, dense, wrong_case)
                with self.assertRaisesRegex(ValueError, "bounded synthetic"):
                    register(dev._json_new(path/"overspend.json", dict(spec, budget=513)), path/"rejected", code)
                wrong_source = dict(spec, weights=dict(spec["weights"], sha256="0"*64))
                with self.assertRaises(ValueError):
                    register(dev._json_new(path/"wrong-source.json", wrong_source), path/"wrong-source", code)
