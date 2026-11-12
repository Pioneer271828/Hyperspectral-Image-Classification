"""Run both fixture class counts through registration and authenticated recovery."""

from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from research_runtime import ROOT, code_snapshot
from tests.test_precision_recovery import fixture
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_real import complete, execute, load_registration, policies, register
from transfer.curriculum_real_audit import audit


class RealCurriculumTests(unittest.TestCase):
    def test_six_arms_on_nine_and_twenty_two_classes(self):
        parent = os.environ.get("WP01_CUR_S_FIXTURES")
        if parent:
            root = Path(parent)
            root.mkdir()
            self.matrix(root)
        else:
            with TemporaryDirectory() as directory:
                self.matrix(Path(directory))

    def matrix(self, parent):
        code = code_snapshot()
        for classes in (9, 22):
            with self.subTest(classes=classes):
                home = parent / str(classes)
                home.mkdir()
                request = fixture(home/"inputs", classes)
                root = home/"run"
                pin = register(request, root, code)
                _, reg, _, scene, record, _, _ = load_registration(pin, code)
                teacher = admission.read(reg["teacher"])
                self.assertEqual(teacher["classes"], list(range(1, classes+1)))
                self.assertEqual(teacher["forward_tiles"], len({r["tile"] for r in reg["support"]}))
                with self.assertRaisesRegex(ValueError, "original state"):
                    policies(dict(teacher, teacher_state="a"*64), scene, record, reg)
                with self.assertRaisesRegex(ValueError, "Teacher hash"):
                    policies(dict(teacher, support=dict(teacher["support"], seed=123)), scene, record, reg)
                self.assertEqual(reg["maximum_updates"], 48)
                cases = [x["case"] for x in reg["cases"]]
                notice = dev._json_new(root/"notice.json", dict(registration=pin, cases=cases, budget=4,
                                       maximum_updates=48, cpu_workers=1, estimate="synthetic fixture"))
                previous = None
                for case in cases:
                    execute(pin, code, notice, case, "full", None)
                    prefix = execute(pin, code, notice, case, "prefix", None)
                    if previous:
                        with self.assertRaisesRegex(ValueError, "Cross-arm"):
                            execute(pin, code, notice, case, "resume", previous)
                    if case.endswith("easy_to_hard"):
                        command = [sys.executable, str(ROOT/"scripts/run_registered_sample_curriculum.py"),
                                   "--expected-commit", code["commit"], "stage", "--registration", pin["path"], pin["sha256"],
                                   "--notice", notice["path"], notice["sha256"], "--case", case, "--stage", "resume",
                                   "--prior", prefix["path"], prefix["sha256"]]
                        result = subprocess.run(command, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        resumed = dev._pin(root/"attempts"/case/"resume/status.json")
                    else:
                        resumed = execute(pin, code, notice, case, "resume", prefix)
                    with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("extra update")):
                        execute(pin, code, notice, case, "completed", resumed)
                    with self.assertRaises(FileExistsError):
                        execute(pin, code, notice, case, "full", None)
                    previous = prefix
                result = admission.read(audit(pin, complete(pin, code, notice), code))
                self.assertEqual(result["actual_updates"], 48)
                self.assertTrue(result["off_balanced_exact"])
                self.assertEqual(len(result["cases"]), 6)
                self.assertEqual(result["cases"]["s64-easy_to_hard"]["selected"]["metrics"]["cnn"]["count"], classes*5)
                self.assertLess(result["cases"]["s64-easy_to_hard"]["exposure"]["minimum_applied_weight"], 1.)
                req = admission.read(request)
                for name, invalid in (("budget", dict(req, budget=17)),
                                      ("source", dict(req, weights=dict(req["weights"], sha256="0"*64)))):
                    new = dev._json_new(home/f"wrong-{name}.json", invalid)
                    with self.assertRaises(ValueError):
                        register(new, home/f"denied-{name}", code)
