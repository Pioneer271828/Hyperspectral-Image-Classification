"""Exercise the distinct registered observation execution on accepted 9/22 fixtures."""
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from research_runtime import ROOT, code_snapshot
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_observation_real import complete, execute, load_registration, register
from transfer.curriculum_observation_audit import audit, observed_exposure
from transfer.xuzhou_multiscale_audit import _packet
from transfer.curriculum_observation_real import MeasuredObservationRun


@unittest.skipUnless(os.environ.get("WP06_O_ADMISSIONS"), "requires accepted observation fixtures")
class RealObservationTests(unittest.TestCase):
    def test_nine_and_twenty_two_real_network_recovery(self):
        fixture = Path(os.environ["WP06_O_ADMISSIONS"])
        output = os.environ.get("WP01_O_FIXTURES")
        if output:
            root = Path(output); root.mkdir()
            self.matrix(fixture, root)
        else:
            with TemporaryDirectory() as d:
                self.matrix(fixture, Path(d))

    def matrix(self, fixture, parent):
        code = code_snapshot()
        for k in (9,22):
            with self.subTest(classes=k):
                prior = dev._pin(fixture/str(k)/"observation/registration.json")
                root = parent/str(k)
                wrong = dict(prior, sha256="0"*64)
                with self.assertRaises(ValueError):
                    register(wrong, parent/f"denied-{k}", code)
                pin = register(prior, root, code)
                _,reg,_,scene,record,plans,trials = load_registration(pin,code)
                names = [x["case"] for x in reg["cases"]]
                notice = dev._json_new(root/"notice.json",dict(registration=pin,cases=names,budget=4,
                                       maximum_updates=32,cpu_workers=1,estimate="synthetic fixture"))
                old = None
                for case in names:
                    with self.assertRaises(ValueError):
                        execute(pin,code,notice,case,"resume",None)
                    execute(pin,code,notice,case,"full",None)
                    prefix = execute(pin,code,notice,case,"prefix",None)
                    if old:
                        with self.assertRaisesRegex(ValueError,"Cross-arm"):
                            execute(pin,code,notice,case,"resume",old)
                    if case.endswith("fixed"):
                        cmd = [sys.executable, str(ROOT/"scripts/run_registered_observation_curriculum.py"),
                               "--expected-commit",code["commit"],"stage","--registration",pin["path"],pin["sha256"],
                               "--notice",notice["path"],notice["sha256"],"--case",case,"--stage","resume",
                               "--prior",prefix["path"],prefix["sha256"]]
                        child = subprocess.run(cmd,capture_output=True,text=True)
                        self.assertEqual(child.returncode,0,child.stderr)
                        resumed = dev._pin(root/"attempts"/case/"resume/status.json")
                    else:
                        resumed = execute(pin,code,notice,case,"resume",prefix)
                    with patch.object(torch.optim.Adam,"step",side_effect=AssertionError("extra update")):
                        execute(pin,code,notice,case,"completed",resumed)
                    old = prefix
                row = admission.read(dev._pin(root/"attempts"/names[0]/"full/status.json"))
                full = _packet(row,reg["contracts"][names[0]],4,packet_format=MeasuredObservationRun.FORMAT)
                full["history"][0]["forward_evidence"]["spectral_sha256"] = "0"*64
                with self.assertRaisesRegex(ValueError,"selected-band input"):
                    observed_exposure(full["history"],plans["64"],trials[names[0]][1],scene,record,{})
                evidence = admission.read(audit(pin,complete(pin,code,notice),code))
                self.assertEqual(evidence["actual_updates"],32)
                self.assertEqual(evidence["training_v5_events"],32)
                self.assertEqual(evidence["restore_v5_events"],16)
                self.assertEqual(evidence["source"]["random_25"]["metrics"]["graph"]["count"],5*k)
                self.assertEqual(len(evidence["guards"]),4)
