"""Synthetic three-class-count factorial execution, original controls and arithmetic oracle."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from research_runtime import ROOT,code_snapshot
from tests.test_curriculum_cli import prepared
from transfer import matched_route_admission as admission,registered_development as dev
from transfer.curriculum_cli import _build
from transfer.curriculum_exposure import ExposurePlan
from transfer.curriculum_factorial import ARMS,FactorialRun,complete,execute,load_registration,register
from transfer.curriculum_factorial_audit import audit,effects
from transfer.curriculum_observation_admission import new_run as old_o
from transfer.curriculum_real import new_run as old_s
from transfer.curriculum_training import CurriculumRun
from transfer.xuzhou_multiscale_audit import _exact,_packet


class FactorialTests(unittest.TestCase):
    def test_independent_effect_arithmetic_and_contract_failures(self):
        rows = [dict(arm=arm,source="old_source",labels="released",exposure="matched",budget=48,metric="oa",
                     seed=64,condition="full",value=value) for arm,value in zip(ARMS,(.25,.40,.35,.65),strict=True)]
        got = effects(rows)
        for key,value in dict(sample_at_full=.15,sample_at_restricted=.30,observation_without_sample=.10,
                              observation_with_sample=.25,interaction=.15).items():
            self.assertAlmostEqual(got[key],value)
        with self.assertRaisesRegex(ValueError,"four complete"):
            effects(rows[:-1])
        for key,value in (("source","other"),("labels","wrong"),("budget",49),("exposure","different")):
            bad = deepcopy(rows); bad[2][key] = value
            with self.assertRaisesRegex(ValueError,"sources, labels"):
                effects(bad)

    @unittest.skipUnless(os.environ.get("WP01_CUR_S_BASE_FIXTURES"),"requires previous accepted sample fixtures")
    def test_all_three_class_counts(self):
        old = Path(os.environ["WP01_CUR_S_BASE_FIXTURES"])
        out = os.environ.get("WP01_FACTORIAL_FIXTURES")
        if out:
            root = Path(out); root.mkdir()
            self.matrix(old,root)
        else:
            with TemporaryDirectory() as d:
                self.matrix(old,Path(d))

    def matrix(self, old, root):
        code = code_snapshot()
        for classes in (3,9,22):
            with self.subTest(classes=classes):
                home = root/str(classes)
                home.mkdir()
                if classes == 3:
                    previous = prepared(home,classes=3,arm="off")
                    req = dict(scope="synthetic",weights=previous["source"],sidecar=previous["metadata"],
                               development=previous["development"],exposure=previous["exposure"])
                else:
                    a = admission.read(dev._pin(old/str(classes)/"run/registration.json"))
                    exposure = dev._json_new(home/"exposure.json",a["exposures"]["64"])
                    req = dict(scope="synthetic",**a["source"],exposure=exposure)
                input_pin = dev._json_new(home/"request.json",req)
                pin = register(input_pin,home/"batch",code)
                _,reg,source,scene,record,plan,trials = load_registration(pin,code)
                sample,obs = trials["s_plus_o"][1:3]
                sref = dev._json_new(home/"old-s-policy.json",sample.to_dict())
                oref = dev._json_new(home/"old-o-policy.json",obs.to_dict())
                with self.assertRaisesRegex(ValueError,"separate experiments"):
                    _build(dict(source=req["weights"],metadata=req["sidecar"],development=req["development"],
                                exposure=req["exposure"],sample_policy=sref,observation_policy=oref),.0001,64)
                with self.assertRaisesRegex(ValueError,"sample policy or budget"):
                    FactorialRun(source,scene,record,trial="s_plus_o",teacher_hash=trials["s_plus_o"][3],
                        plan=plan,policy=sample,lr=.0001,torch_seed=64,expected_source_state=reg["source_state"],
                        expected_plan_hash=plan.plan_hash,expected_policy_hash="0"*64,
                        observation_policy=obs,expected_observation_hash=obs.policy_hash)
                previous = None
                for arm in ARMS:
                    full = execute(pin,code,arm,"full",None)
                    prefix = execute(pin,code,arm,"prefix",None)
                    if previous:
                        with self.assertRaisesRegex(ValueError,"Cross-arm"):
                            execute(pin,code,arm,"resume",previous)
                    argv = [sys.executable,str(ROOT/"scripts/run_curriculum_factorial.py"),
                            "--expected-commit",code["commit"],"stage","--registration",pin["path"],pin["sha256"],
                            "--case",arm,"--stage","resume","--prior",prefix["path"],prefix["sha256"]]
                    ran = subprocess.run(argv,capture_output=True,text=True)
                    self.assertEqual(ran.returncode,0,ran.stderr)
                    resume = dev._pin(home/"batch"/"attempts"/arm/"resume/status.json")
                    with patch.object(torch.optim.Adam,"step",side_effect=AssertionError("extra update")):
                        execute(pin,code,arm,"completed",resume)
                    previous = prefix
                    packet = _packet(admission.read(full),reg["contracts"][arm],plan.updates,packet_format=FactorialRun.FORMAT)
                    if arm != "s_plus_o":
                        self.degenerate(arm,packet,source,scene,record,plan,trials[arm],reg)
                review = admission.read(audit(pin,complete(pin,code),code))
                self.assertEqual(review["synthetic_updates"],plan.updates*8)
                self.assertEqual(review["real_proposal"]["optimizer_updates"],0)
                if classes == 3:
                    shorter = ExposurePlan.freeze(scene,record,updates=24,seed=64,
                        minimum_class_blocks={str(c):1 for c in record["classes"]},minimum_support_visits=1)
                    invalid = dev._json_new(home/"shorter.json",dict(req,exposure=dev._json_new(home/"24.json",shorter.to_dict())))
                    with self.assertRaisesRegex(ValueError,"frozen budget|Frozen budget|fixed per-fixture"):
                        register(invalid,home/"insufficient",code)
                    error_run = register(input_pin,home/"failed-fixture",code)
                    with patch.object(FactorialRun,"_evaluate",side_effect=RuntimeError("synthetic after optimizer")):
                        with self.assertRaisesRegex(RuntimeError,"synthetic after"):
                            execute(error_run,code,"s_plus_o","prefix",None)
                    failed = admission.read(dev._pin(home/"failed-fixture/attempts/s_plus_o/prefix/status.json"))
                    self.assertEqual(failed["status"],"failed")
                    self.assertTrue((home/"failed-fixture/attempts/s_plus_o/prefix/failure.pt").exists())
                    with self.assertRaises(FileExistsError):
                        execute(error_run,code,"s_plus_o","prefix",None)

    def degenerate(self,arm,packet,source,scene,record,plan,trial,reg):
        _,sample,obs,teacher = trial
        case = dict(case="oracle",seed=64,arm="easy_to_hard")
        if arm == "s_only":
            control = old_s(source,scene,record,plan,(case,sample,teacher))
        elif arm == "o_only":
            control = old_o(source,scene,record,plan,(case,obs))
        else:
            control = CurriculumRun(source,scene,record,plan=plan,policy=None,lr=.0001,torch_seed=64,
                expected_source_state=reg["source_state"],expected_plan_hash=plan.plan_hash,expected_policy_hash=None)
        control.advance(plan.updates)
        old = dict(state=control.model.state_dict(), selected_state=control.selected_state, optimizer=control.optimizer.state_dict(),
                   rng=control.rng.bit_generator.state,torch_rng=control.torch_rng,progress=control.ledger.snapshot(),
                   selected_step=control.selected_step)
        for key,value in old.items():
            self.assertTrue(_exact(value,packet[key]),f"{arm} {key}")
        self.assertTrue(all(_exact(left,{k:right[k] for k in left}) for left,right in
                            zip(control.history,packet["history"],strict=True)),f"{arm} shared event history")
