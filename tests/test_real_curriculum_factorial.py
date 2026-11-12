"""Negative joins and an independent prefix and tied-rank oracle."""

from copy import deepcopy
import unittest

from model.research_config import digest
from transfer.curriculum_factorial_controls import _same_arithmetic
from transfer.curriculum_factorial_real import ACCEPTED, CASES
from transfer.curriculum_factorial_real_audit import _freeze_ready
from transfer.curriculum_joint_diagnosis import difficulty, joint_prefix


class Plan:
    updates = 3

    def event(self, step):
        return dict(step=step, tile_id=1, anchor=4, support_ids=[11, 12])


class RealFactorialTests(unittest.TestCase):
    def test_frozen_digests_are_sha256(self):
        self.assertTrue(all(len(v) == 64 for group in ACCEPTED.values() for v in group.values()))

    def test_contract_rejects_identity_order_optimizer_budget_and_wrong_switch(self):
        base = dict(source_state="source", data={"labels": [4, 4]}, updates=178, exposure_hash="frozen",
                    parameters=["a", "b"], optimizer_config=[{"params": [0, 1], "lr": .0001}],
                    policy_hash="sample", observation_policy_hash="obs", factorial_scope="real",
                    trial="s_plus_o", factor_switches=[True,True], teacher_hash="t", applied_weight_history="fp32")
        old = {k:v for k,v in base.items() if k not in {"factorial_scope", "factor_switches", "observation_policy_hash"}}
        old.update(trial={"arm":"off"},policy_hash=None)
        witness = _same_arithmetic(old,base,kind="sample",sample=None,observation=None)
        self.assertEqual(witness,digest({k:v for k,v in old.items() if k not in {"trial","teacher_hash","applied_weight_history"}}))
        for key, value in (("source_state","wrong"),("data",{"labels":[1,4]}),("updates",177),
                           ("exposure_hash","reordered"),("parameters",["b","a"]),("optimizer_config",[]),
                           ("policy_hash","sample"),("observation_policy_hash","extra")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "arithmetic"):
                _same_arithmetic(dict(old, **{key:value}),base,kind="sample",sample=None,observation=None)

    def test_all_endpoint_freeze_precedes_replay(self):
        points = dict(selected=dict(step=100),last=dict(step=178))
        cases = {c["case"]:dict(endpoints=points) for c in CASES}
        frozen = dict(registration="reg", completion="end", optimizer_updates=0,
                      cases={c["case"]: points for c in CASES})
        _freeze_ready(frozen,"reg","end",cases)
        incomplete = deepcopy(frozen)
        incomplete["cases"].pop(next(iter(cases)))
        with self.assertRaisesRegex(ValueError, "before replay"):
            _freeze_ready(incomplete,"reg","end",cases)
        wrong_endpoint = deepcopy(frozen)
        wrong_endpoint["cases"][next(iter(cases))] = dict(selected=dict(step=99), last=points["last"])
        with self.assertRaisesRegex(ValueError, "before replay"):
            _freeze_ready(wrong_endpoint,"reg","end",cases)

    def test_ties_constant_rank_and_wrong_identity(self):
        make = lambda values:{i:dict(label=4,nll=v) for i,v in enumerate(values,1)}
        base = make([0.5,0.5,1.,1.5,2.])
        observed = difficulty(make([2.,2.,1.5,1.,0.5]),base)["4"]
        self.assertAlmostEqual(observed["spearman"],-1.)
        self.assertEqual(observed["migrations"]["1"],dict(full_rank=.5,rank=3.5,delta=3.,full_bin=0,bin=1))
        self.assertIsNone(difficulty(make([2.]*5),base)["4"]["spearman"])
        wrong=make([2.]*5)
        wrong[5]["label"]=6
        with self.assertRaisesRegex(ValueError,"IDs or labels"):
            difficulty(wrong,base)

    def test_joint_early_cutoff_and_independent_fp32_mass(self):
        plan = Plan()
        support = [dict(index=11,label=4),dict(index=12,label=4)]
        tiles = {1:dict(box=[0,2,0,3])}
        strata = {"4":{"migrations":{"11":{"full_bin":0},"12":{"full_bin":2}}}}
        history = [dict(**plan.event(i),weights=list(w),applied_weights=list(w)) for i,w in
                   enumerate(((1.5,.5),(.5,1.5),(1.,1.)),1)]
        early = joint_prefix(history,plan,support,1,None,tiles,8,strata)
        self.assertEqual(early["conditions"]["full"]["support"],{"11":{"raw":1,"effective":1.5},"12":{"raw":1,"effective":.5}})
        self.assertEqual(early["conditions"]["full"]["classes"]["4"],dict(raw=2,effective=2.))
        self.assertEqual((early["conditions"]["full"]["bands"],early["conditions"]["full"]["pixel_bands"]),(8,48))
        self.assertEqual(early["conditions"]["random_25"]["unseen"],2)
        altered = deepcopy(history)
        altered[-1].update(weights=[.5,1.5],applied_weights=[.5,1.5])
        self.assertEqual(early,joint_prefix(altered,plan,support,1,None,tiles,8,strata))
        self.assertNotEqual(joint_prefix(history,plan,support,3,None,tiles,8,strata),
                            joint_prefix(altered,plan,support,3,None,tiles,8,strata))
        altered[0]["support_ids"]=[12,11]
        with self.assertRaisesRegex(ValueError,"ordered support"):
            joint_prefix(altered,plan,support,1,None,tiles,8,strata)
        altered[0]=dict(history[0],weights=[.5,.5],applied_weights=[.5,.5])
        with self.assertRaisesRegex(ValueError,"loss mass"):
            joint_prefix(altered,plan,support,1,None,tiles,8,strata)
