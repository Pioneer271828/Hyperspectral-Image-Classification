"""Blinded hard-support and source isolation checks for the WP05 production path."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from evaluation.frozen_training_stages import json_pin
from scripts.run_native_released_points import arithmetic
from transfer import wp_causal_graph_design as design, wp_region_graph as region
from transfer.native_independent_release import materialize_reviewed
from transfer.wp_causal_graph import verify_trace
from transfer.wp_causal_graph_fixture import generate
from transfer.wp_paired_structure import admit, base_model


class CausalGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.addClassCleanup(torch.set_num_threads,torch.get_num_threads())
        torch.set_num_threads(1)
        cls.root=Path(folder.name)
        cls.code=dict(commit="a"*40,source_hash="b"*64)
        prefix=generate(cls.root/"fixture",cls.root/"runs",cls.code,[])
        cls.request=json_pin(cls.root,prefix["request"])
        cfg=cls.request["releases"]["xz-a"]
        cls.bank,cls.intake,cls.roles=admit(cls.root,cls.root/"runs",cfg,test=True)
        cls.witness=json_pin(cls.root,cfg["causal_graph"])
        baseline,_=base_model(cls.bank,64,None)
        cls.model=region.RegionSupportModel.initialize(baseline,7385).eval()

    def test_directed_witness_and_search_status_are_distinct(self):
        for layout in ("a","b"):
            cert=design.certify(layout)
            self.assertEqual(cert["paths"]["new_path"],
                             dict(original=[False,False],matched=[True,True],placebo=[False,False]))
            self.assertEqual(len(cert["weak"]),1)
            self.assertEqual(len(cert["strong"]),len(design.POSITIONS))
            cases=design.search_controls(layout)
            self.assertEqual(cases["exhaustive"]["status"],"INFEASIBLE")
            self.assertEqual(cases["limited"]["status"],"NOT_FOUND_WITHIN_BUDGET")
        foreign=deepcopy(self.witness)
        foreign["witness"]["witness"]["matched"][1][0][1]=2
        with self.assertRaisesRegex(ValueError,"witness changed"):
            design.verify(foreign,bank=self.bank.pin["sha256"],scene="xuzhou",roles=self.roles)

    def probe(self, policy, band):
        tile=self.roles["evaluation"][0]
        actual,_,targets=materialize_reviewed(self.bank,self.intake,self.roles,"evaluation",tile,
                                                  self.code["commit"],band,use="independent_accuracy")
        program=region.program_for(self.model,policy,19064,24,self.witness)
        results={}
        with torch.no_grad(),arithmetic(self.root,None) as execution:
            for edit in (None,"source_swap","unrelated_swap","cut"):
                heads,graph,_=region.forward_region(self.model,actual,self.code["commit"],"devval",execution,
                                                     program,intervention=edit)
                proof=region.verify_trace(graph["regional_support"],actual.receipt["observation"])
                verify_trace(proof,self.witness,tile,"devval",24)
                selected=[i for i,y in zip(targets.cells,[y for _,y in targets.points]) if y in (5,6)]
                results[edit]=[a[selected].clone() for a in heads[:3]]
        return results

    def test_sources_only_reach_the_controlled_new_path(self):
        baseline=self.probe("causal_original","full")
        matched=self.probe("causal_matched","full")
        for view in (baseline,self.probe("causal_placebo","full"),self.probe("causal_matched","central_gap")):
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(view[None],view["source_swap"],strict=True)))
        self.assertGreater(float((matched[None][1]-matched["source_swap"][1]).abs().max()),1e-6)
        self.assertTrue(torch.equal(matched[None][2],matched["source_swap"][2]))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(matched[None],matched["unrelated_swap"],strict=True)))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(baseline[None],matched["cut"],strict=True)))

    def test_capture_keeps_three_heads_and_separate_graph_info(self):
        from evaluation.wp12_cross_backend import interventions

        tile=self.roles["evaluation"][0]
        actual,_,targets=materialize_reviewed(self.bank,self.intake,self.roles,"evaluation",tile,
                                                self.code["commit"],"full",use="independent_accuracy")
        model=deepcopy(self.model)
        model.graph_program=region.program_for(self.model,"causal_matched",19064,24,self.witness)
        with arithmetic(self.root,None) as execution:
            result=interventions(model,actual,self.code["commit"],execution,targets)
        self.assertEqual(set(result),{"source_swap","unrelated_swap","cut"})
        for row in result.values():
            self.assertEqual(set(row["heads"]),{"graph","fused","cnn"})
            self.assertEqual(len(row["heads"]["fused"]),len(targets.points))

    def test_evolution_has_one_registered_terminal_graph(self):
        self.assertEqual([design.phase("causal_evolving","devtrain",i,24) for i in (1,8,9,16,17,24)],
                         ["original","original","placebo","placebo","matched","matched"])
        self.assertEqual(design.phase("causal_evolving","devval",0,24),"matched")
