"""Exact matching, candidate leakage guards, and protected donor restoration."""

from unittest import TestCase

import torch

from evaluation.matched_edge_controls import freeze, verify
from transfer.band_graph_control import row_coo
from transfer.donor_retrieval_graph import exact_deletions, sample_graph
from transfer.donor_retrieval_training import Data, Run


class DonorRetrievalTests(TestCase):
    def test_independent_matching_fixed_degree_and_reverse_chance(self):
        for direction in ("directed","reciprocal"):
            with self.subTest(direction=direction):
                torch.set_num_threads(1)
                graph,true_order = sample_graph(64,"offset",direction,"confirmation",0)
                matched,item = exact_deletions(graph,true_order)
                assert item["feasible"]==24 and item["independent_alternatives"]==22
                assert item["masks"]["low_weight"]!=item["masks"]["conditional_random:7719"]
                assert item["matched_support"]["important"]["true_direct"]==0
                assert item["matched_support"]["low_weight"]["true_direct"]==4
                assert matched["important"]._nnz()==(32 if direction=="reciprocal" else 20)

    def test_conditional_random_can_be_impossible_while_natural_and_weak_exist(self):
        pairs = [(i,i) for i in range(4)] + [(0,2),(0,3),(1,2),(1,3)]
        a = row_coo(pairs,[1.,1.,1.,1.,5.,.1,.1,5.],4)
        xy = torch.tensor([[0.,0.],[0.,1.],[10.,0.],[10.,1.]])
        r = freeze(a,xy,per_row=1,distance_bins=[1.,20.],seeds=[7719],separate_weak=True)
        verify(r,a,xy,expected_hash=r["plan_hash"])
        assert r["controls"]["random:7719"]["removed"]==r["controls"]["low_weight"]["removed"]
        assert r["controls"]["conditional_random:7719"]["status"]=="no_distinct_control"

    def test_observable_route_has_no_truth_and_cannot_restore_foreign_candidate_graph(self):
        torch.set_num_threads(1)
        first = Data(64,"horizontal","directed")
        run = Run(first,"learned",scope="unit")
        graph,inputs,y = first.released_training_row(0,1)
        assert inputs.shape==(8,3) and y.sum()==1 and len(graph.support.pool)==16
        assert graph.support.last_stats["candidate_pairs"]==16
        with self.assertRaisesRegex(ValueError,"Truth cannot enter"):
            run.route(graph,None,oracle=(0,1,2,3))
        run.validate()
        run.update()
        checkpoint = run.pack()
        same = Run(first,"learned",scope="unit")
        same.restore(checkpoint,expected_cursor=1)
        for data in (Data(512,"horizontal","directed"),Data(64,"offset","directed")):
            with self.assertRaisesRegex(ValueError,"graph"):
                Run(data,"learned",scope="unit").restore(checkpoint,expected_cursor=1)
        run.update()
        same.update()
        assert run.pack()["state_sha256"]==same.pack()["state_sha256"]
