"""Independent dense quadratic forms and readonly original-network probes."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from model.research_config import digest
from transfer.graph_route_context import route_state_hash
from evaluation.fixed_reference_energy import ReferenceGraph,energy_summary,freeze_source_reference,review_smoothness,_capture
from evaluation.graph_edge_intervention import _selected
from tests.test_graph_edge_intervention import completed


def reference(a):return ReferenceGraph.freeze(a.to_sparse().coalesce(),context={'case':'synthetic'},node_ids=list(range(len(a))))
def summary(x,r):return energy_summary(x,r,expected_hash=r.reference_hash,context=r.to_dict()['context'],node_ids=r.to_dict()['node_ids'])


class EnergyTests(unittest.TestCase):
    def test_hand_energy_and_independent_dense_oracle(self):
        a=torch.tensor([[8.,2.],[0.,9.]],dtype=torch.float64);r=reference(a);x=torch.tensor([[1.],[0.]],dtype=torch.float64)
        v=summary(x,r);self.assertAlmostEqual(v['energy'],.5);self.assertAlmostEqual(v['normalized_energy'],.5)
        rng=np.random.default_rng(64)
        for n in (1,2,5,17):
            a=rng.random((n,n));a[a<.6]=0;x=rng.normal(size=(n,7));r=reference(torch.tensor(a))
            w=(a+a.T)/2;np.fill_diagonal(w,1);degree=w.sum(1);lap=np.eye(n)-w/np.sqrt(degree[:,None]*degree[None,:])
            energies=np.diag(x.T@lap@x);got=summary(torch.tensor(x),r)
            np.testing.assert_allclose(got['energy_by_channel'],energies,atol=1e-13,rtol=1e-12)
            if n>1:self.assertAlmostEqual(got['normalized_energy'],energies.sum()/(x*x).sum())

    def test_scale_channel_duplication_rotation_and_reference_immutability(self):
        a=torch.tensor([[0.,3.,1.],[0.,0.,1.],[2.,0.,0.]],dtype=torch.float64);r=reference(a)
        x=torch.tensor([[1.,2.],[3.,-4.],[-1.,5.]],dtype=torch.float64);base=summary(x,r)
        before=r.to_dict();rotation=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
        for y,factor in ((x*3,9),(x@rotation,1),(x.repeat(1,2),2)):
            got=summary(y,r);self.assertAlmostEqual(got['normalized_energy'],base['normalized_energy']);self.assertAlmostEqual(got['energy'],base['energy']*factor)
            self.assertAlmostEqual(got['representation']['effective_dimension'],base['representation']['effective_dimension'])
        self.assertEqual(r.to_dict(),before);altered=r.to_dict();altered['weights'][0]=7
        with self.assertRaises(ValueError):ReferenceGraph(altered,expected_hash=r.reference_hash)
        self.assertEqual(r.to_dict(),before)

    def test_irregular_degree_nullspace_not_constant_and_empty_support(self):
        a=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],dtype=torch.float64);r=reference(a)
        self.assertGreater(summary(torch.ones(3,1),r)['energy'],0)
        self.assertLess(summary(torch.sqrt(torch.tensor([[2.],[3.],[2.]],dtype=torch.float64)),r)['energy'],1e-28)
        self.assertIsNone(summary(torch.zeros(3,1),r)['normalized_energy'])
        empty=summary(torch.ones(3,1),reference(torch.eye(3)));self.assertEqual(empty['reason'],'no_cross_node_edges');self.assertEqual(empty['energy'],0)
        self.assertIsNone(empty['representation']['effective_dimension'])

    def test_identity_invalid_numeric_and_sparse_bounds(self):
        a=torch.eye(3);r=reference(a);x=torch.ones(3,2)
        for kwargs in (dict(expected_hash='0'*64,context={'case':'synthetic'},node_ids=[0,1,2]),dict(expected_hash=r.reference_hash,context={'case':'other'},node_ids=[0,1,2]),dict(expected_hash=r.reference_hash,context={'case':'synthetic'},node_ids=[1,0,2])):
            with self.assertRaises(ValueError):energy_summary(x,r,**kwargs)
        for bad in (torch.full((3,2),float('nan')),torch.full((3,2),1e308,dtype=torch.float64),torch.ones(2,2),torch.ones(3,513),torch.ones(3,2,dtype=torch.int64)):
            with self.assertRaises(ValueError):summary(bad,r)
        for bad in (-a,a*float('nan'),torch.zeros(0,0),torch.ones(2,3)):
            with self.assertRaises(ValueError):reference(bad)
        with self.assertRaises(ValueError):ReferenceGraph.freeze(a,context={'case':'synthetic'},node_ids=[0,1,2])
        v=r.to_dict();v['weights'][0]=2;v['reference_hash']=digest({k:w for k,w in v.items() if k!='reference_hash'})
        with self.assertRaises(ValueError):ReferenceGraph(v,expected_hash=v['reference_hash'])


class NetworkEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.runs=[completed('random',fixed,budget=3) for fixed in (False,True)]

    def test_original_three_stages_fixed_reference_and_prediction_preservation(self):
        for run in self.runs:
            for role in ('devtrain','devval'):
                tid=next(t['id'] for t in run.record['roles'] if t['role']==role);before=route_state_hash(run.model);rng=torch.get_rng_state().clone()
                plan=freeze_source_reference(run,tid)
                with patch('torch.optim.Adam.step',side_effect=AssertionError('optimizer')):got=review_smoothness(run,plan,expected_plan_hash=plan['plan_hash'])
                self.assertEqual({v['reference_hash'] for v in got['stages'].values()},{plan['reference']['reference_hash']})
                self.assertEqual([v['channels'] for v in got['stages'].values()],[64,128,64])
                self.assertEqual(got,review_smoothness(run,plan,expected_plan_hash=plan['plan_hash']))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(before,route_state_hash(run.model))
                self.assertEqual(got['heads']['fused']['count'],sum(c['support'] for c in got['heads']['fused']['per_class']))

    def test_source_reference_survives_current_edge_changes_and_rejects_relabeling(self):
        run=self.runs[0];tid=next(t['id'] for t in run.record['roles'] if t['role']=='devval');plan=freeze_source_reference(run,tid)
        self.assertEqual(plan,freeze_source_reference(run,tid))
        self.assertEqual(plan['reference'],freeze_source_reference(self.runs[1],tid)['reference'])
        bad=copy.deepcopy(plan);bad['reference']['weights'][0]=999;bad['plan_hash']=digest({k:v for k,v in bad.items() if k!='plan_hash'})
        with self.assertRaises(ValueError):review_smoothness(run,bad,expected_plan_hash=bad['plan_hash'])
        bad=copy.deepcopy(plan);bad['source_state']='0'*64;bad['plan_hash']=digest({k:v for k,v in bad.items() if k!='plan_hash'})
        with self.assertRaises(ValueError):review_smoothness(run,bad,expected_plan_hash=bad['plan_hash'])
        query=next(t['id'] for t in run.record['roles'] if t['role']=='test')
        with self.assertRaises(ValueError):freeze_source_reference(run,query)

    def test_capture_failure_cleans_methods_and_original_model(self):
        run=self.runs[1];tid=next(t['id'] for t in run.record['roles'] if t['role']=='devval');model=_selected(run);before=route_state_hash(model)
        with patch.object(run,'_predict',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):_capture(run,model,tid)
        for owner,name in ((model.backbone.learnable_graph,'build_graph'),(model.backbone,'graph_stage1'),(model.backbone,'graph_stage2')):self.assertNotIn(name,owner.__dict__)
        self.assertEqual(before,route_state_hash(model))


if __name__=='__main__':unittest.main()
