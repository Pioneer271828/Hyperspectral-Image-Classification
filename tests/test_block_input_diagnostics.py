"""Frozen local input gradients and counterfactual block geometry."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from evaluation.block_input_diagnostics import input_gradients,intervene_spectrum,BlockInputPlan,review_block,_probe
from evaluation.graph_edge_intervention import _selected
from loadData.spatial_tiles import observe_tile
from model.research_config import digest
from transfer.graph_route_context import route_state_hash
from transfer.graph_route_training import GraphRouteRun
from tests.test_graph_edge_intervention import completed


def spec(run):
    tid=next(t['id'] for t in run.record['roles'] if t['role']=='devval')
    obs=observe_tile(run.scene.cube,run.scene.plan,run.scene.stats,run.scene.plan['tiles'][tid])
    return dict(tile_id=tid,target_ids=sorted([int(obs.global_ids[2,2]),int(obs.global_ids[3,3])]),class_id=1,
        interventions=[dict(kind='identity'),dict(kind='occlusion_min',box=[1,4,2,5]),
                      dict(kind='replace',box=[1,4,2,5],donor_box=[5,8,6,9]),
                      dict(kind='replace',box=[1,4,2,5],donor_box=[1,4,2,5])])


class BlockInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture_run=completed('weight',False,3)

    def test_manual_derivative_and_nonlinear_finite_difference_and_constant(self):
        x=torch.zeros((2,3,2),dtype=torch.float64);x[0,0,0]=.3
        weight=torch.tensor([.04,.02],dtype=torch.float64)
        def forward(v):
            z=(v.reshape(-1,2)*weight).sum(1)
            return torch.stack((.5+z,.5-z),1),torch.stack((z.sigmoid(),(-z).sigmoid()),1),torch.full((6,2),.5,dtype=v.dtype)
        r=input_gradients(forward,x,pixel_indices=[0,2],class_id=1)
        expected=torch.zeros_like(x);expected.reshape(-1,2)[[0,2]]=weight/2
        self.assertTrue(torch.equal(r['gradients'][0],expected));self.assertEqual(r['connected'],[True,True,False])
        self.assertTrue(torch.equal(r['gradients'][2],torch.zeros_like(x)))
        direction=torch.arange(12,dtype=torch.float64).reshape_as(x)/12;eps=1e-5
        for head in (0,1):
            fd=float((forward(x+eps*direction)[head][[0,2],0].mean()-forward(x-eps*direction)[head][[0,2],0].mean())/(2*eps))
            self.assertAlmostEqual(fd,float((r['gradients'][head]*direction).sum()),places=10)

    def test_patch_geometry_original_snapshot_and_no_cross_boundary(self):
        x=torch.arange(4*5*2,dtype=torch.float32).reshape(4,5,2);before=x.clone()
        occluded=intervene_spectrum(x,dict(kind='occlusion_min',box=[0,2,1,4]));self.assertTrue(torch.equal(occluded[:2,1:4],torch.zeros(2,3,2)))
        got=intervene_spectrum(x,dict(kind='replace',box=[0,3,0,4],donor_box=[1,4,1,5]));self.assertTrue(torch.equal(got[:3,:4],x[1:,1:]))
        self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(intervene_spectrum(x,dict(kind='identity')),x))
        for bad in (dict(kind='unknown'),dict(kind='identity',box=[]),dict(kind='occlusion_min',box=[0,5,0,1]),
                    dict(kind='occlusion_min',box=[False,1,0,1]),dict(kind='replace',box=[0,2,0,2],donor_box=[0,1,0,2])):
            with self.assertRaises(ValueError):intervene_spectrum(x,bad)
        with self.assertRaises(ValueError):intervene_spectrum(torch.zeros(257,257,1),dict(kind='identity'))

    def test_network_identity_gradient_and_state_are_preserved(self):
        run=self.fixture_run;state=route_state_hash(run.model);source=route_state_hash(run.source);history=copy.deepcopy(run.history)
        grads={k:None if p.grad is None else p.grad.clone() for k,p in run.model.named_parameters()}
        slots=copy.deepcopy(run.optimizer.state_dict());rng=torch.get_rng_state().clone();nrng=np.random.get_state()
        plan=BlockInputPlan.freeze(run,**spec(run))
        with patch('torch.optim.Adam.step',side_effect=AssertionError('optimization forbidden')):r=review_block(run,plan,expected_plan_hash=plan.plan_hash)
        self.assertEqual(r['interventions'][0]['delta'],[0.]*3);self.assertEqual(r['interventions'][3]['delta'],[0.]*3)
        self.assertTrue(any(v!=0 for v in r['interventions'][1]['delta']))
        gradients=np.array(r['gradients']);np.testing.assert_allclose(r['pixel_norms'],np.linalg.norm(gradients,axis=-1),rtol=1e-6,atol=1e-9)
        np.testing.assert_allclose(r['band_norms'],np.sqrt((gradients**2).sum((1,2))),rtol=1e-6,atol=1e-9)
        self.assertTrue(all(r['connected']));self.assertGreater(np.abs(gradients).sum(),0)
        self.assertEqual((state,source),(route_state_hash(run.model),route_state_hash(run.source)));self.assertEqual(history,run.history)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));np.testing.assert_array_equal(nrng[1],np.random.get_state()[1])
        for key,slot in slots['state'].items():
            for name,t in slot.items():self.assertTrue(torch.equal(t,run.optimizer.state_dict()['state'][key][name]))
        for k,p in run.model.named_parameters():
            if grads[k] is None:self.assertIsNone(p.grad)
            else:self.assertTrue(torch.equal(grads[k],p.grad))
        for row in r['interventions']:
            self.assertEqual(row['added_edges'],sorted(set(row['support']['edges'])-set(r['baseline_support']['edges'])))

    def test_plan_authentication_state_targets_and_query_rejected(self):
        run=self.fixture_run;s=spec(run);plan=BlockInputPlan.freeze(run,**s)
        with self.assertRaises(ValueError):review_block(run,plan,expected_plan_hash='0'*64)
        for change in (dict(class_id=0),dict(target_ids=[-1]),dict(target_ids=s['target_ids'][::-1]),dict(interventions=[]),
                       dict(tile_id=next(t['id'] for t in run.record['roles'] if t['role']=='test'))):
            with self.assertRaises(ValueError):BlockInputPlan.freeze(run,**(s|change))
        for key in ('selected_state_hash','normalized_spectrum_hash','coordinates_hash'):
            value=plan.to_dict();value[key]='wrong';value['plan_hash']=digest({k:v for k,v in value.items() if k!='plan_hash'})
            bad=BlockInputPlan(value,expected_hash=value['plan_hash'])
            with self.assertRaises(ValueError):review_block(run,bad,expected_plan_hash=bad.plan_hash)

    def test_exception_cleans_whole_model_graph_and_route_interceptors(self):
        run=self.fixture_run;model=_selected(run);s=spec(run);obs=observe_tile(run.scene.cube,run.scene.plan,run.scene.stats,run.scene.plan['tiles'][s['tile_id']])
        from research_runtime import tensor_hash
        x=torch.tensor(obs.spectral.copy());state=route_state_hash(model)
        with patch.object(run,'_predict',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):_probe(run,model,s['tile_id'],x,tensor_hash(x))
        self.assertNotIn('forward',model.__dict__);self.assertNotIn('forward',model.backbone.graph_router.__dict__)
        self.assertNotIn('build_graph',model.backbone.learnable_graph.__dict__);self.assertEqual(state,route_state_hash(model))
        _probe(run,model,s['tile_id'],x,tensor_hash(x))

    def test_completed_fixed_random_reload_and_input_gradient_validation(self):
        run=completed('random',True,2);plan=BlockInputPlan.freeze(run,**spec(run));first=review_block(run,plan,expected_plan_hash=plan.plan_hash)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';sha=run.save(path)
            restored=GraphRouteRun.load(run.source,run.scene,run.record,path,plan=run.plan,bank=run.bank,expected_bank_hash=run.bank.bank_hash,
                expected_sha256=sha,expected_contract_hash=run.contract_hash)
            self.assertEqual(first,review_block(restored,plan,expected_plan_hash=plan.plan_hash))
        x=torch.zeros(2,2,1)
        for indices in ([],[4],[1,0],[True]):
            with self.assertRaises(ValueError):input_gradients(lambda v:(),x,pixel_indices=indices,class_id=1)
        with self.assertRaises(ValueError):input_gradients(lambda v:(torch.ones(4,2),)*3,x,pixel_indices=[0],class_id=1)


if __name__=='__main__':unittest.main()
