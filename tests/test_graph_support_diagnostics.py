import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from model.research_config import digest
from monitoring.graph_support import support_summary,support_change,capture_support
from research_runtime import tensor_hash
from transfer.graph_route_context import route_state_hash
from transfer.graph_route_training import GraphRouteRun
from tests.test_graph_route_training import arguments,make,load


def graph(n,edges):
    index=torch.tensor(edges,dtype=torch.int64).T
    count=torch.bincount(index[0],minlength=n)
    return torch.sparse_coo_tensor(index,1./count[index[0]],(n,n)).coalesce()


class SupportDiagnosticsTests(unittest.TestCase):
    def test_hand_directed_components_degrees_soft_mass_and_loops(self):
        a=graph(5,[(i,i) for i in range(5)]+[(0,1),(1,0),(1,2),(2,3),(3,2)])
        q=graph(5,[(0,0),(1,1),(2,2),(3,3),(4,0)])
        result=support_summary(a,q,small_mass_cutoff=1.)
        self.assertEqual(result['weak_components'],[[0,1,2,3],[4]])
        self.assertEqual(result['strong_components'],[[0,1],[2,3],[4]])
        self.assertEqual(result['row_dependency_degree'],[2,3,2,2,1])
        self.assertEqual(result['column_dependency_degree'],[2,2,3,2,1])
        self.assertEqual(result['soft_pixel_mass'],[2.,1.,1.,1.,0.])
        self.assertEqual(result['empty_nodes'],[4]);self.assertEqual(result['small_nodes'],[1,2,3,4])
        self.assertEqual(result['small_row_connections'],8);self.assertEqual(result['small_nonself_row_connections'],4)
        self.assertEqual(result['nonself_isolated_nodes'],[4]);self.assertEqual(result['self_loop_fraction'],.5)

    def test_deep_directed_chain_uses_no_recursion_or_dense_graph(self):
        n=1500;a=graph(n,[(i,i) for i in range(n)]+[(i,i+1) for i in range(n-1)])
        q=graph(n,[(i,i) for i in range(n)])
        with patch.object(torch.Tensor,'to_dense',side_effect=AssertionError('dense graph')):
            result=support_summary(a,q,small_mass_cutoff=.5)
        self.assertEqual(len(result['weak_components']),1);self.assertEqual(len(result['strong_components']),n)
        self.assertEqual(result['small_nodes'],[])

    def test_invalid_support_assignment_and_cutoff_rejected(self):
        good=graph(2,[(0,0),(1,1)])
        for cutoff in (-1,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):support_summary(good,good,small_mass_cutoff=cutoff)
        bad=[good.to_dense(),torch.sparse_coo_tensor(good.indices(),good.values(),good.shape),
             torch.sparse_coo_tensor(good.indices(),torch.tensor([0.,1.]),good.shape).coalesce(),
             torch.sparse_coo_tensor(good.indices(),torch.tensor([float('nan'),1.]),good.shape).coalesce(),
             graph(2,[(0,1),(1,0)]),good*.5]
        for a in bad:
            with self.assertRaises(ValueError):support_summary(a,good,small_mass_cutoff=1)
        with self.assertRaises(ValueError):support_summary(good,graph(3,[(i,i) for i in range(3)]),small_mass_cutoff=1)

    def test_eight_network_modes_readonly_equal_probabilities_and_fixed_support(self):
        for mode in ('weight','random','graph','joint'):
            for fixed in (False,True):
                run=make(arguments(mode,fixed));tid=next(r['id'] for r in run.record['roles'] if r['role']=='devtrain')
                previous=capture_support(run,tid,state_kind='last',small_mass_cutoff=100.)
                run.advance(2);source=route_state_hash(run.source);state=route_state_hash(run.model)
                grads={k:None if p.grad is None else p.grad.clone() for k,p in run.model.named_parameters()}
                rng=torch.get_rng_state().clone();npstate=np.random.get_state()
                now=capture_support(run,tid,state_kind='last',small_mass_cutoff=100.)
                self.assertEqual(source,route_state_hash(run.source));self.assertEqual(state,route_state_hash(run.model))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                after=np.random.get_state();self.assertEqual(npstate[0],after[0]);np.testing.assert_array_equal(npstate[1],after[1]);self.assertEqual(npstate[2:],after[2:])
                for k,p in run.model.named_parameters():
                    if grads[k] is None:self.assertIsNone(p.grad)
                    else:self.assertTrue(torch.equal(grads[k],p.grad))
                selected=copy.deepcopy(run.model).eval()
                with torch.no_grad():probs=run._predict(selected,run.scene.plan['tiles'][tid],0)[:3]
                self.assertEqual(now['probability_hashes'],[tensor_hash(x) for x in probs])
                change=support_change(now,previous)
                if fixed:self.assertEqual(change['all']['added']+change['all']['removed'],0)
                self.assertEqual(now['optimizer_updates'],0);self.assertEqual(now['optimization_step'],2)
                self.assertEqual(len(run.history),2)
                val=next(r['id'] for r in run.record['roles'] if r['role']=='devval')
                review=capture_support(run,val,state_kind='selected',small_mass_cutoff=100.)
                self.assertEqual(review['optimization_step'],run.selected_step)

    def test_comparison_identity_monotonicity_and_no_query(self):
        run=make(arguments('random'));tid=next(r['id'] for r in run.record['roles'] if r['role']=='devtrain')
        first=capture_support(run,tid,state_kind='last',small_mass_cutoff=100.)
        with self.assertRaises(ValueError):support_change(first,first)
        query=next(r['id'] for r in run.record['roles'] if r['role']=='test')
        with self.assertRaises(ValueError):capture_support(run,query,state_kind='last',small_mass_cutoff=100.)
        with self.assertRaises(ValueError):capture_support(run,tid,state_kind='selected',small_mass_cutoff=100.)
        run.advance(1);second=capture_support(run,tid,state_kind='last',small_mass_cutoff=100.)
        for key in ('contract_hash','context','state_kind','probe_route_step'):
            altered=copy.deepcopy(second)
            if key=='context':altered[key]['tile_id']=-1
            elif key=='probe_route_step':altered[key]=1
            else:altered[key]='changed'
            altered['snapshot_hash']=digest({k:v for k,v in altered.items() if k!='snapshot_hash'})
            with self.assertRaises(ValueError):support_change(altered,first)
        altered=copy.deepcopy(second);altered['summary']['edges']=[]
        with self.assertRaises(ValueError):support_change(altered,first)

    def test_exception_cleans_private_interceptors_and_preserves_source(self):
        run=make(arguments());tid=next(r['id'] for r in run.record['roles'] if r['role']=='devtrain')
        state=route_state_hash(run.model);rng=torch.get_rng_state().clone();seen=[]
        def fail(self,model,*args):seen.append(model);torch.rand(4);raise RuntimeError('injected probe failure')
        with patch.object(GraphRouteRun,'_predict',fail):
            with self.assertRaisesRegex(RuntimeError,'injected'):capture_support(run,tid,state_kind='last',small_mass_cutoff=100.)
        self.assertEqual(len(seen),1)
        self.assertNotIn('forward',seen[0].backbone.graph_router.__dict__)
        self.assertNotIn('build_graph',seen[0].backbone.learnable_graph.__dict__)
        self.assertEqual(state,route_state_hash(run.model));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(run.failed)
        capture_support(run,tid,state_kind='last',small_mass_cutoff=100.)

    def test_snapshot_exact_after_authenticated_checkpoint_restore(self):
        args=arguments('joint',True);run=make(args);run.advance(2)
        tid=next(r['id'] for r in run.record['roles'] if r['role']=='devval')
        first=capture_support(run,tid,state_kind='selected',small_mass_cutoff=100.)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt';sha=run.save(path);restored=load(args,path,sha,run.contract_hash)
            self.assertEqual(first,capture_support(restored,tid,state_kind='selected',small_mass_cutoff=100.))


if __name__=='__main__':unittest.main()
