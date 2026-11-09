"""Fixed selected-state, per-row count-matched edge deletion controls."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from evaluation.graph_edge_intervention import deletion_masks,delete_support,EdgeDeletionPlan,review_deletions,_selected,_probe
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.curriculum_exposure import ExposurePlan
from transfer.graph_route_context import RouteContextBank,route_state_hash
from transfer.graph_route_training import GraphRouteRun
from tests.test_int8_model import fixture


def completed(mode='weight',fixed=False,budget=3):
    base,scene,record=fixture();kw=copy.deepcopy(base.backbone_kwargs)
    kw['graph_route']=dict(budget=budget,mode=mode,seed=64)
    source=MultiDatasetVCASAHGCN({scene.dataset:scene.classes},**kw).eval()
    source.load_state_dict(base.state_dict(),strict=False);source.backbone.adapter.cache_attention=False
    ids=[r['id'] for r in record['roles'] if r['role'] in ('devtrain','devval')]
    bank=RouteContextBank.freeze(source,scene,record,tile_ids=ids,step=7,expected_source_state=route_state_hash(source)) if fixed else None
    plan=ExposurePlan.freeze(scene,record,updates=24,seed=64,
        minimum_class_blocks={str(c):1 for c in record['classes']},minimum_support_visits=1)
    run=GraphRouteRun(source,scene,record,plan=plan,bank=bank,expected_bank_hash=None if bank is None else bank.bank_hash,
        lr=.0001,torch_seed=512,expected_source_state=route_state_hash(source),expected_plan_hash=plan.plan_hash)
    run.advance(24);return run


def weighted():
    return torch.tensor([[.1,.6,.3],[.4,.2,.4],[.5,.2,.3]],dtype=torch.float32).to_sparse().coalesce()


def repin(value):
    value['plan_hash']=digest({k:v for k,v in value.items() if k!='plan_hash'})
    return EdgeDeletionPlan(value,expected_hash=value['plan_hash'])


class EdgeInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture_run=completed()

    def test_hand_weight_ranking_ties_counts_self_and_row_normalization(self):
        a=weighted();m=deletion_masks(a,per_row=1,random_seeds=[64,512,2048],tile_id=7)
        self.assertEqual(m['removed']['important'],[1,3,6]);self.assertEqual(m['degenerate_rows'],[])
        for removed in m['removed'].values():
            self.assertEqual(np.bincount(np.array(removed)//3,minlength=3).tolist(),[1,1,1])
            got=delete_support(a,removed);manual=a.to_dense().numpy().copy()
            for e in removed:manual[e//3,e%3]=0
            manual/=manual.sum(1)[:,None]
            np.testing.assert_allclose(got.to_dense().numpy(),manual,rtol=1e-7,atol=1e-7)
            self.assertEqual(got._nnz(),6);self.assertTrue(torch.all(got.to_dense().diag()>0))
        self.assertIs(delete_support(a,[]),a)
        full=deletion_masks(a,per_row=99,random_seeds=[64],tile_id=7)
        self.assertEqual(full['counts_per_row'],[2]*3);self.assertEqual(full['degenerate_rows'],[0,1,2])
        self.assertEqual(full['removed']['important'],full['removed']['random:64'])
        self.assertTrue(torch.equal(delete_support(a,full['removed']['important']).to_dense(),torch.eye(3)))

    def test_local_random_reproducibility_and_invalid_deletions(self):
        a=weighted();state=np.random.get_state();torchstate=torch.get_rng_state().clone()
        first=deletion_masks(a,per_row=1,random_seeds=[64,512],tile_id=7)
        self.assertEqual(first,deletion_masks(a,per_row=1,random_seeds=[64,512],tile_id=7))
        np.testing.assert_array_equal(state[1],np.random.get_state()[1]);self.assertTrue(torch.equal(torchstate,torch.get_rng_state()))
        for per,seeds in ((True,[1]),(-1,[1]),(1,[]),(1,[1,1]),(1,[True]),(1,[-1])):
            with self.assertRaises(ValueError):deletion_masks(a,per_row=per,random_seeds=seeds,tile_id=0)
        for mask in ([0],[1,1],[3,1],[99],[True]):
            with self.assertRaises(ValueError):delete_support(a,mask)
        with self.assertRaises(ValueError):delete_support(torch.eye(3),[])

    def test_complete_network_zero_deletion_exact_and_readonly(self):
        run=self.fixture_run;state=route_state_hash(run.model);source=route_state_hash(run.source);history=copy.deepcopy(run.history)
        slots=copy.deepcopy(run.optimizer.state_dict());rng=torch.get_rng_state().clone();nrng=np.random.get_state()
        grads={k:None if p.grad is None else p.grad.clone() for k,p in run.model.named_parameters()}
        with patch('torch.optim.Adam.step',side_effect=AssertionError('optimization forbidden')):
            plan=EdgeDeletionPlan.freeze(run,per_row=0,random_seeds=[64,512])
            result=review_deletions(run,plan,expected_plan_hash=plan.plan_hash)
        for value in result['controls'].values():self.assertEqual(value,result['controls']['none'])
        self.assertEqual(result['selected_step'],run.selected_step);self.assertEqual(result['optimizer_updates'],0)
        self.assertEqual((state,source),(route_state_hash(run.model),route_state_hash(run.source)))
        self.assertEqual(run.history,history);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        np.testing.assert_array_equal(nrng[1],np.random.get_state()[1]);self.assertEqual(nrng[2:],np.random.get_state()[2:])
        for key,slot in slots['state'].items():
            for name,t in slot.items():self.assertTrue(torch.equal(t,run.optimizer.state_dict()['state'][key][name]))
        for key,p in run.model.named_parameters():
            if grads[key] is None:self.assertIsNone(p.grad)
            else:self.assertTrue(torch.equal(grads[key],p.grad))

    def test_nonzero_network_controls_change_graph_only_and_match_per_row_counts(self):
        run=self.fixture_run;plan=EdgeDeletionPlan.freeze(run,per_row=1,random_seeds=[64,512,2048])
        result=review_deletions(run,plan,expected_plan_hash=plan.plan_hash);baseline=result['controls']['none']
        different=False
        for name,control in result['controls'].items():
            self.assertEqual(control['confusion'][2],baseline['confusion'][2])
            self.assertEqual(control['support'],[[5]*9]*3)
            for original,trace in zip(baseline['traces'],control['traces']):
                self.assertEqual(trace['probability_hashes'][2],original['probability_hashes'][2])
                self.assertEqual(trace['original_nnz']-trace['actual_nnz'],len(trace['removed']))
                if name!='none':
                    self.assertEqual(len(trace['removed']),4)
                    different|=trace['probability_hashes'][1]!=original['probability_hashes'][1]
        self.assertTrue(different);self.assertTrue(any(not x for x in result['identical_important_random'].values()))

    def test_authenticated_plan_rejects_state_context_support_and_query_changes(self):
        run=self.fixture_run;plan=EdgeDeletionPlan.freeze(run,per_row=1,random_seeds=[64])
        with self.assertRaises(ValueError):review_deletions(run,plan,expected_plan_hash='0'*64)
        for key in ('contract_hash','selected_state_hash','selected_step'):
            value=plan.to_dict();value[key]='wrong';bad=repin(value)
            with self.assertRaises(ValueError):review_deletions(run,bad,expected_plan_hash=bad.plan_hash)
        mutations=[lambda v:v['contexts'][0]['context'].update(tile_id=next(r['id'] for r in run.record['roles'] if r['role']=='test')),
            lambda v:v['contexts'][0].update(graph_hash='0'*64),
            lambda v:v['contexts'][0]['masks']['removed'].update(important=[])]
        for change in mutations:
            value=plan.to_dict();change(value);bad=repin(value)
            with self.assertRaises(ValueError):review_deletions(run,bad,expected_plan_hash=bad.plan_hash)

    def test_exception_cleanup_and_reject_unfinished_model(self):
        run=self.fixture_run;model=_selected(run);tid=next(r['id'] for r in run.record['roles'] if r['role']=='devval')
        rng=torch.get_rng_state().clone();state=route_state_hash(model)
        def fail(*args):torch.rand(3);raise RuntimeError('injected')
        with patch.object(run,'_predict',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'injected'):_probe(run,model,tid)
        self.assertNotIn('forward',model.backbone.graph_router.__dict__)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(state,route_state_hash(model))
        early=GraphRouteRun(run.source,run.scene,run.record,plan=run.plan,lr=.0001,torch_seed=512,
            expected_source_state=route_state_hash(run.source),expected_plan_hash=run.plan.plan_hash)
        with self.assertRaises(ValueError):EdgeDeletionPlan.freeze(early,per_row=1,random_seeds=[64])

    def test_fixed_random_degenerate_control_and_completed_reload(self):
        run=completed('random',True,2);plan=EdgeDeletionPlan.freeze(run,per_row=1,random_seeds=[64,512])
        first=review_deletions(run,plan,expected_plan_hash=plan.plan_hash)
        self.assertTrue(all(first['identical_important_random'].values()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';sha=run.save(path)
            restored=GraphRouteRun.load(run.source,run.scene,run.record,path,plan=run.plan,bank=run.bank,
                expected_bank_hash=run.bank.bank_hash,expected_sha256=sha,expected_contract_hash=run.contract_hash)
            again=review_deletions(restored,plan,expected_plan_hash=plan.plan_hash)
        first.pop('elapsed_seconds');again.pop('elapsed_seconds');self.assertEqual(first,again)


if __name__=='__main__':unittest.main()
