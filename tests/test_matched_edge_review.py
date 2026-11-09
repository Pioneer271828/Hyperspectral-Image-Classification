import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from evaluation.matched_edge_review import MatchedReviewPlan,coordinates,review
from evaluation.graph_edge_intervention import EdgeDeletionPlan,review_deletions
from model.research_config import digest
from transfer.graph_route_context import route_state_hash
from transfer.graph_route_training import GraphRouteRun
from tests.test_graph_edge_intervention import completed


def repin(v):
    v['plan_hash']=digest({k:x for k,x in v.items() if k!='plan_hash'})
    return MatchedReviewPlan(v,expected_hash=v['plan_hash'])


class MatchedReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture_run=completed(budget=4)

    def plan(self,k=1):return MatchedReviewPlan.freeze(self.fixture_run,per_row=k,distance_bins=[100.],seeds=[64,512,2048])

    def test_seed_coordinates_row_major_pixel_units(self):
        c={'node_order':'row_major_seed_grid_v1','node_position':'grid_seed_centers','grid_shape':[2,3],'shape':[8,12,7]}
        self.assertEqual(coordinates(c).tolist(),[[1.5,1.5],[1.5,5.5],[1.5,9.5],[5.5,1.5],[5.5,5.5],[5.5,9.5]])
        with self.assertRaises(ValueError):coordinates({**c,'node_position':'learned_centroid'})

    def test_actual_trained_network_three_heads_old_baseline_and_readonly(self):
        run=self.fixture_run;state=route_state_hash(run.model);source=route_state_hash(run.source)
        history=copy.deepcopy(run.history);slots=copy.deepcopy(run.optimizer.state_dict());rng=torch.get_rng_state().clone()
        grads={k:None if p.grad is None else p.grad.clone() for k,p in run.model.named_parameters()}
        with patch('torch.optim.Adam.step',side_effect=AssertionError('no updates')):
            plan=self.plan();result=review(run,plan,expected_plan_hash=plan.plan_hash)
            old=EdgeDeletionPlan.freeze(run,per_row=1,random_seeds=[64]);reference=review_deletions(run,old,expected_plan_hash=old.plan_hash)
        for name in ('none','important'):
            for key in ('confusion','recall','support','oa','aa','nll'):self.assertEqual(result['controls'][name]['metrics'][key],reference['controls'][name][key])
        available=0
        for name,value in result['controls'].items():
            if value['status']=='unavailable':self.assertIsNone(value['metrics']);continue
            available+=1;self.assertEqual(value['metrics']['support'],[[5]*9]*3)
            for trace,base in zip(value['traces'],result['controls']['none']['traces']):
                self.assertEqual(trace['probability_hashes'][2],base['probability_hashes'][2])
                self.assertEqual(trace['original_nnz']-trace['actual_nnz'],len(trace['removed']))
        self.assertGreater(available,2)
        self.assertEqual((state,source),(route_state_hash(run.model),route_state_hash(run.source)));self.assertEqual(history,run.history)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in slots['state'].items():
            for key,t in v.items():self.assertTrue(torch.equal(t,run.optimizer.state_dict()['state'][k][key]))
        for k,p in run.model.named_parameters():
            if grads[k] is None:self.assertIsNone(p.grad)
            else:self.assertTrue(torch.equal(grads[k],p.grad))

    def test_zero_and_single_unavailable_block_never_reduce_population(self):
        plan=self.plan(0);r=review(self.fixture_run,plan,expected_plan_hash=plan.plan_hash)
        self.assertEqual(r['controls']['none']['metrics'],r['controls']['important']['metrics'])
        self.assertTrue(all(r['controls'][n]['status']=='unavailable' for n in ('low_weight','random:64')))
        plan=self.plan();v=plan.to_dict();p=v['contexts'][0]['control_plan']
        p['controls']['low_weight']={'status':'unresolved','removed':None,'reason':'injected timeout'}
        p['plan_hash']=digest({k:x for k,x in p.items() if k!='plan_hash'});bad=repin(v)
        r=review(self.fixture_run,bad,expected_plan_hash=bad.plan_hash)
        self.assertEqual(r['controls']['low_weight']['status'],'unavailable');self.assertIsNone(r['controls']['low_weight']['metrics'])

    def test_plan_context_state_masks_refusals_and_completion_reload(self):
        run=self.fixture_run;plan=self.plan();first=review(run,plan,expected_plan_hash=plan.plan_hash)
        for mutate in (lambda v:v.update(contract_hash='wrong'),lambda v:v.update(selected_step=0),
                       lambda v:v['contexts'][0]['context'].update(role='test'),lambda v:v['contexts'].pop()):
            v=plan.to_dict();mutate(v);bad=repin(v)
            with self.assertRaises(ValueError):review(run,bad,expected_plan_hash=bad.plan_hash)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';sha=run.save(path)
            loaded=GraphRouteRun.load(run.source,run.scene,run.record,path,plan=run.plan,expected_sha256=sha,expected_contract_hash=run.contract_hash)
            restored=MatchedReviewPlan(json.loads(json.dumps(plan.to_dict())),expected_hash=plan.plan_hash)
            again=review(loaded,restored,expected_plan_hash=restored.plan_hash)
        first.pop('elapsed_seconds');again.pop('elapsed_seconds');self.assertEqual(first,again)


if __name__=='__main__':unittest.main()
