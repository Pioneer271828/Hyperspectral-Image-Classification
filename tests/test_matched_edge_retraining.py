import copy, json, tempfile, unittest
from pathlib import Path
import torch
from model.research_config import digest
from transfer.graph_route_context import route_state_hash
from transfer.matched_edge_retraining import MatchedRetrainingPlan
from tests.test_graph_edge_intervention import completed


def repin(v):
    v['plan_hash']=digest({k:x for k,x in v.items() if k!='plan_hash'})
    return MatchedRetrainingPlan(v,expected_hash=v['plan_hash'])


class MatchedRetrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.donor=completed(budget=4)
        cls.plan=MatchedRetrainingPlan.freeze(cls.donor,per_row=1,distance_bins=[2.,4.,100.],seeds=[64,512,2048])

    def test_six_arms_equal_initialization_exposure_full_resume_and_complete_zero(self):
        donor=self.donor;before=route_state_hash(donor.model);source=route_state_hash(donor.source);history=copy.deepcopy(donor.history)
        completed_names=[]
        with tempfile.TemporaryDirectory() as d:
            for name in self.plan.to_dict()['names']:
                a=self.plan.start(donor,name,expected_hash=self.plan.plan_hash)
                self.assertEqual(route_state_hash(a.model),source)
                for key in donor.spec:
                    if key!='graph_route':self.assertEqual(a.spec[key],donor.spec[key],key)
                b=self.plan.start(donor,name,expected_hash=self.plan.plan_hash)
                a.advance(24);b.advance(7);p=Path(d)/'run.pt';sha=b.save(p)
                b=self.plan.restore(donor,name,p,expected_hash=self.plan.plan_hash,expected_sha256=sha);b.advance(17)
                self.assertEqual(a.history,b.history);self.assertEqual(route_state_hash(a.model),route_state_hash(b.model))
                self.assertEqual((a.selected_step,route_state_hash(a.model)),(b.selected_step,route_state_hash(b.model)))
                for k,v in a.selected_state.items():
                    if torch.is_tensor(v):self.assertTrue(torch.equal(v,b.selected_state[k]))
                    else:self.assertEqual(v,b.selected_state[k])
                for k,slot in a.optimizer.state_dict()['state'].items():
                    for field,t in slot.items():self.assertTrue(torch.equal(t,b.optimizer.state_dict()['state'][k][field]))
                self.assertTrue(all(r['gradient_l1']['graph']>0 for r in a.history))
                self.assertTrue(any(r['changed_parameters']['graph']>0 for r in a.history))
                self.assertTrue(all(r['validation']['support']==[5]*9 for r in a.history))
                sha=b.save(p);c=self.plan.restore(donor,name,p,expected_hash=self.plan.plan_hash,expected_sha256=sha)
                c.advance(24);self.assertEqual(c.history,b.history)
                completed_names.append(name)
        self.assertEqual(len(completed_names),6)
        self.assertEqual((before,source,history),(route_state_hash(donor.model),route_state_hash(donor.source),donor.history))

    def test_zero_deletion_exact_banks_and_serialization(self):
        p=MatchedRetrainingPlan.freeze(self.donor,per_row=0,distance_bins=[100.],seeds=[64])
        p=MatchedRetrainingPlan(json.loads(json.dumps(p.to_dict())),expected_hash=p.plan_hash)
        self.assertEqual(p.bank(self.donor,'none',expected_hash=p.plan_hash).to_dict(),p.bank(self.donor,'important',expected_hash=p.plan_hash).to_dict())
        with self.assertRaisesRegex(ValueError,'unavailable'):p.start(self.donor,'random:64',expected_hash=p.plan_hash)

    def test_wrong_source_missing_context_and_unavailable_control(self):
        for mutate in (lambda v:v.update(source_state='wrong'),lambda v:v.update(selected_step=0),
                       lambda v:v['contexts'].pop(),lambda v:v['contexts'][0]['context'].update(role='test')):
            v=self.plan.to_dict();mutate(v);p=repin(v)
            with self.assertRaises(ValueError):p.start(self.donor,'none',expected_hash=p.plan_hash)
        with self.assertRaises(ValueError):self.plan.start(self.donor,'none',expected_hash='wrong')
        v=self.plan.to_dict();p=v['contexts'][0]['control_plan'];p['controls']['low_weight']={'status':'unresolved','removed':None,'reason':'injected timeout'}
        p['plan_hash']=digest({k:x for k,x in p.items() if k!='plan_hash'});bad=repin(v)
        with self.assertRaisesRegex(ValueError,'unavailable'):bad.start(self.donor,'low_weight',expected_hash=bad.plan_hash)


if __name__=='__main__':unittest.main()
