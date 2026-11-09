import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from model.VCASAHGCN import SharedHGCN
from evaluation.controlled_signal import SignalPlan,review_signals
from evaluation.linear_probe import fingerprint


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(64);model=SharedHGCN(2,2,2,dropout=.5).train()
    with torch.no_grad():
        for layer in (model.hgc1,model.hgc2):layer.weight.copy_(torch.eye(2));layer.bias.zero_()
    a=torch.eye(4)
    for i in range(1,4):a[i,i-1]=1
    a/=a.sum(1,keepdim=True)
    x=torch.ones(4,2)
    request={'sources':[[0,0],[0,1],[2,0]],'targets':[[1,1,0],[2,0,0],[2,2,0],[2,3,0]],'epsilon':.01,'cut_nodes':[1]}
    return model,x,a,request


def run(model,x,a,request):
    plan=SignalPlan.freeze(model,x,a,request=request)
    return review_signals(model,x,a,plan,expected_plan_hash=plan.plan_hash)


class ControlledSignalTests(unittest.TestCase):
    def test_analytic_matrix_powers_directed_reach_and_cut(self):
        model,x,a,request=fixture();report=run(model,x,a,request)
        # Independent matrix powers and explicit diagonal mask, positive ReLU region.
        scale=1/np.sqrt(1+1e-5);matrix=a.numpy().astype(np.float64);mask=np.eye(4);mask[1,1]=0
        checked=0
        for row in report['rows']:
            stage,node,channel=row['target'];cut=row['arm']=='cut_intermediate'
            operator=(mask if cut else np.eye(4))@matrix*scale
            if stage==2:operator=matrix@operator
            for response in row['responses']:
                i,c=response['source'];expected=operator[node,i] if c==channel else 0.
                self.assertAlmostEqual(response['jacobian'],expected,places=6)
                self.assertLess(abs(response['finite_difference']-expected),1e-5)
                if not response['within_stage_reach']:self.assertEqual(response['jacobian'],0)
                checked+=1
        self.assertEqual(checked,24)
        original=next(r for r in report['rows'] if r['arm']=='original' and r['target']==[2,2,0])
        cut=next(r for r in report['rows'] if r['arm']=='cut_intermediate' and r['target']==[2,2,0])
        self.assertGreater(original['responses'][0]['jacobian'],0)
        self.assertEqual(cut['responses'][0]['jacobian'],0)
        reversed_row=next(r for r in report['rows'] if r['target']==[2,0,0])
        self.assertIsNone(reversed_row['responses'][2]['hop_distance'])

    def test_private_state_gradients_rng_dense_sparse_and_saved_plan(self):
        model,x,a,request=fixture()
        for p in model.parameters():p.grad=torch.ones_like(p)
        before=fingerprint({'state':model.state_dict(),'grad':[p.grad for p in model.parameters()]})
        flags=[m.training for m in model.modules()];rng=torch.get_rng_state().clone()
        first=run(model,x,a,request);second=run(model,x,a.to_sparse(),request)
        self.assertEqual(first,second);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(flags,[m.training for m in model.modules()])
        self.assertEqual(before,fingerprint({'state':model.state_dict(),'grad':[p.grad for p in model.parameters()]}))
        plan=SignalPlan.freeze(model,x,a,request=request)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'plan.json';path.write_text(json.dumps(plan.to_dict()))
            restored=SignalPlan(json.loads(path.read_text()),expected_hash=plan.plan_hash)
        self.assertEqual(first,review_signals(model,x,a,restored,expected_plan_hash=plan.plan_hash))
        self.assertEqual(first['optimizer_updates'],0)

    def test_identity_cut_and_zero_weights_do_not_fake_signal(self):
        model,x,a,request=fixture();request['cut_nodes']=[];report=run(model,x,a,request)
        original=[r for r in report['rows'] if r['arm']=='original'];cut=[r for r in report['rows'] if r['arm']!='original']
        for l,r in zip(original,cut):self.assertEqual(l['responses'],r['responses']);self.assertEqual(l['baseline'],r['baseline'])
        with torch.no_grad():model.hgc2.weight.zero_()
        report=run(model,x,a,request)
        for row in report['rows']:
            if row['target'][0]==2:
                self.assertTrue(all(z['jacobian']==0 and z['finite_difference']==0 for z in row['responses']))

    def test_frozen_identity_and_invalid_budgets_refused(self):
        model,x,a,request=fixture();plan=SignalPlan.freeze(model,x,a,request=request)
        for changed in (lambda:review_signals(model,x+1,a,plan,expected_plan_hash=plan.plan_hash),
                        lambda:review_signals(model,x,torch.eye(4),plan,expected_plan_hash=plan.plan_hash),
                        lambda:review_signals(model,x,a,plan,expected_plan_hash='bad')):
            with self.assertRaises(ValueError):changed()
        for change in (lambda r:r.update(epsilon=float('nan')),lambda r:r.update(epsilon=0),
                       lambda r:r.update(sources=[[0,0],[0,0]]),lambda r:r.update(cut_nodes=[4]),
                       lambda r:r.update(targets=[[3,0,0]]),lambda r:r.update(query=True)):
            r=copy.deepcopy(request);change(r)
            with self.assertRaises(ValueError):SignalPlan.freeze(model,x,a,request=r)
        for graph in (a*2,a.double(),a-torch.eye(4),torch.full_like(a,float('nan'))):
            with self.assertRaises(ValueError):SignalPlan.freeze(model,x,graph,request=request)
        with torch.no_grad():model.hgc1.weight.add_(1)
        with self.assertRaises(ValueError):review_signals(model,x,a,plan,expected_plan_hash=plan.plan_hash)

    def test_project_nontrivial_stage_weights_and_local_nonlinearity(self):
        model,x,a,request=fixture();g=torch.Generator().manual_seed(512)
        with torch.no_grad():
            model.hgc1.weight.copy_(torch.rand(2,2,generator=g)+.1)
            model.hgc2.weight.copy_(torch.rand(2,2,generator=g)+.1)
        report=run(model,x,a,request)
        self.assertTrue(all(z['absolute_difference']<3e-5 for r in report['rows'] for z in r['responses']))
        # Negative first-stage preactivation suppresses propagation despite reachability.
        with torch.no_grad():model.hgc1.bias.fill_(-100)
        dead=run(model,x,a,request)
        self.assertTrue(any(z['within_stage_reach'] for r in dead['rows'] for z in r['responses']))
        self.assertTrue(all(z['jacobian']==0 for r in dead['rows'] for z in r['responses']))


if __name__=='__main__':unittest.main()
