import copy
import os
import random
import unittest
import numpy as np
import torch
from monitoring import LayerSpec
from monitoring.probes import FrozenProbe
from monitoring.interventions import ChannelIntervention,isolated_probe,_snapshot

DEVICE=os.environ.get('MON3B_DEVICE','cpu')


class Twin(torch.nn.Module):
    def __init__(self):
        super().__init__();self.features=torch.nn.Identity();self.head=torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():self.head.weight.fill_(1)
    def forward(self,x):return self.head(self.features(x))


def fixture(model=None,path='features',call=0):
    model=(Twin() if model is None else model).to(DEVICE)
    x=torch.tensor([[2.,-1.],[1.,1.]],device=DEVICE)
    inputs={'x':x}
    c=dict(run_id='isolated',commit='source',checkpoint_hash='weights',data_hash='data',split_hash='split',dataset='source',partition='val',step=1,sample_id='case',condition='fixed')
    spec=LayerSpec(path,'NC',.1,'post-module','pixel')
    plan=FrozenProbe(c,inputs,{'features':(spec,call)}, {'features':{'all':torch.ones(2,dtype=torch.bool,device=DEVICE)}},threshold_source='frozen',allowed_datasets=['source'])
    return model,inputs,plan


class IsolationTests(unittest.TestCase):
    def run_case(self,model,inputs,plan,targets=(),factory=None,forward=None,**kw):
        return isolated_probe(model,lambda:copy.deepcopy(model) if factory is None else factory(),
                 (lambda m,x:m(x['x'])) if forward is None else forward,inputs,plan,interventions=targets,**kw)

    def test_noop_and_exact_single_group_controls(self):
        model,x,p=fixture();before=_snapshot(model)
        a=self.run_case(model,x,p);b=self.run_case(model,x,p,(ChannelIntervention('features',(0,)),))
        c=self.run_case(model,x,p,(ChannelIntervention('features',(1,)),));d=self.run_case(model,x,p,(ChannelIntervention('features',(0,1)),))
        torch.testing.assert_close(a['outputs'],torch.tensor([[1.],[2.]]),rtol=0,atol=0)
        torch.testing.assert_close(b['outputs'],torch.tensor([[-1.],[1.]]),rtol=0,atol=0)
        torch.testing.assert_close(c['outputs'],torch.tensor([[2.],[1.]]),rtol=0,atol=0)
        self.assertEqual(float(d['outputs'].abs().sum()),0)
        self.assertEqual(before,_snapshot(model));self.assertEqual(d['audit']['optimizer_updates'],0)
        self.assertEqual(d['statistics']['records'][0]['statistics']['response_rate'],[0,0])

    def test_modes_gradients_and_all_rng_preserved(self):
        model=torch.nn.Sequential(torch.nn.BatchNorm1d(2),torch.nn.Dropout(.5),torch.nn.Linear(2,2)).to(DEVICE)
        model.train();model[2].eval()
        for v in model.parameters():v.grad=torch.ones_like(v)
        model,x,p=fixture(model,'0');before=_snapshot(model)
        py=random.getstate();n=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state() if DEVICE=='cuda' else None
        def factory():
            random.random();np.random.rand();torch.rand(2,device=DEVICE)
            return copy.deepcopy(model)
        r=self.run_case(model,x,p,factory=factory)
        self.assertEqual(_snapshot(model),before);self.assertEqual(random.getstate(),py)
        self.assertTrue(np.array_equal(np.random.get_state()[1],n[1]));self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        if cuda is not None:self.assertTrue(torch.equal(cuda,torch.cuda.get_rng_state()))
        self.assertFalse(r['outputs'].requires_grad)

    def test_alias_and_wrong_state_rejected_before_forward(self):
        model,x,p=fixture();before=_snapshot(model)
        with self.assertRaisesRegex(ValueError,'fresh'):self.run_case(model,x,p,factory=lambda:model)
        def shared():
            m=copy.deepcopy(model);m.head.weight=torch.nn.Parameter(model.head.weight.detach());return m
        with self.assertRaisesRegex(ValueError,'shares storage'):self.run_case(model,x,p,factory=shared)
        def changed():
            m=copy.deepcopy(model)
            with torch.no_grad():m.head.weight.add_(1)
            return m
        with self.assertRaisesRegex(ValueError,'state/checkpoint'):self.run_case(model,x,p,factory=changed)
        self.assertEqual(before,_snapshot(model))

    def test_exceptions_cleanup_and_reference_isolation(self):
        model,x,p=fixture();fresh=[];before=_snapshot(model)
        def factory():m=copy.deepcopy(model);fresh.append(m);return m
        def fail(m,inputs):m(inputs['x']);raise RuntimeError('probe failure')
        with self.assertRaisesRegex(RuntimeError,'probe failure'):self.run_case(model,x,p,factory=factory,forward=fail)
        self.assertTrue(all(not m._forward_hooks for m in fresh[0].modules()));self.assertEqual(before,_snapshot(model))
        def mutate(m,inputs):
            m.head.weight.add_(1);return m(inputs['x'])
        with self.assertRaisesRegex(ValueError,'changed model'):self.run_case(model,x,p,forward=mutate)
        self.assertEqual(before,_snapshot(model))
        def mutate_input(m,inputs):inputs['x'].add_(1);return m(inputs['x'])
        with self.assertRaisesRegex(ValueError,'mutated cloned inputs'):self.run_case(model,x,p,forward=mutate_input)
        torch.testing.assert_close(x['x'],torch.tensor([[2.,-1.],[1.,1.]],device=DEVICE))

    def test_invalid_channel_call_and_copy_limits(self):
        model,x,p=fixture()
        for targets in ((ChannelIntervention('features',(9,)),),(ChannelIntervention('missing',(0,)),),
                        (ChannelIntervention('features',(0,)),ChannelIntervention('features',(1,)))):
            with self.assertRaises(ValueError):self.run_case(model,x,p,targets)
        for k in ('max_input_bytes','max_output_bytes','max_activation_bytes'):
            with self.assertRaisesRegex(ValueError,'budget'):self.run_case(model,x,p,**{k:1})
        model,x,p=fixture(call=1)
        with self.assertRaisesRegex(ValueError,'not executed'):
            self.run_case(model,x,p,(ChannelIntervention('features',(0,)),))

    def test_known_group_damage_exceeds_single_damage(self):
        # A redundant two-positive-channel decision survives either single removal.
        model,x,p=fixture();x['x'].copy_(torch.ones_like(x['x']))
        c=p.manifest['context'];spec=LayerSpec('features','NC',.1,'post-module','pixel')
        p=FrozenProbe(c,x,{'features':(spec,0)},{'features':{'all':torch.ones(2,dtype=torch.bool,device=DEVICE)}},threshold_source='fixed',allowed_datasets=['source'])
        a=self.run_case(model,x,p,(ChannelIntervention('features',(0,)),))
        b=self.run_case(model,x,p,(ChannelIntervention('features',(1,)),))
        group=self.run_case(model,x,p,(ChannelIntervention('features',(0,1)),))
        self.assertTrue((a['outputs']>.5).all());self.assertTrue((b['outputs']>.5).all());self.assertFalse((group['outputs']>.5).any())

    def test_sparse_input_budget_uses_stored_values(self):
        from monitoring.interventions import _copy_tree
        x=torch.sparse_coo_tensor(torch.tensor([[0],[99999]],device=DEVICE),torch.tensor([1.],device=DEVICE),(100000,100000)).coalesce()
        copied=_copy_tree({'q':x},[100])['q'];self.assertEqual(copied._nnz(),1)
        self.assertNotEqual(copied.values().data_ptr(),x.values().data_ptr())


if __name__=='__main__':unittest.main()
