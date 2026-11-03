"""MON-2 known arithmetic, lifecycle and optimizer/model noninterference."""
import copy
import json
import os
import unittest
from contextlib import nullcontext
import torch
from monitoring.readonly import LayerSpec
from monitoring.updates import ParameterSpec, UpdateMonitor
from test_monitoring_statistics import context
from joint_training import _tree_fingerprint

DEVICE=os.environ.get('MON2_DEVICE','cpu')
torch.set_num_threads(1)


class UpdateTests(unittest.TestCase):
    def fixture(self, **kwargs):
        model=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False)).to(DEVICE)
        with torch.no_grad():model[0].weight.copy_(torch.tensor([[3.,4.],[0.,2.]],device=DEVICE))
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        rows=[];mon=UpdateMonitor(model,opt,[ParameterSpec('0.weight',0,'output_channel')],rows.append,**kwargs)
        return model,opt,rows,mon

    def check_clean(self,model,opt):
        self.assertFalse(opt._optimizer_step_pre_hooks)
        self.assertFalse(opt._optimizer_step_post_hooks)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_known_sgd_gradient_weight_and_actual_delta(self):
        model,opt,rows,mon=self.fixture()
        before=model[0].weight.detach().clone()
        with mon.sample(context(),contract='fp32_single_backward'):
            model(torch.ones(1,2,device=DEVICE)).sum().backward();opt.step()
        r=rows[0]['parameters'][0]
        self.assertEqual(r['gradient_rms'],[1.,1.]);self.assertEqual(r['weight_l2_before'],[5.,2.])
        delta=model[0].weight.detach()-before
        expected=delta.double().square().sum(1).sqrt()
        self.assertEqual(r['update_l2'],expected.cpu().tolist())
        self.assertEqual(r['relative_update'],(expected/(torch.tensor([5.,2.],dtype=torch.float64,device=DEVICE)+1e-12)).cpu().tolist())
        self.assertEqual(r['changed_elements'],4);self.assertEqual(r['gradient_state'],'present')
        self.check_clean(model,opt);json.dumps(rows,allow_nan=False)

    def test_adamw_zero_gradient_can_still_update(self):
        model,opt,rows,_=self.fixture();opt=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.1)
        mon=UpdateMonitor(model,opt,[ParameterSpec('0.weight',0,'channel')],rows.append)
        with mon.sample(context(),contract='fp32_single_backward'):
            model(torch.ones(1,2,device=DEVICE)).sum().mul(0).backward();opt.step()
        r=rows[0]['parameters'][0]
        self.assertEqual(r['gradient_state'],'zero');self.assertEqual(r['update_state'],'changed')
        self.assertGreater(r['changed_elements'],0)

    def test_missing_frozen_unregistered_and_zero_update_are_distinct(self):
        model=torch.nn.ParameterDict({'used':torch.nn.Parameter(torch.ones(2,2,device=DEVICE)),
          'unused':torch.nn.Parameter(torch.ones(2,2,device=DEVICE)),
          'frozen':torch.nn.Parameter(torch.ones(2,2,device=DEVICE),requires_grad=False),
          'omitted':torch.nn.Parameter(torch.ones(2,2,device=DEVICE))})
        opt=torch.optim.SGD([model['used'],model['unused']],lr=0)
        records=[];mon=UpdateMonitor(model,opt,[ParameterSpec(n,0,'channel') for n in model],records.append)
        with mon.sample(context(),contract='fp32_single_backward'):
            (model['used'].sum()+model['omitted'].sum()).backward();opt.step()
        rows={r['parameter']:r for r in records[0]['parameters']}
        self.assertEqual(rows['unused']['gradient_state'],'missing')
        self.assertEqual(rows['frozen']['gradient_state'],'frozen')
        self.assertIsNone(rows['omitted']['optimizer_group'])
        self.assertEqual(rows['omitted']['gradient_state'],'present')
        self.assertTrue(all(r['update_state']=='zero' for r in rows.values()))

    def test_activation_gradient_masks_shared_calls_and_empty(self):
        model,opt,rows,mon=self.fixture(activations=[LayerSpec('0','NC',0,'linear-output','pixel')])
        mask=torch.tensor([True,False,True],device=DEVICE)
        with mon.sample(context(),contract='fp32_single_backward',masks={'0':mask}):
            first=model(torch.ones(3,2,device=DEVICE));second=model(torch.ones(3,2,device=DEVICE))
            (first.sum()+second.sum()*2).backward();opt.step()
        act=rows[0]['activation_gradients'];self.assertEqual([r['call_index'] for r in act],[0,1])
        self.assertEqual(act[0]['gradient_rms'],[1.,1.]);self.assertEqual(act[1]['gradient_rms'],[2.,2.])
        self.assertEqual(act[0]['count_per_channel'],2)
        opt.zero_grad(set_to_none=True)
        with mon.sample(context(),contract='fp32_single_backward',masks={'0':torch.zeros(3,dtype=torch.bool,device=DEVICE)}):
            model(torch.ones(3,2,device=DEVICE)).sum().backward();opt.step()
        self.assertEqual(rows[1]['activation_gradients'][0]['gradient_state'],'empty')
        self.assertEqual(rows[1]['activation_gradients'][0]['gradient_rms'],[None,None])
        self.check_clean(model,opt)

    def test_exact_adam_noninterference_and_bounded_repeated_sampling(self):
        torch.manual_seed(44)
        first=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.LeakyReLU(.2),torch.nn.Linear(4,2)).to(DEVICE)
        second=copy.deepcopy(first);x=torch.ones(5,3,device=DEVICE)
        states=[];record_counts=[]
        for active,model in [(False,first),(True,second)]:
            opt=torch.optim.Adam(model.parameters(),lr=.001);rows=[]
            mon=UpdateMonitor(model,opt,[ParameterSpec('0.weight',0,'channel')],rows.append,
                activations=[LayerSpec('1','NC',0,'post-leakyrelu','pixel')])
            rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if DEVICE=='cuda' else None
            for step in range(20):
                with mon.sample(context()|{'step':step},contract='fp32_single_backward',enabled=active):
                    opt.zero_grad(set_to_none=True);output=model(x);output.square().mean().backward();opt.step()
                self.check_clean(model,opt)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            if cuda_rng is not None:self.assertTrue(torch.equal(cuda_rng,torch.cuda.get_rng_state()))
            states.append(_tree_fingerprint({'model':model.state_dict(),'optimizer':opt.state_dict(),
                'gradient':{n:p.grad for n,p in model.named_parameters()},'output':output.detach()}))
            record_counts.append(len(rows));self.assertEqual(mon.emitted,len(rows))
            self.assertFalse(hasattr(mon,'records'));self.assertFalse(hasattr(mon,'before'))
        self.assertEqual(states[0],states[1]);self.assertEqual(record_counts,[0,20])

    def test_rejected_protocols_budget_and_identity(self):
        model,opt,rows,mon=self.fixture()
        for contract in ('amp','accumulated','closure'):
            with self.assertRaises(ValueError):
                with mon.sample(context(),contract=contract):pass
        for bad in (context()|{'partition':'test'},context()|{'step':True},context()|{'dataset':''}):
            with self.assertRaises(ValueError):
                with mon.sample(bad,contract='fp32_single_backward'):pass
        for spec in (ParameterSpec('0.weight',2,'channel'),ParameterSpec('missing',0,'channel')):
            other=UpdateMonitor(model,opt,[spec],rows.append)
            with self.assertRaises(ValueError):
                with other.sample(context(),contract='fp32_single_backward'):pass
        small=UpdateMonitor(model,opt,[ParameterSpec('0.weight',0,'channel')],rows.append,max_snapshot_bytes=1)
        with self.assertRaisesRegex(ValueError,'budget'):
            with small.sample(context(),contract='fp32_single_backward'):pass
        self.check_clean(model,opt)

    def test_missing_multiple_step_closure_and_nested_cleanup(self):
        model,opt,rows,mon=self.fixture()
        with self.assertRaisesRegex(RuntimeError,'did not complete'):
            with mon.sample(context(),contract='fp32_single_backward'):pass
        with self.assertRaisesRegex(RuntimeError,'Overlapping'):
            with mon.sample(context(),contract='fp32_single_backward'):
                with mon.sample(context(),contract='fp32_single_backward'):pass
        with self.assertRaisesRegex(ValueError,'closures'):
            with mon.sample(context(),contract='fp32_single_backward'):opt.step(lambda:None)
        with self.assertRaisesRegex(RuntimeError,'Exactly one'):
            with mon.sample(context(),contract='fp32_single_backward'):opt.step();opt.step()
        self.check_clean(model,opt);self.assertEqual(rows,[])
        with self.assertRaisesRegex(RuntimeError,'body failed'):
            with mon.sample(context(),contract='fp32_single_backward'):raise RuntimeError('body failed')
        self.check_clean(model,opt)

    def test_nonfinite_and_sink_error_do_not_leak_hooks(self):
        model,opt,rows,mon=self.fixture();before=model[0].weight.detach().clone()
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            with mon.sample(context(),contract='fp32_single_backward'):
                model[0].weight.grad=torch.full_like(before,float('nan'));opt.step()
        self.assertTrue(torch.equal(before,model[0].weight));self.check_clean(model,opt)
        opt.zero_grad(set_to_none=True)
        def bad_sink(record):raise OSError('sink failed')
        mon=UpdateMonitor(model,opt,[ParameterSpec('0.weight',0,'channel')],bad_sink)
        with self.assertRaisesRegex(OSError,'sink failed'):
            with mon.sample(context(),contract='fp32_single_backward'):
                model(torch.ones(1,2,device=DEVICE)).sum().backward();opt.step()
        self.assertFalse(torch.equal(before,model[0].weight));self.check_clean(model,opt)

    def test_activation_repeated_backward_and_call_bound(self):
        model,opt,rows,mon=self.fixture(activations=[LayerSpec('0','NC',0,'linear-output','pixel')],max_calls=1)
        with self.assertRaisesRegex(RuntimeError,'call budget'):
            with mon.sample(context(),contract='fp32_single_backward'):
                model(torch.ones(1,2,device=DEVICE));model(torch.ones(1,2,device=DEVICE))
        self.check_clean(model,opt)

    def test_actual_three_task_model_state_optimizer_and_rng_noninterference(self):
        from test_joint_baseline import tiny_items
        from test_joint_training import settings
        from model.research_config import ResearchConfig
        from joint_training import train_step
        cfg=ResearchConfig.from_dict(settings()['config'])
        data={n:{k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in i.items()} for n,i in tiny_items().items()}
        torch.manual_seed(45)
        first=cfg.build_model({n:i['class_num'] for n,i in data.items()}).to(DEVICE)
        second=copy.deepcopy(first);results=[];records=[]
        for active,model in [(False,first),(True,second)]:
            opt=torch.optim.SGD(model.parameters(),lr=.001)
            specs=[ParameterSpec('backbone.CNN_Branch.0.point_conv.weight',0,'pixel_feature')]
            mon=UpdateMonitor(model,opt,specs,records.append,
                activations=[LayerSpec('backbone.CNN_Branch.0','NCHW',0,'post-block','pixel')])
            rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if DEVICE=='cuda' else None
            trace=[]
            for step,(name,item) in enumerate(data.items()):
                with mon.sample(context()|{'step':step,'dataset':name},contract='fp32_single_backward',enabled=active):
                    trace.append(train_step(model,opt,name,item,cfg))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            if cuda_rng is not None:self.assertTrue(torch.equal(cuda_rng,torch.cuda.get_rng_state()))
            results.append({'model':model.state_dict(),'optimizer':opt.state_dict(),
                'gradients':{n:p.grad for n,p in model.named_parameters()},'trace':trace})
            self.check_clean(model,opt)
        torch.testing.assert_close(results[0],results[1],rtol=1e-5,atol=2e-6)
        self.assertEqual(len(records),3)
        self.assertTrue(all(r['parameters'][0]['changed_elements']>0 for r in records))
        with self.assertRaisesRegex(RuntimeError,'Repeated'):
            with mon.sample(context(),contract='fp32_single_backward'):
                out=model(torch.ones(1,2,device=DEVICE)).sum();out.backward(retain_graph=True);out.backward()
        self.check_clean(model,opt)


if __name__=='__main__':unittest.main()
