"""Mathematical contracts and actual model noninterference for MON-0/1."""
import copy
import json
import os
import unittest
import torch
from monitoring import ChannelStats, LayerSpec, ReadOnlyMonitor, weight_statistics
from research_runtime import code_snapshot, tensor_hash
from train_spatial import SpatialConfig


def context():
    return dict(run_id='synthetic-monitoring',commit=code_snapshot()['commit'],checkpoint_hash='synthetic-initial-state',
                data_hash='synthetic-fixture',split_hash='synthetic-block',dataset='paviaU',partition='train',
                step=0,sample_id='block-0',condition='full-synthetic-spectrum')


class StatisticsTests(unittest.TestCase):
    def test_signed_masked_stats_empty_and_nonfinite(self):
        x=torch.tensor([[1.,-2.],[0.,.1],[-3.,4.]])
        stats=ChannelStats(2,.1);stats.update(x,'NC',torch.tensor([True,False,True]));r=stats.report()
        self.assertEqual(r['mean'],[-1.,1.]);self.assertEqual(r['variance_population'],[4.,9.])
        self.assertEqual(r['response_rate'],[1.,1.]);self.assertEqual(r['negative_rate'],[.5,.5])
        self.assertEqual(r['rms'],[5**.5,10**.5])
        empty=ChannelStats(2,.1);empty.update(x,'NC',torch.zeros(3,dtype=torch.bool))
        self.assertEqual(empty.report()['mean'],[None,None]);self.assertEqual(empty.report()['valid_blocks'],0)
        json.dumps(empty.report(),allow_nan=False)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):stats.update(torch.tensor([[float('nan'),0.]]),'NC')
        with self.assertRaises(ValueError):stats.update(x,'NC',torch.ones(3))
        for bad in (-1,float('nan'),True):
            with self.assertRaises(ValueError):ChannelStats(2,bad)

    def test_unequal_denominators_and_block_frequency(self):
        a=ChannelStats(1,.5);a.update(torch.tensor([[1.]]),'NC');a.update(torch.zeros(3,1),'NC')
        r=a.report();self.assertEqual(r['response_rate'],[.25]);self.assertEqual(r['block_response_rate'],[.5])
        whole=ChannelStats(1,.5);whole.update(torch.tensor([[1.],[0.],[0.],[0.]]),'NC')
        for key in ('mean','variance_population','rms','response_rate'):
            torch.testing.assert_close(torch.tensor(r[key]),torch.tensor(whole.report()[key]))
        nchw=ChannelStats(2,0);nchw.update(torch.ones(1,2,3,4),'NCHW')
        self.assertEqual(nchw.report()['count_per_channel'],12)
        hwc=ChannelStats(2,0);hwc.update(torch.ones(3,4,2),'HWC')
        self.assertEqual(hwc.report()['count_per_channel'],12)
        with self.assertRaises(ValueError):nchw.update(torch.ones(2,2,3,4),'NCHW')

    def test_static_weights(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,2))
        with torch.no_grad():model[0].weight.copy_(torch.tensor([[3.,4.],[0.,-2.]]))
        before={k:v.clone() for k,v in model.state_dict().items()}
        records=weight_statistics(model,['0'],0)
        w=next(r for r in records if r['parameter']=='weight')
        self.assertEqual(w['l1'],[7.,2.]);self.assertEqual(w['l2'],[5.,2.])
        self.assertEqual(w['near_zero_rate'],[0.,.5])
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(before[key],value))

    def test_selector_shared_calls_and_hook_cleanup(self):
        class Output(torch.nn.Module):
            def forward(self,x):return {'value':(x,x*2)}
        model=torch.nn.Sequential(Output());records=[]
        spec=LayerSpec('0','NC',.1,'returned-double','pixel',('value',1))
        monitor=ReadOnlyMonitor(model,[spec],records.append,max_calls=2)
        self.assertFalse(model[0]._forward_hooks)
        with monitor.sample(context()):
            y=model(torch.ones(3,2));model(torch.zeros(3,2))
            with self.assertRaises(RuntimeError):
                with ReadOnlyMonitor(model,[spec],records.append).sample(context()):pass
        self.assertFalse(model[0]._forward_hooks);self.assertEqual([r['call_index'] for r in records],[0,1])
        self.assertEqual(records[0]['statistics']['mean'],[2.,2.])
        with self.assertRaises(RuntimeError):
            with monitor.sample(context()):
                for _ in range(3):model(torch.ones(3,2))
        self.assertFalse(model[0]._forward_hooks)
        def broken_sink(record):raise OSError('sink error')
        with self.assertRaises(OSError):
            with ReadOnlyMonitor(model,[spec],broken_sink).sample(context()):model(torch.ones(3,2))
        self.assertFalse(model[0]._forward_hooks)
        with self.assertRaisesRegex(RuntimeError,'not called'):
            with monitor.sample(context()):pass
        self.assertFalse(model[0]._forward_hooks)
        bad=context();bad['partition']='test'
        with self.assertRaises(ValueError):
            with monitor.sample(bad):pass
        model[0].checkpoint_chunks=True
        with self.assertRaisesRegex(ValueError,'recomputation'):
            with monitor.sample(context()):pass


class ModelTests(unittest.TestCase):
    def test_actual_model_prediction_gradient_update_and_rng_unchanged(self):
        device=os.environ.get('MON_DEVICE','cpu')
        if device == 'cuda':
            prior=torch.are_deterministic_algorithms_enabled()
            self.addCleanup(torch.use_deterministic_algorithms,prior)
            torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1);torch.manual_seed(17)
        cfg=SpatialConfig({'name':'monitoring_fixture','device':device,'dataset':'paviaU',
            'model':{'graph_mode':'learned','latent_channels':8,'adapter_embed_dim':8,'norm_type':'group_norm',
                     'adapter_checkpoint_chunks':False},
            'interaction':{'interaction_mode':'gated'},'training':{'epochs':0,'steps_per_epoch':2}})
        baseline=cfg.research.build_model({'paviaU':9}).to(device).train();measured=copy.deepcopy(baseline)
        x=torch.randn(12,12,8,device=device);wave=torch.linspace(400,900,6,device=device)
        specs=[LayerSpec('backbone.adapter','HWC',1e-6,'post-adapter','pixel'),
               LayerSpec('backbone.CNN_Branch.0','NCHW',1e-6,'post-block','pixel'),
               LayerSpec('backbone.HGNN_Branch.batch_normalzation2','NC',1e-6,'post-normalization','region')]
        records=[];monitor=ReadOnlyMonitor(measured,specs,records.append)
        initial_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None
        def step(model,enabled):
            torch.set_rng_state(initial_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
            if enabled:
                with monitor.sample(context()):out=model('paviaU',x,wavelengths=wave,superpixel_scale=50)
            else:out=model('paviaU',x,wavelengths=wave,superpixel_scale=50)
            loss=sum(-p[:,0].clamp_min(1e-8).log().mean() for p in out[:3]);loss.backward()
            grads={k:p.grad.clone() if p.grad is not None else None for k,p in model.named_parameters()}
            optimizer.step()
            return out,loss.detach(),grads,torch.get_rng_state(),torch.cuda.get_rng_state_all() if device=='cuda' else []
        left=step(baseline,False);right=step(measured,True)
        # CPU exact; CUDA sparse reductions permit small roundoff, with equal predicted IDs.
        for a,b in zip(left[0][:3],right[0][:3]):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-5);self.assertTrue(torch.equal(a.argmax(-1),b.argmax(-1)))
        torch.testing.assert_close(left[1],right[1],rtol=1e-5,atol=2e-5)
        for k,a in left[2].items():
            b=right[2][k]
            if a is None:self.assertIsNone(b)
            else:torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-5)
        for k,a in baseline.state_dict().items():torch.testing.assert_close(a,measured.state_dict()[k],rtol=1e-4,atol=2e-5)
        self.assertTrue(torch.equal(left[3],right[3]))
        for a,b in zip(left[4],right[4]):self.assertTrue(torch.equal(a,b))
        self.assertEqual(len(records),3);self.assertEqual(records[-1]['unit'],'region')
        self.assertTrue(all(not m._forward_hooks for m in measured.modules()))
        json.dumps(records,allow_nan=False)
        # Repeated bounded sampling: output is streamed, collector retains no per-sample history.
        measured.eval();before={k:tensor_hash(v) for k,v in measured.state_dict().items()}
        for i in range(10):
            with torch.no_grad(),monitor.sample(context()|{'sample_id':str(i)}):
                measured('paviaU',x,wavelengths=wave,superpixel_scale=50)
        self.assertEqual(monitor.emitted,33)
        self.assertEqual(before,{k:tensor_hash(v) for k,v in measured.state_dict().items()})
        self.assertFalse(hasattr(monitor,'records'))

if __name__=='__main__':unittest.main()


