import copy
import os
import unittest
import torch
from monitoring import LayerSpec
from monitoring.probes import FrozenProbe,class_masks


DEVICE=os.environ.get('MON3_DEVICE','cpu')


def fixture():
    c=dict(run_id='probe',commit='frozen-code',checkpoint_hash='fixed-state',data_hash='data',
           split_hash='split',dataset='source',partition='val',step=1,sample_id='scene',condition='full')
    x=torch.tensor([[1.,-2.],[0.,.1],[-3.,4.]],device=DEVICE)
    spec=LayerSpec('cnn','NC',.1,'post-activation','pixel')
    calls={'cnn-call0':(spec,0)}
    masks={'cnn-call0':{'class1':torch.tensor([True,False,True],device=DEVICE),
                         'class2':torch.tensor([False,True,False],device=DEVICE),
                         'empty':torch.zeros(3,dtype=torch.bool,device=DEVICE)}}
    return c,{'cube':x.clone()},calls,masks,x


class ProbeTests(unittest.TestCase):
    def build(self,**kwargs):
        c,inputs,calls,masks,x=fixture()
        args=dict(context=c,inputs=inputs,calls=calls,buckets=masks,
                  threshold_source='source-training-warmup-sha',allowed_datasets=['source'])
        args.update(kwargs)
        return FrozenProbe(**args),c,inputs,x

    def test_signed_frequency_counts_and_empty(self):
        p,c,inputs,x=self.build();r=p.observe(inputs,{'cnn-call0':x},context=c)
        a,b,e=[v['statistics'] for v in r['records']]
        self.assertEqual(a['negative_rate'],[.5,.5]);self.assertEqual(a['response_rate'],[1.,1.])
        self.assertEqual(a['count_per_channel'],2);self.assertEqual(b['count_per_channel'],1)
        # Stored FP32 .1 is slightly above Python .1; use zero channel for equality check.
        self.assertEqual(b['response_rate'][0],0)
        self.assertEqual(e['count_per_channel'],0);self.assertEqual(e['response_rate'],[None,None])
        self.assertFalse(r['automatic_pruning']);self.assertFalse(r['model_execution_verified'])

    def test_frozen_inputs_context_and_thresholds(self):
        p,c,inputs,x=self.build()
        bad={'cube':inputs['cube']+1}
        with self.assertRaisesRegex(ValueError,'inputs'):p.observe(bad,{'cnn-call0':x},context=c)
        for k,v in [('checkpoint_hash','new'),('partition','test'),('condition','subset'),('dataset','target')]:
            with self.assertRaises(ValueError):p.observe(inputs,{'cnn-call0':x},context={**c,k:v})
        p._specs['cnn-call0']=LayerSpec('cnn','NC',.2,'post-activation','pixel')
        with self.assertRaisesRegex(ValueError,'threshold'):p.observe(inputs,{'cnn-call0':x},context=c)

    def test_cloned_masks_and_defensive_manifest(self):
        c,inputs,calls,masks,x=fixture();p,_,_,_=self.build(buckets=masks)
        masks['cnn-call0']['class1'].fill_(False)
        view=p.manifest;view['context']['dataset']='target'
        r=p.observe(inputs,{'cnn-call0':x},context=c)
        self.assertEqual(r['records'][0]['statistics']['count_per_channel'],2)
        p._masks['cnn-call0']['class1'].fill_(False)
        with self.assertRaisesRegex(ValueError,'mask'):p.observe(inputs,{'cnn-call0':x},context=c)

    def test_class_masks_protect_unlabelled_and_units(self):
        labels=torch.tensor([0,1,2,1],device=DEVICE,dtype=torch.int64)
        permitted=torch.tensor([True,True,True,False],device=DEVICE)
        m=class_masks(labels,permitted,[1,2,3],unit='pixel')
        self.assertEqual([int(v.sum()) for v in m.values()],[1,1,0]);self.assertFalse(any(v[0] for v in m.values()))
        for classes in ([0,1,2],[1,1,2],[1]):
            with self.assertRaises(ValueError):class_masks(labels,permitted,classes,unit='pixel')
        with self.assertRaisesRegex(ValueError,'region'):class_masks(labels,permitted,[1,2],unit='region')

    def test_repeated_calls_and_separate_units(self):
        c,inputs,calls,masks,x=fixture()
        calls['region1']=(LayerSpec('graph','NC',.1,'before-broadcast','region'),1)
        masks['region1']={'all':torch.ones(2,dtype=torch.bool,device=DEVICE)}
        p,_,_,_=self.build(calls=calls,buckets=masks)
        r=p.observe(inputs,{'cnn-call0':x,'region1':x[:2]},context=c)
        self.assertEqual(r['records'][-1]['unit'],'region')
        self.assertEqual(r['records'][-1]['statistics']['count_per_channel'],2)
        duplicate={**calls,'duplicate':calls['region1']}
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.build(calls=duplicate,buckets={**masks,'duplicate':masks['region1']})

    def test_invalid_or_uncovered_measurements(self):
        p,c,inputs,x=self.build()
        with self.assertRaises(ValueError):p.observe(inputs,{},context=c)
        with self.assertRaises(ValueError):p.observe(inputs,{'cnn-call0':x[:2]},context=c)
        bad=x.clone();bad[0,0]=float('nan')
        with self.assertRaises(ValueError):p.observe(inputs,{'cnn-call0':bad},context=c)
        for kwargs in ({'max_mask_bytes':1},{'max_calls':0},{'threshold_source':''},{'allowed_datasets':['target']},
                       {'allowed_datasets':'source'},{'context':{**c,'partition':'test'}}):
            with self.assertRaises(ValueError):self.build(**kwargs)

    def test_repeated_observation_preserves_input_rng_and_gradients(self):
        p,c,inputs,x=self.build();x.requires_grad_();x.grad=torch.ones_like(x)
        before=x.detach().clone();grad=x.grad.clone();state=torch.get_rng_state()
        cuda=torch.cuda.get_rng_state() if DEVICE=='cuda' else None
        r=p.observe(inputs,{'cnn-call0':x},context=c)
        for _ in range(10):self.assertEqual(r,p.observe(inputs,{'cnn-call0':x},context=c))
        torch.testing.assert_close(x,before,rtol=0,atol=0);torch.testing.assert_close(x.grad,grad,rtol=0,atol=0)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        if cuda is not None:self.assertTrue(torch.equal(cuda,torch.cuda.get_rng_state()))
        self.assertIsNone(x.grad_fn)


if __name__=='__main__':unittest.main()
