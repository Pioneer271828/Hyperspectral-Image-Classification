"""Support isolation, protected gradients and paired actual-network recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from loadData.spatial_tiles import tile_slice
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from model.research_config import digest
from model.task_losses import probability_loss_components
from transfer.development_training import sparse_labels,evaluate_validation
from transfer.g3_training import _forward,state_hash
from transfer.sample_curriculum import SampleCurriculum,score_training_support,curriculum_loss
from transfer.tile_sampling import SupportTileSampler
from tests.test_int8_model import fixture

ORDERS=('balanced','random','easy_to_hard','hard_to_easy')

def setup():
    m,s,r=fixture();scores=score_training_support(m,s,r,expected_teacher_state=state_hash(m))
    return m,s,r,scores

def policy(scores,order='easy_to_hard',seed=512):
    return SampleCurriculum.freeze(scores,updates=12,order=order,seed=seed,start_fraction=.2,full_at=10,minimum_weight=.1)

class SampleCurriculumTests(unittest.TestCase):
    def test_teacher_train_support_only_manual_scores_state_rng_and_poison(self):
        m,s,r=fixture();initial=state_hash(m);rng=torch.get_rng_state().clone();nr=np.random.get_state()
        with patch('transfer.sample_curriculum._forward',wraps=_forward) as forward:
            scores=score_training_support(m,s,r,expected_teacher_state=initial)
        self.assertEqual(initial,state_hash(m));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        np.testing.assert_array_equal(nr[1],np.random.get_state()[1]);self.assertFalse(m.backbone.adapter.cache_attention)
        truth={};support=dict(zip(r['train_support']['indices'],r['train_support']['labels']))
        with torch.no_grad():
            for call in forward.call_args_list:
                tile=call.args[2];self.assertEqual(r['roles'][tile['id']]['role'],'devtrain')
                y0,y1,x0,x1=tile['box'];yy,xx=np.mgrid[y0:y1,x0:x1];ids=(yy*s.plan['shape'][1]+xx).ravel()
                p=_forward(m,s,tile)[0]
                for j,i in enumerate(ids):
                    if i in support:truth[int(i)]=float(-p[j,support[i]-1].clamp(1e-8,1).log())
        self.assertEqual(scores['losses'],[truth[i] for i in scores['support']['indices']]);self.assertEqual(len(truth),90)
        poisoned=copy.deepcopy(s)
        for row in r['roles']:
            if row['role']!='devtrain':poisoned.cube[tile_slice(row)]=float('nan')
        other=score_training_support(m,poisoned,r,expected_teacher_state=initial)
        for key in ('losses','identity','support','teacher_state'):self.assertEqual(scores[key],other[key])

    def test_teacher_guards_fail_before_forward(self):
        for mode in ('hash','training','cache','dtype'):
            m,s,r=fixture();expected=state_hash(m)
            if mode=='hash':expected='0'*64
            if mode=='training':m.train()
            if mode=='cache':m.backbone.adapter.cache_attention=True
            if mode=='dtype':m.double();expected=state_hash(m)
            with patch('transfer.sample_curriculum._forward') as forward:
                with self.assertRaises(ValueError):score_training_support(m,s,r,expected_teacher_state=expected)
                forward.assert_not_called()

    def test_ranks_pacing_class_mass_positive_floor_and_rng(self):
        _,_,_,scores=setup();ids=scores['support']['indices'];labels=np.array(scores['support']['labels'])
        tr=torch.get_rng_state().clone();nr=np.random.get_state()
        for order in ORDERS:
            p=policy(scores,order);v=p.to_dict();self.assertEqual(v,SampleCurriculum(json.loads(json.dumps(v))).to_dict())
            for c in scores['classes']:
                chosen=[j for j,l in enumerate(labels) if l==c]
                expected=sorted(chosen,key=lambda j:((1 if order=='easy_to_hard' else -1)*scores['losses'][j],ids[j]))
                if order in ('easy_to_hard','hard_to_easy'):
                    self.assertEqual([v['ranks'][str(ids[j])] for j in expected],list(range(10)))
            for step in range(1,13):
                w=p.weights(ids,step=step);self.assertGreaterEqual(w.min(),.1)
                for c in scores['classes']:self.assertAlmostEqual(w[labels==c].sum(),10)
                if step>=10 or order=='balanced':np.testing.assert_array_equal(w,np.ones(90))
        self.assertTrue(torch.equal(tr,torch.get_rng_state()));np.testing.assert_array_equal(nr[1],np.random.get_state()[1])
        self.assertNotEqual(policy(scores,'random',512).to_dict()['ranks'],policy(scores,'random',2048).to_dict()['ranks'])
        self.assertEqual(policy(scores,'random',512).to_dict(),policy(scores,'random',512).to_dict())

    def test_singleton_hard_minority_retains_equal_class_gradient(self):
        _,_,_,s=setup();s['support']={'indices':[0,1,2,3],'labels':[1,1,1,2],'seed':64};s['classes']=[1,2];s['losses']=[.1,.2,10,20]
        s['score_hash']=digest({k:v for k,v in s.items() if k!='score_hash'});p=policy(s)
        # Only a difficult majority pixel and the rare class occur in this block.
        weights=p.weights([2,3],step=1);np.testing.assert_array_equal(weights,[1.,1.])
        logits=torch.zeros(2,2,requires_grad=True);probs=logits.softmax(-1)
        labels=torch.eye(2);mask=torch.tensor(weights,dtype=torch.float32)[:,None].expand(-1,2)
        loss=probability_loss_components((probs,probs,probs),labels,mask,class_balance=True)['total'];loss.backward()
        self.assertGreater(float(logits.grad[1].abs().sum()),0.)
        self.assertTrue(torch.equal(logits.grad[0].abs(),logits.grad[1].abs()))

    def test_off_balanced_exact_original_and_graph_terms_unchanged(self):
        m,s,r,scores=setup();tile=next(t for t in s.plan['tiles'] if r['roles'][t['id']]['role']=='devtrain')
        with torch.no_grad():out=_forward(m,s,tile)
        out=tuple(out[:3])+({'compactness_loss':torch.tensor(.7),'assignment_entropy_loss':torch.tensor(.2),'assignment_balance_loss':torch.tensor(.3)},)
        target=torch.tensor(sparse_labels(r,s.plan,'train_support')[tile_slice(tile)].ravel());mask=(target>0)[:,None].expand(-1,s.classes).float()
        onehot=torch.nn.functional.one_hot((target-1).clamp_min(0),s.classes).float()*mask
        kwargs={'class_balance':True,'graph_loss_weights':{'compactness':.4,'entropy':.5,'balance':.6},'cnn_loss_weight':.7,'graph_branch_loss_weight':.8}
        original=probability_loss_components(out,onehot,mask,**kwargs)
        off=curriculum_loss(out,s,r,tile,**kwargs);balanced=curriculum_loss(out,s,r,tile,policy=policy(scores,'balanced'),step=1,**kwargs)
        for key in original:
            self.assertTrue(torch.equal(original[key],off[key]));self.assertTrue(torch.equal(original[key],balanced[key]))
        weighted=curriculum_loss(out,s,r,tile,policy=policy(scores),step=1,**kwargs)
        for key in original:
            if 'graph_' in key and 'supervised' not in key:self.assertTrue(torch.equal(original[key],weighted[key]))

    def test_invalid_identity_controls_ranks_roles_and_budget(self):
        m,s,r,scores=setup();p=policy(scores);tile=next(t for t in s.plan['tiles'] if r['roles'][t['id']]['role']=='devtrain')
        with torch.no_grad():out=_forward(m,s,tile)
        for key,value in (('ranks',{}),('selection','query'),('sampler','uniform')):
            bad=p.to_dict();bad[key]=value;bad['policy_hash']=digest({k:v for k,v in bad.items() if k!='policy_hash'})
            with self.assertRaises(ValueError):SampleCurriculum(bad)
        for change in ({'minimum_weight':0},{'full_at':13},{'start_fraction':float('nan')},{'seed':True},{'order':'bad'}):
            args=dict(updates=12,order='balanced',seed=64,start_fraction=.2,full_at=10,minimum_weight=.1);args.update(change)
            with self.assertRaises(ValueError):SampleCurriculum.freeze(scores,**args)
        for ids,step in (([scores['support']['indices'][0]],0),([999999],1),([],1),([0],True)):
            with self.assertRaises(ValueError):p.weights(ids,step=step)
        with self.assertRaises(ValueError):curriculum_loss(out,s,r,tile,policy=p,step=1)
        with self.assertRaises(ValueError):curriculum_loss(out[:2],s,r,tile)
        with self.assertRaises(ValueError):curriculum_loss(out,s,r,tile,step=1)
        wrong=next(t for t in s.plan['tiles'] if r['roles'][t['id']]['role']=='devval')
        with self.assertRaises(ValueError):curriculum_loss(out,s,r,wrong,policy=p,step=1,class_balance=True)

    def test_four_arms_12_updates_disk_5_plus_7_exact_sampler_optimizer_and_selection(self):
        source,s,r,scores=setup();initial=state_hash(source);all_draws=[]
        sampler=SupportTileSampler(s.plan,r['roles'],sparse_labels(r,s.plan,'train_support'),r['classes'],'class_first')
        for order in ORDERS:
            p=policy(scores,order);m=copy.deepcopy(source);opt=torch.optim.Adam(m.parameters(),lr=.0001);rng=np.random.default_rng(64);history=[];draws=[]
            def step(model,optimizer,generator,n):
                tile,anchor=sampler.draw(generator);model.train();optimizer.zero_grad()
                out=_forward(model,s,tile);loss=curriculum_loss(out,s,r,tile,policy=p,step=n,class_balance=True)['total'];loss.backward()
                self.assertGreater(sum(float(x.grad.abs().sum()) for x in model.backbone.adapter.parameters() if x.grad is not None),0.)
                optimizer.step();validation=evaluate_validation(model,s,r)
                self.assertEqual(validation['count'],45)
                return validation['nll'],(tile['id'],anchor)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)
                for n in range(1,13):
                    loss,draw=step(m,opt,rng,n);history.append(loss);draws.append(draw)
                    if n==5:
                        save_model_checkpoint(m,path/'model.pt')
                        torch.save({'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state()},path/'resume.pt')
                        (path/'progress.json').write_text(json.dumps({'completed':n,'policy':p.to_dict(),'rng':rng.bit_generator.state,'history':history,'draws':draws}))
                restored=load_model_checkpoint(path/'model.pt');restored.backbone.adapter.cache_attention=False
                saved=torch.load(path/'resume.pt',weights_only=True);progress=json.loads((path/'progress.json').read_text())
                self.assertEqual(SampleCurriculum(progress['policy']).policy_hash,p.policy_hash)
                opt2=torch.optim.Adam(restored.parameters(),lr=.0001);opt2.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
                rng2=np.random.default_rng();rng2.bit_generator.state=progress['rng'];replay=progress['history'];replaydraw=[tuple(x) for x in progress['draws']]
                for n in range(progress['completed']+1,13):
                    loss,draw=step(restored,opt2,rng2,n);replay.append(loss);replaydraw.append(draw)
                self.assertEqual(history,replay);self.assertEqual(draws,replaydraw);self.assertEqual(state_hash(m),state_hash(restored))
                self.assertNotEqual(initial,state_hash(m));self.assertEqual(min(range(12),key=history.__getitem__),min(range(12),key=replay.__getitem__))
                for k,slot in opt.state_dict()['state'].items():
                    for name,v in slot.items():self.assertTrue(torch.equal(v,opt2.state_dict()['state'][k][name]))
                self.assertEqual(int(next(iter(opt.state_dict()['state'].values()))['step']),12)
            all_draws.append(draws)
        self.assertTrue(all(x==all_draws[0] for x in all_draws));self.assertEqual(initial,state_hash(source))

if __name__=='__main__':unittest.main()
