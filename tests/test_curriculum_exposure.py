"""Raw exposure gates, authoritative sequence reconstruction and real-step replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
import numpy as np
import torch
from loadData.spatial_tiles import tile_slice
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from model.research_config import digest
from transfer.curriculum_exposure import ExposurePlan,ExposureLedger
from transfer.sample_curriculum import curriculum_loss
from transfer.development_training import sparse_labels,evaluate_validation
from transfer.tile_sampling import SupportTileSampler
from transfer.g3_training import _forward,state_hash
from tests.test_int8_model import fixture

def freeze(scene,record,updates=24,seed=64,blocks=1,visits=1):
    return ExposurePlan.freeze(scene,record,updates=updates,seed=seed,
        minimum_class_blocks={str(c):blocks for c in record['classes']},minimum_support_visits=visits)

class ExposureTests(unittest.TestCase):
    def test_original_class_first_sequence_rng_and_owned_plan(self):
        _,s,r=fixture();tr=torch.get_rng_state().clone();nr=np.random.get_state();p=freeze(s,r)
        self.assertTrue(torch.equal(tr,torch.get_rng_state()));np.testing.assert_array_equal(nr[1],np.random.get_state()[1])
        sampler=SupportTileSampler(s.plan,r['roles'],sparse_labels(r,s.plan,'train_support'),r['classes'],'class_first');rng=np.random.default_rng(64)
        for n in range(1,25):
            tile,anchor=sampler.draw(rng);event=p.event(n)
            self.assertEqual((event['tile_id'],event['anchor']),(tile['id'],anchor));self.assertEqual(event['step'],n)
            self.assertEqual(r['roles'][event['tile_id']]['role'],'devtrain')
        value=p.to_dict();value['events'].clear();self.assertEqual(len(p.to_dict()['events']),24)
        q=ExposurePlan(s,r,json.loads(json.dumps(p.to_dict())),expected_plan_hash=p.plan_hash)
        self.assertEqual(p.to_dict(),q.to_dict())

    def test_independent_class_pixel_anchor_counts_and_all_prefixes(self):
        _,s,r=fixture();p=freeze(s,r);width=s.plan['shape'][1]
        for n in range(25):
            counts=Counter();blocks=Counter();anchors=Counter()
            for event in p.to_dict()['events'][:n]:
                y0,y1,x0,x1=s.plan['tiles'][event['tile_id']]['box'];present=set();anchors[event['anchor']]+=1
                for i,c in zip(r['train_support']['indices'],r['train_support']['labels']):
                    if y0<=i//width<y1 and x0<=i%width<x1:counts[i]+=1;present.add(c)
                blocks.update(present)
            result=p.exposure(n)
            for i in r['train_support']['indices']:self.assertEqual(result['support_visits'][str(i)],counts[i])
            for c in r['classes']:
                self.assertEqual(result['class_blocks'][str(c)],blocks[c]);self.assertEqual(result['anchor_counts'][str(c)],anchors[c])
            self.assertEqual(result['meets_minimum'],all(counts[i]>=1 for i in r['train_support']['indices']) and all(blocks[c]>=1 for c in r['classes']))
            self.assertEqual(p.resume_step(json.loads(json.dumps(p.progress(n)))),n+1 if n<24 else None)

    def test_underexposure_and_unfinished_budget_never_accepted_or_extended(self):
        _,s,r=fixture()
        for p in (freeze(s,r,updates=1),freeze(s,r,blocks=25)):
            before=p.to_dict();self.assertFalse(p.exposure()['meets_minimum'])
            with self.assertRaises(ValueError):p.require_eligible()
            ledger=ExposureLedger(p)
            with self.assertRaises(ValueError):ledger.finish()
            for n in range(1,p.updates+1):ledger.append(p.event(n))
            with self.assertRaises(ValueError):ledger.finish()
            with self.assertRaises(ValueError):ledger.append(p.event(1))
            self.assertEqual(before,p.to_dict());self.assertEqual(ledger.completed,p.updates)

    def test_semantic_tamper_wrong_registration_and_nontraining_isolation(self):
        _,s,r=fixture();p=freeze(s,r)
        poisoned=copy.deepcopy(s)
        for row in r['roles']:
            if row['role']!='devtrain':poisoned.cube[tile_slice(row)]=float('nan')
        self.assertEqual(freeze(poisoned,r).to_dict(),p.to_dict())
        for kind in ('event','support','sampler','selection'):
            v=p.to_dict()
            if kind=='event':v['events'][0]['tile_id']=next(t['id'] for t in r['roles'] if t['role']=='devval')
            if kind=='support':v['events'][0]['support_ids']=r['validation_support']['indices'].copy()
            if kind=='sampler':v['sampler']['mode']='uniform'
            if kind=='selection':v['selection']='query'
            v['plan_hash']=digest({k:x for k,x in v.items() if k!='plan_hash'})
            with self.assertRaises(ValueError):ExposurePlan(s,r,v,expected_plan_hash=v['plan_hash'])
        q=freeze(s,r,seed=512)
        with self.assertRaises(ValueError):ExposurePlan(s,r,q.to_dict(),expected_plan_hash=p.plan_hash)
        for kwargs in ({'updates':True},{'seed':-1},{'blocks':0},{'visits':0}):
            with self.assertRaises(ValueError):freeze(s,r,**kwargs)

    def test_ledger_rejects_repeats_skips_modified_prefix_and_wrong_plan(self):
        _,s,r=fixture();p=freeze(s,r);ledger=ExposureLedger(p)
        with self.assertRaises(ValueError):ledger.append(p.event(2))
        boolean_step=p.event(1);boolean_step['step']=True
        with self.assertRaises(ValueError):ledger.append(boolean_step)
        ledger.append(p.event(1))
        with self.assertRaises(ValueError):ledger.append(p.event(1))
        for n in range(2,6):ledger.append(p.event(n))
        packet=json.loads(json.dumps(ledger.snapshot()));restored=ExposureLedger(p,packet)
        self.assertEqual(restored.completed,5)
        for n in range(6,25):ledger.append(p.event(n));restored.append(p.event(n))
        self.assertEqual(ledger.snapshot(),restored.snapshot());self.assertEqual(ledger.finish()['status'],'passed_exposure_only')
        self.assertFalse(ledger.finish()['optimizer_execution_verified'])
        self.assertIsNone(p.resume_step(ledger.snapshot()))
        for kind in ('counts','prefix'):
            bad=copy.deepcopy(packet)
            if kind=='counts':bad['exposure']['support_visits'][str(r['train_support']['indices'][0])]+=1
            else:bad['prefix_hash']='0'*64
            bad['progress_hash']=digest({k:v for k,v in bad.items() if k!='progress_hash'})
            with self.assertRaises(ValueError):ExposureLedger(p,bad)
        with self.assertRaises(ValueError):ExposureLedger(freeze(s,r,seed=512),packet)

    def test_actual_network_24_and_disk_5_plus_19_exact_with_post_update_ledger(self):
        source,s,r=fixture();p=freeze(s,r);p.require_eligible();initial=state_hash(source)
        model=copy.deepcopy(source);opt=torch.optim.Adam(model.parameters(),lr=.0001);ledger=ExposureLedger(p);history=[]
        def step(m,o,l,n):
            event=p.event(n);tile=s.plan['tiles'][event['tile_id']];m.train();o.zero_grad()
            loss=curriculum_loss(_forward(m,s,tile),s,r,tile,class_balance=True)['total'];loss.backward()
            self.assertGreater(sum(float(x.grad.abs().sum()) for x in m.backbone.adapter.parameters() if x.grad is not None),0)
            o.step();l.append(event);return evaluate_validation(m,s,r)['nll']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for n in range(1,25):
                history.append(step(model,opt,ledger,n))
                if n==5:
                    save_model_checkpoint(model,path/'model.pt');torch.save({'optimizer':opt.state_dict(),'rng':torch.get_rng_state()},path/'state.pt')
                    (path/'progress.json').write_text(json.dumps({'plan':p.to_dict(),'progress':ledger.snapshot(),'history':history}))
            saved=json.loads((path/'progress.json').read_text());q=ExposurePlan(s,r,saved['plan'],expected_plan_hash=p.plan_hash)
            replay_ledger=ExposureLedger(q,saved['progress']);replay=saved['history'];restored=load_model_checkpoint(path/'model.pt');restored.backbone.adapter.cache_attention=False
            state=torch.load(path/'state.pt',weights_only=True);opt2=torch.optim.Adam(restored.parameters(),lr=.0001);opt2.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
            for n in range(q.resume_step(saved['progress']),25):replay.append(step(restored,opt2,replay_ledger,n))
            self.assertEqual(history,replay);self.assertEqual(state_hash(model),state_hash(restored));self.assertNotEqual(initial,state_hash(model))
            self.assertEqual(ledger.finish(),replay_ledger.finish());self.assertEqual(min(range(24),key=history.__getitem__),min(range(24),key=replay.__getitem__))
            for k,slot in opt.state_dict()['state'].items():
                for key,v in slot.items():self.assertTrue(torch.equal(v,opt2.state_dict()['state'][k][key]))
            self.assertEqual(int(next(iter(opt.state_dict()['state'].values()))['step']),24)
        self.assertEqual(initial,state_hash(source))

if __name__=='__main__':unittest.main()
