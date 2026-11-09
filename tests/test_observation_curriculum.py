"""Equal observation exposure, unchanged validation, real-network resume checks."""
import copy
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from transfer.observation_policy import ObservationPolicy
from transfer.observation_curriculum import CurriculumPolicy,ORDERS
from transfer.development_training import sparse_labels,adapt,evaluate_validation
from transfer.g3_training import _forward,state_hash
from loadData.spatial_tiles import tile_slice
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from model.research_config import digest
from tests.test_int8_model import fixture


def policies(updates=12,configs=None):
    model,scene,record=fixture()
    configs=configs or [{'mode':'none'},{'mode':'random_subset','keep_fraction':.5},{'mode':'random_subset','keep_fraction':.01}]
    base=ObservationPolicy.freeze(scene,record,updates=updates,training_configs=configs,train_seed=64,validation_seed=512)
    severity={digest(c):i for i,c in enumerate(configs)}
    return model,scene,record,base,severity


class CurriculumTests(unittest.TestCase):
    def test_exact_multiset_analytic_order_exposure_and_validation(self):
        _,_,_,base,severity=policies();raw=base.to_dict();keys=[digest(r['config']) for r in raw['training']]
        expected=Counter(r['record_hash'] for r in raw['training'])
        for order in ORDERS:
            p=CurriculumPolicy.freeze(base,order=order,severity=severity,seed=2048);v=p.to_dict()
            self.assertEqual(expected,Counter(r['record_hash'] for r in v['training']))
            self.assertEqual(raw['validation'],v['validation']);self.assertEqual(v['selection_condition'],'full')
            self.assertEqual(p.exposure()['updates'],12);self.assertEqual(set(p.exposure()['by_config'].values()),{4})
            self.assertEqual(p.exposure()['total_retained_band_exposures'],52)
            self.assertEqual(v['identity'],raw['identity'])
            if order=='fixed':self.assertEqual(v['training'],raw['training'])
            if order=='easy_to_hard':self.assertEqual(v['curriculum']['permutation'],[0,3,6,9,1,4,7,10,2,5,8,11])
            if order=='hard_to_easy':self.assertEqual(v['curriculum']['permutation'],[2,5,8,11,1,4,7,10,0,3,6,9])
            if order=='random':self.assertNotEqual(v['curriculum']['permutation'],list(range(12)))
            self.assertEqual(v,CurriculumPolicy(v).to_dict())

    def test_rng_repeat_ownership_disabled_exact_and_seeds(self):
        _,_,_,base,severity=policies();np.random.seed(17);torch.manual_seed(17)
        nr=np.random.get_state();tr=torch.get_rng_state().clone()
        p=CurriculumPolicy.freeze(base,order='random',severity=severity,seed=512)
        self.assertTrue(torch.equal(tr,torch.get_rng_state()));np.testing.assert_array_equal(nr[1],np.random.get_state()[1])
        self.assertEqual(p.to_dict(),CurriculumPolicy.freeze(base,order='random',severity=severity,seed=512).to_dict())
        q=CurriculumPolicy.freeze(base,order='random',severity=severity,seed=2048)
        self.assertNotEqual(p.to_dict()['curriculum']['permutation'],q.to_dict()['curriculum']['permutation'])
        self.assertEqual(p.exposure(),q.exposure())
        off=CurriculumPolicy.freeze(base);self.assertEqual(off.to_dict(),base.to_dict());self.assertEqual(off.policy_hash,base.policy_hash)
        mutated=p.to_dict();mutated['training'].clear();self.assertEqual(p.exposure()['updates'],12)

    def test_invalid_controls_nested_zero_and_semantic_tamper_rejected(self):
        _,_,_,base,severity=policies()
        for kwargs in ({'order':'bad','severity':severity},{'order':'fixed','severity':{}},{'order':'fixed','severity':dict.fromkeys(severity,float('nan'))},
                       {'order':'fixed','severity':severity,'seed':True},{'order':'off','severity':severity}):
            with self.assertRaises(ValueError):CurriculumPolicy.freeze(base,**kwargs)
        p=CurriculumPolicy.freeze(base,order='fixed',severity=severity)
        with self.assertRaises(ValueError):CurriculumPolicy.freeze(p,order='fixed',severity=severity)
        for kind in ('records','permutation','validation','budget','selection'):
            bad=p.to_dict()
            if kind=='records':bad['training'][0]=copy.deepcopy(bad['training'][1])
            elif kind=='permutation':bad['curriculum']['permutation'].reverse()
            elif kind=='validation':bad['validation']['full']=copy.deepcopy(bad['training'][1])
            elif kind=='budget':bad['updates']=11
            else:bad['selection_condition']='random_25'
            bad['policy_hash']=digest({k:v for k,v in bad.items() if k!='policy_hash'})
            with self.assertRaises(ValueError):CurriculumPolicy(bad)
        bad=base.to_dict();bad['training_application']='support_zero';bad['policy_hash']=digest({k:v for k,v in bad.items() if k!='policy_hash'})
        with self.assertRaises(ValueError):CurriculumPolicy.freeze(ObservationPolicy(bad),order='fixed',severity=severity)

    def test_progress_exact_prefix_completed_endpoint_and_wrong_policy(self):
        _,_,_,base,s=policies();p=CurriculumPolicy.freeze(base,order='easy_to_hard',severity=s)
        q=CurriculumPolicy.freeze(base,order='hard_to_easy',severity=s)
        for completed in (0,1,4,5,11,12):
            packet=json.loads(json.dumps(p.progress(completed)))
            self.assertEqual(p.resume_step(packet),completed+1 if completed<12 else None)
            self.assertEqual(packet['exposure']['updates'],completed)
            with self.assertRaises(ValueError):q.resume_step(packet)
        for n in (-1,13,True,1.5):
            with self.assertRaises(ValueError):p.progress(n)
        bad=p.progress(5);bad['exposure']['updates']=4;bad['progress_hash']=digest({k:v for k,v in bad.items() if k!='progress_hash'})
        with self.assertRaises(ValueError):p.resume_step(bad)

    def test_role_isolation_one_band_gap_and_mapped_views(self):
        configs=[{'mode':'none'},{'mode':'contiguous_gap','lower':550.,'upper':850.,'unit':'nm'},{'mode':'random_subset','keep_fraction':.01}]
        _,scene,record,base,s=policies(configs=configs);p=CurriculumPolicy.freeze(base,order='hard_to_easy',severity=s)
        tile=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devtrain')
        for step,oldindex in enumerate(p.to_dict()['curriculum']['permutation'],1):
            a=p.view(scene,record,tile,phase='training',step=step);b=base.view(scene,record,tile,phase='training',step=oldindex+1)
            np.testing.assert_array_equal(a.observation.spectral,b.observation.spectral)
            self.assertEqual(a.bands,b.bands)
            np.testing.assert_array_equal(a.observation.global_ids,b.observation.global_ids)
        self.assertEqual(p.training_record(1)['selected_band_count'],1)
        for role in ('devval','test'):
            wrong=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']==role)
            with self.assertRaises(ValueError):p.view(scene,record,wrong,phase='training',step=1)
        for step in (0,13):
            with self.assertRaises(ValueError):p.view(scene,record,tile,phase='training',step=step)
        vt=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devval')
        for condition in p.conditions:
            a=p.view(scene,record,vt,phase='validation',condition=condition);b=base.view(scene,record,vt,phase='validation',condition=condition)
            np.testing.assert_array_equal(a.observation.spectral,b.observation.spectral)

    def test_nontraining_data_poison_does_not_change_training_order_or_view(self):
        _,scene,record,base,s=policies();p=CurriculumPolicy.freeze(base,order='random',severity=s,seed=64)
        changed=copy.deepcopy(scene)
        for row in record['roles']:
            if row['role']!='devtrain':changed.cube[tile_slice(row)]=float('nan')
        tile=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devtrain')
        p.validate(changed,record,12)
        for step in (1,6,12):
            np.testing.assert_array_equal(p.view(scene,record,tile,phase='training',step=step).observation.spectral,
                p.view(changed,record,tile,phase='training',step=step).observation.spectral)

    def test_disabled_real_training_adapter_equivalence(self):
        _,scene,record,base,_=policies(updates=2,configs=[{'mode':'none'}])
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/screen_m6_none.json').read_text())
        cfg['model'].update(latent_channels=8,adapter_embed_dim=8,adapter_checkpoint_chunks=False)
        with tempfile.TemporaryDirectory() as tmp:
            a=adapt(scene,record,cfg,Path(tmp)/'base',steps=2,seed=64,observation_policy=base)
            b=adapt(scene,record,cfg,Path(tmp)/'off',steps=2,seed=64,observation_policy=CurriculumPolicy.freeze(base))
            for key in ('trace','validation','selected_step','final_backbone_hash','final_head_hash','observation_policy_hash'):
                self.assertEqual(a[key],b[key])

    def test_four_arms_12_updates_and_disk_5_plus_7_exact_model_optimizer_selection(self):
        source,scene,record,base,s=policies();initial=state_hash(source)
        tile=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devtrain')
        y=torch.tensor(sparse_labels(record,scene.plan,'train_support')[tile_slice(tile)].reshape(-1));mask=y>0
        for order in ORDERS:
            policy=CurriculumPolicy.freeze(base,order=order,severity=s,seed=512)
            model=copy.deepcopy(source);optimizer=torch.optim.Adam(model.parameters(),lr=.0001);history=[]
            def step(m,opt,n):
                m.train();opt.zero_grad();view=policy.view(scene,record,tile,phase='training',step=n)
                outputs=_forward(m,scene,tile,observation_view=view)
                loss=sum(F.nll_loss(p[mask].clamp_min(1e-10).log(),y[mask]-1) for p in outputs[:3]);loss.backward();opt.step()
                gradients=sum(float(p.grad.abs().sum()) for p in m.backbone.adapter.parameters() if p.grad is not None)
                self.assertGreater(gradients,0.)
                return evaluate_validation(m,scene,record,observation_policy=policy)['nll']
            with tempfile.TemporaryDirectory() as tmp:
                file=Path(tmp)/'model.pt'
                for n in range(1,13):
                    history.append(step(model,optimizer,n))
                    if n==5:
                        save_model_checkpoint(model,file)
                        torch.save({'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'progress':policy.progress(n),'history':history.copy()},Path(tmp)/'resume.pt')
                        (Path(tmp)/'policy.json').write_text(json.dumps(policy.to_dict()))
                restored=load_model_checkpoint(file);restored.backbone.adapter.cache_attention=False
                saved=torch.load(Path(tmp)/'resume.pt',weights_only=True)
                loaded=CurriculumPolicy(json.loads((Path(tmp)/'policy.json').read_text()))
                self.assertEqual(loaded.policy_hash,policy.policy_hash)
                opt2=torch.optim.Adam(restored.parameters(),lr=.0001);opt2.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
                replay=saved['history']
                for n in range(loaded.resume_step(saved['progress']),13):replay.append(step(restored,opt2,n))
                self.assertEqual(history,replay);self.assertEqual(state_hash(model),state_hash(restored));self.assertNotEqual(initial,state_hash(model))
                self.assertEqual(min(range(12),key=history.__getitem__),min(range(12),key=replay.__getitem__))
                for k,slot in optimizer.state_dict()['state'].items():
                    for name,v in slot.items():self.assertTrue(torch.equal(v,opt2.state_dict()['state'][k][name]))
                self.assertEqual(policy.exposure()['by_config'],dict.fromkeys(s,4))
                self.assertEqual(int(next(iter(optimizer.state_dict()['state'].values()))['step']),12)


if __name__=='__main__':unittest.main()
