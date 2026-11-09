"""Fixed per-class validation contracts reach light training and diagnostics."""
import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.utils import compute_loss
from model.research_config import digest
from loadData.spatial_tiles import tile_slice
from transfer.development_training import sparse_labels
from transfer.g3_training import _forward,state_hash
from transfer.light_head_training import LightHeadRun
from transfer.m4_ablation import validation_readouts
from inference.trained_light import export_trained_light,load_trained_light
from evaluation.graph_edge_intervention import EdgeDeletionPlan,review_deletions
from transfer.graph_route_context import route_state_hash
from tests.test_multiclass_development import fixture as routed_fixture,make

BASE='ffb88163a3952cd04760471e8843904f0c97aedc'
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def fixture(classes):
    _,scene,record,_,_=routed_fixture(classes,512)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(512)
        source=MultiDatasetVCASAHGCN({scene.dataset:classes},latent_channels=8,adapter_embed_dim=8,
            locality_window=None,norm_type='group_norm',graph_mode='learned').eval()
    source.backbone.adapter.cache_attention=False
    return source,scene,record


def light_run(a,seed=64):
    return LightHeadRun(*a,updates=12,seed=seed,lr=.0001,class_balance=True)


def manual_metrics(model,scene,record):
    labels=sparse_labels(record,scene.plan,'validation_support');c=scene.classes
    cm=np.zeros((3,c,c),dtype=np.int64);nll=np.zeros(3);both=np.zeros(c,dtype=int)
    with torch.no_grad():
        for tile in scene.plan['tiles']:
            if record['roles'][tile['id']]['role']!='devval':continue
            target=labels[tile_slice(tile)].ravel();mask=target>0;y=target[mask]-1
            pred=[]
            for hi,p in enumerate(_forward(model,scene,tile)[:3]):
                q=p.numpy()[mask];k=q.argmax(1);pred.append(k)
                for t,v in zip(y,k):cm[hi,t,v]+=1
                nll[hi]-=np.log(np.maximum(q[np.arange(len(y)),y],1e-8)).sum()
            for j,t in enumerate(y):both[t]+=int(pred[1][j]!=t and pred[2][j]!=t)
    return cm,nll/(5*c),both/5


class MulticlassConsumerTests(unittest.TestCase):
    def test_light_equal_budget_full_cnn_recovery_selection_and_publish(self):
        for classes in (3,9,22):
            for seed in (64,512,2048):
                with self.subTest(classes=classes,sampler_seed=seed),tempfile.TemporaryDirectory() as tmp:
                    a=fixture(classes);source,scene,record=a;run=light_run(a,seed);before=state_hash(source)
                    full=copy.deepcopy(source).requires_grad_(False);full.cnn_heads[scene.dataset].requires_grad_(True)
                    opt=torch.optim.Adam(full.cnn_heads[scene.dataset].parameters(),lr=.0001);rng=np.random.default_rng(seed)
                    for step in range(1,13):
                        tile,anchor=run.sampler.draw(rng)
                        y=torch.tensor(sparse_labels(record,scene.plan,'train_support')[tile_slice(tile)].ravel())
                        mask=(y>0)[:,None].expand(-1,classes).float();onehot=torch.nn.functional.one_hot((y-1).clamp_min(0),classes).float()*mask
                        opt.zero_grad();loss=compute_loss(_forward(full,scene,tile)[2],onehot,mask,class_balance=True);loss.backward();opt.step()
                        row=run.step();self.assertEqual(row['loss'],float(loss.detach()))
                        self.assertEqual(state_hash(run.head),state_hash(full.cnn_heads[scene.dataset]))
                        self.assertEqual(row['validation']['support'],[5]*classes)
                        if step==5:partial=Path(tmp)/'partial.pt';pin=run.save(partial)
                    resumed=LightHeadRun.load(*a,partial,expected_sha256=pin,expected_contract_hash=run.contract_hash);resumed.advance(12)
                    self.assertEqual(run.history,resumed.history);self.assertEqual(state_hash(run.head),state_hash(resumed.head))
                    for k,v in run.optimizer.state_dict()['state'].items():
                        for name,value in v.items():self.assertTrue(torch.equal(value,resumed.optimizer.state_dict()['state'][k][name]))
                    self.assertEqual(run.selected_step,min(range(12),key=lambda i:run.history[i]['validation']['nll'])+1)
                    self.assertEqual(before,state_hash(source));self.assertNotEqual(run.spec['initial_head_state'],state_hash(run.head))
                    done=Path(tmp)/'done.pt';pin=run.save(done);out=Path(tmp)/'published.pt'
                    with patch('torch.optim.Adam.step',side_effect=AssertionError('completed/publish update')):
                        zero=LightHeadRun.load(*a,done,expected_sha256=pin,expected_contract_hash=run.contract_hash);zero.advance(12)
                        published=export_trained_light(*a,done,out,expected_training_sha256=pin,expected_contract_hash=run.contract_hash)
                        result=load_trained_light(source,out,expected_sha256=published,expected_training_sha256=pin,expected_contract_hash=run.contract_hash)
                    self.assertEqual(result.specification['selected_validation'],run.history[run.selected_step-1]['validation'])
                    for key,val in run.selected_head.items():self.assertTrue(torch.equal(val,result.model.heads[scene.dataset].state_dict()[key]))
                    self.assertTrue(all(p.grad is None for p in source.parameters()))

    def test_m4_three_head_readouts_use_every_class_and_no_updates(self):
        for classes in (3,9,22):
            with self.subTest(classes=classes):
                model,scene,record=fixture(classes);before=state_hash(model);rng=torch.get_rng_state().clone()
                with patch('torch.optim.Adam.step',side_effect=AssertionError('readout update')):
                    result=validation_readouts(model,scene,record);cm,nll,both=manual_metrics(model,scene,record)
                for i,name in enumerate(('fused','graph','cnn')):
                    self.assertEqual(result[name]['confusion_matrix'],cm[i].tolist())
                    self.assertEqual(result[name]['nll'],nll[i])
                    self.assertEqual([r['support'] for r in result[name]['per_class']],[5]*classes)
                self.assertEqual(result['both_branches_wrong_by_class'],both.tolist())
                self.assertEqual(before,state_hash(model));self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_graph_interventions_dynamic_fixed_and_multiclass_baseline(self):
        for classes in (3,9,22):
            for fixed in (False,True):
                with self.subTest(classes=classes,fixed=fixed):
                    run=make(routed_fixture(classes,512,fixed));run.advance(24)
                    before=route_state_hash(run.model);history=copy.deepcopy(run.history)
                    with patch('torch.optim.Adam.step',side_effect=AssertionError('intervention update')):
                        plan=EdgeDeletionPlan.freeze(run,per_row=1,random_seeds=[64,512,2048])
                        result=review_deletions(run,plan,expected_plan_hash=plan.plan_hash)
                    for row in result['controls'].values():
                        self.assertEqual(row['support'],[[5]*classes]*3)
                        for i,cm in enumerate(np.asarray(row['confusion'])):
                            self.assertEqual(row['oa'][i],np.trace(cm)/(5*classes))
                    self.assertEqual(result['controls']['none']['confusion'][0],run.history[run.selected_step-1]['validation']['confusion'])
                    self.assertEqual(before,route_state_hash(run.model));self.assertEqual(history,run.history)

    def test_original_nine_class_light_and_m4_exact(self):
        a=fixture(9);current=light_run(a);old=types.ModuleType('old_light')
        exec(subprocess.check_output(['git','show',BASE+':transfer/light_head_training.py'],cwd=ROOT,text=True),old.__dict__)
        previous=old.LightHeadRun(*a,updates=12,seed=64,lr=.0001,class_balance=True)
        self.assertEqual(current.spec,previous.spec)
        for _ in range(12):self.assertEqual(current.step(),previous.step())
        self.assertEqual(state_hash(current.head),state_hash(previous.head))
        old_m4=types.ModuleType('old_m4')
        exec(subprocess.check_output(['git','show',BASE+':transfer/m4_ablation.py'],cwd=ROOT,text=True),old_m4.__dict__)
        self.assertEqual(validation_readouts(*a),old_m4.validation_readouts(*a))

    def test_bad_counts_history_and_spatial_isolation(self):
        a=fixture(22);source,scene,record=a;changed=copy.deepcopy(scene)
        for row in record['roles']:
            if row['role'] not in ('devtrain','devval'):changed.cube[tile_slice(row)]=np.nan
        left,right=light_run(a),light_run((source,changed,record))
        self.assertEqual(left.step(),right.step());self.assertEqual(validation_readouts(*a),validation_readouts(source,changed,record))
        bad=copy.deepcopy(record);bad['validation_labels']=45;bad['validation_hash']=digest({k:v for k,v in bad.items() if k!='validation_hash'})
        with self.assertRaises(ValueError):light_run((source,scene,bad))
        with self.assertRaises(ValueError):validation_readouts(source,scene,bad)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'good.pt';left.save(path);packet=torch.load(path,weights_only=True)
            packet['history'][0]['validation']['count']=45;badfile=Path(tmp)/'bad.pt';torch.save(packet,badfile)
            with self.assertRaises(ValueError):LightHeadRun.load(*a,badfile,expected_sha256=hashlib.sha256(badfile.read_bytes()).hexdigest(),expected_contract_hash=left.contract_hash)


if __name__=='__main__':unittest.main()
