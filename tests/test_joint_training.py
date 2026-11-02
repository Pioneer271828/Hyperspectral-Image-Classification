"""Legacy schedule/selection arithmetic, exact CPU continuation and safe resume."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from joint_baseline import NAMES,load_config
from joint_training import (PREPARATION_COMMIT,_tree_fingerprint,audit_preparation_source,
                            load_plan,options,run_training,schedule_for,train_step,
                            validation_epochs,validation_snapshot)
from model.research_config import ResearchConfig,digest
from research_runtime import tensor_hash
from test_joint_baseline import tiny_items

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'configs/experiments/base97_training.json'
BASE=ROOT/'configs/experiments/base97_joint.json'
SOURCE={'commit':'fixture','source_hash':'fixture','files':{}}
PROVENANCE={'prepared_manifest_sha256':'fixture','preparation_code':{'commit':'fixture'}}
torch.set_num_threads(1)


def settings(seed=64):
    return options(load_plan(PLAN),'engineering',seed,load_config(BASE))


def checkpoints(path):
    info=json.loads((path/'progress.json').read_text())
    return torch.load(path/info['state_file'],map_location='cpu',weights_only=True)


def hashes(path):
    from loadData.dataset_registry import file_hash
    return {p.name:file_hash(p) for p in path.iterdir() if p.is_file()}


class JointTrainingTests(unittest.TestCase):
    def test_legacy_random_schedule_and_validation_budget(self):
        base=load_config(BASE); plan=load_plan(PLAN)
        for seed in (64,512,2048):
            opts=options(plan,'full',seed,base); actual=schedule_for(opts)
            rng=np.random.default_rng(seed)
            expected=[]
            for _ in range(501): expected.extend(rng.choice(list(NAMES),size=3,replace=True).tolist())
            self.assertEqual(actual,expected)
            self.assertEqual(len(actual),1503)
            self.assertEqual(validation_epochs(opts),list(range(0,501,10)))
            self.assertEqual(set(Counter(actual)),set(NAMES))
            self.assertEqual(schedule_for(settings(seed)),actual[:12])
        self.assertEqual(validation_epochs(settings()),[0,1,2,3])

    def test_reject_modified_protocol_and_unregistered_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'plan.json'; raw=json.loads(PLAN.read_text())
            for group,key,value in [('profiles','full',{'epochs':999,'val_interval':10}),
                                    ('scheduler','patience',0)]:
                bad=copy.deepcopy(raw); bad[group][key]=value; p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load_plan(p)
        with self.assertRaises(ValueError): options(load_plan(PLAN),'full',2026,load_config(BASE))

    def test_preparation_producer_audit_rejects_changed_or_missing_files(self):
        files={'producer.py':'hash','config.json':'config-hash'}
        producer={'commit':PREPARATION_COMMIT,'files':files,'source_hash':digest(files)}
        consumer={'files':{**files,'new_runner.py':'new'}}
        audit_preparation_source(producer,consumer)
        for changed in ({'files':{'producer.py':'modified'}},{'files':{}},
                        {'files':{**files,'producer.py':'different'}}):
            with self.assertRaises(ValueError): audit_preparation_source(producer,changed)
        with self.assertRaises(ValueError): audit_preparation_source({**producer,'commit':'unknown'},consumer)

    def test_validation_matches_legacy_fused_loss_and_oa_arithmetic(self):
        from train_vca_multidataset import evaluate_split
        opts=settings(); config=ResearchConfig.from_dict(opts['config'])
        items=tiny_items()
        for item in items.values(): item['zeros']=torch.zeros(item['net_input'].shape[0]*item['net_input'].shape[1])
        torch.manual_seed(64); model=config.build_model({n:i['class_num'] for n,i in items.items()})
        expected_loss,expected_oa,_=evaluate_split(model,items,'val')
        actual,_=validation_snapshot(model,items)
        self.assertEqual(actual['val_loss'],float(expected_loss))
        self.assertEqual(actual['legacy_val_oa'],expected_oa)
        self.assertEqual(actual['macro_oa'],float(np.mean(list(expected_oa.values()))))

    def test_mid_epoch_and_boundary_resume_match_uninterrupted_cpu(self):
        opts=settings(); items=tiny_items()
        for item in items.values(): item['zeros']=torch.zeros(item['height']*item['width'])
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'reference'
            expected=run_training(opts,items,base,PROVENANCE,SOURCE,device='cpu')
            reference=checkpoints(base)
            for cursor in (2,3,5):
                path=Path(temp)/('split'+str(cursor))
                first=run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu',stop_after=cursor)
                self.assertEqual(first['operation'],'interrupted')
                self.assertFalse((path/'result.json').exists())
                result=run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu',resume=True)
                state=checkpoints(path)
                for key in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities'):
                    self.assertEqual(_tree_fingerprint(state[key]),_tree_fingerprint(reference[key]),key)
                self.assertEqual(result['result']['selected_step'],expected['result']['selected_step'])
                self.assertEqual(result['updates_this_call'],12-cursor)
                self.assertIsNotNone(result['restored_state_fingerprints'])

    def test_completed_resume_is_readonly_and_rejects_artifact_tamper(self):
        opts=settings(); items=tiny_items()
        for item in items.values(): item['zeros']=torch.zeros(item['height']*item['width'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run'
            run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu')
            before=hashes(path)
            with patch('model.research_config.ResearchConfig.build_model',side_effect=AssertionError('must skip')):
                result=run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu',resume=True)
            self.assertEqual(result['operation'],'verified_completed')
            self.assertEqual(result['result']['status'],'completed')
            self.assertEqual(result['updates_this_call'],0)
            self.assertEqual(before,hashes(path))
            with (path/'primary.pt').open('ab') as stream: stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu',resume=True)

    def test_resume_identity_lock_and_state_corruption_do_not_overwrite(self):
        opts=settings(); items=tiny_items()
        for item in items.values(): item['zeros']=torch.zeros(item['height']*item['width'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run'
            run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu',stop_after=2)
            before=hashes(path)
            with self.assertRaises(ValueError):
                run_training(settings(512),items,path,PROVENANCE,SOURCE,device='cpu',resume=True)
            with self.assertRaises(FileExistsError):
                run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu')
            self.assertEqual(before,hashes(path))
            (path/'.lock').write_text('another owner')
            with self.assertRaises(FileExistsError):
                run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu',resume=True)
            (path/'.lock').unlink()
            p=json.loads((path/'progress.json').read_text()); cp=path/p['state_file']
            with cp.open('ab') as stream: stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                run_training(opts,items,path,PROVENANCE,SOURCE,device='cpu',resume=True)

    def test_formal_step_matches_original_supervised_adam_update(self):
        from joint_baseline import forward
        from train_vca_multidataset import supervised_loss
        opts=settings(); config=ResearchConfig.from_dict(opts['config']); items=tiny_items()
        torch.manual_seed(64); model=config.build_model({n:i['class_num'] for n,i in items.items()})
        reference=copy.deepcopy(model)
        def optimizer(m):
            heads=list(m.heads.parameters())+list(m.hgnn_heads.parameters())+list(m.cnn_heads.parameters())
            return torch.optim.Adam([{'params':m.backbone.parameters(),'lr':5e-5},
                                     {'params':heads,'lr':5e-4}],weight_decay=.001)
        a,b=optimizer(model),optimizer(reference)
        name=NAMES[0]; item=items[name]
        torch.manual_seed(700); state=torch.get_rng_state()
        train_step(model,a,name,item,config)
        torch.set_rng_state(state)
        reference.train(); b.zero_grad(); out=forward(reference,name,item)
        supervised_loss(out,item,'train',class_balance=True).backward(); b.step()
        self.assertEqual(_tree_fingerprint(model.state_dict()),_tree_fingerprint(reference.state_dict()))
        self.assertEqual(_tree_fingerprint(a.state_dict()),_tree_fingerprint(b.state_dict()))

    def test_gpu_interrupt_resume_and_checkpoint_runtime(self):
        device=os.environ.get('BASE97B_DEVICE','cpu')
        opts=settings(); items=tiny_items()
        for item in items.values(): item['zeros']=torch.zeros(item['height']*item['width'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run'
            first=run_training(opts,items,path,PROVENANCE,SOURCE,device=device,stop_after=5)
            self.assertEqual(first['cursor'],5)
            result=run_training(opts,items,path,PROVENANCE,SOURCE,device=device,resume=True)
            self.assertEqual(result['result']['optimizer_steps'],12)
            self.assertEqual(result['updates_this_call'],7)
            meta=json.loads((path/'primary.pt.json').read_text())['training_config']
            self.assertEqual(meta['joint_runtime']['executed_optimizer_steps'],12)
            self.assertEqual(meta['checkpoint_optimizer_step'],result['result']['selected_step'])


if __name__=='__main__': unittest.main()
