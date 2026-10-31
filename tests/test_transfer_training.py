"""Independent G3 stage correctness, frozen backbone and target label access."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.io import savemat
from research_runtime import hash_file
from train_cross_sensor import campaign,synthetic_task
from transfer.g3_training import build_model,build_target,state_hash,head_hash,adapt_target,evaluate_target,METHODS

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    config=json.loads((ROOT/'configs/protocols/paviau_to_xuzhou_g3.json').read_text(encoding='utf-8-sig'))
    base=json.loads((ROOT/'configs/experiments/screen_m6_none.json').read_text())
    base['model'].update(latent_channels=8,adapter_embed_dim=8,adapter_checkpoint_chunks=False)
    return config,base,synthetic_task(config)


class TransferTrainingTests(unittest.TestCase):
    def device(self):return os.environ.get('G3_DEVICE','cpu')

    def test_only_backbone_transfers_new_three_heads_are_shared(self):
        config,base,(scene,labels,target,_,support)=fixture()
        source,_=build_model(base,scene.dataset,9,config['model_seed'],self.device())
        with torch.no_grad():
            for name in ('heads','hgnn_heads','cnn_heads'):
                getattr(source,name)[scene.dataset].weight.fill_(99.)
            next(source.backbone.parameters()).add_(.25)
        models=[]
        for method in METHODS:
            model,_=build_target(base,target,method,source,model_seed=config['model_seed'],head_seed=config['target_head_seed'],device=self.device())
            models.append(model)
            for name in ('heads','hgnn_heads','cnn_heads'):
                self.assertEqual(list(getattr(model,name)),['xuzhou'])
                self.assertFalse(torch.all(getattr(model,name)['xuzhou'].weight==99))
        self.assertEqual(len({head_hash(m) for m in models}),1)
        self.assertEqual(state_hash(models[1].backbone),state_hash(source.backbone))
        self.assertEqual(state_hash(models[2].backbone),state_hash(source.backbone))
        self.assertNotEqual(state_hash(models[0].backbone),state_hash(source.backbone))
        self.assertTrue(all(not p.requires_grad for p in models[1].backbone.parameters()))
        self.assertFalse(models[1].backbone.training)

    def test_stage_order_source_selection_and_test_only_after_all_frozen(self):
        config,base,(source,source_labels,target,target_labels,support)=fixture()
        config.update(source_updates=2,target_updates=2)
        task={'config':config,'task_hash':'fixture-task','source_identity':{'name':'paviaU'},
              'target_identity':{'name':'xuzhou','label_path':'target_gt.mat'}}
        from transfer.g3_training import evaluate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';out.mkdir()
            savemat(root/'target_gt.mat',{'xuzhou_gt':target_labels})
            task['target_identity']['label_sha256']=hash_file(root/'target_gt.mat')
            loader_order=[];evaluations=[]
            def loader(data_root,identity,plan,*,with_labels):
                loader_order.append((identity['name'],with_labels))
                if identity['name']=='paviaU':return source,source_labels
                self.assertTrue((out/'source/model.pt').exists())
                return target,None
            def guarded_evaluate(*args,**kwargs):
                partition,dataset=args[5],args[6];evaluations.append((dataset,partition))
                if dataset=='paviaU':self.assertEqual(partition,'val')
                else:
                    self.assertEqual(partition,'test')
                    self.assertTrue((out/'target-checkpoints-frozen.json').exists())
                    self.assertTrue(all((out/m/'model.pt').exists() for m in METHODS))
                return evaluate(*args,**kwargs)
            with patch('train_cross_sensor.load_scene',side_effect=loader),patch('transfer.g3_training.evaluate',side_effect=guarded_evaluate):
                result=campaign(task,{'source':source.plan,'target':target.plan},support,base,out,
                    device=self.device(),mode='run',data_root=root)
            self.assertEqual(loader_order,[('paviaU',True),('xuzhou',False)])
            self.assertEqual(evaluations,[('paviaU','val')]+[('xuzhou','test')]*3)
            self.assertEqual(result['actual_source_updates'],2)
            self.assertEqual(result['actual_target_updates_per_method'],2)
            self.assertTrue(result['targets']['source_pretrained_head_only']['backbone_unchanged'])
            self.assertTrue(all(r['initial_head_hash']!=r['final_head_hash'] for r in result['targets'].values()))
            self.assertEqual(len({r['initial_head_hash'] for r in result['targets'].values()}),1)
            self.assertTrue(all(r['target_support_labels']==90 and r['target_validation_labels']==0 for r in result['targets'].values()))

    def test_adaptation_receives_only_released_support_and_rejects_final_tamper(self):
        config,base,(_,_,target,target_labels,support)=fixture()
        from transfer.g3_training import _update
        seen=[]
        def inspect_update(model,scene,tile,labels,*args,**kwargs):
            seen.append(tile['id']);self.assertEqual(tile['partition'],'train')
            y0,y1,x0,x1=tile['box'];ids=np.flatnonzero(labels.reshape(-1)>0)
            released_global=[(y0+i//(x1-x0))*scene.plan['shape'][1]+x0+i%(x1-x0) for i in ids]
            self.assertTrue(set(released_global)<=set(support['indices']))
            return _update(model,scene,tile,labels,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory,patch('transfer.g3_training._update',side_effect=inspect_update):
            folder=Path(directory)/'target'
            result=adapt_target(target,support,base,folder,method=METHODS[0],source=None,steps=2,
                model_seed=config['model_seed'],head_seed=config['target_head_seed'],sampler_seed=config['target_sampler_seed'],
                device=self.device(),task_hash='task')
            self.assertEqual(len(seen),2)
            path=folder/'model.pt';before=hash_file(path)
            # Final evaluation must leave the frozen checkpoint unchanged.
            metrics=evaluate_target(target,support,target_labels,path,expected_checkpoint_hash=result['checkpoint_sha256'],
                expected_sidecar_hash=result['sidecar_sha256'],task_hash='task',method=METHODS[0],device=self.device())
            self.assertGreater(metrics['count'],0);self.assertEqual(before,hash_file(path))
            for field in ('checkpoint','sidecar','task'):
                with self.assertRaises(ValueError):
                    evaluate_target(target,support,target_labels,path,
                        expected_checkpoint_hash='0'*64 if field=='checkpoint' else result['checkpoint_sha256'],
                        expected_sidecar_hash='0'*64 if field=='sidecar' else result['sidecar_sha256'],
                        task_hash='wrong' if field=='task' else 'task',method=METHODS[0],device=self.device())
            with self.assertRaises(FileExistsError):
                adapt_target(target,support,base,folder,method=METHODS[0],source=None,steps=2,
                    model_seed=1,head_seed=2,sampler_seed=3,device=self.device(),task_hash='task')

if __name__=='__main__':unittest.main()


