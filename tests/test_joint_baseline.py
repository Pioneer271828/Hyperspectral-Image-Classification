"""BASE-97 identity/access checks and a real synthetic three-task training loop."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from joint_baseline import (NAMES, _item_fingerprint, graph_random_state, load_config, ratio_split,
                            readout, short_loop, validate_split, validation_score, verify_prepared)
from loadData.dataset_registry import file_hash
from loadData.split_data import split_data
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from research_runtime import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/experiments/base97_joint.json'
torch.set_num_threads(1)


def tiny_items():
    rng = torch.Generator().manual_seed(123)
    items = {}
    for name, bands in zip(NAMES, (7,11,13)):
        item = {'name': name, 'seed': 64, 'height': 6, 'width': 6, 'bands': bands,
                'class_num': 9, 'superpixel_scale': 6,
                'Q': torch.nn.functional.one_hot(torch.arange(36)%6,6).float().to_sparse().coalesce(),
                'A': torch.eye(6)+torch.ones(6,6)/6,
                'net_input': torch.rand(6,6,bands+2,generator=rng),
                'wavelengths': torch.linspace(450,950,bands), 'fwhm': None}
        for phase, offset in (('train',0),('val',9)):
            labels = torch.zeros(36)
            labels[offset:offset+9] = torch.arange(1,10)
            onehot = torch.zeros(36,9)
            onehot[torch.arange(offset,offset+9),torch.arange(9)] = 1
            item[phase+'_samples_gt'] = labels
            item[phase+'_gt_onehot'] = onehot
            item[phase+'_label_mask'] = (labels>0).float()[:,None].expand(36,9).clone()
        items[name] = item
    return items


class JointBaselineTests(unittest.TestCase):
    def test_graph_walk_python_rng_is_repeatable_and_restored_on_failure(self):
        random.seed(901)
        initial = random.getstate()
        values = []
        for _ in range(2):
            with graph_random_state(64):
                values.append(random.choices(range(100),k=30))
            self.assertEqual(random.getstate(),initial)
        self.assertEqual(values[0],values[1])
        with self.assertRaises(RuntimeError):
            with graph_random_state(65):
                random.random()
                raise RuntimeError('preparation failed')
        self.assertEqual(random.getstate(),initial)

    def test_legacy_ratio_permutation_and_rng_state(self):
        labels = np.array([0]*3+[1]*947+[2]*2436).reshape(2,-1)
        for seed in (0,64,2026):
            np.random.seed(902)
            before = np.random.get_state()
            manifest = ratio_split(labels,seed)
            after = np.random.get_state()
            self.assertTrue(all(np.array_equal(a,b) for a,b in zip(before,after)))
            np.random.seed(seed)
            train,val,test = split_data(labels.ravel(),2,.01,.005,30,10,'ratio')
            for phase, values in zip(('train','val','test'),(train,val,test)):
                self.assertEqual(manifest['indices'][phase],values.tolist())
            self.assertEqual(manifest['counts'],{'1':[10,5,932],'2':[25,13,2398]})
            validate_split(manifest,labels)

    def test_reject_split_tampering_and_invalid_population(self):
        labels = np.tile(np.arange(1,4),(4,1))
        manifest = ratio_split(labels,64)
        for field, value in (('seed',65),('protocol','another'),('train_ratio',.1)):
            altered = copy.deepcopy(manifest); altered[field] = value
            with self.assertRaises(ValueError): validate_split(altered,labels)
        altered = copy.deepcopy(manifest)
        altered['indices']['val'][0] = altered['indices']['train'][0]
        with self.assertRaises(ValueError): validate_split(altered,labels)
        for invalid in (np.array([[1,1]]),np.array([[2,2,2]]),np.array([[0,0]]),
                        np.array([[1.,float('nan')]]),np.array([[1.,-1.]])):
            with self.assertRaises(ValueError): ratio_split(invalid,64)

    def test_frozen_config_rejects_budget_architecture_or_label_changes(self):
        raw = json.loads(CONFIG.read_text())
        self.assertEqual(load_config(CONFIG).budget['optimizer_steps'],1503)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            for group,key,value in [('data','train_ratio',.1),('data','datasets',list(reversed(NAMES))),
                                    ('training','lr',.001),('training','seed',512),
                                    ('model','region_residual_mode','x_residual'),
                                    ('loss','cnn_loss_weight',0),('augmentation','train_drop_band_ratio',.3)]:
                altered = copy.deepcopy(raw); altered.setdefault(group,{})[key] = value
                path.write_text(json.dumps(altered))
                with self.assertRaises(ValueError): load_config(path)

    def test_readout_hand_computed_and_test_access_rejected(self):
        p = torch.tensor([[.8,.2],[.4,.6],[.1,.9],[.9,.1]])
        item = {'class_num':2,'train_samples_gt':torch.tensor([1,1,2,0])}
        metrics, values = readout((p,p,p),item,'train')
        self.assertEqual(metrics['fused']['confusion'],[[1,1],[0,1]])
        self.assertAlmostEqual(metrics['fused']['oa'],2/3)
        self.assertAlmostEqual(metrics['fused']['aa'],.75)
        self.assertEqual(len(values['cnn']),3)
        with self.assertRaises(ValueError): readout((p,p,p),item,'test')

    def test_selection_uses_only_validation_macro_oa(self):
        metrics = {n:{'val':{'fused':{'oa':v}},'train':{'fused':{'oa':1}},
                      'test':{'fused':{'oa':float('nan')}}} for n,v in zip(NAMES,(.1,.2,.6))}
        self.assertAlmostEqual(validation_score(metrics),.3)

    def test_verify_rejects_manifest_data_and_prepared_tampering(self):
        config = load_config(CONFIG)
        labels = np.tile(np.arange(1,10),(4,1))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); records = {}
            for offset,(name,item) in enumerate(tiny_items().items()):
                write_json(root/(name+'-split.json'),ratio_split(labels,64+offset))
                torch.save(item,root/(name+'-prepared.pt'))
                records[name] = {'raw':{'name':name},'split_file':name+'-split.json',
                                 'prepared_file':name+'-prepared.pt',
                                 'split_sha256':file_hash(root/(name+'-split.json')),
                                 'prepared_sha256':file_hash(root/(name+'-prepared.pt')),
                                 'prepared_fields':_item_fingerprint(item)}
            manifest = {'schema_version':1,'protocol':'acc1p_joint_prepared_v1','test_scoring':False,
                        'config':config.to_dict(),'config_hash':config.config_hash,
                        'code':{'source_hash':'fixture'},'datasets':records}
            write_json(root/'manifest.json',manifest)
            h = file_hash(root/'manifest.json')
            inspector = lambda root,name: ({'name':name},labels)
            with patch('joint_baseline.inspect_dataset',side_effect=inspector):
                items,_ = verify_prepared(root,h,config,root,{'source_hash':'fixture'})
                self.assertEqual(set(items),set(NAMES))
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    verify_prepared(root,'0'*64,config,root,{'source_hash':'fixture'})
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    verify_prepared(root,h,config,root,{'source_hash':'changed'})
                with (root/'paviaU-prepared.pt').open('ab') as stream: stream.write(b'tamper')
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    verify_prepared(root,h,config,root,{'source_hash':'fixture'})
            with patch('joint_baseline.inspect_dataset',return_value=({'name':'changed'},labels)):
                with self.assertRaisesRegex(ValueError,'Raw dataset identity'):
                    verify_prepared(root,h,config,root,{'source_hash':'fixture'})

    def test_three_task_update_and_selected_classifier_roundtrip(self):
        config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            result = short_loop(config,tiny_items(),Path(temp)/'new',device=os.environ.get('BASE97_DEVICE','cpu'))
            self.assertEqual(result['runtime']['actual_optimizer_steps'],6)
            self.assertEqual(result['status'],'passed')
            best = max(result['history'],key=lambda x:x['validation_macro_oa'])
            self.assertEqual(best['cycle'],result['selected_cycle'])
            trace = json.loads((Path(temp)/'new/trace.json').read_text())
            self.assertEqual([x['dataset'] for x in trace],list(NAMES)*2)
            self.assertTrue(all(x['changed_elements']['shared_adapter']>0 for x in trace))
            sidecar = json.loads((Path(temp)/'new/primary.pt.json').read_text())
            self.assertEqual(sidecar['training_config']['short_loop']['actual_optimizer_steps'],6)
            with self.assertRaises(FileExistsError):
                short_loop(config,tiny_items(),Path(temp)/'new',device='cpu')

    def test_short_loop_rejects_missing_task_or_test_supervision(self):
        config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): short_loop(config,{},Path(temp)/'missing',device='cpu')
            items = tiny_items(); items['paviaU']['test_samples_gt'] = torch.zeros(36)
            with self.assertRaisesRegex(ValueError,'Test supervision'):
                short_loop(config,items,Path(temp)/'forbidden',device='cpu')

    def test_chunked_projection_preserves_three_task_outputs_and_gradients(self):
        # Memory chunking is execution-only; use FP64 to assess numerical equivalence.
        torch.manual_seed(64)
        kwargs = dict(latent_channels=8,adapter_embed_dim=8,adapter_attention_mode='index',norm_type='domain_bn')
        plain = MultiDatasetVCASAHGCN(dict.fromkeys(NAMES,9),**kwargs).double()
        chunks = MultiDatasetVCASAHGCN(dict.fromkeys(NAMES,9),adapter_chunk_pixels=5,**kwargs).double()
        chunks.load_state_dict(plain.state_dict())
        for name,item in tiny_items().items():
            outputs = []
            for model in (plain,chunks):
                model.zero_grad(set_to_none=True)
                output = model(name,item['net_input'].double(),item['Q'].double(),item['A'].double().to_sparse(),
                               wavelengths=item['wavelengths'].double())
                sum(-x[:,0].clamp_min(1e-8).log().mean() for x in output).backward()
                outputs.append(output)
            for a,b in zip(*outputs): torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
            for (key,a),(_,b) in zip(plain.named_parameters(),chunks.named_parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad,b.grad,rtol=1e-7,atol=1e-9,msg=key)


if __name__ == '__main__':
    unittest.main()
