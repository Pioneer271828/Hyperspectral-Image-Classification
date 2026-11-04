import copy,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
import source_data_context as m
from joint_baseline import NAMES,_item_fingerprint
from joint_training import PREPARATION_COMMIT
from model.research_config import ResearchConfig,digest
from research_runtime import write_json
from loadData.dataset_registry import file_hash
from test_joint_training import settings
from test_calibrated_selection import items


class SourceContextTests(unittest.TestCase):
    def fixture(self,temp):
        root=Path(temp);prepared=root/'prepared';data=root/'data';prepared.mkdir();data.mkdir()
        cfg=ResearchConfig.from_dict(settings()['config']);cfg=replace(cfg,data=replace(cfg.data,datasets=NAMES))
        producer={'commit':PREPARATION_COMMIT,'files':{'fixture.py':'known'}};producer['source_hash']=digest(producer['files'])
        consumer={'files':{**producer['files'],'consumer.py':'extra'}};records={}
        for index,(name,item) in enumerate(items().items()):
            split=prepared/(name+'-split.json');write_json(split,{'seed':cfg.training.seed+NAMES.index(name)})
            packed=prepared/(name+'-prepared.pt');torch.save(item,packed)
            image=data/(name+'.mat');label=data/(name+'_gt.mat');image.write_bytes(b'image');label.write_bytes(b'label')
            records[name]={'split_file':split.name,'split_sha256':file_hash(split),'prepared_file':packed.name,
                'prepared_sha256':file_hash(packed),'prepared_fields':_item_fingerprint(item),
                'raw':{'image_path':image.name,'label_path':label.name,'image_sha256':file_hash(image),
                       'label_sha256':file_hash(label),'class_count':item['class_num']},
                'normalization':{'scope':'full_scene_global_minmax'}}
        manifest={'schema_version':1,'protocol':'acc1p_joint_prepared_v1','config':cfg.to_dict(),'config_hash':cfg.config_hash,
                  'code':producer,'datasets':records,'context':'full_scene_transductive_train_label_lda_slic','test_scoring':False}
        write_json(prepared/'manifest.json',manifest)
        return prepared,data,manifest,consumer

    def load(self,fixture,held='xuzhou',sources=None):
        prepared,data,_,consumer=fixture
        return m.load_source_context(prepared,data,manifest_sha256=file_hash(prepared/'manifest.json'),held_out=held,consumer=consumer,sources=sources)

    def test_never_opens_missing_or_poisoned_target_payload_and_model_excludes_target(self):
        with tempfile.TemporaryDirectory() as temp:
            f=self.fixture(temp);p,d,manifest,_=f;held='xuzhou';entry=manifest['datasets'][held]
            forbidden={p/entry['split_file'],p/entry['prepared_file'],d/entry['raw']['image_path'],d/entry['raw']['label_path']}
            original=self.load(f)
            for path in forbidden:path.unlink()
            real_hash=file_hash
            def guarded(path):
                self.assertNotIn(Path(path),forbidden);return real_hash(path)
            with patch.object(m,'file_hash',side_effect=guarded):context=self.load(f)
            self.assertEqual(context.record['source_identity'],original.record['source_identity'])
            self.assertEqual(tuple(context.items),('paviaU','whu-honghu'))
            self.assertEqual(tuple(context.config.data.datasets),context.sources)
            model=context.build_model();self.assertEqual(set(model.heads),set(context.sources))
            self.assertFalse(any(held in n.split('.') for n in model.state_dict()))
            self.assertFalse(context.record['target_payload_accessed']);self.assertTrue(context.record['cohort_target_metadata_visible'])

    def test_all_rotations_single_source_and_exact_canonical_source_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            f=self.fixture(temp)
            for held in NAMES:
                context=self.load(f,held)
                self.assertEqual(len(context.items),2);self.assertNotIn(held,context.class_counts)
                for n,item in context.items():self.assertEqual(_item_fingerprint(item),f[2]['datasets'][n]['prepared_fields'])
            a=self.load(f,'xuzhou',('paviaU',));b=self.load(f,'whu-honghu',('paviaU',))
            self.assertEqual(a.record['source_identity'],b.record['source_identity'])

    def test_role_and_source_tamper_rejected(self):
        for sources in ((),('xuzhou',),('paviaU','paviaU'),('unknown',)):
            with self.assertRaises(ValueError):m.roles('xuzhou',sources)
        with self.assertRaises(ValueError):m.roles('unknown')
        with tempfile.TemporaryDirectory() as temp:
            f=self.fixture(temp);path=f[0]/f[2]['datasets']['paviaU']['prepared_file'];path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'artifact changed'):self.load(f)

    def test_source_test_fields_and_renumbered_split_seeds_rejected(self):
        for change in ('test_field','renumber_seed','path_escape','config'):
            with tempfile.TemporaryDirectory() as temp:
                f=self.fixture(temp);p,_,manifest,_=f;entry=manifest['datasets']['whu-honghu']
                if change=='test_field':
                    item=torch.load(p/entry['prepared_file'],weights_only=True);item['test_labels']=torch.tensor([1]);torch.save(item,p/entry['prepared_file'])
                    entry['prepared_sha256']=file_hash(p/entry['prepared_file']);entry['prepared_fields']=_item_fingerprint(item)
                elif change=='renumber_seed':
                    write_json(p/entry['split_file'],{'seed':manifest['config']['training']['seed']+1});entry['split_sha256']=file_hash(p/entry['split_file'])
                elif change=='path_escape':entry['raw']['image_path']='../escape'
                else:manifest['config_hash']='changed'
                write_json(p/'manifest.json',manifest)
                with self.assertRaises(ValueError):self.load(f)


if __name__=='__main__':unittest.main()
