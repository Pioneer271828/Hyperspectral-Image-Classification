"""Selected trained light publication must preserve frozen pair and lazy graph."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from inference.light_cnn import LightCNN,load_light
from inference.tile_cascade import cascade
from inference.trained_light import export_trained_light,load_trained_light,trained_cascade
from transfer.light_head_training import LightHeadRun
from transfer.g3_training import state_hash
from loadData.spatial_tiles import observe_tile
from model.research_config import digest
from tests.test_int8_model import fixture


class TrainedLightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.full,self.scene,self.record=fixture();self.run=LightHeadRun(self.full,self.scene,self.record,updates=3,seed=64,lr=.0001,class_balance=True)

    def publish(self):
        self.run.advance(3);p=self.root/'train.pt';self.trainsha=self.run.save(p);self.path=self.root/'published.pt'
        self.sha=export_trained_light(self.full,self.scene,self.record,p,self.path,expected_training_sha256=self.trainsha,expected_contract_hash=self.run.contract_hash)
        return self.load()

    def load(self,**kwargs):
        args=dict(expected_sha256=self.sha,expected_training_sha256=self.trainsha,expected_contract_hash=self.run.contract_hash);args.update(kwargs)
        return load_trained_light(self.full,self.path,**args)

    def rows(self):
        rows=[]
        for tile in self.scene.plan['tiles'][:2]:
            obs=observe_tile(self.scene.cube,self.scene.plan,self.scene.stats,tile)
            rows.append(dict(id=str(tile['id']),dataset=self.scene.dataset,x=torch.tensor(np.concatenate((obs.spectral,obs.coordinates),-1)),
                wavelengths=torch.tensor(self.scene.bands.wavelengths,dtype=torch.float32),scale=50,coverage_known=True,coverage_fraction=1.))
        return rows

    def test_selected_head_owned_frozen_and_roundtrip_source_unchanged(self):
        before=state_hash(self.full);rng=torch.get_rng_state().clone();pub=self.publish();again=self.load()
        self.assertEqual(before,state_hash(self.full));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(state_hash(pub.model),state_hash(again.model));self.assertFalse(any(p.requires_grad for p in pub.model.parameters()))
        self.assertEqual(pub.specification['selected_validation'],self.run.history[self.run.selected_step-1]['validation'])
        for k,v in self.run.selected_head.items():self.assertTrue(torch.equal(pub.model.heads[self.scene.dataset].state_dict()[k],v))
        self.assertNotEqual(pub.model.heads[self.scene.dataset].weight.data_ptr(),self.run.head.weight.data_ptr())
        source=pub.specification;source['selected_step']=999;self.assertNotEqual(pub.specification['selected_step'],999)
        for row in self.rows():
            features=self.run.path.frozen_features(row['dataset'],row['x'],wavelengths=row['wavelengths'])
            head=copy.deepcopy(self.run.head);head.load_state_dict(self.run.selected_head)
            self.assertTrue(torch.equal(head(features).softmax(-1),pub.model(row['dataset'],row['x'],wavelengths=row['wavelengths'])))

    def test_old_copy_boundary_stays_strict_default_full_and_fast_no_graph(self):
        pub=self.publish();rows=self.rows();graph=self.full.backbone.learnable_graph
        with self.assertRaises(ValueError):cascade(self.full,pub.model,rows,mode='fast')
        with self.assertRaises(ValueError):load_light(self.path,self.sha)
        with patch.object(pub.model,'forward',side_effect=AssertionError('light called')),patch.object(graph,'build_graph',wraps=graph.build_graph) as spy:
            out=trained_cascade(self.full,pub,rows);self.assertEqual(spy.call_count,2)
        for row,actual in zip(rows,out['outputs']):
            with torch.no_grad():expected=self.full(row['dataset'],row['x'],wavelengths=row['wavelengths'],superpixel_scale=row['scale'])[0]
            self.assertTrue(torch.equal(expected,actual['probabilities']))
        with patch.object(graph,'build_graph',side_effect=AssertionError('graph called')),patch.object(self.full,'forward',side_effect=AssertionError('full called')):
            out=trained_cascade(self.full,pub,rows,mode='fast')
        self.assertEqual(out['counts']['full_calls'],0);self.assertFalse(out['calibrated'])
        for row,actual in zip(rows,out['outputs']):self.assertTrue(torch.equal(actual['probabilities'],pub.model(row['dataset'],row['x'],wavelengths=row['wavelengths'])))

    def test_mixed_unknown_low_coverage_all_fallback_and_order(self):
        pub=self.publish();rows=self.rows()[::-1];rows[1]['coverage_known']=False
        out=trained_cascade(self.full,pub,rows,mode='adaptive',criterion='margin',threshold=0)
        self.assertEqual([o['id'] for o in out['outputs']],[r['id'] for r in rows]);self.assertEqual([o['route'] for o in out['outputs']],['fast','full'])
        rows[1]['coverage_known']=True;rows[1]['coverage_fraction']=.25
        self.assertEqual(trained_cascade(self.full,pub,rows,mode='fast')['counts']['full_calls'],1)
        rows[1]['coverage_fraction']=1.
        out=trained_cascade(self.full,pub,rows,mode='adaptive',criterion='max_probability',threshold=1)
        self.assertEqual(out['counts'],dict(light_calls=2,full_calls=2,accepted_tiles=0,total_tiles=2))
        expected=trained_cascade(self.full,pub,rows)
        for a,b in zip(out['outputs'],expected['outputs']):self.assertTrue(torch.equal(a['probabilities'],b['probabilities']))
        self.assertEqual(trained_cascade(self.full,pub,[])['outputs'],[])

    def test_incomplete_wrong_training_hash_contract_and_no_overwrite(self):
        p=self.root/'partial.pt';self.run.advance(1);sha=self.run.save(p)
        with self.assertRaises(ValueError):export_trained_light(self.full,self.scene,self.record,p,self.root/'bad.pt',expected_training_sha256=sha,expected_contract_hash=self.run.contract_hash)
        self.assertFalse((self.root/'bad.pt').exists());self.publish()
        with self.assertRaises(FileExistsError):export_trained_light(self.full,self.scene,self.record,self.root/'train.pt',self.path,expected_training_sha256=self.trainsha,expected_contract_hash=self.run.contract_hash)
        for key in ('expected_training_sha256','expected_contract_hash'):
            with self.assertRaises(ValueError):self.load(**{key:'0'*64})
        with patch('torch.load',side_effect=AssertionError('unauthenticated deserialize')):
            with self.assertRaises(ValueError):self.load(expected_sha256='0'*64)

    def test_source_prefix_head_runtime_and_other_task_rejected(self):
        pub=self.publish();rows=self.rows()
        for target in ('head','prefix','train','grad','cache','spec'):
            obj=copy.deepcopy(pub)
            if target=='head':
                with torch.no_grad():obj.model.heads[self.scene.dataset].bias.add_(.01)
            elif target=='prefix':
                with torch.no_grad():next(obj.model.adapter.parameters()).add_(.01)
            elif target=='train':obj.model.train()
            elif target=='grad':obj.model.heads[self.scene.dataset].requires_grad_(True)
            elif target=='cache':obj.model.adapter.cache_attention=True
            else:obj._spec['selected_step']=999
            with self.assertRaises(ValueError):trained_cascade(self.full,obj,rows)
        bad=[dict(rows[0],dataset='honghu')]
        with patch.object(self.full,'forward',side_effect=AssertionError('executed')):
            with self.assertRaises(ValueError):trained_cascade(self.full,pub,bad)
        with torch.no_grad():next(self.full.parameters()).add_(.01)
        with self.assertRaises(ValueError):self.load()

    def test_corrupted_packet_schema_and_rehashed_prefix_rejected(self):
        self.publish();packet=torch.load(self.path,weights_only=True)
        for kind in ('dtype','nan','extra','selection','prefix','contract'):
            bad=copy.deepcopy(packet);key=next(iter(bad['state']))
            if kind=='dtype':bad['state'][key]=bad['state'][key].double()
            elif kind=='nan':bad['state'][key].flatten()[0]=float('nan')
            elif kind=='extra':bad['state']['extra']=torch.zeros(1)
            elif kind=='selection':bad['spec']['selected_step']=0
            elif kind=='contract':bad['spec']['training_spec']['lr']=1.
            else:
                bad['state'][key].add_(.01);model=LightCNN(self.full);model.load_state_dict(bad['state']);bad['spec']['light_state']=state_hash(model)
            bad['spec_hash']=digest(bad['spec']);p=self.root/(kind+'.pt');torch.save(bad,p)
            with self.assertRaises(ValueError):load_trained_light(self.full,p,expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),expected_training_sha256=self.trainsha,expected_contract_hash=self.run.contract_hash)

    def test_prevalidates_every_request_without_partial_execution(self):
        pub=self.publish();rows=self.rows()
        with patch.object(pub.model,'forward',side_effect=AssertionError('executed')),patch.object(self.full,'forward',side_effect=AssertionError('executed')):
            for requests in (rows+rows,[rows[0],dict(rows[1],x=rows[1]['x'].double())],iter(rows)):
                with self.assertRaises(ValueError):trained_cascade(self.full,pub,requests,mode='fast')


if __name__=='__main__':unittest.main()
