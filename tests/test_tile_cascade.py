"""Whole-block routes must actually skip graph work and preserve frozen outputs."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from inference.light_cnn import LightCNN, save_light, load_light
from inference.tile_cascade import cascade
from loadData.spatial_tiles import observe_tile
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from tests.test_int8_model import fixture
from transfer.g3_training import state_hash


def setup(norm=None):
    full,scene,_=fixture()
    if norm:
        cfg=full.export_config();cfg['backbone_kwargs']['norm_type']=norm
        full=MultiDatasetVCASAHGCN(cfg['class_counts'],**cfg['backbone_kwargs']).eval()
        full.backbone.adapter.cache_attention=False
    rows=[]
    for tile in scene.plan['tiles'][:2]:
        obs=observe_tile(scene.cube,scene.plan,scene.stats,tile)
        rows.append(dict(id=str(tile['id']),dataset=scene.dataset,
            x=torch.tensor(np.concatenate((obs.spectral,obs.coordinates),-1)),
            wavelengths=torch.tensor(scene.bands.wavelengths,dtype=torch.float32),scale=50,
            coverage_known=True,coverage_fraction=1.0))
    return full,LightCNN(full),rows


def original(full,row):
    with torch.no_grad():
        return full(row['dataset'],row['x'],wavelengths=row['wavelengths'],superpixel_scale=row['scale'])


class TileCascadeTests(unittest.TestCase):
    def test_light_exact_original_cnn_both_norms_owned_no_graph(self):
        for norm in (None,'domain_bn'):
            full,light,rows=setup(norm);before=state_hash(full)
            self.assertLess(sum(p.numel() for p in light.parameters()),sum(p.numel() for p in full.parameters()))
            self.assertFalse(any('graph' in k.lower() or 'hgnn' in k.lower() for k in light.state_dict()))
            for row in rows:
                self.assertTrue(torch.equal(light(row['dataset'],row['x'],wavelengths=row['wavelengths']),original(full,row)[2]))
            self.assertEqual(before,state_hash(full));self.assertFalse(any(p.requires_grad for p in light.parameters()))
            self.assertNotEqual(next(full.backbone.adapter.parameters()).data_ptr(),next(light.adapter.parameters()).data_ptr())

    def test_default_full_exact_no_light_and_fast_no_graph(self):
        full,light,rows=setup();expected=[original(full,row) for row in rows]
        graph=full.backbone.learnable_graph
        with patch.object(light,'forward',side_effect=AssertionError('light invoked')),patch.object(graph,'build_graph',wraps=graph.build_graph) as spy:
            result=cascade(full,light,rows);self.assertEqual(spy.call_count,2)
        self.assertEqual(result['counts'],dict(light_calls=0,full_calls=2,accepted_tiles=0,total_tiles=2))
        for row,exp in zip(result['outputs'],expected):self.assertTrue(torch.equal(row['probabilities'],exp[0]))
        with patch.object(graph,'build_graph',side_effect=AssertionError('graph invoked')),patch.object(full,'forward',side_effect=AssertionError('full invoked')):
            result=cascade(full,light,rows,mode='fast')
        self.assertEqual(result['counts']['full_calls'],0)
        for row,exp in zip(result['outputs'],expected):self.assertTrue(torch.equal(row['probabilities'],exp[2]))

    def test_mixed_unknown_low_coverage_and_order(self):
        full,light,rows=setup();rows=rows[::-1];rows[1]['coverage_known']=False
        result=cascade(full,light,rows,mode='adaptive',criterion='max_probability',threshold=0)
        self.assertEqual([r['id'] for r in result['outputs']],[r['id'] for r in rows])
        self.assertEqual([r['route'] for r in result['outputs']],['fast','full'])
        self.assertEqual(result['counts'],dict(light_calls=1,full_calls=1,accepted_tiles=1,total_tiles=2))
        rows[1]['coverage_known']=True;rows[1]['coverage_fraction']=.5
        result=cascade(full,light,rows,mode='fast',min_coverage=.75)
        self.assertEqual([r['route'] for r in result['outputs']],['fast','full'])
        self.assertTrue(torch.equal(result['outputs'][1]['probabilities'],original(full,rows[1])[0]))

    def test_three_gate_scores_endpoints_and_empty(self):
        full,light,rows=setup()
        for criterion,accept,reject in [('max_probability',0,1),('margin',0,1),('entropy',100,0)]:
            a=cascade(full,light,rows,mode='adaptive',criterion=criterion,threshold=accept)
            b=cascade(full,light,rows,mode='adaptive',criterion=criterion,threshold=reject)
            self.assertEqual(a['counts']['full_calls'],0);self.assertEqual(b['counts']['full_calls'],2)
            self.assertEqual(b['counts']['light_calls'],2);self.assertFalse(a['calibrated'])
            p=light(rows[0]['dataset'],rows[0]['x'],wavelengths=rows[0]['wavelengths']).numpy()
            score=(p.max(1).min() if criterion=='max_probability' else
                (np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2]).min() if criterion=='margin' else -(p*np.log(p)).sum(1).max())
            if criterion=='entropy':score=(-(p*np.log(p)).sum(1)).max()
            self.assertAlmostEqual(a['outputs'][0]['score'],float(score),places=5)
        self.assertEqual(cascade(full,light,[])['outputs'],[])

    def test_prevalidate_all_requests_before_any_execution(self):
        full,light,rows=setup()
        bad=[rows+rows,[rows[0],dict(rows[1],x=rows[1]['x'].double())],
             [dict(rows[0],coverage_fraction=float('nan'))],[dict(rows[0],wavelengths=torch.tensor([-1.]))],
             [dict(rows[0],scale=True)],[dict(rows[0],dataset='missing')]]
        with patch.object(full,'forward',side_effect=AssertionError('executed')),patch.object(light,'forward',side_effect=AssertionError('executed')):
            for request in bad:
                with self.assertRaises(ValueError):cascade(full,light,request,mode='fast')
            with self.assertRaises(ValueError):cascade(full,light,iter(rows))
            for kw in ({'mode':'bad'},{'mode':'full','threshold':.5},{'mode':'adaptive','criterion':'margin','threshold':float('nan')},
                       {'mode':'adaptive','criterion':'max_probability','threshold':1.1},{'min_coverage':True}):
                with self.assertRaises(ValueError):cascade(full,light,rows,**kw)

    def test_source_training_cache_coupling_and_light_mutation_rejected(self):
        full,light,rows=setup();full.train()
        with self.assertRaises(ValueError):LightCNN(full)
        full.eval();full.backbone.adapter.cache_attention=True
        with self.assertRaises(ValueError):LightCNN(full)
        full.backbone.adapter.cache_attention=False;full.backbone.region_residual=torch.nn.Identity()
        with self.assertRaises(ValueError):LightCNN(full)
        full.backbone.region_residual=None
        with torch.no_grad():next(light.parameters()).add_(.01)
        with self.assertRaises(ValueError):cascade(full,light,rows)
        light=LightCNN(full)
        with torch.no_grad():next(full.parameters()).add_(.01)
        with self.assertRaises(ValueError):cascade(full,light,rows)

    def test_checkpoint_exact_rng_hash_before_load_and_no_overwrite(self):
        full,light,rows=setup();rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'light.pt';sha=save_light(light,path);restored=load_light(path,sha)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for row in rows:
                self.assertTrue(torch.equal(light(row['dataset'],row['x'],wavelengths=row['wavelengths']),restored(row['dataset'],row['x'],wavelengths=row['wavelengths'])))
            self.assertEqual(cascade(full,restored,rows,mode='fast')['counts']['full_calls'],0)
            with self.assertRaises(FileExistsError):save_light(light,path)
            with patch('torch.load',side_effect=AssertionError('deserialized')):
                with self.assertRaises(ValueError):load_light(path,'0'*64)

    def test_checkpoint_bad_schema_dtype_nonfinite_and_metadata(self):
        _,light,_=setup()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'light.pt';save_light(light,path);packet=torch.load(path,weights_only=True)
            for kind in ('extra','dtype','nan','spec','source'):
                bad=copy.deepcopy(packet);key=next(iter(bad['state']))
                if kind=='extra':bad['state']['extra']=torch.zeros(1)
                elif kind=='dtype':bad['state'][key]=bad['state'][key].double()
                elif kind=='nan':bad['state'][key].flatten()[0]=float('nan')
                elif kind=='spec':bad['spec']['method']='other';bad['spec_hash']=digest(bad['spec'])
                else:bad['spec']['source_state']='bad';bad['spec_hash']=digest(bad['spec'])
                target=Path(tmp)/(kind+'.pt');torch.save(bad,target)
                with self.assertRaises(ValueError):load_light(target,hashlib.sha256(target.read_bytes()).hexdigest())

    def test_mixed_shapes_repeat_input_state_and_cache_unchanged(self):
        full,light,rows=setup();rows[1]['x']=rows[1]['x'][:9,:11].clone()
        before=(state_hash(full),state_hash(light));inputs=[r['x'].clone() for r in rows]
        a=cascade(full,light,rows,mode='fast');b=cascade(full,light,rows,mode='fast')
        for left,right,x,row in zip(a['outputs'],b['outputs'],inputs,rows):
            self.assertTrue(torch.equal(left['probabilities'],right['probabilities']));self.assertTrue(torch.equal(x,row['x']))
            self.assertEqual(len(left['probabilities']),row['x'].shape[0]*row['x'].shape[1])
        self.assertEqual(before,(state_hash(full),state_hash(light)))
        self.assertEqual(light.adapter._eval_attention_cache,{})
        self.assertEqual(full.backbone.adapter._eval_attention_cache,{})


if __name__=='__main__':unittest.main()
