import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from loadData.spatial_tiles import make_tile_plan,tile_slice
from loadData.spatial_development import make_development
from loadData.observation_sampling import BandMetadata
from transfer.development_training import prepare_scene,adapt,validate,evaluate_validation
from transfer.g3_training import _update
from transfer.multiscale_training import MultiscaleTrial, load_multiscale_checkpoint
from model.research_config import digest


class DevelopmentTrainingTests(unittest.TestCase):
    def fixture(self):
        plan=make_tile_plan((96,96),32,2,2026);labels=np.resize(np.arange(1,10),(96,96))
        record=make_development(labels,plan,list(range(1,10)))
        cube=np.random.default_rng(3).normal(size=(96,96,6)).astype(np.float32)
        bands=BandMetadata('fixture',tuple(range(6)),tuple(np.linspace(400,1100,6)),'nm','approximate',(True,)*6)
        return plan,record,cube,bands

    def test_heldout_inputs_do_not_change_train_statistics_and_wrong_support_rejected(self):
        plan,record,cube,bands=self.fixture();scene=prepare_scene(cube,bands,plan,record)
        changed=cube.copy()
        for tile in record['roles']:
            if tile['role']!='devtrain':changed[tile_slice(tile)]=999999
        self.assertEqual(scene.stats,prepare_scene(changed,bands,plan,record).stats)
        bad=copy.deepcopy(record);bad['train_support']['indices'][0]=record['validation_support']['indices'][0]
        bad['train_support']['indices'].sort();bad['validation_hash']=digest({k:v for k,v in bad.items() if k!='validation_hash'})
        with self.assertRaises(ValueError):validate(bad,plan)

    def test_short_loop_update_labels_and_selected_checkpoint(self):
        plan,record,cube,bands=self.fixture();scene=prepare_scene(cube,bands,plan,record)
        base=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/screen_m6_none.json').read_text())
        base['model'].update(latent_channels=8,adapter_embed_dim=8,adapter_checkpoint_chunks=False)
        allowed=set(record['train_support']['indices']);width=plan['shape'][1]
        def guarded(model,scene,tile,local,*args,**kwargs):
            self.assertEqual(record['roles'][tile['id']]['role'],'devtrain')
            yy,xx=np.nonzero(local);y0,_,x0,_=tile['box']
            self.assertTrue(set(((yy+y0)*width+xx+x0).tolist())<=allowed)
            return _update(model,scene,tile,local,*args,**kwargs)
        with tempfile.TemporaryDirectory() as root,patch('transfer.development_training._update',side_effect=guarded):
            for mode in ('uniform','class_first'):
                with self.subTest(sampling=mode):
                    result=adapt(scene,record,base,Path(root)/mode,device=os.environ.get('G3_DEVICE','cpu'),sampling=mode)
                    self.assertFalse(result['test_evaluated'])
                    self.assertEqual(len(result['validation']),2)
                    self.assertEqual(result['selected_step'],min(result['validation'],key=lambda x:x['nll'])['step'])
                    self.assertEqual(result['sampler']['mode'],mode)
                    self.assertNotEqual(result['initial_backbone_hash'],result['final_backbone_hash'])
                    if mode=='class_first':self.assertTrue(all('anchor_class' in x for x in result['trace']))

    def test_multiscale_variants_train_and_reload_under_matched_spatial_protocol(self):
        plan, record, cube, approximate = self.fixture()
        bands = BandMetadata(approximate.source_id, approximate.band_ids, approximate.wavelengths,
                             "nm", "verified", approximate.valid)
        scene = prepare_scene(cube, bands, plan, record)
        base = json.loads((Path(__file__).resolve().parents[1] / 'configs/experiments/screen_m6_none.json').read_text())
        base['model'].update(latent_channels=8, adapter_embed_dim=8, adapter_checkpoint_chunks=False)
        updated = []

        def record_update(model, *args, **kwargs):
            selector = getattr(model.backbone, 'range_metadata', None)
            previous = selector.weight.detach().clone() if selector is not None else None
            result = _update(model, *args, **kwargs)
            if selector is not None:
                updated.append(not torch.equal(previous, selector.weight.detach()))
            return result

        with tempfile.TemporaryDirectory() as root, patch('transfer.development_training._update', side_effect=record_update):
            baseline = adapt(scene, record, base, Path(root) / 'baseline', steps=2)
            control = adapt(scene, record, base, Path(root) / 'control', steps=2, multiscale=MultiscaleTrial())
            self.assertEqual(baseline['trace'], control['trace'])
            self.assertEqual(baseline['validation'], control['validation'])
            candidate_dir = Path(root) / 'candidate'
            candidate = adapt(scene, record, base, candidate_dir, steps=2,
                              multiscale=MultiscaleTrial(cnn_multiscale='joint', graph_multiscale='joint'))
            self.assertTrue(any(updated))
            self.assertEqual(candidate['initial_head_hash'], baseline['initial_head_hash'])
            self.assertEqual([row['tile_id'] for row in candidate['trace']], [row['tile_id'] for row in baseline['trace']])
            self.assertFalse(candidate['test_evaluated'])
            selected = load_multiscale_checkpoint(candidate_dir / 'model.pt')
            selected_row = min(candidate['validation'], key=lambda row: row['nll'])
            self.assertEqual(candidate['selected_step'], selected_row['step'])
            self.assertEqual(evaluate_validation(selected, scene, record), {k: v for k, v in selected_row.items() if k != 'step'})
