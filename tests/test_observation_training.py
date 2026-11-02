"""M1-B: actual classifier training, disabled regression, role isolation and replay."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

from loadData.observation_sampling import BandMetadata
from loadData.spatial_tiles import make_tile_plan, tile_slice, observe_tile
from loadData.spatial_development import make_development
from model.checkpoint_io import load_model_checkpoint
from transfer.development_training import prepare_scene, adapt, evaluate_validation
from transfer.g3_training import build_target, _forward, state_hash
from transfer.observation_policy import ObservationPolicy, evaluate_conditions
from transfer.run_provenance import verify_run_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASE = '39459615f2fb3c84d7a362fa5cf7421ca60dd766'


class ObservationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('M1_DEVICE', 'cpu')
        self.plan = make_tile_plan((96, 96), 32, 2, 2026)
        self.record = make_development(np.resize(np.arange(1, 10), (96, 96)), self.plan, list(range(1, 10)))
        self.cube = np.random.default_rng(3).normal(size=(96, 96, 8)).astype(np.float32)
        self.bands = BandMetadata('fixture', tuple(range(8)), tuple(np.linspace(400, 1100, 8)), 'nm', 'approximate', (True,)*8)
        self.scene = prepare_scene(self.cube, self.bands, self.plan, self.record)
        self.base = json.loads((ROOT/'configs/experiments/screen_m6_none.json').read_text())
        self.base['model'].update(latent_channels=8, adapter_embed_dim=8, adapter_checkpoint_chunks=False)

    def policy(self, configs=None, updates=2, **kwargs):
        return ObservationPolicy.freeze(self.scene, self.record, updates=updates,
            training_configs=configs or [{'mode': 'random_subset', 'keep_fraction': .5}, {'mode': 'random_subset', 'keep_fraction': .25}],
            train_seed=64, validation_seed=512, **kwargs)

    def tile(self, role): return next(t for t in self.plan['tiles'] if self.record['roles'][t['id']]['role'] == role)

    def test_frozen_replay_deterministic_independent_rng_and_validation_budget(self):
        np.random.seed(44); torch.manual_seed(44); n=np.random.get_state(); t=torch.get_rng_state().clone()
        p=self.policy(); q=self.policy(updates=4)
        self.assertEqual(p.to_dict(), self.policy().to_dict())
        self.assertEqual(p.to_dict()['validation'], q.to_dict()['validation'])
        np.testing.assert_array_equal(np.random.get_state()[1], n[1]); self.assertTrue(torch.equal(t, torch.get_rng_state()))
        a=p.view(self.scene,self.record,self.tile('devtrain'),phase='training',step=1)
        b=p.view(self.scene,self.record,self.tile('devtrain'),phase='training',step=1)
        np.testing.assert_array_equal(a.observation.spectral,b.observation.spectral)
        exported=p.to_dict(); exported['training'][0]['selected_positions'].clear()
        self.assertEqual(p.training_record(1)['selected_band_count'],4)

    def test_metadata_geometry_and_original_statistics_slices(self):
        self.scene.bands=replace(self.bands,valid=(True,False,True,True,True,True,True,True),fwhm=(2.,)*8,fwhm_unit='nm')
        p=self.policy(); tile=self.tile('devtrain'); obs=observe_tile(self.cube,self.plan,self.scene.stats,tile)
        view=p.view(self.scene,self.record,tile,phase='training',step=1)
        ids=list(view.bands.band_ids)
        self.assertNotIn(1,ids); self.assertEqual(view.bands.fwhm,(2.,)*4); self.assertTrue(all(view.bands.valid))
        np.testing.assert_array_equal(view.observation.spectral,obs.spectral[...,ids])
        np.testing.assert_array_equal(view.observation.coordinates,obs.coordinates)
        np.testing.assert_array_equal(view.observation.global_ids,obs.global_ids)
        self.assertEqual(p.selected_statistics(self.scene,step=1)['minimum'],[self.scene.stats['minimum'][i] for i in ids])

    def test_wrong_roles_empty_tampered_identity_and_budget_rejected_before_output(self):
        p=self.policy()
        for role in ('devval','test'):
            with self.assertRaises(ValueError):p.view(self.scene,self.record,self.tile(role),phase='training',step=1)
        with self.assertRaises(ValueError):p.view(self.scene,self.record,self.tile('devtrain'),phase='validation')
        with self.assertRaises(ValueError):p.view(self.scene,self.record,self.tile('devval'),phase='validation',condition='unregistered')
        bad=p.to_dict();bad['training'][0]['selected_positions'][0]=7
        with self.assertRaises(ValueError):ObservationPolicy(bad)
        with self.assertRaises(ValueError):self.policy([{'mode':'range_crop','lower':1,'upper':2,'unit':'nm'}])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'bad'
            with self.assertRaises(ValueError):adapt(self.scene,self.record,self.base,path,steps=3,observation_policy=p)
            self.assertFalse(path.exists())
        changed=copy.deepcopy(self.scene);changed.bands=replace(self.bands,source_id='wrong')
        with self.assertRaises(ValueError):p.validate(changed,self.record)

    def test_actual_forward_variable_b_one_band_fwhm_and_gradients(self):
        self.scene.bands=replace(self.bands,fwhm=(2.,)*8,fwhm_unit='nm')
        p=self.policy([{'mode':'random_subset','keep_fraction':.01}]); tile=self.tile('devtrain')
        model,_=build_target(self.base,self.scene,'target_scratch_full',None,model_seed=64,head_seed=66,device=self.device)
        seen=[]
        def hook(_,args,kwargs):seen.append((args[0].shape[-1],len(kwargs['wavelengths']),kwargs['fwhm'].detach().cpu().tolist()))
        h=model.backbone.adapter.register_forward_pre_hook(hook,with_kwargs=True)
        view=p.view(self.scene,self.record,tile,phase='training',step=1)
        output=_forward(model,self.scene,tile,observation_view=view)[0]
        (-output[:,0].clamp_min(1e-8).log().mean()).backward();h.remove()
        self.assertEqual(seen,[(1,1,[2.])]);self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(any(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max()>0 for x in model.backbone.adapter.parameters()))

    def test_default_matches_actual_parent_short_training(self):
        old=types.ModuleType('old_development');old_g3=types.ModuleType('old_g3')
        for module,name in ((old,'transfer/development_training.py'),(old_g3,'transfer/g3_training.py')):
            source=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+name],text=True)
            exec(compile(source,name,'exec'),module.__dict__)
        old._update=old_g3._update;old._forward=old_g3._forward
        with tempfile.TemporaryDirectory() as root:
            a=old.adapt(self.scene,self.record,self.base,Path(root)/'old',device=self.device,seed=64)
            b=adapt(self.scene,self.record,self.base,Path(root)/'new',device=self.device,seed=64)
            self.assertEqual(a['validation'],b['validation']);self.assertEqual(a['trace'],b['trace'])
            self.assertEqual(a['final_backbone_hash'],b['final_backbone_hash']);self.assertEqual(a['final_head_hash'],b['final_head_hash'])

    def test_full_policy_equivalence_and_checkpoint_roundtrip(self):
        p=self.policy([{'mode':'none'}])
        with tempfile.TemporaryDirectory() as root:
            a=adapt(self.scene,self.record,self.base,Path(root)/'off',device=self.device,seed=64)
            b=adapt(self.scene,self.record,self.base,Path(root)/'on',device=self.device,seed=64,observation_policy=p)
            self.assertEqual(a['validation'],b['validation']);self.assertEqual(a['final_backbone_hash'],b['final_backbone_hash'])
            model=load_model_checkpoint(Path(root)/'on/model.pt',self.device)
            with self.assertRaises(ValueError):evaluate_validation(model,self.scene,self.record)
            metric=evaluate_validation(model,self.scene,self.record,observation_policy=p)
            self.assertAlmostEqual(metric['nll'],min(b['validation'],key=lambda x:x['nll'])['nll'],places=6)
            self.assertEqual(verify_run_artifacts(Path(root)/'on')['status'],'verified')

    def test_subset_training_actual_widths_sampling_and_fixed_three_head_review(self):
        from transfer import development_training as dev
        p=self.policy();seen=[];grad=[];builder=dev.build_target
        def build(*args,**kwargs):
            model,config=builder(*args,**kwargs)
            model.backbone.adapter.register_forward_pre_hook(lambda m,args:seen.append((model.training,args[0].shape[-1])))
            next(model.backbone.adapter.parameters()).register_hook(lambda x:grad.append(bool(torch.isfinite(x).all())))
            return model,config
        before=copy.deepcopy(self.scene.stats)
        with tempfile.TemporaryDirectory() as root,patch.object(dev,'build_target',side_effect=build):
            result=adapt(self.scene,self.record,self.base,Path(root)/'aug',device=self.device,seed=64,sampling='class_first',observation_policy=p)
            self.assertEqual([b for train,b in seen if train],[4,2]);self.assertTrue(all(b==8 for train,b in seen if not train));self.assertEqual(grad,[True,True])
            model=load_model_checkpoint(Path(root)/'aug/model.pt',self.device)
            review=evaluate_conditions(model,self.scene,self.record,ObservationPolicy(json.loads((Path(root)/'aug/observation-policy.json').read_text())))
            self.assertEqual(set(review['conditions']),set(p.conditions));self.assertFalse(review['test_evaluated'])
            self.assertAlmostEqual(review['conditions']['full']['fused']['nll'],min(result['validation'],key=lambda x:x['nll'])['nll'],places=6)
            self.assertEqual(before,self.scene.stats)
            bad=json.loads((Path(root)/'aug/observation-policy.json').read_text());bad['training'][0]['selected_band_ids'][0]=99
            (Path(root)/'aug/observation-policy.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):verify_run_artifacts(Path(root)/'aug')

    def test_heldout_changes_cannot_change_schedule_or_statistics(self):
        changed=self.cube.copy()
        for t in self.record['roles']:
            if t['role']!='devtrain':changed[tile_slice(t)]=99999
        scene=prepare_scene(changed,self.bands,self.plan,self.record)
        a=self.policy();b=ObservationPolicy.freeze(scene,self.record,updates=2,training_configs=a.to_dict()['training_configs'],train_seed=64,validation_seed=512)
        self.assertEqual(a.to_dict(),b.to_dict())
