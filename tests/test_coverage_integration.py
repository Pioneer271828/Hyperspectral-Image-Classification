"""E1-B registered backbone, identity binding, production training and restoration."""
import copy
from dataclasses import replace
from functools import lru_cache
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
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from model.coverage_config import coverage_forward_kwargs
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, tensor_hash, hash_file
from train_spatial import SpatialConfig, execute
from train_vca_multidataset import build_parser
from transfer import development_training as dev
from transfer.g3_training import build_target, build_model, state_hash, _forward
from transfer.observation_policy import ObservationPolicy
from transfer.run_provenance import verify_run_artifacts

ROOT = Path(__file__).resolve().parents[1]
BASE = '70edf753518dc50b3ac4f212cd79d1e593433f54'


@lru_cache(maxsize=None)
def parent(path):
    module=types.ModuleType('e1b_parent_'+path.replace('/','_'))
    source=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+path],text=True)
    exec(compile(source,path,'exec'),module.__dict__)
    return module


class CoverageIntegrationTests(unittest.TestCase):
    def setUp(self):self.device=os.environ.get('E1_DEVICE','cpu')

    def model(self, mode='affine', **kwargs):
        torch.manual_seed(9)
        return MultiDatasetVCASAHGCN({'a':3},latent_channels=8,adapter_embed_dim=8,
            norm_type='group_norm',coverage_mode=mode,**{'graph_mode':'learned',**kwargs}).to(self.device)

    def inputs(self):
        x=torch.randn(6,7,10,device=self.device)
        meta=BandMetadata('fixture',tuple(range(8)),tuple(np.linspace(400,1100,8)),
                          'nm','approximate',(True,)*8)
        return x,torch.tensor(meta.wavelengths,device=self.device,dtype=x.dtype),meta

    def call(self,model,x,wave,meta,**kw):
        return model('a',x,wavelengths=wave,superpixel_scale=9,
            **coverage_forward_kwargs(model,meta),**kw)

    def test_default_actual_parent_rng_state_outputs_gradients_and_buffers(self):
        old_type=parent('model/VCASAHGCN.py').MultiDatasetVCASAHGCN
        opts=dict(latent_channels=8,adapter_embed_dim=8,norm_type='group_norm',graph_mode='learned')
        torch.manual_seed(9);old=old_type({'a':3},**opts).to(self.device);rng=torch.get_rng_state().clone()
        new=self.model('none');self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(old.export_config(),new.export_config())
        self.assertEqual(state_hash(old),state_hash(new))
        x,wave,meta=self.inputs();rtol=atol=0
        if self.device.startswith('cuda'):
            old.eval();new.eval()
            with torch.no_grad():
                for a,b in zip(self.call(old,x,wave,meta),self.call(new,x,wave,meta)):
                    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            old.train().double();new.train().double();x=x.double();wave=wave.double();rtol,atol=1e-9,1e-10
        a=x.clone().requires_grad_();b=x.clone().requires_grad_()
        left=self.call(old,a,wave,meta,return_graph_info=True)
        right=self.call(new,b,wave,meta,return_graph_info=True)
        self.assertEqual(set(left[3]),set(right[3]))
        for p,q in zip(left[:3],right[:3]):torch.testing.assert_close(p,q,rtol=rtol,atol=atol)
        for outputs in (left,right):sum(-p[:,0].clamp_min(1e-8).log().mean() for p in outputs[:3]).backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=rtol,atol=atol)
        for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=rtol,atol=atol,msg=name)
        for name,value in old.named_buffers():torch.testing.assert_close(value,dict(new.named_buffers())[name],rtol=rtol,atol=atol)

    def test_default_canonical_configs_and_cli_roundtrip(self):
        old=parent('model/research_config.py').ResearchConfig.from_dict({})
        new=ResearchConfig.from_dict({})
        self.assertEqual(old.to_dict(),new.to_dict());self.assertEqual(old.config_hash,new.config_hash)
        self.assertEqual(old.model_kwargs(),new.model_kwargs())
        hashes=[]
        for mode in ('none','affine'):
            args=build_parser().parse_args(['--coverage-mode',mode,'--coverage-init-seed','512','--device','cpu'])
            cfg=ResearchConfig.from_legacy(args)
            self.assertEqual(ResearchConfig.from_legacy(cfg.trainer_args('/tmp/m.pt','/tmp/m.json')).to_dict(),cfg.to_dict())
            hashes.append(cfg.config_hash)
            spatial=SpatialConfig({'model':{'graph_mode':'learned','coverage_mode':mode},'training':{'steps_per_epoch':2}})
            self.assertEqual(spatial.to_dict(),SpatialConfig(spatial.to_dict()).to_dict())
        self.assertNotEqual(*hashes)
        for key,value in [('coverage_mode','bad'),('coverage_hidden',0),('coverage_strength',2),('coverage_window',0),('coverage_init_seed',True),('coverage_sigma',float('nan'))]:
            with self.assertRaises(ValueError):ResearchConfig.from_dict({'model':{key:value}})

    def test_default_short_training_fp64_with_same_version_control(self):
        # Native adaptive-pool CUDA backward is nondeterministic. Keep original
        # M1 tests/thresholds unchanged; explicitly type both model and inputs,
        # and impose stricter bounds on these three numerical comparisons.
        from test_observation_training import ObservationTrainingTests
        original_forward=MultiDatasetVCASAHGCN.forward
        original_compare=ObservationTrainingTests.compare_runs
        def typed(model,dataset,x,*a,**kw):return original_forward(model,dataset,x.to(torch.float64),*a,**kw)
        def strict(fixture,a,b,left,right):
            original_compare(fixture,a,b,left,right)
            np.testing.assert_allclose([r['nll'] for r in a['validation']],[r['nll'] for r in b['validation']],rtol=0,atol=1e-10)
            x=load_model_checkpoint(left/'model.pt','cpu').state_dict();y=load_model_checkpoint(right/'model.pt','cpu').state_dict()
            for k in x:torch.testing.assert_close(x[k],y[k],rtol=0,atol=1e-10,msg=k)
        previous=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with patch.object(MultiDatasetVCASAHGCN,'forward',typed),patch.object(ObservationTrainingTests,'compare_runs',strict):
                for name in ('test_default_matches_actual_parent_short_training','test_same_version_repeat_numerical_control','test_full_policy_equivalence_and_checkpoint_roundtrip'):
                    fixture=ObservationTrainingTests(name);fixture.setUp();fixture.device=self.device
                    getattr(fixture,name)()
        finally:torch.set_default_dtype(previous)

    def test_enabled_initialization_preserves_existing_weights_rng_and_heads(self):
        off=self.model('none');rng=torch.get_rng_state().clone();on=self.model()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,p in off.state_dict().items():self.assertTrue(torch.equal(p,on.state_dict()[name]),name)
        x,wave,meta=self.inputs();off.eval();on.eval()
        with torch.no_grad():
            for a,b in zip(self.call(off,x,wave,meta),self.call(on,x,wave,meta)):
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_binding_rejects_wrong_subset_values_units_widths_and_invalid_columns(self):
        model=self.model();x,wave,meta=self.inputs()
        bad=[meta.subset([0,1]),replace(meta,wavelengths=tuple(v+1 for v in meta.wavelengths)),
             replace(meta,valid=(False,)+(True,)*7),replace(meta,fwhm=(2.,)*8,fwhm_unit='nm')]
        for item in bad:
            with self.assertRaises(ValueError):self.call(model,x,wave,item)
        with self.assertRaises(ValueError):model('a',x,wavelengths=wave,coverage_metadata=meta,wavelength_unit='normalized')
        small=replace(meta,wavelengths=(1.,)*8)
        with self.assertRaisesRegex(ValueError,'units differ'):model('a',x,wavelengths=torch.ones(8,device=self.device),coverage_metadata=small)

    def test_missing_index_normalized_and_nonphysical_adapter_fallback(self):
        for attention in ('wavelength','index','uniform'):
            model=self.model(adapter_attention_mode=attention).eval();x,wave,meta=self.inputs()
            with torch.no_grad():model.backbone.coverage_modulation.generator[2].bias.fill_(.7)
            indexed=replace(meta,wavelengths=tuple(np.linspace(0,1,8)),wavelength_unit='normalized',wavelength_quality='index_only')
            cases=[(wave,None),(torch.tensor(indexed.wavelengths,device=self.device,dtype=x.dtype),indexed)]
            if attention!='wavelength':cases.append((wave,meta))
            for actual_wave,item in cases:
                out=self.call(model,x,actual_wave,item,return_graph_info=True,return_diagnostics=True)
                self.assertEqual(out[3]['coverage']['usable_anchors'],0)
                self.assertIsNotNone(out[3]['coverage']['fallback_reason'])
            norm=replace(meta,wavelengths=tuple((w-400)/2100 for w in meta.wavelengths),wavelength_unit='normalized')
            nw=torch.tensor(norm.wavelengths,device=self.device,dtype=x.dtype)
            out=self.call(model,x,nw,norm,return_graph_info=True)
            self.assertEqual(out[3]['coverage']['usable_anchors'],0)
            if attention=='wavelength':
                out=self.call(model,x,nw,norm,return_graph_info=True,coverage_reference_nm=(400.,2500.))
                self.assertGreater(out[3]['coverage']['usable_anchors'],0)
                with self.assertRaises(ValueError):self.call(model,x,nw,norm,coverage_reference_nm=(400.,1100.))

    def test_modulation_precedes_denoise_and_preserves_vca_unsupported_channels(self):
        model=self.model(coverage_window=1.).eval();x,wave,meta=self.inputs();seen={};order=[]
        layer=model.backbone.coverage_modulation
        with torch.no_grad():layer.generator[2].bias.fill_(.7)
        def capture(m,args,out):
            order.append('coverage');seen.update(before=args[0].detach(),after=out.detach(),description=args[1])
        def denoise(m,args):order.append('denoise');seen['denoise']=args[0].detach()
        a=layer.register_forward_hook(capture);b=model.backbone.CNN_denoise[0].register_forward_pre_hook(denoise)
        try:self.call(model,x,wave,meta,return_graph_info=True)
        finally:a.remove();b.remove()
        self.assertEqual(order[:2],['coverage','denoise'])
        unused=~seen['description'].usable
        self.assertTrue(unused.any());self.assertTrue(torch.equal(seen['after'][...,unused],seen['before'][...,unused]))
        self.assertGreater(float((seen['before']-seen['after']).abs().max()),0)
        torch.testing.assert_close(seen['denoise'][0].permute(1,2,0),torch.cat([seen['after'],x[...,-2:]],-1),rtol=0,atol=0)

    def test_static_learned_injected_graph_interaction_and_m4_compatibility(self):
        from model.graph_state import GraphState
        x,wave,meta=self.inputs();q=torch.zeros(42,6,device=self.device);q[torch.arange(42),torch.arange(42)%6]=1
        a=torch.eye(6,device=self.device)
        for mode in ('static','learned','injected'):
            model=self.model(graph_mode='static' if mode=='static' else 'learned')
            kw={'Q':q,'A':a} if mode=='static' else {}
            if mode=='injected':kw['graph_state']=GraphState(q.to_sparse().coalesce(),a.to_sparse().coalesce(),torch.zeros(6,2,device=self.device),(6,7),(2,3))
            out=self.call(model,x,wave,meta,return_graph_info=True,**kw)
            self.assertIn('coverage',out[3]);self.assertTrue(all(p.shape==(42,3) for p in out[:3]))
        for extra in ({'interaction_mode':'gated'},{'region_residual_mode':'x_residual'}):
            model=self.model(**extra);self.assertEqual(self.call(model,x,wave,meta)[0].shape,(42,3))

    def test_checkpoint_strict_restore_old_default_and_enabled(self):
        x,wave,meta=self.inputs()
        with tempfile.TemporaryDirectory() as root:
            for mode in ('none','affine'):
                model=self.model(mode).eval()
                if mode=='affine':
                    with torch.no_grad():model.backbone.coverage_modulation.generator[2].bias.fill_(.2)
                path=Path(root)/(mode+'.pt');save_model_checkpoint(model,path)
                original=Path(str(path)+'.json').read_bytes();restored=load_model_checkpoint(path,self.device)
                self.assertEqual(original,Path(str(path)+'.json').read_bytes());self.assertEqual(model.export_config(),restored.export_config())
                with torch.no_grad():
                    for a,b in zip(self.call(model,x,wave,meta),self.call(restored,x,wave,meta)):
                        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            raw=Path(root)/'raw.pt';torch.save(model.state_dict(),raw)
            with self.assertRaises(RuntimeError):load_model_checkpoint(raw,self.device,config=self.model('none').export_config()['backbone_kwargs'])
            restored=load_model_checkpoint(raw,self.device,config=model.export_config()['backbone_kwargs'])
            self.assertEqual(state_hash(model),state_hash(restored))

    def test_source_target_same_shape_semantic_mismatch_and_frozen_enrollment(self):
        from test_transfer_training import fixture
        _,base,(source,_,target,_,_)=fixture();base['model']['coverage_mode']='affine'
        model,_=build_model(base,source.dataset,9,64,self.device)
        for method in ('source_pretrained_head_only','source_pretrained_full_finetune'):
            candidate,_=build_target(base,target,method,model,model_seed=64,head_seed=66,device=self.device)
            self.assertEqual(state_hash(candidate.backbone),state_hash(model.backbone))
            self.assertTrue(all(p.requires_grad==(method!='source_pretrained_head_only') for p in candidate.backbone.coverage_modulation.parameters()))
        for key,value in [('coverage_mode','none'),('coverage_strength',.2),('coverage_sigma',.2),('wavelength_max',2600),('adapter_attention_mode','index')]:
            bad=copy.deepcopy(base);bad['model'][key]=value
            with self.assertRaisesRegex(ValueError,'coverage settings'):build_target(bad,target,'source_pretrained_full_finetune',model,model_seed=64,head_seed=66,device=self.device)

    def development(self):
        from test_observation_training import ObservationTrainingTests
        fixture=ObservationTrainingTests();fixture.setUp()
        base=copy.deepcopy(fixture.base);base['model']['coverage_mode']='affine'
        return fixture.scene,fixture.record,base

    def test_real_subset_training_registered_gradients_trace_restore_and_tamper(self):
        scene,record,base=self.development()
        policy=ObservationPolicy.freeze(scene,record,updates=3,training_configs=[{'mode':'none'},
            {'mode':'range_crop','lower':400,'upper':750,'unit':'nm'},
            {'mode':'contiguous_gap','lower':650,'upper':850,'unit':'nm'}],train_seed=64,validation_seed=512)
        retained={};original=dev.build_target;seen=[]
        def build(*args,**kwargs):
            model,cfg=original(*args,**kwargs);layer=model.backbone.coverage_modulation
            retained.update(model=model,before={n:p.detach().clone() for n,p in layer.named_parameters()})
            layer.register_forward_hook(lambda m,args,out:seen.append((model.training,args[1].diagnostics.get('metadata_hash'))))
            return model,cfg
        with tempfile.TemporaryDirectory() as root,patch.object(dev,'build_target',side_effect=build):
            folder=Path(root)/'run';result=dev.adapt(scene,record,base,folder,steps=3,seed=64,device=self.device,observation_policy=policy)
            self.assertEqual([h for train,h in seen if train],[r['selected_metadata_hash'] for r in policy.to_dict()['training']])
            self.assertEqual(verify_run_artifacts(folder)['status'],'verified')
            runtime=result['runtime_config']['coverage'];self.assertEqual(runtime['optimized_parameters'],194)
            for n,p in retained['model'].backbone.coverage_modulation.named_parameters():
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float((p-retained['before'][n]).abs().max()),0)
            restored=load_model_checkpoint(folder/'model.pt',self.device)
            metric=dev.evaluate_validation(restored,scene,record,observation_policy=policy)
            self.assertAlmostEqual(metric['nll'],min(r['nll'] for r in result['validation']),places=5)
            self.assertFalse(result['test_evaluated'])
            pristine=(folder/'model.pt.json').read_bytes()
            for mutation in ('trace','metadata','signature','count'):
                r=copy.deepcopy(result);s=json.loads((folder/'model.pt.json').read_text())
                if mutation=='trace':r['trace'][0]['coverage_metadata_hash']='wrong'
                else:
                    c=r['runtime_config']['coverage']
                    if mutation=='metadata':c['input_metadata']['source_id']='wrong'
                    if mutation=='signature':c['settings']['coverage_strength']=.2
                    if mutation=='count':c['optimized_parameters']=0
                    r['runtime_config_hash']=digest(r['runtime_config'])
                    for key in ('runtime_config','runtime_config_hash'):s['training_config'][key]=r[key]
                    (folder/'model.pt.json').write_text(json.dumps(s));r['sidecar_sha256']=hash_file(folder/'model.pt.json')
                (folder/'result.json').write_text(json.dumps(r))
                with self.assertRaisesRegex(ValueError,'E1'):verify_run_artifacts(folder)
                (folder/'result.json').write_text(json.dumps(result))
                (folder/'model.pt.json').write_bytes(pristine)

    def test_spatial_loader_metadata_optimization_resume_and_changed_identity(self):
        from test_spatial_training import fixture
        cfg,cube,labels=fixture();value=cfg.to_dict();value['device']=self.device;value['model']['coverage_mode']='affine';cfg=SpatialConfig(value)
        meta=BandMetadata('spatial',tuple(range(cube.shape[-1])),tuple(np.linspace(400,1100,cube.shape[-1])),'nm','approximate',(True,)*cube.shape[-1])
        wave=np.asarray(meta.wavelengths,dtype=np.float32);retained={}
        def init(model):retained.update(model=model,before=state_hash(model.backbone.coverage_modulation))
        loader=lambda:(cube,labels,{'cube':tensor_hash(cube),'labels':tensor_hash(labels)},wave,meta)
        with tempfile.TemporaryDirectory() as root:
            result=execute(cfg,root,code_snapshot(),loader,model_initializer=init)
            self.assertEqual(result['status'],'completed');self.assertNotEqual(retained['before'],state_hash(retained['model'].backbone.coverage_modulation))
            with patch.object(torch.optim,'Adam',side_effect=AssertionError('completed run must skip')):
                self.assertEqual(execute(cfg,root,code_snapshot(),loader,True)['status'],'skipped')
            changed=lambda:(cube,labels,loader()[2],wave,replace(meta,wavelength_quality='verified'))
            with self.assertRaises(ValueError):execute(cfg,root,code_snapshot(),changed,True)

    def test_source_and_target_production_stage_records(self):
        from test_transfer_training import fixture
        from transfer.g3_training import train_source,adapt_target
        config,base,(source,labels,target,_,support)=fixture();base['model']['coverage_mode']='affine'
        with tempfile.TemporaryDirectory() as root:
            model,result=train_source(source,labels,base,Path(root)/'source',steps=1,val_every=1,
                model_seed=64,sampler_seed=68,device=self.device,task_hash='fixture')
            self.assertFalse(result['source_test_evaluated'])
            result=adapt_target(target,support,base,Path(root)/'target',method='source_pretrained_head_only',source=model,
                steps=1,model_seed=64,head_seed=66,sampler_seed=68,device=self.device,task_hash='fixture')
            self.assertEqual(result['runtime_config']['coverage']['optimized_parameters'],0)
            self.assertEqual(verify_run_artifacts(Path(root)/'target')['status'],'verified')

    def test_legacy_metadata_loader_does_not_guess_physical_quality(self):
        from loadData.data_read import read_band_metadata
        self.assertEqual(read_band_metadata('xuzhou',8).wavelength_quality,'approximate')
        self.assertEqual(read_band_metadata('unknown',8).wavelength_quality,'index_only')

    def test_legacy_trainer_optimizes_and_resumes_with_metadata_identity(self):
        from test_interaction_modules import synthetic_item
        from test_research_c0 import tiny_config
        from run_research_ablation import execute_one
        import train_vca_multidataset as train
        value=tiny_config().to_dict();value['device']=self.device
        value['model']['coverage_mode']='affine';cfg=ResearchConfig.from_dict(value)
        item=synthetic_item()
        for k,v in item.items():
            if isinstance(v,torch.Tensor):item[k]=v.to(self.device)
        item['coverage_metadata']=BandMetadata('legacy',tuple(range(12)),tuple(item['wavelengths'].cpu().tolist()),'nm','approximate',(True,)*12)
        with tempfile.TemporaryDirectory() as root,patch.object(train,'prepare_dataset',return_value=item):
            result=execute_one(cfg,root,code_snapshot())
            self.assertEqual(result['status'],'completed')
            folder=Path(root)/result['run_id']/'attempts'/result['attempt']
            model=load_model_checkpoint(folder/'model.pt',self.device)
            self.assertGreater(float(model.backbone.coverage_modulation.generator[2].weight.abs().max()),0)
            self.assertIn('coverage_input_metadata',model.training_config)
            with patch.object(torch.optim,'Adam',side_effect=AssertionError('must skip')):
                self.assertEqual(execute_one(cfg,root,code_snapshot(),resume=True)['status'],'skipped')
            item['coverage_metadata']=replace(item['coverage_metadata'],wavelength_quality='verified')
            with self.assertRaises(ValueError):execute_one(cfg,root,code_snapshot(),resume=True)

    def test_dynamic_head_and_legacy_single_wrapper_preserve_device_and_mode(self):
        from model.VCASAHGCN import VCASAHGCN
        model=self.model().double().eval();model.add_dataset_head('b',4)
        self.assertFalse(model.training)
        self.assertTrue(all(p.device.type==torch.device(self.device).type and p.dtype==torch.float64 for p in model.parameters()))
        self.assertFalse(model.backbone.coverage_modulation.training)
        x,wave,meta=self.inputs();q=torch.nn.functional.one_hot(torch.arange(42)%6,6).float().to(self.device);a=torch.eye(6,device=self.device)
        wrapper=VCASAHGCN(6,7,8,3,q,a,.8,wavelengths=wave,latent_channels=8,adapter_embed_dim=8,
            norm_type='group_norm',coverage_mode='affine').to(self.device).eval()
        out=wrapper(x,coverage_metadata=meta)
        self.assertTrue(all(p.shape==(42,3) for p in out))


if __name__=='__main__':unittest.main()
