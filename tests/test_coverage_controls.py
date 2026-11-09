"""E1-C negative controls, unchanged continuous path and checkpoint semantics."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
import numpy as np
import torch

from loadData.observation_sampling import BandMetadata
from model.coverage_controls import generator_features
from model.coverage_config import CONDITIONS,coverage_signature,coverage_forward_kwargs
from model.observation_coverage import CoverageModulation,CoverageConfig,describe
from model.research_config import ResearchConfig
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from train_vca_multidataset import build_parser
from transfer.g3_training import build_model,build_target

ROOT=Path(__file__).resolve().parents[1]
BASE='2d9b7b20b1c970a43a9f5829aaf239d66c34fe1b'


def old_module(path):
    module=types.ModuleType('e1c_old')
    text=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+path],text=True)
    exec(compile(text,path,'exec'),module.__dict__);return module


class CoverageControlTests(unittest.TestCase):
    def setUp(self):
        self.device=os.environ.get('E1_DEVICE','cpu')
        self.bands=BandMetadata('control',tuple(range(8)),tuple(np.linspace(400,1700,8)),'nm','approximate',(True,)*8)
        self.anchors=torch.linspace(0,1,8,device=self.device,dtype=torch.float64)
        self.desc=describe(self.bands,self.anchors,CoverageConfig())

    def test_anchor_basis_is_observation_independent_but_varies_by_anchor(self):
        x=generator_features(self.desc,'anchor_only',0)
        other=replace(self.desc,features=self.desc.features+2,diagnostics={**self.desc.diagnostics,'metadata_hash':'wrong'})
        other.features[:,0].copy_(self.anchors)
        torch.testing.assert_close(x,generator_features(other,'anchor_only',512),rtol=0,atol=0)
        self.assertEqual(x.shape,(8,9));self.assertGreater(float(x.std(0).max()),0)
        self.assertFalse(x.requires_grad)

    def test_binary_uses_actual_vca_bits_and_keeps_physical_description_intact(self):
        saved=self.desc.features.clone();valid=torch.tensor([True,False]*4,device=self.device)
        x=generator_features(self.desc,'binary',0,valid)
        torch.testing.assert_close(x[:,1:5],valid[:,None].expand(-1,4).double(),rtol=0,atol=0)
        torch.testing.assert_close(x[:,[0,5,6,7,8]],saved[:,[0,5,6,7,8]],rtol=0,atol=0)
        self.assertTrue(torch.equal(saved,self.desc.features))
        for wrong in (None,valid.float(),valid[:2]):
            with self.assertRaises(ValueError):generator_features(self.desc,'binary',0,wrong)

    def test_shuffle_joint_geometry_is_stable_rng_neutral_and_preserves_quality(self):
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state() if self.device.startswith('cuda') else None
        saved=self.desc.features.clone();x=generator_features(self.desc,'shuffled',0);y=generator_features(self.desc,'shuffled',0)
        self.assertTrue(torch.equal(x,y));self.assertFalse(torch.equal(x[:,1:5],saved[:,1:5]))
        self.assertEqual(sorted(map(tuple,x[:,1:5].cpu().tolist())),sorted(map(tuple,saved[:,1:5].cpu().tolist())))
        self.assertTrue(torch.equal(x[:,[0,5,6,7,8]],saved[:,[0,5,6,7,8]]))
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        if cuda is not None:self.assertTrue(torch.equal(cuda,torch.cuda.get_rng_state()))
        self.assertTrue(torch.equal(saved,self.desc.features))
        with self.assertRaises(ValueError):generator_features(replace(self.desc,diagnostics={}), 'shuffled',0)

    def test_continuous_exact_actual_e1b_operator_outputs_gradients_and_export(self):
        old=old_module('model/observation_coverage.py').CoverageModulation(8,mode='affine').to(self.device).double()
        new=CoverageModulation(8,mode='affine').to(self.device).double()
        self.assertEqual(old.export_config(),new.export_config())
        with torch.no_grad():
            old.generator[2].weight.fill_(.03);old.generator[2].bias.fill_(.02);new.load_state_dict(old.state_dict())
        a=torch.randn(5,8,device=self.device,dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_()
        # Parent's dataclass has its own class identity; bind the same tensors.
        # Use the class expected by old.forward, not a separately loaded copy.
        old_desc=old.forward.__globals__['CoverageDescription'](self.desc.features,self.desc.usable,self.desc.local_count,self.desc.max_gap,self.desc.diagnostics)
        left=old(a,old_desc);right=new(b,self.desc)
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        left.square().sum().backward();right.square().sum().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        for p,q in zip(old.parameters(),new.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_all_controls_equal_parameter_init_learn_and_protect_unsupported(self):
        previous=None
        for condition in CONDITIONS:
            layer=CoverageModulation(8,mode='affine',condition=condition).to(self.device).double()
            self.assertEqual(sum(p.numel() for p in layer.parameters()),194)
            initial={n:p.detach().clone() for n,p in layer.named_parameters()}
            if previous is not None:
                for n in initial:self.assertTrue(torch.equal(initial[n],previous[n]))
            previous=initial;x=torch.randn(5,8,device=self.device,dtype=torch.float64)
            kwargs={'anchor_valid':self.desc.usable} if condition=='binary' else {}
            self.assertTrue(torch.equal(layer(x,self.desc,**kwargs),x))
            optimizer=torch.optim.Adam(layer.parameters(),lr=.01)
            for i in range(3):
                optimizer.zero_grad();layer(x,self.desc,**kwargs).square().mean().backward();optimizer.step()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))
            self.assertTrue(any(not torch.equal(p,initial[n]) for n,p in layer.named_parameters()))
            with torch.no_grad():layer.generator[2].bias.fill_(1.)
            out=layer(x,self.desc,**kwargs);self.assertTrue(torch.equal(out[:,~self.desc.usable],x[:,~self.desc.usable]))
            missing=describe(None,self.anchors);self.assertIs(layer(x,missing,**kwargs),x)

    def test_config_and_cli_hashes_keep_accepted_none_and_affine_canonical_forms(self):
        parent=old_module('model/research_config.py').ResearchConfig
        for model in ({},{'coverage_mode':'affine'}):
            old=parent.from_dict({'model':model});new=ResearchConfig.from_dict({'model':model})
            self.assertEqual(old.to_dict(),new.to_dict());self.assertEqual(old.config_hash,new.config_hash)
            self.assertNotIn('coverage_condition',new.to_dict()['model']);self.assertNotIn('coverage_condition',coverage_signature(new.model_kwargs()))
        hashes=[]
        for c in CONDITIONS:
            args=build_parser().parse_args(['--coverage-mode','affine','--coverage-condition',c,'--device','cpu'])
            cfg=ResearchConfig.from_legacy(args);hashes.append(cfg.config_hash)
            self.assertEqual(ResearchConfig.from_legacy(cfg.trainer_args('/tmp/a.pt','/tmp/b.json')).to_dict(),cfg.to_dict())
        self.assertEqual(len(set(hashes)),4)
        with self.assertRaises(ValueError):ResearchConfig.from_dict({'model':{'coverage_condition':'unknown'}})

    def test_actual_backbone_routes_binary_bits_and_checkpoint_restores_each_control(self):
        x=torch.randn(6,7,10,device=self.device);wave=torch.tensor(self.bands.wavelengths,device=self.device,dtype=x.dtype)
        with tempfile.TemporaryDirectory() as d:
            for c in CONDITIONS:
                model=MultiDatasetVCASAHGCN({'a':3},graph_mode='learned',norm_type='group_norm',latent_channels=8,
                    adapter_embed_dim=8,coverage_mode='affine',coverage_condition=c).to(self.device).eval()
                with torch.no_grad():model.backbone.coverage_modulation.generator[2].bias.fill_(.2)
                out=model('a',x,wavelengths=wave,superpixel_scale=9,**coverage_forward_kwargs(model,self.bands))
                path=Path(d)/(c+'.pt');save_model_checkpoint(model,path);restored=load_model_checkpoint(path,self.device)
                self.assertEqual(restored.backbone.coverage_modulation.condition,c)
                again=restored('a',x,wavelengths=wave,superpixel_scale=9,**coverage_forward_kwargs(restored,self.bands))
                for a,b in zip(out,again):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_transfer_rejects_same_shape_different_condition(self):
        from test_transfer_training import fixture
        _,base,(source,_,target,_,_)=fixture();base['model'].update(coverage_mode='affine',coverage_condition='binary')
        model,_=build_model(base,source.dataset,9,64,self.device)
        base['model']['coverage_condition']='shuffled'
        with self.assertRaisesRegex(ValueError,'coverage settings'):build_target(base,target,'source_pretrained_full_finetune',model,model_seed=64,head_seed=66,device=self.device)


if __name__=='__main__':unittest.main()
