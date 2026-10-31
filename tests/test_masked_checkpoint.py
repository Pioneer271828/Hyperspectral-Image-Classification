"""Continuous versus checkpoint/resume training, identities and recovery snapshots."""
import copy
import hashlib
import os
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from compression import MaskedWeights,save_masked_training,load_masked_training
from train_spatial import SpatialConfig


def seed():
    torch.manual_seed(31);np.random.seed(32);random.seed(33)


def optimizer_for(model,kind=torch.optim.Adam):
    named=dict(model.named_parameters())
    return kind([{'params':[named['0.weight']],'lr':.01},{'params':[named['0.bias']],'lr':.02}],weight_decay=.01)


class CheckpointTests(unittest.TestCase):
    def device(self):return os.environ.get('MASK_DEVICE','cpu')
    def fixture(self):
        return torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Dropout(.25)).to(self.device()).train()
    def step(self,model,opt,scheduler,sampler):
        x=torch.randn(5,3,device=self.device())*(random.random()+np.random.random()+sampler.random())
        opt.zero_grad(set_to_none=True);loss=model(x).square().mean();loss.backward();opt.step()
        scheduler.step();opt.zero_grad(set_to_none=True)
        return loss.detach().clone()

    def test_continuous_resume_exact_with_rng_groups_scheduler_and_recovery(self):
        seed();model=self.fixture();initial=copy.deepcopy(model.state_dict());opt=optimizer_for(model)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9);sampler=np.random.default_rng(34)
        keep=torch.ones_like(model[0].weight,dtype=torch.bool);keep[0,0]=False
        controller=MaskedWeights(model,{'0.weight':keep},recovery='restore_masked_values').attach()
        self.step(model,opt,scheduler,sampler)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'masked.pt'
            saved=save_masked_training(path,controller,opt,identity={'code':'fixture','data':'fixed'},progress={'step':1},scheduler=scheduler,sampler=sampler)
            # Saving must leave the legacy guard, active mask, weights and optimizer unchanged.
            with self.assertRaisesRegex(RuntimeError,'legacy state_dict'):model.state_dict()
            expected_loss=self.step(model,opt,scheduler,sampler)
            expected=controller._checkpoint_payload()[0];expected_optimizer=copy.deepcopy(opt.state_dict())
            expected_draw=(random.random(),np.random.random(),sampler.random(),torch.rand(3,device=self.device()))
            fresh=self.fixture();fresh_opt=optimizer_for(fresh);fresh_scheduler=torch.optim.lr_scheduler.StepLR(fresh_opt,step_size=1,gamma=.9)
            fresh_sampler=np.random.default_rng(0)
            restored,progress=load_masked_training(path,fresh,fresh_opt,expected_sha256=saved['sha256'],expected_identity={'code':'fixture','data':'fixed'},scheduler=fresh_scheduler,sampler=fresh_sampler)
            self.assertEqual(progress,{'step':1})
            actual_loss=self.step(fresh,fresh_opt,fresh_scheduler,fresh_sampler)
            torch.testing.assert_close(expected_loss,actual_loss,rtol=0,atol=0)
            for key,value in expected.items():torch.testing.assert_close(value,restored._checkpoint_payload()[0][key],rtol=0,atol=0)
            actual_optimizer=fresh_opt.state_dict()
            self.assertEqual(expected_optimizer['param_groups'],actual_optimizer['param_groups'])
            for key,state in expected_optimizer['state'].items():
                for name,value in state.items():torch.testing.assert_close(value,actual_optimizer['state'][key][name],rtol=0,atol=0)
            actual_draw=(random.random(),np.random.random(),fresh_sampler.random(),torch.rand(3,device=self.device()))
            self.assertEqual(expected_draw[:3],actual_draw[:3]);torch.testing.assert_close(expected_draw[3],actual_draw[3],rtol=0,atol=0)
            self.assertEqual(scheduler.state_dict(),fresh_scheduler.state_dict())
            controller.detach();restored.detach()
            torch.testing.assert_close(fresh[0].weight[~keep],initial['0.weight'][~keep],rtol=0,atol=0)

    def test_reject_identity_groups_config_hash_and_overwrite(self):
        seed();model=self.fixture();opt=optimizer_for(model);keep=torch.ones_like(model[0].weight,dtype=torch.bool)
        with MaskedWeights(model,{'0.weight':keep},recovery='keep_latent') as controller,tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'masked.pt';saved=save_masked_training(path,controller,opt,identity={'data':'fixed'},progress={'step':0})
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):save_masked_training(path,controller,opt,identity={'data':'fixed'},progress={'step':0})
            self.assertEqual(original,path.read_bytes())
            for bad in ('identity','groups','config','hash'):
                fresh=self.fixture();before=copy.deepcopy(fresh.state_dict());fresh_opt=optimizer_for(fresh)
                identity={'data':'wrong'} if bad=='identity' else {'data':'fixed'}
                if bad=='groups':fresh_opt.param_groups.reverse()
                if bad=='config':fresh[1].p=.5
                with self.assertRaises(ValueError):
                    load_masked_training(path,fresh,fresh_opt,expected_sha256='0'*64 if bad=='hash' else saved['sha256'],expected_identity=identity)
                for key,value in before.items():torch.testing.assert_close(value,fresh.state_dict()[key],rtol=0,atol=0)

    def test_adamw_and_sgd_state_and_pending_gradient_rejection(self):
        for kind in (torch.optim.AdamW,torch.optim.SGD):
            seed();model=self.fixture();opt=optimizer_for(model,kind)
            with MaskedWeights(model,{'0.weight':torch.zeros_like(model[0].weight,dtype=torch.bool)},recovery='keep_latent') as controller,tempfile.TemporaryDirectory() as directory:
                model(torch.ones(2,3,device=self.device())).sum().backward()
                with self.assertRaisesRegex(ValueError,'boundary'):save_masked_training(Path(directory)/'bad.pt',controller,opt,identity={'x':1},progress={'step':0})
                opt.step();opt.zero_grad(set_to_none=True)
                path=Path(directory)/'ok.pt';saved=save_masked_training(path,controller,opt,identity={'x':1},progress={'step':1})
                fresh=self.fixture();fresh_opt=optimizer_for(fresh,kind)
                restored,_=load_masked_training(path,fresh,fresh_opt,expected_sha256=saved['sha256'],expected_identity={'x':1})
                self.assertEqual(fresh[0].weight.count_nonzero().item(),0)
                self.assertEqual(opt.state_dict()['param_groups'],fresh_opt.state_dict()['param_groups']);restored.detach()

    def test_actual_backbone_resume_and_plateau_scheduler(self):
        device=self.device();torch.set_num_threads(1);seed()
        config=SpatialConfig({'name':'masked_checkpoint_fixture','device':device,'dataset':'paviaU',
            'model':{'graph_mode':'learned','latent_channels':8,'adapter_embed_dim':8,'norm_type':'group_norm','adapter_checkpoint_chunks':False},
            'training':{'epochs':0,'steps_per_epoch':2}})
        model=config.research.build_model({'paviaU':9}).to(device).train();model.training_config=config.to_dict()
        opt=torch.optim.Adam(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=0)
        keep=torch.ones_like(model.backbone.CNN_Branch[0].point_conv.weight,dtype=torch.bool);keep[0]=False
        masks={'backbone.CNN_Branch.0.point_conv.weight':keep}
        x=torch.randn(12,12,8,device=device);wave=torch.linspace(400,900,6,device=device)
        def step(m,o,s):
            outputs=m('paviaU',x,wavelengths=wave,superpixel_scale=50)
            loss=sum(-p[:,0].clamp_min(1e-8).log().mean() for p in outputs[:3]);loss.backward();o.step();o.zero_grad(set_to_none=True);s.step(1.)
            return loss.detach()
        with MaskedWeights(model,masks,recovery='keep_latent') as controller,tempfile.TemporaryDirectory() as directory:
            step(model,opt,scheduler)
            path=Path(directory)/'actual.pt';saved=save_masked_training(path,controller,opt,identity={'fixture':True},progress={'step':1},scheduler=scheduler)
            expected=step(model,opt,scheduler);state=controller._checkpoint_payload()[0]
            fresh=config.research.build_model({'paviaU':9}).to(device);fresh_opt=torch.optim.Adam(fresh.parameters(),lr=.001)
            fresh_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(fresh_opt,patience=0)
            restored,_=load_masked_training(path,fresh,fresh_opt,expected_sha256=saved['sha256'],expected_identity={'fixture':True},scheduler=fresh_scheduler)
            self.assertEqual(fresh.training_config,model.training_config);self.assertTrue(fresh.training)
            actual=step(fresh,fresh_opt,fresh_scheduler)
            tolerance=2e-4 if device=='cuda' else 0
            torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
            for key,value in state.items():torch.testing.assert_close(value,restored._checkpoint_payload()[0][key],rtol=tolerance,atol=tolerance)
            self.assertEqual(fresh.backbone.CNN_Branch[0].point_conv.weight[~keep].count_nonzero().item(),0)
            restored.detach()

if __name__=='__main__':unittest.main()
