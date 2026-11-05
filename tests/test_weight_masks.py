"""Fixed-mask behavior under training, existing optimizer state, and recovery."""
import copy
import io
import os
import unittest
import torch
from torch.nn.utils import parametrize
from compression import MaskedWeights
from train_spatial import SpatialConfig

class WeightMaskTests(unittest.TestCase):
    def device(self):return os.environ.get('MASK_DEVICE','cpu')

    def test_all_ones_matches_unmasked_adam_and_identity(self):
        torch.manual_seed(3);device=self.device()
        first=torch.nn.Sequential(torch.nn.Linear(3,2)).to(device);second=copy.deepcopy(first)
        x=torch.ones(4,3,device=device);a=torch.optim.Adam(first.parameters(),lr=.01);b=torch.optim.Adam(second.parameters(),lr=.01)
        parameter=second[0].weight
        with MaskedWeights(second,{'0.weight':torch.ones_like(parameter,dtype=torch.bool)},recovery='keep_latent'):
            self.assertIs(second[0].parametrizations.weight.original,parameter)
            self.assertTrue(any(p is parameter for g in b.param_groups for p in g['params']))
            for _ in range(3):
                for model,opt in ((first,a),(second,b)):
                    opt.zero_grad();model(x).square().mean().backward();opt.step()
                torch.testing.assert_close(first(x),second(x),rtol=0,atol=0)
        self.assertIs(second[0].weight,parameter)
        for name,value in first.state_dict().items():torch.testing.assert_close(value,second.state_dict()[name],rtol=0,atol=0)

    def test_hot_adamw_latent_drift_and_persistent_effective_zero(self):
        torch.manual_seed(4);device=self.device()
        model=torch.nn.Linear(3,2).to(device);x=torch.ones(4,3,device=device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.1)
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
        before=model.weight.detach().clone();keep=torch.ones_like(before,dtype=torch.bool);keep[0,0]=False
        original=model.weight
        with MaskedWeights(model,{'weight':keep},recovery='keep_latent'):
            for _ in range(4):
                optimizer.zero_grad();model(x).square().mean().backward()
                self.assertEqual(original.grad[0,0].item(),0)
                optimizer.step();self.assertEqual(model.weight[0,0].item(),0)
            self.assertNotEqual(original[0,0].item(),before[0,0].item())
            latent=original.detach().clone()
        torch.testing.assert_close(model.weight,latent,rtol=0,atol=0)
        self.assertIn(original,optimizer.state)

    def test_restore_masked_values_keeps_unmasked_learning_and_bias(self):
        device=self.device();torch.manual_seed(5)
        model=torch.nn.Linear(3,2).to(device);before=model.weight.detach().clone()
        keep=torch.ones_like(before,dtype=torch.bool);keep[0,:]=False
        optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
        with MaskedWeights(model,{'weight':keep},recovery='restore_masked_values'):
            self.assertEqual(model.weight[0].count_nonzero().item(),0)
            torch.testing.assert_close(model(torch.zeros(1,3,device=device)),model.bias[None,:])
            for _ in range(3):
                optimizer.zero_grad();model(torch.ones(2,3,device=device)).square().mean().backward();optimizer.step()
            updated=model.parametrizations.weight.original.detach().clone()
        torch.testing.assert_close(model.weight[~keep],before[~keep],rtol=0,atol=0)
        torch.testing.assert_close(model.weight[keep],updated[keep],rtol=0,atol=0)
        self.assertFalse(torch.equal(model.weight[keep],before[keep]))

    def test_mask_copy_plan_roundtrip_exception_and_reentry(self):
        model=torch.nn.Linear(3,2).to(self.device());before=model.weight.detach().clone()
        keep=torch.zeros_like(before,dtype=torch.bool)
        controller=MaskedWeights(model,{'weight':keep},recovery='restore_masked_values');keep.fill_(True)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with controller:
                self.assertEqual(model.weight.count_nonzero().item(),0)
                with self.assertRaisesRegex(RuntimeError,'legacy state_dict'):model.state_dict()
                with self.assertRaises(RuntimeError):controller.attach()
                raise RuntimeError('injected')
        self.assertFalse(parametrize.is_parametrized(model));torch.testing.assert_close(model.weight,before)
        stream=io.BytesIO();torch.save(controller.mask_plan(),stream);stream.seek(0)
        restored=MaskedWeights.from_plan(model,torch.load(stream,weights_only=True),recovery='keep_latent')
        with restored:self.assertEqual(model.weight.count_nonzero().item(),0)
        with controller:self.assertEqual(model.weight.count_nonzero().item(),0)

    def test_reject_invalid_shared_or_existing_before_mutation(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Linear(2,1))
        for mask in (torch.ones(2,3),torch.ones(1,dtype=torch.bool)):
            with self.assertRaises(ValueError):
                with MaskedWeights(model,{'0.weight':mask},recovery='keep_latent'):pass
        with self.assertRaises(ValueError):
            MaskedWeights(model,{'0.weight':torch.ones(2,3,dtype=torch.bool)},recovery='bad')
        with self.assertRaises(ValueError):
            with MaskedWeights(model,{'0.weight':torch.ones(2,3,dtype=torch.bool),'1.bias':torch.ones(1,dtype=torch.bool)},recovery='keep_latent'):pass
        self.assertFalse(parametrize.is_parametrized(model[0]))
        model[1].weight=model[0].weight
        with self.assertRaisesRegex(ValueError,'Shared'):
            with MaskedWeights(model,{'0.weight':torch.ones(2,3,dtype=torch.bool)},recovery='keep_latent'):pass
        single=torch.nn.Linear(3,2)
        with MaskedWeights(single,{'weight':torch.ones(2,3,dtype=torch.bool)},recovery='keep_latent'):
            with self.assertRaisesRegex(ValueError,'overlapping'):
                with MaskedWeights(single,{'weight':torch.ones(2,3,dtype=torch.bool)},recovery='keep_latent'):pass

    def test_actual_backbone_short_training_and_restored_export(self):
        device=self.device();torch.set_num_threads(1);torch.manual_seed(8)
        config=SpatialConfig({'name':'mask_fixture','device':device,'dataset':'paviaU',
            'model':{'graph_mode':'learned','latent_channels':8,'adapter_embed_dim':8,'norm_type':'group_norm','adapter_checkpoint_chunks':False},
            'training':{'epochs':0,'steps_per_epoch':2}})
        model=config.research.build_model({'paviaU':9}).to(device).train()
        layer=model.backbone.CNN_Branch[0].point_conv;before=layer.weight.detach().clone()
        keep=torch.ones_like(before,dtype=torch.bool);keep[0]=False
        x=torch.randn(12,12,8,device=device);wave=torch.linspace(400,900,6,device=device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        parameter=layer.weight
        with MaskedWeights(model,{'backbone.CNN_Branch.0.point_conv.weight':keep},recovery='restore_masked_values'):
            for _ in range(3):
                optimizer.zero_grad();outputs=model('paviaU',x,wavelengths=wave,superpixel_scale=50)
                loss=sum(-p[:,0].clamp_min(1e-8).log().mean() for p in outputs[:3]);self.assertTrue(torch.isfinite(loss))
                loss.backward();self.assertEqual(parameter.grad[~keep].count_nonzero().item(),0)
                optimizer.step();self.assertEqual(layer.weight[~keep].count_nonzero().item(),0)
            self.assertIs(layer.parametrizations.weight.original,parameter)
        self.assertIs(layer.weight,parameter);torch.testing.assert_close(layer.weight[~keep],before[~keep],rtol=0,atol=0)
        self.assertFalse(any(parametrize.is_parametrized(m) for m in model.modules()))
        # The original model/checkpoint parameter naming contract is recovered after detaching.
        self.assertFalse(any('parametrizations' in k for k in model.state_dict()))
        from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';save_model_checkpoint(model,path)
            restored=load_model_checkpoint(path,device)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()

