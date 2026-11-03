"""Population moments, protected domains and persisted BN policy acceptance."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.bn_calibration import (CONTEXT,POLICY,calibrate_denoise,observation_id,
                                  state_digest,verify_calibration_binding)
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from joint_baseline import NAMES,forward
from test_joint_baseline import tiny_items

torch.set_num_threads(1)


class BatchNormCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.device=os.environ.get('BNALIGN_DEVICE','cpu')
        self.items={n:{k:v.to(self.device) if torch.is_tensor(v) else v for k,v in item.items()}
                    for n,item in tiny_items().items()}
        torch.manual_seed(300)
        self.model=MultiDatasetVCASAHGCN({n:i['class_num'] for n,i in self.items.items()},
            latent_channels=8,adapter_embed_dim=4,norm_type='domain_bn').to(self.device).eval()
        with torch.no_grad():
            for block in self.model.backbone.CNN_denoise:
                for bn in block.norm.norm.norms.values():
                    bn.running_mean.fill_(4.);bn.running_var.fill_(2.)
        self.observations={n:{k:i[k] for k in ('net_input','wavelengths','fwhm')} for n,i in self.items.items()}

    def apply(self,model=None,observations=None,**kw):
        return calibrate_denoise(model or self.model,self.observations if observations is None else observations,
            mode=POLICY,context=CONTEXT,source_checkpoint_sha256='a'*64,**kw)

    def test_disabled_path_is_exact_identity_without_observation_access(self):
        before=state_digest(self.model)
        candidate,receipt=calibrate_denoise(self.model,{'bad':'never read'})
        self.assertIs(candidate,self.model);self.assertEqual(receipt['policy'],'none')
        self.assertEqual(state_digest(self.model),before)

    def test_matches_live_denoise_batch_statistics_and_second_layer_moments(self):
        candidate,_=self.apply()
        live=copy.deepcopy(self.model)
        for block in live.backbone.CNN_denoise:
            for bn in block.norm.norm.norms.values():bn.training=True;bn.track_running_stats=False
        with torch.no_grad():
            for n,item in self.items.items():
                reference=forward(live,n,item);actual=forward(candidate,n,item)
                for a,b in zip(reference,actual):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
                b=candidate.backbone;sp,extra=b._split_spectral_extra(item['net_input'],item['wavelengths'])
                latent=b.adapter(sp,wavelengths=item['wavelengths'],fwhm=item['fwhm'])
                noise=torch.cat((latent,extra),2).permute(2,0,1).unsqueeze(0)
                for block in b.CNN_denoise:
                    bn=block.norm.norm.norms[n];v,m=torch.var_mean(noise,dim=(0,2,3),unbiased=False)
                    torch.testing.assert_close(bn.running_mean,m,rtol=0,atol=0)
                    torch.testing.assert_close(bn.running_var,v,rtol=0,atol=0)
                    noise=block(noise,n)

    def test_only_selected_mean_variance_change_source_parameters_and_counters_unchanged(self):
        before=state_digest(self.model);rng=torch.get_rng_state().clone()
        one={NAMES[0]:self.observations[NAMES[0]]};candidate,r=self.apply(observations=one)
        self.assertEqual(state_digest(self.model),before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(len(r['calibrated_buffers']),4)
        for n,p in candidate.state_dict().items():
            if n not in r['calibrated_buffers']:self.assertTrue(torch.equal(p,self.model.state_dict()[n]),n)
        self.assertFalse(any(m.training for m in candidate.modules()))
        self.assertEqual(r['optimizer_updates'],0)
        with torch.no_grad():
            for n in NAMES[1:]:
                for a,b in zip(forward(candidate,n,self.items[n]),forward(self.model,n,self.items[n])):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_domain_order_and_repeated_calibration_do_not_mix_statistics(self):
        a,_=self.apply();b,_=self.apply(observations=dict(reversed(list(self.observations.items()))))
        c,_=self.apply(model=a)
        self.assertEqual(state_digest(a),state_digest(b));self.assertEqual(state_digest(a),state_digest(c))

    def test_fp64_strict_reference_with_identical_sparse_graph_coefficients(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from validate_bn_calibration import double_reference_check
        self.assertLess(double_reference_check(self.model,self.items,'a'*64),1e-10)

    def test_invalid_policy_context_hash_and_training_mode_rejected(self):
        for key,value in [('mode','unknown'),('context','strict_target_holdout'),('source_checkpoint_sha256','bad')]:
            kwargs=dict(mode=POLICY,context=CONTEXT,source_checkpoint_sha256='a'*64);kwargs[key]=value
            with self.assertRaises(ValueError):calibrate_denoise(self.model,self.observations,**kwargs)
        self.model.backbone.CNN_denoise[0].train()
        with self.assertRaises(ValueError):self.apply()

    def test_labels_nonfinite_unknown_domain_and_failure_leave_source_unchanged(self):
        before=state_digest(self.model)
        with self.assertRaises(ValueError):self.apply(observations={'unknown':self.observations[NAMES[0]]})
        for key,value in [('val_samples_gt',torch.ones(1)),('net_input',torch.full((6,6,9),float('nan')))]:
            bad=copy.deepcopy(self.observations);bad[NAMES[1]][key]=value
            with self.assertRaises(ValueError):self.apply(observations=bad)
        bad=copy.deepcopy(self.observations);bad[NAMES[1]]['wavelengths']=torch.arange(3.)
        with self.assertRaises(ValueError):self.apply(observations=bad)
        self.assertEqual(state_digest(self.model),before)

    def test_zero_variance_scene_is_finite_and_epsilon_preserved(self):
        observations=copy.deepcopy(self.observations)
        for item in observations.values():item['net_input'].fill_(1.)
        candidate,_=self.apply(observations=observations)
        for block,old in zip(candidate.backbone.CNN_denoise,self.model.backbone.CNN_denoise):
            for name,bn in block.norm.norm.norms.items():
                self.assertEqual(bn.eps,old.norm.norm.norms[name].eps)
                self.assertTrue(torch.isfinite(bn.running_var).all())
                self.assertTrue((bn.running_var>=0).all())

    def test_checkpoint_reload_and_scene_context_state_tamper_rejection(self):
        candidate,r=self.apply()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calibrated.pt';save_model_checkpoint(candidate,path)
            restored=load_model_checkpoint(path,device=self.device)
            self.assertEqual(verify_calibration_binding(restored,self.observations,context=CONTEXT,source_checkpoint_sha256='a'*64),r)
            self.assertEqual(state_digest(restored),state_digest(candidate))
            with torch.no_grad():
                for n in NAMES:
                    for a,b in zip(forward(restored,n,self.items[n]),forward(candidate,n,self.items[n])):
                        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            changed=copy.deepcopy(self.observations);changed[NAMES[0]]['net_input'][0,0,0]+=.1
            with self.assertRaises(ValueError):verify_calibration_binding(restored,changed,context=CONTEXT,source_checkpoint_sha256='a'*64)
            with self.assertRaises(ValueError):verify_calibration_binding(restored,self.observations,context='inductive',source_checkpoint_sha256='a'*64)
            with self.assertRaises(ValueError):verify_calibration_binding(restored,self.observations,context=CONTEXT,source_checkpoint_sha256='b'*64)
            with torch.no_grad():next(restored.parameters()).add_(.01)
            with self.assertRaises(ValueError):verify_calibration_binding(restored,self.observations,context=CONTEXT,source_checkpoint_sha256='a'*64)


if __name__=='__main__':unittest.main()
