"""E1-A geometry, provenance, exact bypass, learning and standalone lifecycle."""
from dataclasses import replace
import io
import math
import os
import unittest

import torch

from loadData.observation_sampling import BandMetadata
from model.observation_coverage import CoverageConfig, CoverageModulation, describe
from model.variable_channel_adapter import VariableChannelAdapter


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get('E1_DEVICE', 'cpu'))
        self.cfg = CoverageConfig(sigma=.1, window=.2)

    def tensor(self, values, **kwargs):
        return torch.tensor(values, device=self.device, dtype=torch.float64, **kwargs)

    def bands(self, positions=(0, .25, .5, .75, 1), **kwargs):
        base = dict(source_id='fixture', band_ids=tuple(range(len(positions))),
            wavelengths=tuple(400+2100*x for x in positions), wavelength_unit='nm',
            wavelength_quality='verified', valid=(True,)*len(positions))
        return BandMetadata(**{**base, **kwargs})

    def layer(self, k=3, **kwargs):
        return CoverageModulation(k, mode='affine', **kwargs).to(device=self.device, dtype=torch.float64)

    def test_hand_computed_distance_counts_gap_boundaries_kernel_and_flags(self):
        a = self.tensor([.125, .5, .9], requires_grad=True)
        d = describe(self.bands(), a, self.cfg)
        torch.testing.assert_close(d.features[:, 1], self.tensor([.125, 0, .1]))
        torch.testing.assert_close(d.features[:, 2], self.tensor([math.exp(-.78125), 1, math.exp(-.5)]))
        self.assertEqual(d.local_count.tolist(), [2, 1, 2])
        torch.testing.assert_close(d.max_gap, self.tensor([.25, .2, .25]))
        torch.testing.assert_close(d.features[:, 4], self.tensor([.25/.325, .5, .25/.3]))
        torch.testing.assert_close(d.features[:, 5:], self.tensor([[1, 1, 0, 0]]*3))
        self.assertFalse(d.features.requires_grad); self.assertIsNone(a.grad)

    def test_global_reference_distinguishes_same_count_crops(self):
        a = self.tensor([0, .5, 1])
        full = describe(self.bands((0, .5, 1)), a, self.cfg)
        crop = describe(self.bands((0, .25, .5)), a, self.cfg)
        self.assertEqual(full.usable.tolist(), [True]*3)
        self.assertEqual(crop.usable.tolist(), [True, True, False])
        self.assertEqual(float(crop.features[-1, 1]), .5)
        self.assertFalse(torch.equal(full.features, crop.features))

    def test_permutation_duplicates_masks_and_actual_subset_agree(self):
        a = self.tensor([0, .5, 1]); b = self.bands()
        perm = BandMetadata(b.source_id, b.band_ids[::-1], b.wavelengths[::-1], b.wavelength_unit,
                            b.wavelength_quality, b.valid)
        torch.testing.assert_close(describe(b, a, self.cfg).features, describe(perm, a, self.cfg).features, rtol=0, atol=0)
        mask = [True, False, True, False, True]
        selected = describe(b, a, self.cfg, band_mask=mask)
        torch.testing.assert_close(selected.features, describe(b.subset([0, 2, 4]), a, self.cfg).features, rtol=0, atol=0)
        invalid = replace(b, valid=tuple(mask))
        torch.testing.assert_close(selected.features, describe(invalid, a, self.cfg).features, rtol=0, atol=0)
        dup = describe(self.bands((.5, .5)), a, self.cfg)
        single = describe(self.bands((.5,)), a, self.cfg)
        self.assertEqual(dup.local_count.tolist(), [0, 2, 0])
        torch.testing.assert_close(dup.max_gap, single.max_gap, rtol=0, atol=0)

    def test_nm_normalized_explicit_reference_and_quality(self):
        a = self.tensor([0, .5, 1]); b = self.bands()
        norm = replace(b, wavelengths=(0, .25, .5, .75, 1), wavelength_unit='normalized')
        d = describe(norm, a, self.cfg, normalized_reference_nm=(400, 2500))
        torch.testing.assert_close(d.features, describe(b, a, self.cfg).features, rtol=0, atol=0)
        unknown = describe(norm, a, self.cfg)
        self.assertEqual(unknown.diagnostics['fallback_reason'], 'unresolved_normalized_reference')
        self.assertFalse(unknown.usable.any())
        approx = describe(replace(b, wavelength_quality='approximate'), a, self.cfg)
        self.assertEqual(approx.features[:, 6].tolist(), [0]*3)
        self.assertEqual(approx.features[:, 7].tolist(), [1]*3)

    def test_fwhm_presence_does_not_change_design_geometry(self):
        a = self.tensor([0, .5, 1]); b = self.bands()
        without = describe(b, a, self.cfg)
        wide = describe(replace(b, fwhm=(1000.,)*5, fwhm_unit='nm'), a, self.cfg)
        torch.testing.assert_close(without.features[:, :8], wide.features[:, :8], rtol=0, atol=0)
        self.assertEqual(wide.features[:, 8].tolist(), [1]*3)
        self.assertFalse(wide.diagnostics['fwhm_used_in_geometry'])

    def test_missing_index_empty_and_out_of_range_are_finite_bypass(self):
        a = self.tensor([0, .5, 1]); index = self.bands(wavelengths=(0, .25, .5, .75, 1),
            wavelength_unit='normalized', wavelength_quality='index_only')
        descriptors = [describe(None, a, self.cfg), describe(index, a, self.cfg),
            describe(self.bands(), a, self.cfg, band_mask=[False]*5),
            describe(self.bands((2, 3)), a, self.cfg)]
        layer = self.layer(); x = self.tensor([[2, 0, -1]])
        with torch.no_grad(): layer.generator[-1].bias.fill_(2.)
        for d in descriptors:
            self.assertTrue(torch.isfinite(d.features).all()); self.assertFalse(d.usable.any())
            self.assertIs(layer(x, d), x)
        self.assertEqual(descriptors[-1].diagnostics['out_of_reference_count'], 2)
        self.assertIs(layer(x, None), x)

    def test_bad_geometry_metadata_and_configuration_fail(self):
        b = self.bands()
        for a in (self.tensor([]), self.tensor([float('nan')]), self.tensor([-1]), self.tensor([[0]]), torch.tensor([1])):
            with self.assertRaises(ValueError): describe(b, a, self.cfg)
        for kwargs in ({'sigma':0}, {'window':1e-20}, {'wavelength_min':2500}, {'sigma':float('nan')}):
            with self.assertRaises(ValueError): CoverageConfig(**kwargs)
        for kwargs in ({'band_mask':[1]*5}, {'band_mask':[True]}, {'normalized_reference_nm':(0, 1)}):
            with self.assertRaises(ValueError): describe(b, self.tensor([.5]), self.cfg, **kwargs)
        with self.assertRaises(ValueError): describe({}, self.tensor([.5]), self.cfg)
        with self.assertRaises(ValueError): describe(None, self.tensor([.5]), self.cfg, band_mask=[True])

    def test_descriptors_have_no_cache_or_input_alias(self):
        a = self.tensor([0, .5, 1]); saved = a.clone(); b = self.bands()
        first = describe(b, a, self.cfg); expected = first.features.clone()
        first.features.fill_(99)
        torch.testing.assert_close(describe(b, a, self.cfg).features, expected, rtol=0, atol=0)
        torch.testing.assert_close(a, saved, rtol=0, atol=0)
        a[1] = .8
        self.assertFalse(torch.equal(describe(b, a, self.cfg).features, expected))

    def test_off_and_zero_strength_are_exact_objects_and_initialization_preserves_rng(self):
        x = self.tensor([[1, 2, 3]])
        cpu = torch.get_rng_state().clone()
        gpu = torch.cuda.get_rng_state(self.device).clone() if self.device.type == 'cuda' else None
        off = CoverageModulation(3); on = self.layer(); zero = self.layer(strength=0)
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        if gpu is not None:self.assertTrue(torch.equal(gpu, torch.cuda.get_rng_state(self.device)))
        self.assertEqual(list(off.parameters()), []); self.assertEqual(dict(off.state_dict()), {})
        self.assertIs(off(x, 'ignored while disabled'), x); self.assertIs(zero(x, None), x)
        d = describe(self.bands(), self.tensor([0, .5, 1]), self.cfg)
        torch.testing.assert_close(on(x, d), x, rtol=0, atol=0)

    def test_analytic_affine_and_unsupported_zero_protection(self):
        layer = self.layer(k=2, strength=.5)
        d = describe(self.bands((.25,)), self.tensor([.25, .75]), CoverageConfig(window=.1))
        with torch.no_grad(): layer.generator[-1].bias.copy_(self.tensor([math.atanh(.2), math.atanh(-.3)]))
        x = self.tensor([[2, 0], [4, 9]], requires_grad=True)
        y = layer(x, d)
        torch.testing.assert_close(y, self.tensor([[2.05, 0], [4.25, 9]]))
        y.sum().backward();torch.testing.assert_close(x.grad, self.tensor([[1.1, 1], [1.1, 1]]))

    def test_gradients_actual_updates_and_conditional_dependence(self):
        d = describe(self.bands(), self.tensor([0, .5, 1]), self.cfg); layer = self.layer()
        x = self.tensor([[2, -1, 3], [1, 4, 2]], requires_grad=True)
        before = [p.detach().clone() for p in layer.parameters()]
        optimizer = torch.optim.SGD(layer.parameters(), lr=.05)
        gradients = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True); y = layer(x, d); (y+.5).square().mean().backward()
            gradients.append(float(layer.generator[0].weight.grad.abs().max()));optimizer.step()
        self.assertEqual(gradients[0], 0); self.assertGreater(gradients[1], 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in layer.parameters()))
        self.assertTrue(all(not torch.equal(a,p) for a,p in zip(before,layer.parameters())))
        altered = replace(d, features=d.features.clone());altered.features[:, 3] += 2
        self.assertGreater(float((layer(x,d)-layer(x,altered)).abs().max()), 0)

    def test_standalone_config_state_roundtrip_and_version_rejection(self):
        layer = self.layer(value_dim=2); d = describe(self.bands(),self.tensor([0,.5,1]),self.cfg)
        with torch.no_grad():layer.generator[-1].weight.fill_(.03)
        x = self.tensor([[1,2,3,4,5,6]])
        f = io.BytesIO();torch.save({'config':layer.export_config(),'state_dict':layer.state_dict()},f);f.seek(0)
        payload = torch.load(f,map_location=self.device,weights_only=True)
        restored = CoverageModulation(**payload['config']).to(device=self.device,dtype=x.dtype)
        restored.load_state_dict(payload['state_dict'],strict=True)
        torch.testing.assert_close(layer(x,d),restored(x,d),rtol=0,atol=0)
        self.assertEqual(self.cfg.export_config(),CoverageConfig(**self.cfg.export_config()).export_config())
        with self.assertRaises(ValueError):self.layer(descriptor_version=2)
        with self.assertRaises(RuntimeError):restored.load_state_dict({},strict=True)

    def test_layouts_noncontiguous_and_bad_modulation_inputs(self):
        layer = self.layer();d = describe(self.bands(),self.tensor([0,.5,1]),self.cfg)
        with torch.no_grad():layer.generator[-1].bias.fill_(.2)
        hwc = torch.arange(18,device=self.device,dtype=torch.float64).reshape(2,3,3).transpose(0,1)
        nchw = hwc.permute(2,0,1).unsqueeze(0)
        torch.testing.assert_close(layer(hwc,d),layer(nchw,d,channel_dim=1)[0].permute(1,2,0),rtol=0,atol=0)
        for x,desc,kw in ((hwc.float(),d,{}),(hwc,d,{'channel_dim':99}),
                         (hwc,replace(d,features=d.features[:2]),{}),(hwc,replace(d,usable=d.usable.float()),{})):
            with self.assertRaises(ValueError):layer(x,desc,**kw)
        with self.assertRaises(ValueError):CoverageModulation(3,mode='unknown')

    def test_vca_composition_variable_b_value_dim_and_anchor_support(self):
        adapter = VariableChannelAdapter(latent_channels=4,embed_dim=8,value_dim=2).to(device=self.device,dtype=torch.float64).eval()
        layer = self.layer(k=4,value_dim=2).eval(); base = self.bands()
        with torch.no_grad():
            for indices in ([0,1,2,3,4],[0,2,4],[2]):
                b=base.subset(indices);x=self.tensor([[[1.]*len(indices)]])
                latent,info=adapter(x,wavelengths=b.wavelengths,wavelength_unit='nm',return_attention=True)
                d=describe(b,info['anchors'],CoverageConfig())
                self.assertEqual(d.usable.tolist(),info['anchor_valid'].bool().tolist())
                torch.testing.assert_close(layer(latent,d),latent,rtol=0,atol=0)


if __name__ == '__main__':unittest.main()
