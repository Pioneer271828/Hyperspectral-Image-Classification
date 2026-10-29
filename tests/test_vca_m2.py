"""VCA observation contracts and cache lifecycle regressions (CPU)."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.variable_channel_adapter import VariableChannelAdapter


torch.set_num_threads(1)


class VCAContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.adapter = VariableChannelAdapter(latent_channels=8, embed_dim=8, locality_window=None)
        self.x = torch.randn(4, 5, 12)
        self.wave = torch.linspace(400, 2500, 12)

    def assert_fresh(self, **kwargs):
        with torch.no_grad():
            cached = self.adapter(self.x, self.wave, **kwargs)
            self.adapter.clear_attention_cache()
            fresh = self.adapter(self.x, self.wave, **kwargs)
        torch.testing.assert_close(cached, fresh, rtol=0, atol=0)

    def test_inference_reuses_attention_and_bounds_storage(self):
        self.adapter.eval()
        with patch.object(self.adapter, '_attention', wraps=self.adapter._attention) as compute:
            with torch.no_grad():
                first = self.adapter(self.x, self.wave)
                second = self.adapter(self.x + 1, self.wave)
                self.assertEqual(compute.call_count, 1)
                self.assertFalse(torch.equal(first, second))  # values are never cached
                self.adapter.cache_attention = False
                torch.testing.assert_close(first, self.adapter(self.x, self.wave), rtol=0, atol=0)
                self.adapter.cache_attention = True
                for bands in (1, 5, 12, 20, 103):
                    self.adapter(torch.randn(2, 3, bands), torch.linspace(400, 2500, bands))
                    self.assertEqual(len(self.adapter._eval_attention_cache), 1)

    def test_optimizer_train_validation_cycles(self):
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=.05)
        for _ in range(3):
            self.adapter.eval()
            with torch.no_grad():
                self.adapter(self.x, self.wave)
            self.adapter.train()
            optimizer.zero_grad()
            self.adapter(self.x, self.wave).square().mean().backward()
            optimizer.step()
            self.adapter.eval()
            self.assert_fresh()

    def test_eval_optimizer_and_buffer_updates_without_mode_switch(self):
        self.adapter.eval()
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=.05)
        with torch.no_grad():
            before = self.adapter(self.x, self.wave)
        self.adapter(self.x, self.wave).square().mean().backward()
        optimizer.step()
        self.assert_fresh()
        with torch.no_grad():
            after = self.adapter(self.x, self.wave)
            self.assertGreater((after - before).abs().max().item(), 1e-6)
            self.adapter.anchor_base.mul_(.8)
            self.adapter.frequencies.mul_(.9)
        self.assert_fresh()
        with torch.no_grad():
            self.adapter.anchor_offsets = nn.Parameter(torch.ones(8))
        self.assert_fresh()

    def test_warm_eval_autograd_matches_uncached_repeated_backward(self):
        self.adapter.eval()
        with torch.no_grad():
            self.adapter(self.x, self.wave)
        reference = copy.deepcopy(self.adapter)
        reference.cache_attention = False
        for _ in range(2):
            self.adapter.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            wave = self.wave.clone().requires_grad_()
            reference_wave = self.wave.clone().requires_grad_()
            self.adapter(self.x, wave).square().sum().backward()
            reference(self.x, reference_wave).square().sum().backward()
            torch.testing.assert_close(wave.grad, reference_wave.grad)
            for (name, parameter), (_, expected) in zip(self.adapter.named_parameters(), reference.named_parameters()):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.testing.assert_close(parameter.grad, expected.grad)
            self.assertGreater(self.adapter.query_encoder[0].weight.grad.abs().sum().item(), 0)
            self.assertGreater(self.adapter.anchor_offsets.grad.abs().sum().item(), 0)

    def test_nested_state_load_and_assign_into_warmed_adapter(self):
        parent = nn.Sequential(self.adapter).eval()
        donor = copy.deepcopy(parent)
        with torch.no_grad():
            donor[0].anchor_offsets.add_(1.)
        state = donor.state_dict()
        for assign in (False, True):
            with torch.no_grad():
                self.adapter(self.x, self.wave)
            parent.load_state_dict(state, assign=assign)
            with torch.no_grad():
                torch.testing.assert_close(self.adapter(self.x, self.wave), donor[0](self.x, self.wave))
            self.assert_fresh()
        self.assertFalse(any('cache' in name for name in parent.state_dict()))

    def test_metadata_config_dtype_and_diagnostic_mutation(self):
        self.adapter.eval()
        with torch.no_grad():
            _, info = self.adapter(self.x, self.wave, return_attention=True)
            info['attention'].zero_()
            info['anchors'].fill_(-1)
        self.assert_fresh()
        for attribute, value in (('locality_sigma', .3), ('locality_window', .01),
                                 ('max_anchor_offset', .4), ('attention_mode', 'uniform'),
                                 ('attention_mode', 'index'), ('attention_mode', 'wavelength')):
            setattr(self.adapter, attribute, value)
            self.assert_fresh()
        self.adapter.double()
        self.x, self.wave = self.x.double(), self.wave.double()
        self.assert_fresh(fwhm=torch.linspace(1., 20., 12))
        self.assert_fresh(fwhm=torch.linspace(2., 40., 12))
        self.wave = self.wave.flip(0)
        self.assert_fresh()

    def test_sub_rounding_metadata_change_does_not_hit_cache(self):
        self.adapter.eval()
        wave = torch.linspace(.1, .9, 12)
        changed = wave.clone()
        changed[2] += 1e-7
        with patch.object(self.adapter, '_attention', wraps=self.adapter._attention) as compute:
            with torch.no_grad():
                self.adapter(self.x, wave)
                self.adapter(self.x, changed)
            self.assertEqual(compute.call_count, 2)

    def test_autocast_and_inference_tensor_safely_bypass_cache(self):
        self.adapter.eval()
        with torch.no_grad():
            self.adapter(self.x, self.wave)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = self.adapter(self.x, self.wave)
                self.adapter.clear_attention_cache()
                expected = self.adapter(self.x, self.wave)
            torch.testing.assert_close(actual, expected)
        with torch.inference_mode():
            adapter = VariableChannelAdapter(latent_channels=8, embed_dim=8).eval()
            output = adapter(self.x, self.wave)
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(len(adapter._eval_attention_cache), 0)

    def test_modes_variable_bands_and_3d_4d_equivalence(self):
        for mode in ('wavelength', 'index', 'uniform'):
            self.adapter.attention_mode = mode
            for bands in (1, 12, 20, 103):
                x = torch.randn(2, 3, bands)
                wave = torch.linspace(400, 2500, bands)
                y, info = self.adapter(x, wave, return_attention=True)
                self.assertEqual(tuple(y.shape), (2, 3, 8))
                self.assertEqual(tuple(info['attention'].shape), (8, bands))
                self.assertTrue(torch.isfinite(y).all())
                batched = self.adapter(x.permute(2, 0, 1).unsqueeze(0), wave)
                torch.testing.assert_close(y, batched[0].permute(1, 2, 0))
                self.adapter.zero_grad(set_to_none=True)
                y.square().mean().backward()
                self.assertIsNotNone(self.adapter.value_proj[0].weight.grad)

    def test_wavelength_permutation_and_mask_equal_explicit_subset(self):
        permutation = torch.randperm(12)
        fwhm = torch.linspace(3., 15., 12)
        y = self.adapter(self.x, self.wave, fwhm)
        permuted = self.adapter(self.x[..., permutation], self.wave[permutation], fwhm[permutation])
        torch.testing.assert_close(y, permuted)
        mask = torch.arange(12) % 3 != 0
        damaged = self.x.clone()
        damaged[..., ~mask] = float('nan')
        damaged_wave = self.wave.clone()
        damaged_wave[~mask] = float('nan')
        output, info = self.adapter(damaged, damaged_wave, fwhm, True, band_mask=mask)
        subset = self.adapter(self.x[..., mask], self.wave[mask], fwhm[mask])
        torch.testing.assert_close(output, subset)
        torch.testing.assert_close(info['band_indices'], torch.arange(12)[mask])
        torch.testing.assert_close(info['attention'].sum(1), torch.ones(8))

    def test_explicit_units_and_missing_metadata_diagnostics(self):
        nm = self.adapter(self.x, self.wave, torch.ones(12), wavelength_unit='nm', fwhm_unit='nm')
        normalized = self.adapter(self.x, (self.wave - 400) / 2100, torch.ones(12) / 2100,
                                  wavelength_unit='normalized', fwhm_unit='normalized')
        torch.testing.assert_close(nm, normalized)
        _, info = self.adapter(self.x, return_attention=True)
        self.assertEqual(info['wavelength_unit'], 'normalized_index')
        self.assertEqual(info['wavelength_source'], 'index_fallback')
        self.assertEqual(info['fwhm_unit'], 'missing')
        _, info = self.adapter(self.x, self.wave, return_attention=True, wavelength_unit='nm')
        self.assertEqual(info['wavelength_unit'], 'nm')
        self.assertEqual(info['wavelength_source'], 'provided')

    def test_invalid_metadata_and_empty_observation_rejected(self):
        cases = [dict(wavelengths=[1, 2]), dict(fwhm=[1, 2]),
                 dict(wavelengths=[float('nan')] * 12), dict(fwhm=[-1.] * 12),
                 dict(wavelength_unit='meters'), dict(fwhm_unit='meters'),
                 dict(wavelength_unit='nm'), dict(band_mask=[False] * 12),
                 dict(band_mask=[1] * 12), dict(band_mask=[True] * 11)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.adapter(self.x, **kwargs)
        with self.assertRaises(ValueError):
            self.adapter(self.x[..., :0])

    def test_uncovered_anchors_are_finite_and_zero(self):
        self.adapter.locality_window = 1e-5
        y, info = self.adapter(self.x[..., :1], [1000.], return_attention=True)
        self.assertEqual(info['anchor_valid'].sum().item(), 0)
        torch.testing.assert_close(y, torch.zeros_like(y))
        self.assertTrue(torch.isfinite(info['attention']).all())

    def test_chunk_checkpoint_outputs_and_all_gradients_match(self):
        chunked = copy.deepcopy(self.adapter)
        chunked.chunk_pixels = 3
        chunked.checkpoint_chunks = True
        reference_x = self.x.clone().requires_grad_()
        chunked_x = self.x.clone().requires_grad_()
        reference = self.adapter(reference_x, self.wave)
        actual = chunked(chunked_x, self.wave)
        torch.testing.assert_close(actual, reference)
        reference.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(chunked_x.grad, reference_x.grad)
        for (name, parameter), (_, expected) in zip(chunked.named_parameters(), self.adapter.named_parameters()):
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, expected.grad)


if __name__ == '__main__':
    unittest.main()
