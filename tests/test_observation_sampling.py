"""M1 real spectral dimensions, synchronized metadata and fixed replay."""
import copy
from dataclasses import replace
import json
import os
import unittest
import numpy as np
import torch
from loadData.observation_sampling import BandMetadata,sample,replay,subset_statistics
from loadData.spatial_tiles import make_tile_plan,fit_tile_statistics,observe_tile
from model.variable_channel_adapter import VariableChannelAdapter


def fixture():
    cube=np.arange(60*60*8,dtype=np.float32).reshape(60,60,8)
    plan=make_tile_plan((60,60),12,1);stats=fit_tile_statistics(cube,plan)
    observation=observe_tile(cube,plan,stats,plan['tiles'][0])
    bands=BandMetadata('synthetic-source',tuple(range(8)),tuple(np.linspace(400,1100,8)),
                       'nm','approximate',(True,)*8,tuple(range(1,9)),'nm')
    return observation,bands,stats


class SamplingTests(unittest.TestCase):
    def test_none_identity_and_all_modes_sync(self):
        obs,bands,stats=fixture();before=obs.spectral.copy()
        view,record=sample(obs,bands,{'mode':'none'})
        self.assertIs(view.observation,obs);self.assertIs(view.bands,bands)
        cases=[({'mode':'uniform_stride','stride':2,'offset':1},[1,3,5,7]),
               ({'mode':'contiguous_gap','lower':600,'upper':800,'unit':'nm'},[0,1,5,6,7]),
               ({'mode':'range_crop','lower':600,'upper':800,'unit':'nm'},[2,3,4])]
        for config,positions in cases:
            view,record=sample(obs,bands,config)
            np.testing.assert_array_equal(view.observation.spectral,obs.spectral[...,positions])
            self.assertEqual(record['selected_positions'],positions)
            self.assertEqual(view.bands.band_ids,tuple(positions))
            self.assertEqual(view.bands.wavelengths,tuple(bands.wavelengths[i] for i in positions))
            self.assertEqual(view.bands.fwhm,tuple(bands.fwhm[i] for i in positions))
            self.assertIs(view.observation.coordinates,obs.coordinates);self.assertIs(view.observation.global_ids,obs.global_ids)
            self.assertEqual(view.observation.plan_hash,obs.plan_hash)
            selected_stats=subset_statistics(stats,bands,record)
            self.assertEqual(selected_stats['minimum'],[stats['minimum'][i] for i in positions])
        np.testing.assert_array_equal(obs.spectral,before)

    def test_random_reproducible_and_frozen_replay_rng_isolated(self):
        obs,bands,_=fixture();evaluation_rng=np.random.default_rng(7);before=copy.deepcopy(evaluation_rng.bit_generator.state)
        global_before=np.random.get_state()
        first,record=sample(obs,bands,{'mode':'random_subset','keep_fraction':.5},np.random.default_rng(12))
        global_after=np.random.get_state()
        np.testing.assert_array_equal(global_before[1],global_after[1]);self.assertEqual(global_before[2:],global_after[2:])
        second,record2=sample(obs,bands,{'mode':'random_subset','keep_fraction':.5},np.random.default_rng(12))
        self.assertEqual(record,record2);self.assertEqual(len(first.bands.band_ids),4)
        restored=replay(obs,bands,json.loads(json.dumps(record)))
        np.testing.assert_array_equal(restored.observation.spectral,first.observation.spectral)
        self.assertEqual(before,evaluation_rng.bit_generator.state)
        _,evaluation_record=sample(obs,bands,{'mode':'random_subset','keep_fraction':.5},evaluation_rng)
        _,isolated_record=sample(obs,bands,{'mode':'random_subset','keep_fraction':.5},np.random.default_rng(7))
        self.assertEqual(evaluation_record,isolated_record)
        corrupt=copy.deepcopy(record);corrupt['selected_positions'][0]=7
        with self.assertRaises(ValueError):replay(obs,bands,corrupt)
        with self.assertRaises(ValueError):replay(replace(obs,partition='test' if obs.partition!='test' else 'val'),bands,record)
        with self.assertRaises(ValueError):replay(obs,replace(bands,source_id='other'),record)

    def test_validity_single_band_and_original_id_stride(self):
        obs,bands,_=fixture();bands=replace(bands,valid=(True,False,False,False,False,False,False,False))
        view,_=sample(obs,bands,{'mode':'none'});self.assertEqual(view.observation.spectral.shape[-1],1)
        self.assertEqual(view.bands.valid,(True,))
        bands=replace(bands,band_ids=tuple(range(1,9)),valid=(True,)*8)
        view,_=sample(obs,bands,{'mode':'uniform_stride','stride':2,'offset':0})
        self.assertEqual(view.bands.band_ids,(2,4,6,8))
        view,_=sample(obs,bands,{'mode':'random_subset','keep_fraction':1e-9},np.random.default_rng(0))
        self.assertEqual(view.observation.spectral.shape[-1],1)

    def test_metadata_unknown_units_and_empty_rejected(self):
        obs,bands,stats=fixture()
        for kwargs in ({'band_ids':(0,)*8},{'valid':(False,)*8},{'wavelengths':(float('nan'),)*8},
                       {'wavelength_unit':'auto'},{'wavelength_quality':'index_only'},
                       {'fwhm_unit':'normalized'},{'fwhm':None}):
            with self.assertRaises(ValueError):replace(bands,**kwargs)
        indexed=BandMetadata('unknown',tuple(range(8)),tuple(np.linspace(0,1,8)),'normalized','index_only',(True,)*8)
        view,_=sample(obs,indexed,{'mode':'uniform_stride','stride':2,'offset':0});self.assertIsNone(view.bands.fwhm)
        for config in ({'mode':'range_crop','lower':0,'upper':1,'unit':'nm'},
                       {'mode':'random_subset','keep_fraction':0},{'mode':'none','extra':1},
                       {'mode':'uniform_stride','stride':0,'offset':0},
                       {'mode':'range_crop','lower':9999,'upper':10000,'unit':'nm'},
                       {'mode':'contiguous_gap','lower':0,'upper':10000,'unit':'nm'}):
            with self.assertRaises(ValueError):sample(obs,bands,config)
        with self.assertRaises(ValueError):sample(obs,indexed,{'mode':'range_crop','lower':0,'upper':1,'unit':'normalized'})
        _,record=sample(obs,bands,{'mode':'none'});bad=copy.deepcopy(stats);bad['minimum'][0]+=1
        with self.assertRaises(ValueError):subset_statistics(bad,bands,record)

    def test_normalize_then_slice_equals_original_statistics_subset(self):
        obs,bands,stats=fixture();config={'mode':'random_subset','keep_fraction':.75}
        view,record=sample(obs,bands,config,np.random.default_rng(6));positions=record['selected_positions']
        self.assertEqual(view.bands.wavelength_quality,bands.wavelength_quality)
        np.testing.assert_array_equal(view.observation.spectral,np.take(obs.spectral,positions,axis=-1))
        self.assertEqual(subset_statistics(stats,bands,record)['maximum'],[stats['maximum'][i] for i in positions])
        self.assertFalse(view.observation.spectral.flags.writeable)

    def test_vca_cropped_vs_masked_forward_backward_and_variable_b(self):
        device=os.environ.get('M1_DEVICE','cpu');torch.set_num_threads(1);torch.manual_seed(5)
        obs,bands,_=fixture()
        for fraction in (1,.75,.5,.25,1e-9):
            view,record=sample(obs,bands,{'mode':'random_subset','keep_fraction':fraction},np.random.default_rng(2))
            adapter=VariableChannelAdapter(latent_channels=8,embed_dim=8,locality_window=None).to(device)
            reference=copy.deepcopy(adapter)
            full=torch.tensor(np.array(obs.spectral,copy=True),device=device)
            crop=torch.tensor(np.array(view.observation.spectral,copy=True),device=device)
            wave=torch.tensor(bands.wavelengths,device=device,dtype=torch.float32)
            widths=torch.tensor(bands.fwhm,device=device,dtype=torch.float32)
            positions=record['selected_positions'];mask=torch.zeros(8,dtype=torch.bool,device=device);mask[positions]=True
            actual=adapter(crop,wave[positions],widths[positions],wavelength_unit='nm',fwhm_unit='nm')
            expected=reference(full,wave,widths,wavelength_unit='nm',fwhm_unit='nm',band_mask=mask)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
            actual.square().mean().backward();expected.square().mean().backward()
            for (_,a),(_,b) in zip(adapter.named_parameters(),reference.named_parameters()):
                if a.grad is None:self.assertIsNone(b.grad)
                else:torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)
            self.assertEqual(crop.shape[-1],len(positions));self.assertEqual(actual.shape[-1],8)

if __name__=='__main__':unittest.main()

