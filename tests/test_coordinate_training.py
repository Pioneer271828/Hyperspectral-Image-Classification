"""Persisted coordinate policies through optimization, selection and reload."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from loadData.coordinate_observation import coordinate_observation
from loadData.spatial_tiles import make_tile_plan, fit_tile_statistics, observe_tile
from train_spatial import SpatialConfig, execute, evaluate, restore_spatial_checkpoint
from research_runtime import code_snapshot
from scripts.run_coordinate_screening import prepare

class CoordinateTrainingTests(unittest.TestCase):
    def test_policy_preserves_observation_identity(self):
        cube=np.random.default_rng(1).normal(size=(60,60,6)).astype(np.float32)
        plan=make_tile_plan((60,60),12,1);stats=fit_tile_statistics(cube,plan)
        obs=observe_tile(cube,plan,stats,plan['tiles'][6])
        before=obs.coordinates.copy()
        self.assertIs(coordinate_observation(obs),obs)
        for mode in ('relative','zero'):
            other=coordinate_observation(obs,mode)
            self.assertIs(other.spectral,obs.spectral);self.assertIs(other.global_ids,obs.global_ids)
            self.assertEqual(other.partition,obs.partition);self.assertEqual(other.plan_hash,obs.plan_hash)
            self.assertFalse(other.coordinates.flags.writeable)
            if mode=='zero':self.assertFalse(other.coordinates.any())
            else:
                np.testing.assert_allclose(other.coordinates[0,0],[0,0])
                np.testing.assert_allclose(other.coordinates[-1,-1],[1,1])
        np.testing.assert_array_equal(obs.coordinates,before)
        with self.assertRaises(ValueError):coordinate_observation(obs,'bad')

    def test_legacy_config_and_paired_preparation(self):
        path=Path('configs/experiments/screen_m6_none.json')
        old=SpatialConfig(json.loads(path.read_text()))
        self.assertNotIn('coordinate_mode',old.to_dict())
        self.assertEqual(old.config_hash,SpatialConfig(old.to_dict()).config_hash)
        for invalid in (None,'bad',1):
            with self.assertRaises(ValueError):SpatialConfig(old.to_dict()|{'coordinate_mode':invalid})
        with tempfile.TemporaryDirectory() as directory:
            configs,state,digest=prepare(Path(directory),code_snapshot())
            self.assertEqual(set(configs),{'absolute','relative','zero'})
            for mode,c in configs.items():
                self.assertEqual(c.values['coordinate_mode'],mode)
                self.assertEqual(c.budget['optimizer_steps'],10000)
            again=prepare(Path(directory),code_snapshot())
            self.assertEqual(digest,again[2])

    def test_training_eval_checkpoint_policy_closure(self):
        torch.set_num_threads(1)
        cube=np.random.default_rng(4).normal(size=(60,60,6)).astype(np.float32)
        labels=np.tile(np.arange(1,10),400).reshape(60,60)
        for mode in ('absolute','relative','zero'):
            config=SpatialConfig({'name':'coordinate_'+mode,'coordinate_mode':mode,'dataset':'paviaU',
                'model':{'graph_mode':'learned','latent_channels':8,'adapter_embed_dim':8,'norm_type':'group_norm'},
                'training':{'epochs':0,'steps_per_epoch':2,'primary_checkpoint':'val_loss'},'tile_size':12,'gap':1})
            calls=[]
            def observed(obs, policy):
                calls.append((obs.partition,policy))
                return coordinate_observation(obs,policy)
            with tempfile.TemporaryDirectory() as directory:
                with patch('train_spatial.coordinate_observation',side_effect=observed):
                    result=execute(config,directory,code_snapshot(),lambda:(cube,labels,{},None))
                self.assertTrue({'train','val','test'}<=set(p for p,_ in calls))
                self.assertTrue(all(m==mode for _,m in calls))
                folder=Path(directory)/result['run_id']/'attempts'/result['attempt']
                model,plan,stats=restore_spatial_checkpoint(folder/'model.pt',folder/'protocol.json',folder/'statistics.json')
                self.assertEqual(model.training_config['spatial_config']['coordinate_mode'],mode)
                auto=evaluate(model,cube,labels,plan,stats,'val','paviaU',9,None,50)
                explicit=evaluate(model,cube,labels,plan,stats,'val','paviaU',9,None,50,mode)
                self.assertEqual(auto,explicit)

if __name__=='__main__':unittest.main()
