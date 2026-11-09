import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from loadData.spatial_tiles import make_tile_plan, validate_tile_plan, fit_tile_statistics, observe_tile, tile_slice
from model.checkpoint_io import load_model_checkpoint
from research_runtime import code_snapshot, tensor_hash
from train_spatial import SpatialConfig, execute, evaluate, restore_spatial_checkpoint
import json

torch.set_num_threads(1)

def fixture():
    cube=np.random.default_rng(9).normal(size=(60,60,6)).astype(np.float32)
    labels=np.tile(np.arange(1,10),400).reshape(60,60)
    config=SpatialConfig({'name':'tiny_spatial','dataset':'paviaU','model':{'graph_mode':'learned','latent_channels':8,'adapter_embed_dim':8,'norm_type':'domain_bn'},
        'training':{'epochs':0,'steps_per_epoch':2,'primary_checkpoint':'val_loss'},'tile_size':12,'gap':1})
    return config,cube,labels

class TileTests(unittest.TestCase):
    def test_geometry_reproducible_disjoint_no_label_access(self):
        _,cube,_=fixture();plan=make_tile_plan((60,60),12,1,2026)
        self.assertEqual(plan,make_tile_plan((60,60),12,1,2026))
        stats=fit_tile_statistics(cube,plan);ids=set()
        for tile in plan['tiles']:
            obs=observe_tile(cube,plan,stats,tile)
            self.assertFalse(ids&set(obs.global_ids.ravel()));ids.update(obs.global_ids.ravel())
        self.assertLess(len(ids),60*60)
        altered=copy.deepcopy(plan);altered['tiles'][0]['partition']='other'
        with self.assertRaises(ValueError):validate_tile_plan(altered)
        with self.assertRaises(ValueError):make_tile_plan((4,4),8,1)

    def test_target_perturbation_does_not_refit_train_statistics(self):
        _,cube,_=fixture();plan=make_tile_plan((60,60),12,1)
        stats=fit_tile_statistics(cube,plan);other=cube.copy()
        for tile in plan['tiles']:
            if tile['partition']!='train':other[tile_slice(tile)]=1e6
        self.assertEqual(stats,fit_tile_statistics(other,plan))
        for tile in plan['tiles']:
            if tile['partition']=='train':np.testing.assert_array_equal(observe_tile(cube,plan,stats,tile).spectral,observe_tile(other,plan,stats,tile).spectral)

    def test_unknown_config_and_static_rejected(self):
        for value in ({'unknown':1},{'model':{'graph_mode':'static'}},{'model':{'graph_mode':'learned'},'training':{'steps_per_epoch':0}}):
            with self.assertRaises(ValueError):SpatialConfig(value)

class SpatialTrainingTests(unittest.TestCase):
    def test_training_reload_skip_and_eval_state(self):
        config,cube,labels=fixture();code=code_snapshot()
        loader=lambda:(cube,labels,{'cube':tensor_hash(cube),'labels':tensor_hash(labels)},None)
        with tempfile.TemporaryDirectory() as root:
            result=execute(config,root,code,loader)
            self.assertEqual(result['status'],'completed')
            folder=Path(root)/result['run_id']/'attempts'/result['attempt']
            model=load_model_checkpoint(folder/'model.pt')
            restored,_,_=restore_spatial_checkpoint(folder/'model.pt',folder/'protocol.json',folder/'statistics.json')
            self.assertEqual(restored.export_config(),model.export_config())
            plan=make_tile_plan((60,60),12,1);stats=fit_tile_statistics(cube,plan)
            before={k:v.clone() for k,v in model.state_dict().items()}
            val=evaluate(model,cube,labels,plan,stats,'val','paviaU',9,None,50)
            self.assertGreater(val['count'],0)
            for k,v in before.items():self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
            with patch.object(torch.optim,'Adam',side_effect=AssertionError('must skip')):
                resumed=execute(config,root,code,loader,True)
            self.assertEqual(resumed['status'],'skipped')
            bad=json.loads((folder/'statistics.json').read_text());bad['minimum'][0]-=1
            (folder/'bad.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):restore_spatial_checkpoint(folder/'model.pt',folder/'protocol.json',folder/'bad.json')

    def test_test_labels_cannot_change_optimization_or_selection(self):
        config,cube,labels=fixture();code=code_snapshot();plan=make_tile_plan((60,60),12,1)
        other=labels.copy()
        for tile in plan['tiles']:
            if tile['partition']=='test':other[tile_slice(tile)]=1
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second:
            results=[]
            for root,target in ((first,labels),(second,other)):
                run=execute(config,root,code,lambda:(cube,target,{'labels':tensor_hash(target)},None))
                folder=Path(root)/run['run_id']/'attempts'/run['attempt']
                results.append(torch.load(folder/'model.pt',weights_only=True))
            for key,value in results[0].items():self.assertTrue(torch.equal(value,results[1][key]),key)

    def test_missing_training_class_rejected_without_redraw(self):
        config,cube,labels=fixture();plan=make_tile_plan((60,60),12,1)
        for tile in plan['tiles']:
            if tile['partition']=='train':labels[tile_slice(tile)]=1
        with tempfile.TemporaryDirectory() as root,self.assertRaisesRegex(ValueError,'missing classes'):
            execute(config,root,code_snapshot(),lambda:(cube,labels,{},None))

    def test_small_batchnorm_graph_rejected_early(self):
        config,cube,labels=fixture();value=config.to_dict();value['tile_size']=8
        with tempfile.TemporaryDirectory() as root,self.assertRaisesRegex(ValueError,'BatchNorm requires'):
            execute(SpatialConfig(value),root,code_snapshot(),lambda:(cube,labels,{},None))

if __name__=='__main__':unittest.main()
