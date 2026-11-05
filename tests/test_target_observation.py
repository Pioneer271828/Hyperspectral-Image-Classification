import copy,json,os,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
import torch
import target_observation as m
from createGraph import rdSLIC
from loadData.dataset_registry import file_hash
from joint_baseline import load_config
from research_runtime import tensor_hash

DEVICE=os.environ.get('G4D_DEVICE','cpu')
torch.set_num_threads(1)


class TargetObservationTests(unittest.TestCase):
    def image(self,bands=4):
        rng=np.random.default_rng(7);y,x=np.mgrid[:32,:32]
        return rng.normal(0,.1,(32,32,bands))+y[:,:,None]+x[:,:,None]*.7

    def test_sparse_matches_existing_no_lda_and_restores_rng(self):
        image=self.image();python=random.getstate();numpy=copy.deepcopy(np.random.get_state())
        with m.isolated_graph_rng(64):
            Q,_,A,segments=rdSLIC.LDA_SLIC(image,np.zeros(image.shape[:2]),3).simple_superpixel_no_LDA(50)
        with patch.object(rdSLIC.LDA_SLIC,'LDA_Process',side_effect=AssertionError('No label-guided fit')):
            q,a,s=m.label_free_graph(image,50,64)
        self.assertTrue(np.array_equal(segments,s));np.testing.assert_array_equal(q.to_dense().numpy(),Q)
        np.testing.assert_allclose(a.numpy(),np.asarray(A,dtype=np.float32),rtol=0,atol=0)
        self.assertEqual(q._nnz(),image.shape[0]*image.shape[1]);self.assertEqual(random.getstate(),python)
        now=np.random.get_state();self.assertEqual(now[0],numpy[0]);np.testing.assert_array_equal(now[1],numpy[1]);self.assertEqual(now[2:],numpy[2:])

    def test_image_loader_does_not_need_any_label_split_or_prepared_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared=root/'prepared';prepared.mkdir();image=self.image(103);savemat(root/'image.mat',{'paviaU':image})
            raw={'image_path':'image.mat','image_sha256':file_hash(root/'image.mat'),'shape':list(image.shape),
                'class_count':9,'label_path':'missing-label.mat','label_sha256':'unavailable'}
            manifest={'protocol':'acc1p_joint_prepared_v1','schema_version':1,'datasets':{'paviaU':{'raw':raw}}}
            (prepared/'manifest.json').write_text(json.dumps(manifest));h=file_hash(prepared/'manifest.json')
            actual,record=m.load_target_image(prepared,root,'paviaU',manifest_sha256=h)
            np.testing.assert_array_equal(actual,image);self.assertFalse(record['target_label_payload_accessed'])
            with self.assertRaises(ValueError):m.load_target_image(prepared,root,'paviaU',manifest_sha256='wrong')
            with (root/'image.mat').open('ab') as stream:stream.write(b'change')
            with self.assertRaises(ValueError):m.load_target_image(prepared,root,'paviaU',manifest_sha256=h)

    def test_target_observation_only_model_forward_backward_interface(self):
        image=self.image(103);before=image.copy();item,record=m.build_target_observation(image,'paviaU',64)
        np.testing.assert_array_equal(image,before)
        self.assertFalse(any(k.startswith(('train_','val_','test_')) for k in item));self.assertEqual(record['graph_fit_labels'],0)
        self.assertEqual(record['assignment_nnz'],32*32);self.assertEqual(item['net_input'].shape,(32,32,105))
        cfg=load_config('configs/experiments/base97_joint.json');model=cfg.build_model({'paviaU':9}).to(DEVICE)
        moved={k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in item.items()}
        from joint_baseline import forward
        outputs=forward(model,'paviaU',moved);loss=-torch.log(outputs[0][:,0].clamp_min(1e-8)).mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.backbone.HGNN_Branch.parameters()))

    def test_invalid_images_seeds_and_targets_fail(self):
        for image in (np.zeros((4,4,3)),np.zeros((3,3)),np.full((4,4,3),np.nan)):
            with self.assertRaises(ValueError):m.label_free_graph(image,50,64)
        with self.assertRaises(ValueError):m.label_free_graph(self.image(),50,True)
        with self.assertRaises(ValueError):m.build_target_observation(self.image(),'unknown',64)


if __name__=='__main__':unittest.main()
