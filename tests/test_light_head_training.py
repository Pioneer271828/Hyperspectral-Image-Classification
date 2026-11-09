"""Independent head training must preserve full source and frozen prefix."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from inference.light_cnn import LightCNN
from transfer.light_head_training import LightHeadRun
from transfer.g3_training import _forward,state_hash
from transfer.development_training import sparse_labels
from loadData.spatial_tiles import tile_slice,observe_tile
from model.utils import compute_loss
from tests.test_int8_model import fixture


def run_fixture(updates=12):
    full,s,r=fixture();run=LightHeadRun(full,s,r,updates=updates,seed=64,lr=.0001,class_balance=True)
    return full,s,r,run


class LightHeadTests(unittest.TestCase):
    def test_feature_exact_old_cnn_no_graph_no_grad_and_ownership(self):
        full,s,r,run=run_fixture();before=state_hash(full);prefix=state_hash(run.path)
        tile=next(t for t in s.plan['tiles'] if r['roles'][t['id']]['role']=='devtrain')
        with torch.no_grad():expected=_forward(full,s,tile)[2]
        with patch.object(full.backbone.learnable_graph,'build_graph',side_effect=AssertionError('graph called')),patch.object(full,'forward',side_effect=AssertionError('full called')):
            features=run._features(tile);actual=run.head(features).softmax(-1)
            self.assertFalse(features.requires_grad);self.assertTrue(torch.equal(actual,expected));run.step()
        self.assertEqual(before,state_hash(full));self.assertEqual(prefix,state_hash(run.path))
        self.assertNotEqual(run.spec['initial_head_state'],state_hash(run.head))
        self.assertTrue(all(p.grad is None for p in full.parameters()));self.assertTrue(all(p.grad is None for p in run.path.parameters()))
        self.assertGreater(run.history[0]['head_gradient_l1'],0)

    def test_12_vs_5_plus_7_model_optimizer_history_selection_and_completed_zero(self):
        full,s,r,run=run_fixture();source=state_hash(full);tr=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'run.pt';run.advance(5);sha=run.save(file);run.advance(12)
            other=LightHeadRun.load(full,s,r,file,expected_sha256=sha,expected_contract_hash=run.contract_hash);other.advance(12)
            self.assertEqual(run.history,other.history);self.assertEqual(run.selected_step,other.selected_step)
            self.assertEqual(state_hash(run.head),state_hash(other.head));self.assertEqual(state_hash(run.path),state_hash(other.path))
            for k,slot in run.optimizer.state_dict()['state'].items():
                for name,v in slot.items():self.assertTrue(torch.equal(v,other.optimizer.state_dict()['state'][k][name]))
            for k,v in run.selected_head.items():self.assertTrue(torch.equal(v,other.selected_head[k]))
            file2=Path(tmp)/'done.pt';sha2=run.save(file2)
            done=LightHeadRun.load(full,s,r,file2,expected_sha256=sha2,expected_contract_hash=run.contract_hash)
            before=state_hash(done.head);self.assertEqual(done.advance(12),run.history)
            with self.assertRaises(ValueError):done.step()
            self.assertEqual(before,state_hash(done.head));self.assertEqual(done.completed,12)
        self.assertEqual(source,state_hash(full));self.assertTrue(torch.equal(tr,torch.get_rng_state()))

    def test_full_cnn_head_only_equal_budget_loss_gradient_update_control(self):
        full,s,r,run=run_fixture();control=copy.deepcopy(full).eval().requires_grad_(False)
        control.cnn_heads[s.dataset].requires_grad_(True);opt=torch.optim.Adam(control.cnn_heads[s.dataset].parameters(),lr=.0001)
        before={k:v.clone() for k,v in control.state_dict().items()};rng=np.random.default_rng(64)
        for n in range(12):
            tile,anchor=run.sampler.draw(rng);target=torch.tensor(sparse_labels(r,s.plan,'train_support')[tile_slice(tile)].ravel())
            mask=(target>0)[:,None].expand(-1,s.classes).float();onehot=torch.nn.functional.one_hot((target-1).clamp_min(0),s.classes).float()*mask
            opt.zero_grad();loss=compute_loss(_forward(control,s,tile)[2],onehot,mask,class_balance=True);loss.backward();opt.step()
            row=run.step();self.assertEqual(row['loss'],float(loss.detach()));self.assertEqual((row['tile_id'],row['anchor']),(tile['id'],anchor))
            self.assertEqual(state_hash(run.head),state_hash(control.cnn_heads[s.dataset]))
        for k,v in control.state_dict().items():
            if not k.startswith('cnn_heads.'+s.dataset+'.'):self.assertTrue(torch.equal(before[k],v))

    def test_zero_checkpoint_external_hash_before_load_and_no_overwrite(self):
        full,s,r,run=run_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'zero.pt';sha=run.save(file)
            other=LightHeadRun.load(full,s,r,file,expected_sha256=sha,expected_contract_hash=run.contract_hash)
            self.assertEqual(other.completed,0);self.assertEqual(state_hash(other.head),state_hash(run.head))
            with self.assertRaises(FileExistsError):run.save(file)
            with patch('transfer.light_head_training.torch.load',side_effect=AssertionError('deserialized before auth')):
                with self.assertRaises(ValueError):LightHeadRun.load(full,s,r,file,expected_sha256='0'*64,expected_contract_hash=run.contract_hash)
            with self.assertRaises(ValueError):LightHeadRun.load(full,s,r,file,expected_sha256=sha,expected_contract_hash='0'*64)
            with torch.no_grad():full.cnn_heads[s.dataset].bias.add_(1)
            with self.assertRaises(ValueError):LightHeadRun.load(full,s,r,file,expected_sha256=sha,expected_contract_hash=run.contract_hash)

    def test_tampered_adam_progress_rng_selection_or_finite_head_rejected(self):
        full,s,r,run=run_fixture();run.advance(2)
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'valid.pt';run.save(original);packet=torch.load(original,weights_only=True)
            for kind in ('budget','rng','optimizer','selected','head'):
                bad=copy.deepcopy(packet)
                if kind=='budget':bad['completed']=13
                if kind=='rng':bad['rng']['state']['state']+=1
                if kind=='optimizer':bad['optimizer']['state'][0]['step'].add_(1)
                if kind=='selected':bad['selected_step']=None
                if kind=='head':bad['head']['weight'].fill_(float('nan'))
                file=Path(tmp)/(kind+'.pt');torch.save(bad,file);sha=hashlib.sha256(file.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):LightHeadRun.load(full,s,r,file,expected_sha256=sha,expected_contract_hash=run.contract_hash)

    def test_nontraining_pixels_isolated_from_training_and_test_tiles_forbidden(self):
        full,s,r,a=run_fixture(updates=1);changed=copy.deepcopy(s)
        for row in r['roles']:
            if row['role']=='test':changed.cube[tile_slice(row)]=float('nan')
        b=LightHeadRun(full,changed,r,updates=1,seed=64,lr=.0001,class_balance=True)
        self.assertEqual(a.step(),b.step());self.assertEqual(state_hash(a.head),state_hash(b.head))
        test=next(t for t in s.plan['tiles'] if r['roles'][t['id']]['role']=='test')
        with self.assertRaises(ValueError):a._features(test)
        with self.assertRaises(ValueError):a.advance(2)
        full2,s2,r2,c=run_fixture();c.path.train()
        with self.assertRaises(ValueError):c.step()

    def test_domain_bn_buffers_remain_frozen_and_optimizer_changes_rejected(self):
        from model.VCASAHGCN import MultiDatasetVCASAHGCN
        full,s,r=fixture();cfg=full.export_config();cfg['backbone_kwargs']['norm_type']='domain_bn'
        full=MultiDatasetVCASAHGCN(cfg['class_counts'],**cfg['backbone_kwargs']).eval();full.backbone.adapter.cache_attention=False
        run=LightHeadRun(full,s,r,updates=2,seed=64,lr=.0001,class_balance=True)
        source=state_hash(full);prefix=state_hash(run.path);run.advance(2)
        self.assertEqual(source,state_hash(full));self.assertEqual(prefix,state_hash(run.path))
        _,_,_,other=run_fixture();other.optimizer.param_groups[0]['lr']=.1
        with self.assertRaises(ValueError):other.step()

if __name__=='__main__':unittest.main()
