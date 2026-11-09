"""Opt-in three-head integration, packed restore, and matched synthetic budgets."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from compression.int8_checkpoint import save_int8, load_int8
from compression.int8_model import calibration_contract, calibrate_model
from compression.ternary import TernaryMatrix
from compression.ternary_model import ternarize_model, pack_model, PackedTernary, validate_model
from compression.ternary_checkpoint import save_ternary, load_ternary
from loadData.spatial_tiles import tile_slice
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from model.research_config import digest
from transfer.development_training import sparse_labels
from transfer.g3_training import _forward, state_hash
from test_int8_model import fixture, calibrate, PATHS


def predict(model, scene, record):
    tile=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devval')
    with torch.no_grad(): return _forward(model.eval(),scene,tile)[:3]


class TernaryModelTests(unittest.TestCase):
    def test_disabled_exact_and_legacy_checkpoint(self):
        model,scene,record=fixture(); disabled=ternarize_model(model)
        self.assertEqual(state_hash(model),state_hash(disabled))
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'old.pt'; save_model_checkpoint(disabled,file); reloaded=load_model_checkpoint(file)
            for a,b,c in zip(predict(model,scene,record),predict(disabled,scene,record),predict(reloaded,scene,record)):
                self.assertTrue(torch.equal(a,b) and torch.equal(a,c))
        with self.assertRaises(ValueError): ternarize_model(model,threshold_ratio=.05)

    def test_source_whitelist_three_heads_and_no_master_in_packed(self):
        source,scene,record=fixture(); before=state_hash(source)
        rng=torch.get_rng_state().clone(); tq=ternarize_model(source,paths=PATHS,threshold_ratio=.05)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        packed=pack_model(tq)
        for k,p in source.state_dict().items():
            if not any(k.startswith(path+'.') for path in PATHS): self.assertTrue(torch.equal(p,tq.state_dict()[k]))
        for path in PATHS:
            self.assertIsInstance(tq.get_submodule(path),TernaryMatrix)
            self.assertIsInstance(packed.get_submodule(path),PackedTernary)
            self.assertEqual(list(packed.get_submodule(path).parameters()),[])
            self.assertTrue(torch.equal(tq.get_submodule(path).quantized_weight(),packed.get_submodule(path).decoded_weight()))
        for a,b in zip(predict(tq,scene,record),predict(packed,scene,record)):
            self.assertTrue(torch.equal(a,b)); self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a.sum(1),torch.ones(len(a)))
        self.assertEqual(before,state_hash(source))

    def test_invalid_paths_and_nested_modes_rejected(self):
        source,scene,record=fixture()
        for paths in ([PATHS[0],PATHS[0]],['heads.xuzhou'],['backbone.adapter']):
            with self.assertRaises(ValueError): ternarize_model(source,paths=paths,threshold_ratio=.05)
        tq=ternarize_model(source,paths=PATHS,threshold_ratio=.05)
        iq,_=calibrate(source,scene,record,qat=True)
        for model in (tq,pack_model(tq),iq):
            with self.assertRaises(ValueError): ternarize_model(model,paths=PATHS,threshold_ratio=.05)
        with self.assertRaises(ValueError): pack_model(pack_model(tq))

    def test_reverse_int8_wrapping_rejects_disjoint_ternary_layer(self):
        source,scene,record=fixture()
        tq=ternarize_model(source,paths=PATHS[:1],threshold_ratio=.05)
        ids=[r['id'] for r in record['roles'] if r['role']=='devtrain'][:1]
        with self.assertRaises(ValueError): calibration_contract(tq,scene,record,PATHS[1:],ids)
        with self.assertRaises(ValueError): calibrate_model(tq)

    def test_two_phases_disk_reload_trainability_rng_and_hash_boundary(self):
        model,scene,record=fixture(); model.heads[scene.dataset].weight.requires_grad_(False)
        model.get_submodule(PATHS[0]).weight.requires_grad_(False)
        tq=ternarize_model(model,paths=PATHS,threshold_ratio=.2)
        self.assertFalse(tq.get_submodule(PATHS[0]).positive.requires_grad)
        self.assertFalse(tq.get_submodule(PATHS[0]).negative.requires_grad)
        for candidate in (tq,pack_model(tq)):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.pt'; pin=save_ternary(candidate,path); rng=torch.get_rng_state().clone()
                restored=load_ternary(path,pin); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual({n:p.requires_grad for n,p in candidate.named_parameters()},
                                 {n:p.requires_grad for n,p in restored.named_parameters()})
                for a,b in zip(predict(candidate,scene,record),predict(restored,scene,record)): self.assertTrue(torch.equal(a,b))
                with self.assertRaises(FileExistsError): save_ternary(candidate,path)
                with patch('compression.ternary_checkpoint.torch.load',side_effect=AssertionError):
                    with self.assertRaises(ValueError): load_ternary(path,'0'*64)
                with self.assertRaises((KeyError,ValueError,RuntimeError)): load_model_checkpoint(path,config={})

    def test_plan_dtype_payload_and_trainability_tamper_rejected(self):
        model,scene,record=fixture(); tq=ternarize_model(model,paths=PATHS,threshold_ratio=.05)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'good.pt'; save_ternary(pack_model(tq),path); packet=torch.load(path,weights_only=True)
            def bad_code(p): p['state'][PATHS[0]+'.payload'][-1] |= 3
            def bad_scale(p): p['state'][PATHS[0]+'.payload'][21:25]=torch.tensor(list(struct.pack('<f',-1.)),dtype=torch.uint8)
            def bad_shape(p): p['state'][PATHS[0]+'.payload'][5:9]=torch.tensor(list(struct.pack('<I',1)),dtype=torch.uint8)
            for i,mutate in enumerate((lambda p:p['plan'].update(spec_hash='0'*64),lambda p:p['plan'].update(phase='training'),
                lambda p:p['trainable'].clear(),lambda p:p['state'].__setitem__('heads.xuzhou.weight',p['state']['heads.xuzhou.weight'].double()),
                lambda p:p['state'].__setitem__(PATHS[0]+'.payload',p['state'][PATHS[0]+'.payload'].float()),bad_code,bad_scale,bad_shape)):
                p=copy.deepcopy(packet); mutate(p); f=Path(tmp)/f'bad{i}.pt';torch.save(p,f)
                with self.assertRaises((ValueError,RuntimeError,KeyError)): load_ternary(f,hashlib.sha256(f.read_bytes()).hexdigest())
        with torch.no_grad(): tq.get_submodule(PATHS[0]).threshold_ratio.fill_(.3)
        with self.assertRaises(ValueError): validate_model(tq)

    def test_packed_rejects_training_and_grad_input(self):
        model,scene,record=fixture(); packed=pack_model(ternarize_model(model,paths=PATHS,threshold_ratio=.05))
        layer=packed.get_submodule(PATHS[0]); x=torch.ones(1,layer.weight_shape[1],2,2)
        with self.assertRaises(ValueError): layer(x.requires_grad_())
        layer.train()
        with self.assertRaises(ValueError): layer(x.detach())

    def test_three_arms_equal_12_step_nll_budget_and_exact_5_plus_7_restore(self):
        source,scene,record=fixture(); iq,_=calibrate(source,scene,record,qat=True)
        tq=ternarize_model(source,paths=PATHS,threshold_ratio=.05)
        tile=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devtrain')
        valtile=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devval')
        y=torch.tensor(sparse_labels(record,scene.plan,'train_support')[tile_slice(tile)].reshape(-1));mask=y>0
        vy=torch.tensor(sparse_labels(record,scene.plan,'validation_support')[tile_slice(valtile)].reshape(-1));vmask=vy>0
        def loss(model): return sum(F.nll_loss(p[mask].clamp_min(1e-10).log(),y[mask]-1) for p in _forward(model,scene,tile)[:3])
        def val(model):
            with torch.no_grad(): return float(F.nll_loss(_forward(model.eval(),scene,valtile)[0][vmask].clamp_min(1e-10).log(),vy[vmask]-1))
        for name,model in (('fp32',copy.deepcopy(source)),('int8-qat',iq),('tq',tq)):
            opt=torch.optim.Adam(model.parameters(),lr=.0001);history=[];initial=copy.deepcopy(model.state_dict())
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'step5.pt'
                for step in range(12):
                    model.train();opt.zero_grad();loss(model).backward();opt.step();history.append(val(model))
                    if step==4:
                        if name=='tq':pin=save_ternary(model,path)
                        elif name=='int8-qat':pin=save_int8(model,path)
                        else:save_model_checkpoint(model,path)
                        torch.save(opt.state_dict(),Path(tmp)/'opt.pt')
                resumed=load_ternary(path,pin) if name=='tq' else load_int8(path,pin) if name=='int8-qat' else load_model_checkpoint(path)
                resumed.backbone.adapter.cache_attention=False
                opt2=torch.optim.Adam(resumed.parameters(),lr=.0001);opt2.load_state_dict(torch.load(Path(tmp)/'opt.pt',weights_only=True))
                replay=history[:5]
                for _ in range(7):
                    resumed.train();opt2.zero_grad();loss(resumed).backward();opt2.step();replay.append(val(resumed))
                self.assertEqual(state_hash(model),state_hash(resumed));self.assertEqual(history,replay)
                self.assertEqual(min(range(12),key=history.__getitem__),min(range(12),key=replay.__getitem__))
                for key,slot in opt.state_dict()['state'].items():
                    for name2,v in slot.items(): self.assertTrue(torch.equal(v,opt2.state_dict()['state'][key][name2]))
                if name=='tq':
                    for path in PATHS:
                        for leaf in ('layer.weight','positive','negative'):
                            key=path+'.'+leaf;self.assertFalse(torch.equal(initial[key],model.state_dict()[key]))
                            self.assertGreater(float(dict(model.named_parameters())[key].grad.abs().sum()),0.)
                    for a,b in zip(predict(model,scene,record),predict(pack_model(model),scene,record)):self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main()
