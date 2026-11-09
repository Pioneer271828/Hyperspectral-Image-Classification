"""INT8 integration: existing spatial roles, untouched defaults, model recovery."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from compression.int8 import CalibratedMatrix, IntegerMatrix
from compression.int8_model import ALLOWED_PATHS, calibration_contract, calibrate_model, convert_model
from compression.int8_checkpoint import save_int8, load_int8
from loadData.observation_sampling import BandMetadata
from loadData.spatial_tiles import make_tile_plan, tile_slice
from loadData.spatial_development import make_development
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from model.research_config import digest
from transfer.development_training import prepare_scene, sparse_labels
from transfer.g3_training import _forward, state_hash

torch.set_num_threads(1)
PATHS = sorted(ALLOWED_PATHS)


def fixture():
    plan = make_tile_plan((60, 60), 20, 1, 2026)
    record = make_development(np.resize(np.arange(1, 10), (60, 60)), plan, list(range(1, 10)))
    cube = np.random.default_rng(3).normal(size=(60, 60, 8)).astype(np.float32)
    bands = BandMetadata('int8-fixture', tuple(range(8)), tuple(np.linspace(400, 1000, 8)),
                         'nm', 'approximate', (True,)*8)
    scene = prepare_scene(cube, bands, plan, record)
    torch.manual_seed(512)
    model = MultiDatasetVCASAHGCN({scene.dataset: 9}, latent_channels=8, adapter_embed_dim=8,
        locality_window=None, norm_type='group_norm', graph_mode='learned').eval()
    model.backbone.adapter.cache_attention = False
    return model, scene, record


def calibrate(model, scene, record, qat=False):
    ids = [row['id'] for row in record['roles'] if row['role'] == 'devtrain'][:2]
    contract = calibration_contract(model, scene, record, PATHS, ids, qat=qat)
    return calibrate_model(model, scene, record, paths=PATHS, contract=contract,
                           expected_hash=digest(contract)), contract


def predict(model, scene, record, role='devval'):
    tile = next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role'] == role)
    with torch.no_grad(): return _forward(model.eval(), scene, tile)[:3]


class Int8ModelTests(unittest.TestCase):
    def test_disabled_exact_original_and_legacy_checkpoint(self):
        model, scene, record = fixture()
        clone = calibrate_model(model)
        for a, b in zip(predict(model, scene, record), predict(clone, scene, record)):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(hasattr(clone, '_int8_plan'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy.pt'; save_model_checkpoint(model, path)
            restored = load_model_checkpoint(path)
            for a, b in zip(predict(model, scene, record), predict(restored, scene, record)):
                self.assertTrue(torch.equal(a, b))

    def test_whitelist_integer_three_heads_and_source_unchanged(self):
        model, scene, record = fixture(); before = state_hash(model)
        calibrated, contract = calibrate(model, scene, record)
        integer = convert_model(calibrated)
        for path in PATHS:
            self.assertIsInstance(calibrated.get_submodule(path), CalibratedMatrix)
            self.assertIsInstance(integer.get_submodule(path), IntegerMatrix)
        self.assertEqual(sum(isinstance(m, IntegerMatrix) for m in integer.modules()), 2)
        for name, value in model.state_dict().items():
            if not any(name.startswith(path+'.') for path in PATHS):
                self.assertTrue(torch.equal(value, integer.state_dict()[name]), name)
        for outputs in (predict(calibrated, scene, record), predict(integer, scene, record)):
            self.assertEqual(len(outputs), 3)
            for p in outputs:
                self.assertTrue(torch.isfinite(p).all())
                torch.testing.assert_close(p.sum(1), torch.ones(len(p)))
        self.assertEqual(before, state_hash(model))
        self.assertEqual(contract['fitting_role'], 'devtrain')

    def test_nontraining_cube_poison_cannot_affect_calibration(self):
        model, scene, record = fixture()
        left, pin = calibrate(model, scene, record)
        other = copy.deepcopy(scene)
        for row in record['roles']:
            if row['role'] != 'devtrain': other.cube[tile_slice(row)] = float('nan')
        right, pin2 = calibrate(model, other, record)
        self.assertEqual(pin, pin2)
        self.assertEqual(state_hash(left), state_hash(right))

    def test_role_plan_input_and_model_identity_fail_before_forward(self):
        model, scene, record = fixture()
        ids = [row['id'] for row in record['roles'] if row['role'] == 'devtrain'][:1]
        contract = calibration_contract(model, scene, record, PATHS, ids)
        for role in ('devval', 'test', 'unused'):
            denied = next(row['id'] for row in record['roles'] if row['role'] == role)
            with self.assertRaises(ValueError): calibration_contract(model, scene, record, PATHS, [denied])
        for paths in ([PATHS[0], PATHS[0]], ['heads.xuzhou'], ['backbone.adapter']):
            with self.assertRaises(ValueError): calibration_contract(model, scene, record, paths, ids)
        changed = copy.deepcopy(scene)
        changed.cube[tile_slice(changed.plan['tiles'][ids[0]])] += .1
        bad_model = copy.deepcopy(model)
        with torch.no_grad(): next(bad_model.parameters()).add_(.1)
        bad_record = copy.deepcopy(record); bad_record['roles'][ids[0]]['role'] = 'devval'
        with patch('compression.int8_model._forward', side_effect=AssertionError('must reject before compute')):
            for m, s, r, pin in ((model, scene, record, '0'*64), (model, changed, record, digest(contract)),
                                  (bad_model, scene, record, digest(contract)),
                                  (model, scene, bad_record, digest(contract))):
                with self.assertRaises(ValueError):
                    calibrate_model(m, s, r, paths=PATHS, contract=contract, expected_hash=pin)

    def test_checkpoint_phases_exact_three_heads_rng_trainability_and_no_clobber(self):
        model, scene, record = fixture()
        model.heads[scene.dataset].weight.requires_grad_(False)
        ptq, _ = calibrate(model, scene, record)
        qat, _ = calibrate(model, scene, record, qat=True)
        for candidate in (ptq, qat, convert_model(ptq), convert_model(qat)):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'model.pt'; pin = save_int8(candidate, path)
                rng = torch.get_rng_state().clone(); restored = load_int8(path, pin)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertFalse(restored.heads[scene.dataset].weight.requires_grad)
                for a, b in zip(predict(candidate, scene, record), predict(restored, scene, record)):
                    self.assertTrue(torch.equal(a, b))
                with self.assertRaises(FileExistsError): save_int8(candidate, path)
                with patch('compression.int8_checkpoint.torch.load', side_effect=AssertionError):
                    with self.assertRaises(ValueError): load_int8(path, '0'*64)
                with self.assertRaises((ValueError, KeyError, RuntimeError)):
                    load_model_checkpoint(path, config={})

    def test_checkpoint_metadata_tampering_rejected(self):
        model, scene, record = fixture(); cal, _ = calibrate(model, scene, record)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'good.pt'; save_int8(convert_model(cal), path)
            original = torch.load(path, weights_only=True)
            for index, change in enumerate((
                    lambda p: p.update(torch_version='other'),
                    lambda p: p['plan'].update(contract_hash='0'*64),
                    lambda p: p['plan'].update(phase='qat'),
                    lambda p: p['state'].__setitem__(PATHS[0]+'.input_scale', torch.tensor([-1.])),
                    lambda p: p['trainable'].clear())):
                packet = copy.deepcopy(original); change(packet)
                damaged = Path(tmp)/f'bad{index}.pt'; torch.save(packet, damaged)
                with self.assertRaises((ValueError, RuntimeError, KeyError)):
                    load_int8(damaged, hashlib.sha256(damaged.read_bytes()).hexdigest())

    def test_fp32_qat_equal_12_steps_first_min_nll_and_split_resume(self):
        source, scene, record = fixture()
        qat, _ = calibrate(source, scene, record, qat=True)
        tile = next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role'] == 'devtrain')
        valtile = next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role'] == 'devval')
        train_labels = torch.tensor(sparse_labels(record, scene.plan, 'train_support')[tile_slice(tile)].reshape(-1))
        val_labels = torch.tensor(sparse_labels(record, scene.plan, 'validation_support')[tile_slice(valtile)].reshape(-1))
        mask, valmask = train_labels > 0, val_labels > 0
        def loss(model):
            return sum(nn.functional.nll_loss(p[mask].clamp_min(1e-10).log(), train_labels[mask]-1)
                       for p in _forward(model, scene, tile)[:3])
        def val(model):
            model.eval()
            with torch.no_grad():
                p = _forward(model, scene, valtile)[0]
                return float(nn.functional.nll_loss(p[valmask].clamp_min(1e-10).log(), val_labels[valmask]-1))
        for name, model in (('fp32', copy.deepcopy(source)), ('qat', qat)):
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            history = []; initial = {n:p.detach().clone() for n,p in model.named_parameters()}
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'step5.pt'
                for step in range(12):
                    model.train(); opt.zero_grad(); loss(model).backward(); opt.step(); history.append(val(model))
                    if step == 4:
                        if name == 'qat': pin = save_int8(model, path)
                        else: save_model_checkpoint(model, path)
                        torch.save(opt.state_dict(), Path(tmp)/'opt.pt')
                resumed = load_int8(path, pin) if name == 'qat' else load_model_checkpoint(path)
                resumed.backbone.adapter.cache_attention = False
                opt2 = torch.optim.Adam(resumed.parameters(), lr=.001)
                opt2.load_state_dict(torch.load(Path(tmp)/'opt.pt', weights_only=True))
                replay = history[:5]
                for _ in range(7):
                    resumed.train(); opt2.zero_grad(); loss(resumed).backward(); opt2.step(); replay.append(val(resumed))
            self.assertEqual(state_hash(model), state_hash(resumed))
            self.assertEqual(history, replay)
            self.assertEqual(min(range(12), key=history.__getitem__), min(range(12), key=replay.__getitem__))
            if name == 'qat':
                for path in PATHS:
                    key = path+'.layer.weight'
                    self.assertFalse(torch.equal(initial[key], model.state_dict()[key]))
                    self.assertGreater(float(dict(model.named_parameters())[key].grad.abs().sum()), 0.)
                for p in predict(convert_model(model), scene, record): self.assertTrue(torch.isfinite(p).all())


if __name__ == '__main__':
    unittest.main()
