"""Context isolation and actual-network fixed-support update/recovery."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from loadData.spatial_tiles import tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.budgeted_graph_router import BudgetedGraphRouter
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from model.research_config import digest
from model.task_losses import probability_loss_components
from transfer.development_training import sparse_labels
from transfer.g3_training import _forward, state_hash
from transfer.graph_route_context import RouteContextBank, route_state_hash
from tests.test_int8_model import fixture


def setup(mode='weight'):
    base, scene, record = fixture()
    kw = copy.deepcopy(base.backbone_kwargs); kw['graph_route'] = dict(budget=2, mode=mode, seed=64)
    model = MultiDatasetVCASAHGCN({scene.dataset: scene.classes}, **kw).eval()
    model.load_state_dict(base.state_dict(), strict=False)
    model.backbone.adapter.cache_attention = False
    ids = [r['id'] for r in record['roles'] if r['role'] in ('devtrain', 'devval')]
    bank = RouteContextBank.freeze(model, scene, record, tile_ids=ids, step=7,
                                  expected_source_state=route_state_hash(model))
    return model, scene, record, ids, bank


def update(model, bank, scene, record, tile_id, optimizer, step):
    model.train(); optimizer.zero_grad(set_to_none=True)
    out = bank.forward(model, scene, record, tile_id, step=step)
    labels = torch.tensor(sparse_labels(record, scene.plan, 'devtrain')[tile_slice(scene.plan['tiles'][tile_id])].reshape(-1))
    good = labels > 0
    mask = good[:, None].expand(-1, scene.classes)
    onehot = torch.nn.functional.one_hot((labels-1).clamp_min(0), scene.classes).float() * mask
    loss = probability_loss_components(out[:3], onehot, mask.float(), class_balance=True)['total']
    loss.backward(); optimizer.step()
    return float(loss.detach())


class RouteContextTests(unittest.TestCase):
    def test_four_modes_original_route_matches_freeze_and_source_unchanged(self):
        for mode in BudgetedGraphRouter.MODES:
            model, scene, record, ids, bank = setup(mode)
            original = route_state_hash(model); rng = torch.get_rng_state().clone()
            r = model.backbone.graph_router; prior = r.get_extra_state()
            for tid in ids:
                r.set_step(7)
                with torch.no_grad(): direct = _forward(model, scene, scene.plan['tiles'][tid])
                r.set_extra_state(prior)
                with torch.no_grad(): bound = bank.forward(model, scene, record, tid, step=100)
                for a, b in zip(direct[:3], bound[:3]): self.assertTrue(torch.equal(a, b))
            self.assertEqual(route_state_hash(model), original)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(r._forward_hooks); self.assertFalse(hasattr(r, '_context_busy'))
            self.assertEqual(len(bank.to_dict()['contexts']), len(ids))

    def test_authenticated_bank_cross_tile_data_node_role_and_test_isolation(self):
        model, scene, record, ids, bank = setup('joint')
        altered = bank.to_dict(); rows = list(altered['contexts'].values())
        rows[0]['context'], rows[1]['context'] = rows[1]['context'], rows[0]['context']
        with self.assertRaises(ValueError): RouteContextBank(altered, expected_hash=bank.bank_hash)
        altered['bank_hash'] = digest({k:v for k,v in altered.items() if k != 'bank_hash'})
        with self.assertRaises(ValueError): RouteContextBank(altered, expected_hash=altered['bank_hash'])
        other = copy.deepcopy(scene); other.cube[tile_slice(scene.plan['tiles'][ids[0]])] += .01
        with self.assertRaises(ValueError): bank.forward(model, other, record, ids[0], step=0)
        other = copy.deepcopy(scene)
        for row in record['roles']:
            if row['role'] not in ('devtrain', 'devval'):
                other.cube[tile_slice(scene.plan['tiles'][row['id']])] = np.nan
                with self.assertRaises(ValueError): bank.forward(model, other, record, row['id'], step=0)
        again = RouteContextBank.freeze(model, other, record, tile_ids=ids, step=7,
                                       expected_source_state=route_state_hash(model))
        self.assertEqual(again.bank_hash, bank.bank_hash)
        for bad in (True, -1, .5):
            with self.assertRaises(ValueError): bank.forward(model, scene, record, ids[0], step=bad)
        with self.assertRaises(ValueError): RouteContextBank(bank.to_dict(), expected_hash='0'*64)
        altered = bank.to_dict(); row = next(iter(altered['contexts'].values())); row['fixed']['nodes'] += 1
        altered['bank_hash'] = digest({k:v for k,v in altered.items() if k != 'bank_hash'})
        with self.assertRaises(ValueError): RouteContextBank(altered, expected_hash=altered['bank_hash'])

    def test_validation_no_gradient_and_exception_restores_router(self):
        model, scene, record, ids, bank = setup()
        tid = next(i for i in ids if record['roles'][i]['role'] == 'devval')
        model.train()
        with self.assertRaises(ValueError): bank.forward(model, scene, record, tid, step=1)
        model.eval(); before = model.backbone.graph_router.get_extra_state()
        out = bank.forward(model, scene, record, tid, step=1)
        self.assertTrue(all(not p.requires_grad for p in out[:3]))
        with patch('transfer.graph_route_context._forward', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'): bank.forward(model, scene, record, tid, step=2)
        self.assertEqual(before, model.backbone.graph_router.get_extra_state())
        self.assertFalse(hasattr(model.backbone.graph_router, '_context_busy'))
        bank.forward(model, scene, record, tid, step=3)

    def test_actual_updates_keep_support_and_checkpoint_optimizer_recovery(self):
        for mode in BudgetedGraphRouter.MODES:
            model, scene, record, ids, bank = setup(mode)
            training = [i for i in ids if record['roles'][i]['role'] == 'devtrain']
            source_hash = route_state_hash(model); opt = torch.optim.Adam(model.parameters(), lr=.001)
            graphs = []; forward = BudgetedGraphRouter.forward
            def capture(router, *args):
                out = forward(router, *args); graphs.append(out[0].detach().coalesce().clone()); return out
            with patch.object(BudgetedGraphRouter, 'forward', capture), tempfile.TemporaryDirectory() as d:
                losses = [update(model, bank, scene, record, training[i % len(training)], opt, i) for i in range(2)]
                path = Path(d)/'model.pt'; save_model_checkpoint(model, path)
                pin = hashlib.sha256(path.read_bytes()).hexdigest(); slots = copy.deepcopy(opt.state_dict())
                losses += [update(model, bank, scene, record, training[i % len(training)], opt, i) for i in range(2, 6)]
                self.assertEqual(pin, hashlib.sha256(path.read_bytes()).hexdigest())
                restored = load_model_checkpoint(path); restored.backbone.adapter.cache_attention = False
                resumed = torch.optim.Adam(restored.parameters(), lr=.001); resumed.load_state_dict(slots)
                rb = RouteContextBank(json.loads(json.dumps(bank.to_dict())), expected_hash=bank.bank_hash)
                replay = [update(restored, rb, scene, record, training[i % len(training)], resumed, i) for i in range(2, 6)]
                self.assertEqual(losses[2:], replay); self.assertEqual(route_state_hash(model), route_state_hash(restored))
                for k, state in opt.state_dict()['state'].items():
                    for name, tensor in state.items(): self.assertTrue(torch.equal(tensor, resumed.state_dict()['state'][k][name]))
            self.assertNotEqual(route_state_hash(model), source_hash)
            for i, graph in enumerate(graphs[:6]):
                tid = training[i % len(training)]
                fixed = next(r['fixed'] for r in bank.to_dict()['contexts'].values() if r['context']['tile_id'] == tid)
                self.assertEqual((graph.indices()[0]*graph.shape[0]+graph.indices()[1]).tolist(), fixed['selected'])
                self.assertTrue(torch.all(graph.values() > 0)); self.assertLessEqual(graph._nnz(), 2*graph.shape[0])
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                                for n,p in model.named_parameters() if 'graph_builder' in n))

    def test_mutated_candidates_or_bank_fail_and_state_restored(self):
        model, scene, record, ids, bank = setup()
        original = BudgetedGraphRouter.forward; before = model.backbone.graph_router.get_extra_state()
        def wrong(router, adjacency, *features):
            return original(router, torch.eye(adjacency.shape[0]), *features)
        with patch.object(BudgetedGraphRouter, 'forward', wrong):
            with self.assertRaisesRegex(ValueError, 'candidate'): bank.forward(model, scene, record, ids[0], step=0)
        self.assertEqual(before, model.backbone.graph_router.get_extra_state())
        bank._value['freeze_step'] += 1
        with self.assertRaises(ValueError): bank.forward(model, scene, record, ids[0], step=0)

    def test_disabled_legacy_path_untouched(self):
        model, scene, record = fixture(); before = state_hash(model)
        tid = next(r['id'] for r in record['roles'] if r['role'] == 'devtrain')
        with torch.no_grad(): a = _forward(model, scene, scene.plan['tiles'][tid])
        with self.assertRaises(ValueError): RouteContextBank.freeze(model, scene, record, tile_ids=[tid], step=0,
                                                                  expected_source_state=route_state_hash(model))
        with torch.no_grad(): b = _forward(model, scene, scene.plan['tiles'][tid])
        for x,y in zip(a[:3],b[:3]): self.assertTrue(torch.equal(x,y))
        self.assertEqual(before,state_hash(model))


if __name__ == '__main__': unittest.main()
