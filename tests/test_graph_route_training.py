"""Registered eight-arm E3 execution and legacy curriculum recovery boundaries."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transfer.graph_route_training import GraphRouteRun
from transfer.graph_route_context import RouteContextBank, route_state_hash
from transfer.curriculum_exposure import ExposurePlan
from transfer.sample_curriculum import curriculum_loss
from transfer.g3_training import _forward
from transfer.development_training import evaluate_validation
from tests.test_graph_route_context import setup


def arguments(mode='weight', fixed=False):
    source,scene,record,ids,bank = setup(mode)
    plan = ExposurePlan.freeze(scene,record,updates=24,seed=64,
            minimum_class_blocks={str(c):1 for c in record['classes']},minimum_support_visits=1)
    return source,scene,record,plan,bank if fixed else None


def make(args):
    source,scene,record,plan,bank=args
    return GraphRouteRun(source,scene,record,plan=plan,bank=bank,
        expected_bank_hash=None if bank is None else bank.bank_hash,lr=.0001,torch_seed=512,
        expected_source_state=route_state_hash(source),expected_plan_hash=plan.plan_hash)


def load(args,path,sha,contract):
    source,scene,record,plan,bank=args
    return GraphRouteRun.load(source,scene,record,path,plan=plan,bank=bank,
        expected_bank_hash=None if bank is None else bank.bank_hash,
        expected_sha256=sha,expected_contract_hash=contract)


class GraphRouteTrainingTests(unittest.TestCase):
    def test_eight_arms_24_vs_7_plus_17_exact_and_complete_zero(self):
        draws=[]
        for mode in ('weight','random','graph','joint'):
            for fixed in (False,True):
                args=arguments(mode,fixed);run=make(args);initial=route_state_hash(args[0])
                rng=torch.get_rng_state().clone()
                with tempfile.TemporaryDirectory() as d:
                    zero=Path(d)/'zero.pt';zs=run.save(zero)
                    self.assertEqual(load(args,zero,zs,run.contract_hash).completed,0)
                    run.advance(7);partial=Path(d)/'partial.pt';ps=run.save(partial);run.advance(24)
                    restored=load(args,partial,ps,run.contract_hash);restored.advance(24)
                    self.assertEqual(run.history,restored.history);self.assertEqual(run.finish(),restored.finish())
                    self.assertEqual(route_state_hash(run.model),route_state_hash(restored.model))
                    self.assertTrue(torch.equal(run.torch_rng,restored.torch_rng))
                    for key,slot in run.optimizer.state_dict()['state'].items():
                        for name,t in slot.items():self.assertTrue(torch.equal(t,restored.optimizer.state_dict()['state'][key][name]))
                    end=Path(d)/'complete.pt';es=run.save(end)
                    with patch('torch.optim.Adam.step',side_effect=AssertionError('extra update')):
                        done=load(args,end,es,run.contract_hash);done.advance(24);done.finish()
                        with self.assertRaises(ValueError):done.step()
                self.assertEqual(route_state_hash(args[0]),initial);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual(run.selected_step,min(range(24),key=lambda i:run.history[i]['validation']['nll'])+1)
                draws.append([(r['tile_id'],r['anchor']) for r in run.history])
                for r in run.history:
                    self.assertEqual(r['route']['route_step'],r['step']);self.assertEqual(r['route']['route_fixed'],fixed)
                    self.assertEqual(r['route']['route_mode'],mode)
                for group in ('adapter','cnn','graph','heads'):
                    self.assertGreater(sum(r['gradient_l1'][group] for r in run.history),0)
                    self.assertGreater(sum(r['changed_parameters'][group] for r in run.history),0)
        self.assertTrue(all(v==draws[0] for v in draws))

    def test_dynamic_random_matches_original_manual_updates_and_fixed_val_rule(self):
        args=arguments('random');run=make(args);source,scene,record,plan,_=args
        manual=copy.deepcopy(source);opt=torch.optim.Adam(manual.parameters(),lr=.0001)
        for n in range(1,25):
            row=run.step();tile=scene.plan['tiles'][plan.event(n)['tile_id']]
            manual.train();manual.backbone.graph_router.set_step(n);opt.zero_grad(set_to_none=True)
            loss=curriculum_loss(_forward(manual,scene,tile),scene,record,tile,policy=None,class_balance=True)['total']
            loss.backward();opt.step();manual.backbone.graph_router.set_step(0)
            self.assertEqual(float(loss.detach()),row['loss'])
            self.assertEqual(evaluate_validation(manual,scene,record),row['validation'])
            self.assertEqual(route_state_hash(manual),route_state_hash(run.model))

    def test_support_source_and_retraining_initialization_are_separate(self):
        args=arguments('joint');early=make(args);early.advance(2)
        selected=copy.deepcopy(early.model);selected.load_state_dict(early.selected_state);selected.eval()
        source,scene,record,plan,_=args;ids=[r['id'] for r in record['roles'] if r['role'] in ('devtrain','devval')]
        bank=RouteContextBank.freeze(selected,scene,record,tile_ids=ids,step=0,expected_source_state=route_state_hash(selected))
        retrain=make((source,scene,record,plan,bank));spec=retrain.spec['graph_route']
        self.assertNotEqual(spec['support_source_state'],spec['initialization_state'])
        self.assertEqual(route_state_hash(retrain.model),route_state_hash(source))
        self.assertEqual(spec['bank_hash'],bank.bank_hash)
        retrain.advance(1)

    def test_corrupt_state_optimizer_route_and_wrong_hash_rejected(self):
        args=arguments('random',True);run=make(args);run.advance(1)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'run.pt';sha=run.save(path)
            with patch('torch.load',side_effect=AssertionError('read before authentication')):
                with self.assertRaises(ValueError):load(args,path,'0'*64,run.contract_hash)
            mutations=[lambda v:v['state']['backbone.graph_router._extra_state'].update(step=1),
                       lambda v:v['history'][0]['route'].update(route_step=2),
                       lambda v:next(iter(v['optimizer']['state'].values()))['step'].add_(1),
                       lambda v:v.update(selected_step=0)]
            for i,mutate in enumerate(mutations):
                packet=torch.load(path,weights_only=True);mutate(packet);bad=Path(d)/f'bad{i}.pt';torch.save(packet,bad)
                with self.assertRaises(ValueError):load(args,bad,hashlib.sha256(bad.read_bytes()).hexdigest(),run.contract_hash)
            wrong=arguments('random',False)
            with self.assertRaises(ValueError):load(wrong,path,sha,run.contract_hash)

    def test_partial_failure_is_preserved_and_never_resumed(self):
        args=arguments();run=make(args);initial=route_state_hash(run.model)
        with patch.object(run,'_evaluate',side_effect=RuntimeError('injected after update')):
            with self.assertRaises(RuntimeError):run.advance(1)
        self.assertNotEqual(initial,route_state_hash(run.model));self.assertEqual(run.completed,0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'failure.pt';sha=run.save_failure(path)
            self.assertFalse(torch.load(path,weights_only=True)['resumable'])
            with self.assertRaises(ValueError):load(args,path,sha,run.contract_hash)
            with self.assertRaises(ValueError):run.save(Path(d)/'normal.pt')
        with self.assertRaises(ValueError):run.advance(1)

    def test_data_and_policies_cannot_change_during_execution(self):
        args=arguments('graph',True);run=make(args)
        source,scene,record,plan,bank=args
        with self.assertRaises(ValueError):GraphRouteRun(source,scene,record,plan=plan,policy=object(),lr=.001,
            torch_seed=1,expected_source_state=route_state_hash(source),expected_plan_hash=plan.plan_hash)
        with self.assertRaises(ValueError):GraphRouteRun(source,scene,record,plan=plan,bank=bank,expected_bank_hash='0'*64,lr=.001,
            torch_seed=1,expected_source_state=route_state_hash(source),expected_plan_hash=plan.plan_hash)
        run.bank._value['freeze_step']+=1
        with self.assertRaises(ValueError):run.advance(1)
        run=make(args);run.spec['graph_route']['validation_step']=1
        with self.assertRaises(ValueError):run.advance(1)
        run=make(args);tid=plan.event(1)['tile_id'];y0,y1,x0,x1=scene.plan['tiles'][tid]['box'];scene.cube[y0:y1,x0:x1]+=1
        with self.assertRaises(ValueError):run.advance(1)


if __name__=='__main__':unittest.main()
