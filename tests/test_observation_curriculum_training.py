"""Actual observation-order optimization, cross exposure and compatible recovery."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from model.research_config import digest
from transfer.curriculum_training import CurriculumRun
from transfer.curriculum_exposure import ExposurePlan
from transfer.observation_policy import ObservationPolicy
from transfer.observation_curriculum import CurriculumPolicy, ORDERS
from transfer.g3_training import state_hash, _forward
from tests.test_curriculum_training import setup as sample_setup
from tests.test_int8_model import fixture


def setup(order='easy_to_hard', seed=64):
    source,scene,record=fixture()
    configs=[{'mode':'none'},{'mode':'random_subset','keep_fraction':.5},
             {'mode':'random_subset','keep_fraction':.01}, {'mode':'contiguous_gap','lower':625.,'upper':775.,'unit':'nm'}]
    base=ObservationPolicy.freeze(scene,record,updates=24,training_configs=configs,train_seed=64,validation_seed=512)
    severity={digest(c):i for i,c in enumerate(configs)}
    obs=base if order is None else CurriculumPolicy.freeze(base,order=order,severity=severity,seed=2048)
    plan=ExposurePlan.freeze(scene,record,updates=24,seed=seed,minimum_class_blocks={str(c):1 for c in record['classes']},minimum_support_visits=1)
    return source,scene,record,plan,obs


def make(args):
    source,scene,record,plan,obs=args
    return CurriculumRun(source,scene,record,plan=plan,policy=None,lr=.0001,torch_seed=64,
            expected_source_state=state_hash(source),expected_plan_hash=plan.plan_hash,expected_policy_hash=None,
            observation_policy=obs,expected_observation_hash=obs.policy_hash)


def load(args,path,sha,contract):
    source,scene,record,plan,obs=args
    return CurriculumRun.load(source,scene,record,path,plan=plan,policy=None,observation_policy=obs,
            expected_sha256=sha,expected_contract_hash=contract)


class ObservationExecutionTests(unittest.TestCase):
    def test_four_orders_actual_bands_24_vs_7_plus_17_exact(self):
        shared_draws=None;multisets=[]
        for order in ORDERS:
            args=setup(order);run=make(args);before=state_hash(args[0]);rng=torch.get_rng_state().clone()
            with tempfile.TemporaryDirectory() as d:
                with patch('transfer.curriculum_training._forward',wraps=_forward) as forward:
                    run.advance(7);path=Path(d)/'partial.pt';sha=run.save(path);run.advance(24)
                    self.assertEqual(forward.call_count,24)
                    for n,call in enumerate(forward.call_args_list,1):
                        view=call.kwargs['observation_view'];self.assertEqual(len(view.bands.band_ids),args[4].training_record(n)['selected_band_count'])
                        self.assertEqual(args[2]['roles'][call.args[2]['id']]['role'],'devtrain')
                resumed=load(args,path,sha,run.contract_hash);resumed.advance(24)
                self.assertEqual(run.history,resumed.history);self.assertEqual(run.finish(),resumed.finish())
                self.assertEqual(state_hash(run.model),state_hash(resumed.model))
                for i,slot in run.optimizer.state_dict()['state'].items():
                    for k,v in slot.items():self.assertTrue(torch.equal(v,resumed.optimizer.state_dict()['state'][i][k]))
                path=Path(d)/'complete.pt';sha=run.save(path)
                with patch('torch.optim.Adam.step',side_effect=AssertionError('extra update')):
                    finished=load(args,path,sha,run.contract_hash);finished.advance(24);finished.finish()
            self.assertEqual(before,state_hash(args[0]));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            draws=[(r['tile_id'],r['anchor']) for r in run.history]
            if shared_draws is None:shared_draws=draws
            self.assertEqual(shared_draws,draws)
            multisets.append(Counter(r['observation']['record_hash'] for r in run.history))
            self.assertEqual({1,4,6,8},{r['observation']['selected_band_count'] for r in run.history})
            self.assertEqual(24,run.finish()['observations']['completed'])
        self.assertTrue(all(m==multisets[0] for m in multisets))

    def test_fixed_order_exact_original_observation_policy(self):
        a=make(setup(None));b=make(setup('fixed'))
        for _ in range(24):
            x=a.step();y=b.step();x.pop('observation');y.pop('observation')
            self.assertEqual(x,y);self.assertEqual(state_hash(a.model),state_hash(b.model))
        self.assertEqual(a.selected_step,b.selected_step)

    def test_cross_exposure_not_hidden_by_equal_marginals(self):
        results=[]
        for order in ('fixed','easy_to_hard'):
            args=setup(order);run=make(args);run.advance(24);result=run.observation_exposure();results.append(result)
            labels=dict(zip(args[2]['train_support']['indices'],args[2]['train_support']['labels']))
            source=args[4].to_dict()['training']
            for key,counts in result['by_config'].items():
                rows=[r for r in run.history if digest(source[r['step']-1]['config'])==key]
                self.assertEqual(counts['updates'],len(rows))
                self.assertEqual(counts['tiles'],Counter(str(r['tile_id']) for r in rows))
                self.assertEqual(counts['support_visits'],Counter(str(i) for r in rows for i in r['support_ids']))
                self.assertEqual(counts['class_blocks'],Counter(str(c) for r in rows for c in {labels[i] for i in r['support_ids']}))
                self.assertEqual(counts['class_label_visits'],Counter(str(labels[i]) for r in rows for i in r['support_ids']))
            json.dumps(result,allow_nan=False)
        self.assertEqual(results[0]['record_multiset_hash'],results[1]['record_multiset_hash'])
        self.assertEqual([v['updates'] for v in results[0]['by_config'].values()],[v['updates'] for v in results[1]['by_config'].values()])
        self.assertNotEqual(results[0]['by_config'],results[1]['by_config'])

    def test_completed_selected_condition_review_never_reselects_or_updates(self):
        run=make(setup())
        with self.assertRaises(ValueError):run.review_observation_conditions()
        run.advance(24);before=state_hash(run.model);step=run.selected_step;history=copy.deepcopy(run.history);rng=torch.get_rng_state().clone()
        with patch('torch.optim.Adam.step',side_effect=AssertionError('review update')):
            result=run.review_observation_conditions()
        self.assertEqual(result['selected_step'],step);self.assertFalse(result['policy_selection'])
        self.assertEqual(before,state_hash(run.model));self.assertEqual(history,run.history);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(len(result['review']['conditions']),6)
        self.assertEqual(result['review']['conditions']['full']['fused']['nll'],history[step-1]['validation']['nll'])
        json.dumps(result,allow_nan=False)

    def test_wrong_policy_dual_curriculum_zero_control_and_resume_tamper(self):
        args=setup();s,scene,r,p,obs=args;sample=sample_setup()[-1]
        kw=dict(plan=p,policy=None,lr=.0001,torch_seed=64,expected_source_state=state_hash(s),expected_plan_hash=p.plan_hash,expected_policy_hash=None,observation_policy=obs,expected_observation_hash=obs.policy_hash)
        for changes in ({'expected_observation_hash':'0'*64},{'policy':sample,'expected_policy_hash':sample.policy_hash},{'observation_policy':None}):
            with self.assertRaises(ValueError):CurriculumRun(s,scene,r,**(kw|changes))
        zero=ObservationPolicy.freeze(scene,r,updates=24,training_configs=[{'mode':'none'}],train_seed=64,validation_seed=512,training_application='support_zero')
        with self.assertRaises(ValueError):CurriculumRun(s,scene,r,**(kw|dict(observation_policy=zero,expected_observation_hash=zero.policy_hash)))
        run=make(args);run.advance(2)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'partial.pt';sha=run.save(path)
            with self.assertRaises(ValueError):load(setup('hard_to_easy'),path,sha,run.contract_hash)
            v=torch.load(path,weights_only=True);v['history'][0]['observation']['selected_band_count']=99
            bad=Path(d)/'bad.pt';torch.save(v,bad)
            with self.assertRaisesRegex(ValueError,'observation'):load(args,bad,hashlib.sha256(bad.read_bytes()).hexdigest(),run.contract_hash)


if __name__=='__main__':unittest.main()
