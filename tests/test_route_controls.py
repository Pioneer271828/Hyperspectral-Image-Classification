"""Random route controls preserve per-stratum counts and execute all charged work."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from inference.route_controls import random_plan,random_cascade,benchmark_controls,request_hash
from inference.trained_light import export_trained_light,load_trained_light
from transfer.light_head_training import LightHeadRun
from transfer.g3_training import state_hash
from model.research_config import digest
from tests.test_tile_cascade import setup
from tests.test_int8_model import fixture


class RouteControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full,scene,record=fixture();run=LightHeadRun(full,scene,record,updates=3,seed=64,lr=.0001,class_balance=True);run.advance(3)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'run.pt';pin=run.save(p);out=Path(tmp)/'light.pt'
            sha=export_trained_light(full,scene,record,p,out,expected_training_sha256=pin,expected_contract_hash=run.contract_hash)
            cls.published=load_trained_light(full,out,expected_sha256=sha,expected_training_sha256=pin,expected_contract_hash=run.contract_hash)
        cls.full=full;_,_,rows=setup();cls.rows=rows+[dict(copy.deepcopy(r),id=str(i+2),x=r['x'][:9,:11].clone()) for i,r in enumerate(rows)]

    def setUp(self):
        self.f=copy.deepcopy(self.full);self.p=copy.deepcopy(self.published);self.r=copy.deepcopy(self.rows)
        self.ref=[dict(id=r['id'],route='fast' if i%2==0 else 'full') for i,r in enumerate(self.r)]

    def plan(self,**kwargs):
        args=dict(seed=64,min_coverage=.8);args.update(kwargs);return random_plan(self.f,self.p,self.r,self.ref,**args)

    def test_per_shape_counts_order_invariance_and_local_determinism(self):
        rng=torch.get_rng_state().clone()
        for seed in (64,512,2048):
            a=self.plan(seed=seed);b=random_plan(self.f,self.p,self.r[::-1],self.ref[::-1],seed=seed,min_coverage=.8)
            self.assertEqual(a,b);self.assertEqual(len(a['strata']),2);self.assertEqual([g['fast_count'] for g in a['strata']],[1,1])
            self.assertEqual(len(a['selected']),2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_actual_light_and_graph_counts_and_outputs_match_route(self):
        self.r.append(dict(copy.deepcopy(self.r[0]),id='unknown',coverage_known=False));self.ref.append(dict(id='unknown',route='full'))
        plan=self.plan();before=(state_hash(self.f),state_hash(self.p.model),request_hash(self.r));graph=self.f.backbone.learnable_graph
        with patch.object(self.p.model,'forward',wraps=self.p.model.forward) as light,patch.object(graph,'build_graph',wraps=graph.build_graph) as spy:
            out=random_cascade(self.f,self.p,self.r,plan,expected_plan_hash=digest(plan));self.assertEqual(light.call_count,4);self.assertEqual(spy.call_count,3)
        self.assertEqual(out['counts'],dict(light_calls=4,full_calls=3,accepted_tiles=2,total_tiles=5))
        for r,a in zip(self.r,out['outputs']):
            with torch.no_grad():expected=(self.p.model(r['dataset'],r['x'],wavelengths=r['wavelengths']) if a['route']=='fast' else self.f(r['dataset'],r['x'],wavelengths=r['wavelengths'],superpixel_scale=r['scale'])[0])
            self.assertTrue(torch.equal(expected,a['probabilities']))
        self.assertEqual(before,(state_hash(self.f),state_hash(self.p.model),request_hash(self.r)))

    def test_zero_and_all_acceptance_endpoints_keep_light_cost_on_rejection(self):
        for mode in ('full','fast'):
            self.ref=[dict(id=r['id'],route=mode) for r in self.r];plan=self.plan()
            out=random_cascade(self.f,self.p,self.r,plan,expected_plan_hash=digest(plan))
            self.assertEqual(out['counts']['light_calls'],4);self.assertEqual(out['counts']['full_calls'],4 if mode=='full' else 0)

    def test_hash_tamper_and_mutated_input_rejected_before_execution(self):
        plan=self.plan();pin=digest(plan)
        with patch.object(self.f,'forward',side_effect=AssertionError('executed')),patch.object(self.p.model,'forward',side_effect=AssertionError('executed')):
            with self.assertRaises(ValueError):random_cascade(self.f,self.p,self.r,plan,expected_plan_hash='0'*64)
            bad=copy.deepcopy(plan);bad['selected']=[]
            with self.assertRaises(ValueError):random_cascade(self.f,self.p,self.r,bad,expected_plan_hash=digest(bad))
            self.r[0]['x'][0,0,0]+=.01
            with self.assertRaises(ValueError):random_cascade(self.f,self.p,self.r,plan,expected_plan_hash=pin)

    def test_reference_coverage_task_and_source_rejections(self):
        for ref in (self.ref[:-1],self.ref+[self.ref[0]],[dict(id='missing',route='fast')]+self.ref[1:]):
            with self.assertRaises(ValueError):random_plan(self.f,self.p,self.r,ref,seed=64,min_coverage=.8)
        self.r[0]['coverage_fraction']=.1
        with self.assertRaises(ValueError):self.plan()
        self.r[0]['coverage_fraction']=1.;self.r[0]['dataset']='unknown'
        with self.assertRaises(ValueError):self.plan()
        self.r=copy.deepcopy(self.rows)
        with torch.no_grad():next(self.f.parameters()).add_(.1)
        with self.assertRaises(ValueError):self.plan()

    def test_plan_json_roundtrip_and_request_order_restore(self):
        plan=self.plan();loaded=json.loads(json.dumps(plan));self.assertEqual(plan,loaded)
        out=random_cascade(self.f,self.p,self.r[::-1],loaded,expected_plan_hash=digest(plan))
        self.assertEqual([r['id'] for r in out['outputs']],[r['id'] for r in self.r[::-1]])
        self.assertEqual(sorted(r['id'] for r in out['outputs'] if r['route']=='fast'),plan['selected'])

    def test_actual_single_core_paired_cpu_measurement_preserves_outputs(self):
        old=os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0,{min(old)})
            result=benchmark_controls(self.f,self.p,self.r[:2],adaptive_kwargs=dict(mode='adaptive',criterion='max_probability',threshold=1,min_coverage=.8),seed=64,warmups=1,pairs=4)
        finally:os.sched_setaffinity(0,old)
        self.assertEqual(result['summary']['adaptive']['calls'],result['summary']['random']['calls'])
        self.assertEqual(result['summary']['random']['calls']['light_calls'],2)
        self.assertEqual(result['summary']['random']['calls']['full_calls'],2)
        self.assertEqual(len(result['samples']),4);self.assertFalse(result['scientific_speed_claim'])
        for name in result['summary']:
            self.assertEqual(len({r['order'].index(name) for r in result['samples']}),4)
            self.assertGreater(result['summary'][name]['p50_ms'],0)
        self.assertEqual(result['optimizer_updates'],0);self.assertEqual(result['query'],0)

    def test_benchmark_budget_and_fixed_configuration_rejected(self):
        with patch('inference.route_controls.os.sched_getaffinity',return_value={24}):
            for kw in ({'pairs':3},{'pairs':0},{'warmups':0},{'adaptive_kwargs':{'mode':'fast'}}):
                args=dict(adaptive_kwargs=dict(mode='adaptive',criterion='margin',threshold=.5),seed=64);args.update(kw)
                with self.assertRaises(ValueError):benchmark_controls(self.f,self.p,self.r,**args)
        with patch('inference.route_controls.os.sched_getaffinity',return_value={24,25}):
            with self.assertRaises(ValueError):benchmark_controls(self.f,self.p,self.r,adaptive_kwargs=dict(mode='adaptive',criterion='margin',threshold=.5),seed=64)


if __name__=='__main__':unittest.main()
