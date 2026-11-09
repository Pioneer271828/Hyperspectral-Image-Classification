"""CPU cost accounting with invariant gradients, state and stage cleanup."""
import copy
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from evaluation.graph_route_cost import benchmark_route,_once,_stages,_summary
from model.research_config import digest
from transfer.graph_route_context import route_state_hash
from tests.test_graph_edge_intervention import completed


class GraphCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.affinity=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(cls.affinity)})
        cls.fixture_run=completed('joint',False,3)
    @classmethod
    def tearDownClass(cls):os.sched_setaffinity(0,cls.affinity)

    def test_manual_timing_summary_and_nonfinite_rejection(self):
        r=_summary([dict(wall_ms=x,process_cpu_ms=x/2) for x in (1.,2.,3.,4.)])
        self.assertEqual(r['p50_ms'],2.5);self.assertAlmostEqual(r['p95_ms'],3.85)
        self.assertEqual(r['process_cpu_p50_ms'],1.25)
        for bad in (float('nan'),float('inf'),0.,-1.,True):
            with self.assertRaises(ValueError):_summary([dict(wall_ms=bad,process_cpu_ms=1)]*4)
        with self.assertRaises(ValueError):_summary([dict(wall_ms=1,process_cpu_ms=1)]*3)

    def test_actual_forward_backward_all_costs_preserve_original_state_and_gradients(self):
        run=self.fixture_run;tid=next(t['id'] for t in run.record['roles'] if t['role']=='devtrain')
        before=route_state_hash(run.model);source=route_state_hash(run.source);history=copy.deepcopy(run.history)
        slots=copy.deepcopy(run.optimizer.state_dict());rng=torch.get_rng_state().clone();nrng=np.random.get_state()
        grads={k:None if p.grad is None else p.grad.clone() for k,p in run.model.named_parameters()}
        with patch('torch.optim.Adam.step',side_effect=AssertionError('optimizer step forbidden')):
            r=benchmark_route(run,state_kind='selected',tile_ids=[tid],route_step=9,warmups=1,repeats=4)
        self.assertEqual((r['forward_calls'],r['backward_calls']),(20,10));self.assertEqual(r['optimizer_updates'],0)
        plan=r['plan'];self.assertEqual(plan['optimization_step'],run.selected_step)
        self.assertEqual(plan['plan_hash'],digest({k:v for k,v in plan.items() if k!='plan_hash'}))
        row=r['rows'][0]
        for arm in ('forward','forward_backward'):
            raw=[v['timings'][arm] for v in row['samples']];self.assertEqual(_summary(raw),row['summary'][arm])
            self.assertEqual(len(row['profiles'][arm]['stage_summary']),4)
            for s in row['profiles'][arm]['stage_summary'].values():self.assertGreater(s['p50_ms'],0)
        self.assertTrue(any(v is not None for v in row['baseline']['forward_backward']['gradients'].values()))
        self.assertTrue(all(v is None for v in row['baseline']['forward']['gradients'].values()))
        self.assertEqual(row['baseline']['forward']['probabilities'],row['baseline']['forward_backward']['probabilities'])
        self.assertEqual((before,source),(route_state_hash(run.model),route_state_hash(run.source)));self.assertEqual(history,run.history)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));np.testing.assert_array_equal(nrng[1],np.random.get_state()[1])
        for key,slot in slots['state'].items():
            for name,t in slot.items():self.assertTrue(torch.equal(t,run.optimizer.state_dict()['state'][key][name]))
        for k,p in run.model.named_parameters():
            if grads[k] is None:self.assertIsNone(p.grad)
            else:self.assertTrue(torch.equal(grads[k],p.grad))

    def test_fixed_random_route_cost_and_state_selection_are_explicit(self):
        run=completed('random',True,2);tid=next(t['id'] for t in run.record['roles'] if t['role']=='devtrain')
        r=benchmark_route(run,state_kind='last',tile_ids=[tid],route_step=17,warmups=1,repeats=4)
        self.assertEqual(r['plan']['optimization_step'],24);self.assertEqual(r['plan']['state_hash'],route_state_hash(run.model))
        for sig in r['rows'][0]['baseline'].values():
            self.assertTrue(sig['route']['route_fixed']);self.assertEqual(sig['route']['route_step'],17)
            self.assertEqual(sig['route']['route_output_nnz'],8)

    def test_instrumentation_is_exact_and_exception_cleans_private_methods(self):
        run=self.fixture_run;model=copy.deepcopy(run.model);tile=next(t for t in run.scene.plan['tiles'] if run.record['roles'][t['id']]['role']=='devtrain')
        _,base=_once(run,model,tile,3,True)
        with _stages(model) as times:_,got=_once(run,model,tile,3,True)
        self.assertEqual(base,got);self.assertTrue(all(len(v)==1 for v in times.values()))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with _stages(model):raise RuntimeError('injected')
        for owner,method in ((model.backbone,'graph_stage1'),(model.backbone,'graph_stage2'),(model.backbone.graph_router,'forward'),(model.backbone.learnable_graph,'build_graph')):
            self.assertNotIn(method,owner.__dict__)
        with _stages(model):
            with self.assertRaises(ValueError):
                with _stages(model):pass
        _,again=_once(run,model,tile,3,True);self.assertEqual(base,again)

    def test_reject_query_validation_malformed_scope_or_unfinished(self):
        run=self.fixture_run;tid=next(t['id'] for t in run.record['roles'] if t['role']=='devtrain')
        base=dict(state_kind='selected',tile_ids=[tid],route_step=0,warmups=1,repeats=4)
        invalid=[dict(state_kind='source'),dict(tile_ids=[]),dict(tile_ids=[tid,tid]),dict(tile_ids=[True]),
                 dict(route_step=True),dict(route_step=-1),dict(warmups=0),dict(repeats=3)]
        invalid += [dict(tile_ids=[t['id']]) for t in run.record['roles'] if t['role'] in ('devval','test')]
        for changes in invalid:
            with self.assertRaises(ValueError):benchmark_route(run,**(base|changes))
        with patch('evaluation.graph_route_cost.os.sched_getaffinity',return_value={0,1}):
            with self.assertRaises(ValueError):benchmark_route(run,**base)
        with patch.object(run,'finish',side_effect=ValueError('unfinished')):
            with self.assertRaisesRegex(ValueError,'unfinished'):benchmark_route(run,**base)

    def test_modified_output_and_instrumented_gradient_are_rejected(self):
        run=self.fixture_run;tid=next(t['id'] for t in run.record['roles'] if t['role']=='devtrain');calls=[]
        original=_once
        def change(*args,**kwargs):
            value,signature=original(*args,**kwargs);calls.append(1)
            if len(calls)==3:signature['probabilities'][0]='changed'
            return value,signature
        with patch('evaluation.graph_route_cost._once',side_effect=change):
            with self.assertRaisesRegex(ValueError,'Warmup'):benchmark_route(run,state_kind='selected',tile_ids=[tid],route_step=0,warmups=1,repeats=4)
        calls.clear()
        def change_profile(*args,**kwargs):
            value,signature=original(*args,**kwargs)
            if 'graph_stage1' in args[1].backbone.__dict__:signature['gradients']['fake']='changed'
            return value,signature
        with patch('evaluation.graph_route_cost._once',side_effect=change_profile):
            with self.assertRaisesRegex(ValueError,'Instrumentation'):benchmark_route(run,state_kind='selected',tile_ids=[tid],route_step=0,warmups=1,repeats=4)


if __name__=='__main__':unittest.main()
