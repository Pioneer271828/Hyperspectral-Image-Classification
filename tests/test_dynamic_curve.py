"""Analytic development curve oracles and evidence/selection boundary failures."""
import copy
import json
import math
import unittest
import numpy as np
from model.research_config import digest
from inference.gate_calibration import GateProtocol
from evaluation.dynamic_curve import CurveProtocol, area, timing_summary, quality
from tests.test_gate_calibration import fixture as gate_fixture, data_pin


def fixture():
    gate, _, records = gate_fixture()
    methods = [('full','full',False,False,30.), ('fast','fast',True,True,10.),
               ('a0','adaptive',True,True,10.), ('a1','adaptive',True,False,20.), ('a2','adaptive',False,False,30.),
               ('r0','random',True,True,10.), ('r1','random',False,True,20.), ('r2','random',False,False,30.)]
    s = dict(schema='dy-development-curve-v1', gate_protocol=digest(gate), role='confirmation',
             points=[dict(id=n,method=m,policy_hash=digest(n)) for n,m,*_ in methods], environment_hash='3'*64,
             timing_scope='resident_cpu_batch_public_api_validation_and_compute', warmups=2, repeats=4,
             cost_interval_ms=[10.,30.], area_definition='all_points_cost_sorted_piecewise_linear_trapezoid_no_envelope',
             ece_bins=2,nll_epsilon=1e-8)
    results=[]
    for n,m,a,b,t in methods:
        results.append(dict(id=n,policy_hash=digest(n),full_state=gate['full_state'],light_artifact=gate['light_artifact'],
            data_hash=gate['data_hashes']['confirmation'],environment_hash=s['environment_hash'],timing_scope=s['timing_scope'],
            warmups=2,routes=[dict(id=records[0]['id'],fast=a),dict(id=records[1]['id'],fast=b)],
            wall_ms=[t-3,t-1,t+1,t+3],execution_receipt_sha256=digest(['synthetic_oracle',n])))
    return gate,s,records,results


def run(g,s,records,results):
    return CurveProtocol(GateProtocol(g,digest(g)),s,expected_hash=digest(s)).report(records,results,expected_results_hash=digest(results))


class DynamicCurveTests(unittest.TestCase):
    def test_analytic_all_points_classes_routes_and_area(self):
        g,s,r,v=fixture(); result=run(g,s,r,v);by={p['id']:p for p in result['points']}
        self.assertEqual(by['fast']['quality']['confusion'],[[2,0],[1,1]])
        self.assertEqual(by['fast']['quality']['recall'],[1.,.5]);self.assertFalse(by['fast']['all_groups_protected'])
        self.assertTrue(by['a1']['all_groups_protected']);self.assertEqual(by['a1']['fallback_corrected'],1)
        self.assertEqual(by['a1']['correction_given_fallback_fast_error'],1.)
        self.assertEqual(by['a0']['wrong_given_accepted'],.25)
        self.assertIsNone(by['full']['wrong_given_accepted'])
        self.assertEqual(by['a1']['accepted_tile_fraction'],.5)
        self.assertAlmostEqual(by['fast']['quality']['nll'],-(3*math.log(float(np.float32(.9)))+math.log(float(np.float32(.4))))/4)
        self.assertAlmostEqual(by['fast']['quality']['ece'],.075,places=6)
        curves={c['method']:c for c in result['curves']}
        self.assertEqual(curves['adaptive']['areas']['oa']['value'],18.75)
        self.assertEqual(curves['random']['areas']['oa']['value'],16.25)
        self.assertEqual(curves['full']['areas']['oa']['status'],'insufficient_points')
        self.assertFalse(result['selection_performed']);self.assertFalse(result['real_speed_claim'])
        self.assertEqual(len(result['points']),8);json.dumps(result,allow_nan=False)

    def test_order_and_json_are_stable_and_no_mutation(self):
        g,s,r,v=fixture();before=copy.deepcopy(v);a=run(g,s,r,v)
        b=run(g,json.loads(json.dumps(s)),r[::-1],json.loads(json.dumps(v))[::-1])
        # Evidence hash preserves supplied order; summaries use registration order.
        self.assertEqual(a['points'],b['points']);self.assertEqual(a['curves'],b['curves']);self.assertEqual(v,before)

    def test_fixed_interval_clipping_no_envelope_and_no_extrapolation(self):
        g,s,r,v=fixture();s['cost_interval_ms']=[15.,25.];out=run(g,s,r,v)
        adaptive=next(c for c in out['curves'] if c['method']=='adaptive')
        self.assertEqual(adaptive['areas']['oa']['value'],9.6875)
        s['cost_interval_ms']=[5.,35.];out=run(g,s,r,v)
        self.assertTrue(all(c['areas']['oa']['value'] is None for c in out['curves']))
        points=[dict(cost=dict(p50_ms=x),quality=dict(oa=y)) for x,y in [(1.,1.),(2.,0.),(3.,1.)]]
        self.assertEqual(area(points,[1.,3.],'oa')['value'],1.)

    def test_duplicate_cost_keeps_points_but_disables_area(self):
        g,s,r,v=fixture();v[3]['wall_ms']=v[2]['wall_ms'][:];out=run(g,s,r,v)
        c=next(c for c in out['curves'] if c['method']=='adaptive')
        self.assertEqual(c['areas']['oa']['status'],'duplicate_cost');self.assertEqual(len(out['points']),8)

    def test_missing_small_class_is_not_success_or_zero_aa(self):
        g,s,r,v=fixture()
        for row in r:row['labels'][1]=1
        g['data_hashes']['confirmation']=data_pin(r,g['blocks']);s['gate_protocol']=digest(g)
        for row in v:row['data_hash']=g['data_hashes']['confirmation']
        out=run(g,s,r,v);self.assertTrue(all(p['quality']['aa'] is None and not p['all_groups_protected'] for p in out['points']))
        self.assertEqual(next(c for c in out['curves'] if c['method']=='adaptive')['areas']['aa']['status'],'undefined_metric')

    def test_ece_bins_zero_probability_and_no_labels(self):
        p=np.array([[0.,1.],[.5,.5],[1.,0.]])
        q=quality(np.array([1,2,1]),p,2,2,1e-8)
        self.assertAlmostEqual(q['nll'],(-math.log(1e-8)-math.log(.5))/3)
        self.assertEqual(q['calibration_bins'][0]['count'],0);self.assertEqual(q['calibration_bins'][1]['count'],3)
        self.assertAlmostEqual(q['ece'],.5)
        self.assertIsNone(quality(np.array([],dtype=int),np.empty((0,2)),2,2,1e-8)['nll'])

    def test_evidence_identity_and_point_deletion_rejected(self):
        g,s,r,v=fixture();p=CurveProtocol(GateProtocol(g,digest(g)),s,expected_hash=digest(s))
        with self.assertRaises(ValueError):p.report(r,v,expected_results_hash='0'*64)
        for key in ('policy_hash','full_state','light_artifact','data_hash','environment_hash','timing_scope','execution_receipt_sha256'):
            bad=copy.deepcopy(v);bad[0][key]='wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):run(g,s,r,bad)
        for bad in (v[:-1],v[:-1]+v[:1]):
            with self.assertRaises(ValueError):run(g,s,r,bad)
        p._spec['ece_bins']=10
        with self.assertRaises(ValueError):p.report(r,v,expected_results_hash=digest(v))

    def test_coverage_endpoints_routes_and_measurement_budget_rejected(self):
        g,s,r,v=fixture()
        for mutate in (lambda x:x[0]['routes'][0].update(fast=True),lambda x:x[1]['routes'][0].update(fast=False),
                       lambda x:x[2]['routes'][0].update(fast=1),lambda x:x[2]['routes'].pop(),
                       lambda x:x[2].update(warmups=True),lambda x:x[2].update(wall_ms=[1,2,3]),
                       lambda x:x[2].update(wall_ms=[0,1,2,3]),lambda x:x[2].update(wall_ms=[True,1,2,3])):
            bad=copy.deepcopy(v);mutate(bad)
            with self.assertRaises(ValueError):run(g,s,r,bad)
        r[0]['coverage_known']=False;g['data_hashes']['confirmation']=data_pin(r,g['blocks']);s['gate_protocol']=digest(g)
        for row in v:row['data_hash']=g['data_hashes']['confirmation']
        with self.assertRaises(ValueError):run(g,s,r,v)

    def test_contract_role_interval_and_required_controls(self):
        g,s,r,v=fixture()
        for key,value in [('role','query'),('gate_protocol','0'*64),('repeats',3),('warmups',0),('ece_bins',True),
                          ('nll_epsilon',0),('cost_interval_ms',[30,10]),('area_definition','best_envelope')]:
            bad=copy.deepcopy(s);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):run(g,bad,r,v)
        bad=copy.deepcopy(s);bad['points']=bad['points'][2:]
        with self.assertRaises(ValueError):run(g,bad,r,v[2:])

    def test_raw_timing_statistics_are_measured_not_call_counts(self):
        out=timing_summary([1.,2.,3.,4.],repeats=4,blocks=6)
        self.assertEqual(out['p50_ms'],2.5);self.assertAlmostEqual(out['p95_ms'],3.85)
        self.assertAlmostEqual(out['sample_sd_ms'],math.sqrt(5/3));self.assertEqual(out['blocks_per_second'],2400.)


if __name__=='__main__':unittest.main()
