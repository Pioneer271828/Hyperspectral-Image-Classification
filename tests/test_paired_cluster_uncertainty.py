"""Paired cluster draws, rare-class failures and conditional interval boundaries."""
import copy
import itertools
import json
import unittest
import numpy as np
from model.research_config import digest
from evaluation.paired_cluster_uncertainty import PairedClusterProtocol, statistics_from_counts


def fixture():
    units=[];records=[]
    for condition in ('full','gap'):
        for cluster,k in enumerate((-2,-1,0,1)):
            uid=condition+str(cluster)
            units.append(dict(id=uid,scene='scene1',condition=condition,cluster=str(cluster)))
            records.append(dict(id=uid,candidate_policy_hash='1'*64,reference_state_hash='2'*64,
                                candidate=[[8+k,2-k],[2-k,8+k]],reference=[[8,2],[2,8]]))
    s=dict(schema='dy-paired-cluster-v1',role='development_confirmation',candidate_policy_hash='1'*64,reference_state_hash='2'*64,
           data_hash=digest(records),units=units,classes=2,alpha=.2,replicates=1200,seed=64,min_clusters=4,min_class_clusters=4,
           min_tail_count=10,max_drop=dict(oa=.3,aa=.3,recall=.3),independence_evidence_sha256='3'*64,
           method='paired_scene_cluster_percentile_bonferroni_lower')
    return s,records


def run(s,r):
    return PairedClusterProtocol(s,expected_hash=digest(s)).report(r,expected_records_hash=digest(r))


class PairedClusterTests(unittest.TestCase):
    def test_independent_draw_oracle_shared_conditions_and_pairing(self):
        s,r=fixture();out=run(s,r);indices=np.random.default_rng(64).integers(0,4,size=(1200,4))
        oracle=np.array([-2,-1,0,1])[indices].mean(1)/10
        self.assertEqual(out['draws']['scene1']['indices_hash'],digest(indices.tolist()))
        self.assertEqual(out['simultaneous_dimensions'],12)
        for e in out['entries']:
            np.testing.assert_allclose(e['replicates'],oracle,rtol=0,atol=1e-15)
            self.assertAlmostEqual(e['difference'],-.05)
            self.assertAlmostEqual(e['lower_percentile'],sorted(oracle)[int(np.floor((len(oracle)-1)*.2/12))])
            self.assertTrue(e['approximate_bound_clears_margin'])
        self.assertFalse(out['automatic_deployment_authorized']);self.assertFalse(out['statistical_noninferiority_verified'])
        json.dumps(out,allow_nan=False)

    def test_exact_small_enumeration_statistic_and_missing_class(self):
        data=np.array([[[9,1],[3,7]],[[5,5],[1,9]]])
        for indices in itertools.product(range(2),repeat=2):
            matrix=data[list(indices)].sum(0);stat=statistics_from_counts(matrix)
            expected=[(matrix[0,0]+matrix[1,1])/40,(matrix[0,0]/20+matrix[1,1]/20)/2,matrix[0,0]/20,matrix[1,1]/20]
            np.testing.assert_array_equal(stat,expected)
        self.assertTrue(np.isnan(statistics_from_counts([[1,0],[0,0]])[1]))

    def test_identical_predictions_degenerate_does_not_prove_noninferiority(self):
        s,r=fixture()
        for row in r:row['candidate']=copy.deepcopy(row['reference'])
        s['data_hash']=digest(r);out=run(s,r)
        self.assertEqual(out['status'],'inconclusive_or_not_protected')
        self.assertTrue(all(e['lower_percentile']==0 and 'degenerate_resampling_distribution' in e['issues'] for e in out['entries']))

    def test_pooled_cancellation_does_not_hide_condition_harm(self):
        s,r=fixture()
        for row in r:
            if row['id'].startswith('gap'):
                k=8-row['candidate'][0][0];row['candidate']=[[8+k,2-k],[2-k,8+k]]
        s['data_hash']=digest(r);s['max_drop']=dict(oa=.01,aa=.01,recall=.01);out=run(s,r)
        pooled=[e for e in out['entries'] if e['group']['scene']=='__pooled__']
        self.assertTrue(all(e['difference']==0 and not e['approximate_bound_clears_margin'] for e in pooled))
        self.assertTrue(any(e['difference']<-.01 for e in out['entries']))

    def test_rare_class_cluster_and_invalid_bootstrap_replicates_retained(self):
        s,r=fixture()
        for row in r:
            if not row['id'].endswith('0'):
                row['candidate'][1]=[0,0];row['reference'][1]=[0,0]
        s['data_hash']=digest(r);out=run(s,r)
        e=next(e for e in out['entries'] if e['metric']=='recall_2')
        self.assertIn('too_few_class_clusters',e['issues']);self.assertIn('missing_class_in_resample',e['issues'])
        self.assertIsNone(e['lower_percentile']);self.assertLess(e['finite_replicates'],1200)
        self.assertTrue(any(x is None for x in e['replicates']));json.dumps(out,allow_nan=False)

    def test_tail_resolution_cluster_floor_and_independence_are_not_optional(self):
        s,r=fixture();s['replicates']=20;s['min_clusters']=5;s['independence_evidence_sha256']=None;out=run(s,r)
        for e in out['entries']:
            self.assertIn('insufficient_tail_resolution',e['issues']);self.assertIn('too_few_scene_clusters',e['issues'])
            self.assertIn('independence_evidence_missing',e['issues']);self.assertFalse(e['approximate_bound_clears_margin'])

    def test_local_rng_json_replay_order_and_protocol_mutation(self):
        s,r=fixture();np.random.seed(999);before=np.random.get_state();out=run(s,r);after=np.random.get_state()
        for a,b in zip(before,after):np.testing.assert_equal(a,b)
        self.assertEqual(out,run(json.loads(json.dumps(s)),json.loads(json.dumps(r))))
        s['units'].reverse();s['data_hash']=digest(r[::-1]);self.assertEqual(out['entries'],run(s,r[::-1])['entries'])
        p=PairedClusterProtocol(s,expected_hash=digest(s));p._spec['seed']=512
        with self.assertRaises(ValueError):p.report(r[::-1],expected_records_hash=digest(r[::-1]))

    def test_identity_pairs_counts_roles_and_complete_conditions_reject(self):
        s,r=fixture()
        for field,value in [('role','test'),('alpha',True),('replicates',True),('min_clusters',1),('min_tail_count',0),('method','unpaired_pixels')]:
            bad=copy.deepcopy(s);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):run(bad,r)
        for mutate in (lambda x:x[0].update(candidate_policy_hash='0'*64),lambda x:x[0]['candidate'][0].__setitem__(0,8.0),
                       lambda x:x[0]['candidate'][0].__setitem__(0,True),lambda x:x[0]['candidate'][0].__setitem__(0,9),
                       lambda x:x.pop(),lambda x:x.__setitem__(0,copy.deepcopy(x[1]))):
            bad=copy.deepcopy(r);mutate(bad);ss=copy.deepcopy(s);ss['data_hash']=digest(bad)
            with self.assertRaises(ValueError):run(ss,bad)
        ss=copy.deepcopy(s);ss['units'].pop()
        with self.assertRaises(ValueError):PairedClusterProtocol(ss,expected_hash=digest(ss))
        p=PairedClusterProtocol(s,expected_hash=digest(s))
        with self.assertRaises(ValueError):p.report(r,expected_records_hash='0'*64)


if __name__=='__main__':unittest.main()
