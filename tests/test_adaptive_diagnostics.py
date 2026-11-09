import copy,json,unittest
import numpy as np
from evaluation.adaptive_diagnostics import diagnose,cost_report


def record(tile=0):
    return dict(scene='scene',condition='full',tile_id=tile,ids=np.arange(4)+tile*4,
                labels=np.array([2,2,7,7]),mask=np.ones(4,dtype=bool),
                fast=np.array([[.9,.1,0],[.8,.2,0],[.6,.4,0],[.7,.3,0]]),
                full=np.array([[.8,.2,0],[.1,.9,0],[.2,.8,0],[.7,.3,0]]))


def run(rows,**kwargs):
    return diagnose(rows,[2,7,11],role=kwargs.get('role','validation'),readout_kind=kwargs.get('kind','terminal_auxiliary'))


class OfflineDiagnosticTests(unittest.TestCase):
    def test_four_cells_harm_correction_and_distinct_oracles(self):
        r=run([record()]);self.assertEqual(r['four_way'],dict(both_correct=1,fast_only_correct=1,full_only_correct=1,both_wrong=1))
        self.assertEqual(r['full_corrects_fast'],1);self.assertEqual(r['full_harms_fast'],1)
        self.assertEqual(r['pixel_oracle_accuracy'],.75);self.assertEqual(r['block_oracle']['oa'],.5)
        self.assertFalse(r['blocks'][0]['oracle_use_full']);self.assertFalse(r['deployment_speed_evidence'])
        self.assertEqual(r['fast']['per_class'][1]['recall'],0)
        self.assertEqual(r['full']['per_class'][0]['recall'],.5)
        self.assertIsNone(r['fast']['per_class'][2]['recall']);self.assertIsNone(r['fast']['aa_full'])
        np.testing.assert_allclose(r['blocks'][0]['fast_uncertainty']['margin'],[.8,.6,.2,.4])
        self.assertAlmostEqual(r['blocks'][0]['fast_uncertainty']['entropy_nats'][0],-.9*np.log(.9)-.1*np.log(.1))
        json.dumps(r,allow_nan=False)

    def test_block_population_groups_and_order(self):
        a=record();b=record(1);b['full']=b['fast'].copy();b['full'][2]=[0,1,0];b['mask'][3]=False;b['condition']='missing'
        r=run([a,b]);self.assertEqual(r['count'],7);self.assertEqual(r['block_oracle']['oa'],5/7)
        self.assertTrue(r['blocks'][1]['oracle_use_full']);self.assertEqual(len(r['groups']),2)
        p=[2,0,3,1];shuffled=copy.deepcopy(a)
        for key in ('ids','labels','mask','fast','full'):shuffled[key]=shuffled[key][p]
        q=run([b,shuffled])
        for key in ('fast','full','four_way','block_oracle','pixel_oracle_accuracy','groups'):self.assertEqual(r[key],q[key])

    def test_unlabeled_and_masked_poison(self):
        a=record();a['mask'][2:]=False;b=copy.deepcopy(a)
        b['labels']=b['labels'].astype(float);b['labels'][2:]=np.nan;b['fast'][2:]=np.nan;b['full'][2:]=np.nan
        self.assertEqual(run([a]),run([b]))
        a['mask'][:]=False;r=run([a]);self.assertEqual(r['unlabeled_blocks'],1)
        self.assertIsNone(r['blocks'][0]['oracle_use_full']);self.assertIsNone(r['block_oracle']['oa'])
        self.assertIsNone(run([])['pixel_oracle_accuracy']);json.dumps(r,allow_nan=False)

    def test_invalid_roles_ids_alignment_and_labels(self):
        for role in ('test','query','devval',''):
            with self.assertRaises(ValueError):run([record()],role=role)
        for kind in ('','early','cnn'):
            with self.assertRaises(ValueError):run([record()],kind=kind)
        with self.assertRaises(ValueError):run([record(),record()])
        b=record(1);b['ids']=record()['ids']
        with self.assertRaises(ValueError):run([record(),b])
        for field,value in [('ids',[0,0,1,2]),('ids',[0.,1.,2.,3.]),('tile_id',True),('labels',[2,0,7,7]),('mask',[1,1,1,1]),('fast',np.ones((4,2)))]:
            b=record();b[field]=value
            with self.assertRaises(ValueError):run([b])
        with self.assertRaises(ValueError):diagnose([record()],[2,2,7],role='validation',readout_kind='terminal_auxiliary')

    def test_invalid_probabilities(self):
        for bad in ([np.nan,0,1],[-.1,.1,1],[.5,.5,.5],[1.1,-.1,0]):
            b=record();b['fast'][0]=bad
            with self.assertRaises(ValueError):run([b])
        b=record();b['fast'][0]=[.5,.5,0]
        self.assertEqual(run([b])['fast']['per_class'][0]['recall'],1)

    def test_cost_independent_endpoints_and_shared_prefix(self):
        c={'fast':[1,2],'full':[10,20],'gate':[.1,.2],'reorder':[.3,.4]}
        fast=cost_report(c,np.array([False,False]),kind='independent_light')
        np.testing.assert_allclose(fast['per_block'],[1.4,2.6]);self.assertEqual(fast['run_full_fraction'],0)
        full=cost_report(c,np.array([True,True]),kind='independent_light')
        np.testing.assert_allclose(full['per_block'],[11.4,22.6]);self.assertGreater(full['mean'],full['full_baseline_mean'])
        self.assertAlmostEqual(full['sample_sd'],np.std([11.4,22.6],ddof=1))
        c={'prefix':[2,3],'exit':[1,1],'remaining':[10,20],'gate':[.1,.1],'reorder':[.2,.2]}
        q=cost_report(c,np.array([False,True]),kind='shared_prefix_exit')
        np.testing.assert_allclose(q['per_block'],[3.3,24.3]);self.assertEqual(q['run_full_fraction'],.5)

    def test_cost_missing_empty_bad_and_terminal(self):
        self.assertIsNone(cost_report(None,None,kind='terminal_auxiliary')['total'])
        with self.assertRaises(ValueError):cost_report({},[],kind='terminal_auxiliary')
        c={k:[] for k in ('fast','full','gate','reorder')}
        self.assertIsNone(cost_report(c,np.array([],dtype=bool),kind='independent_light')['mean'])
        c={k:[1] for k in c}
        self.assertIsNone(cost_report(c,np.array([True]),kind='independent_light')['sample_sd'])
        for bad in ([np.nan],[-1],[np.inf],[1,2]):
            d=copy.deepcopy(c);d['fast']=bad
            with self.assertRaises(ValueError):cost_report(d,np.array([True]),kind='independent_light')
        with self.assertRaises(ValueError):cost_report(c,[1],kind='independent_light')
        with self.assertRaises(ValueError):cost_report({'fast':[1]},[True],kind='independent_light')

    def test_diagnostic_does_not_mutate_inputs_or_select_threshold(self):
        a=record();before=copy.deepcopy(a);r=run([a])
        for key in ('ids','labels','mask','fast','full'):np.testing.assert_array_equal(a[key],before[key])
        self.assertNotIn('threshold',r);self.assertEqual(r['readout_kind'],'terminal_auxiliary')

if __name__=='__main__':unittest.main()
