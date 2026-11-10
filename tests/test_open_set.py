"""Numeric and information-boundary checks for E4c-1."""
import copy
import json
import math
import unittest

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score, roc_curve

from evaluation.open_set import (OpenSetScorer, calibrate, detection_metrics,
                                 digest, evaluate)


KNOWN = [dict(id=2, name='crop'), dict(id=7, name='road')]
UNKNOWN = [dict(id=9, name='water'), dict(id=11, name='roof')]


def scorer(method='msp', **kwargs):
    return OpenSetScorer.freeze(method, KNOWN, UNKNOWN, model_sha256='a'*64,
                               representation='frozen-head-1', **kwargs)


def calibrated(method='msp'):
    s = scorer(method)
    c, h = calibrate(s, [[4., 0.], [0., 4.]], [2, 7], ['cal/0', 'cal/1'],
                     input_kind='raw_logits', role='source_validation',
                     provenance_sha256='b'*64, known_coverage=1.)
    return s, c, h


class OpenSetTests(unittest.TestCase):
    def test_logits_scores_and_gauge(self):
        x = np.array([[1., 2.], [900., 899.], [-900., -901.], [0., 0.]])
        original = x.copy()
        for t in (.1, 1., 3., 100.):
            m = scorer('msp', temperature=t); e = scorer('energy', temperature=t)
            ms, p = m.score(x, input_kind='raw_logits')
            es, ep = e.score(x, input_kind='raw_logits')
            np.testing.assert_allclose(ms, 1-softmax(x/t, axis=1).max(1), atol=2e-13)
            np.testing.assert_allclose(es, -t*logsumexp(x/t, axis=1), atol=2e-12)
            np.testing.assert_array_equal(p, ep)
            np.testing.assert_allclose(m.score(x+13, input_kind='raw_logits')[0], ms)
            np.testing.assert_allclose(e.score(x+13, input_kind='raw_logits')[0], es-13)
        np.testing.assert_array_equal(x, original)
        # Same probabilities cannot determine absolute energy.
        for s in (m, e):
            with self.assertRaises(ValueError): s.score(softmax(x, axis=1), input_kind='probabilities')

    def test_prototypes(self):
        x = np.array([[0., 0.], [2., 0.], [8., 2.], [10., 2.]])
        kw = dict(source_features=x, source_labels=[2,2,7,7], source_ids=['s0','s1','s2','s3'],
                  source_role='source_train', source_provenance='c'*64)
        s = scorer('prototype_squared_distance', **kw)
        np.testing.assert_array_equal(s.as_dict()['prototype'], [[1.,0.],[9.,2.]])
        q = np.array([[2., 0.], [8., 3.], [5., 1.]])
        score, pred = s.score(q, input_kind='features')
        np.testing.assert_array_equal(score, [1., 2., 17.])
        np.testing.assert_array_equal(pred, [2, 7, 2]) # fixed column-order tie
        self.assertEqual(s.sha256, OpenSetScorer(json.loads(json.dumps(s.as_dict())), s.sha256).sha256)
        for change in ({'source_role':'query'}, {'source_labels':[2,2,2,2]},
                       {'source_labels':[2,0,7,7]}, {'source_ids':['s0']*4}):
            with self.assertRaises(ValueError): scorer('prototype_squared_distance', **(kw | change))
        with self.assertRaises(ValueError): s.score([[1.,2.,3.]], input_kind='features')

    def test_calibration_policy_and_ties(self):
        s = scorer(); x = [[4.,0.],[0.,3.],[2.,0.],[0.,0.]]; y=[2,7,2,7]
        ids=['a','b','c','d']
        for coverage in (.01, .5, .95, 1.):
            c,h=calibrate(s,x,y,ids,input_kind='raw_logits',role='independent_development',
                          provenance_sha256='b'*64,known_coverage=coverage)
            self.assertGreaterEqual(c['empirical_known_coverage'], coverage)
            self.assertEqual(h,digest(c))
        for role in ('query','target_validation','test','source_train'):
            with self.assertRaises(ValueError):
                calibrate(s,x,y,ids,input_kind='raw_logits',role=role,
                          provenance_sha256='b'*64,known_coverage=.95)
        for bad in ([2,7,9,7],[2,7,0,7],[2,2,2,2]):
            with self.assertRaises(ValueError):
                calibrate(s,x,bad,ids,input_kind='raw_logits',role='source_validation',
                          provenance_sha256='b'*64,known_coverage=.95)
        s,c,h=calibrated(); out=evaluate(s,c,h,[[4.,0.],[0.,4.]], [2,7], ['e0','e1'],input_kind='raw_logits')
        self.assertEqual(out['confusion'], [[1,0,0],[0,1,0],[0,0,0]])

    def test_metrics_hand_count_and_unlabeled(self):
        s,c,h=calibrated()
        x=[[4.,0.],[0.,0.],[0.,4.],[4.,0.],[0.,0.],[4.,0.],[100.,-100.]]
        y=[2,2,7,7,9,11,0]
        r=evaluate(s,c,h,x,y,[f'e{i}' for i in range(len(y))],input_kind='raw_logits')
        self.assertEqual(r['confusion'], [[1,0,1],[1,1,0],[1,0,1]])
        self.assertEqual(r['unlabeled_excluded'],1)
        self.assertEqual(r['known_per_class'][0]['f1'], .4) # unknown FP included
        self.assertAlmostEqual(r['known_macro_f1'],(.4+2/3)/2)
        self.assertEqual(r['osstar_class_macro_recall'],.5)
        self.assertEqual(r['unknown_sample_recall'],.5)
        self.assertEqual(r['hos'],.5)
        self.assertEqual([a['recall'] for a in r['unknown_per_class']], [1.,0.])
        r2=evaluate(s,c,h,x[:-1],y[:-1],[f'f{i}' for i in range(6)],input_kind='raw_logits')
        self.assertEqual(r['detection'],r2['detection'])

    def test_roc_oracle_ties_and_fpr_convention(self):
        rng=np.random.default_rng(64)
        for n in (2,5,37,101):
            for tied in (False,True):
                x=rng.normal(size=n); x=np.round(x) if tied else x
                y=np.arange(n)%2 == 0
                r=detection_metrics(x,y)
                self.assertAlmostEqual(r['auroc'],roc_auc_score(y,x),places=14)
                fpr,tpr,_=roc_curve(y,x,drop_intermediate=False)
                self.assertEqual(r['fpr95'],float(fpr[np.flatnonzero(tpr>=.95)[0]]))
        self.assertEqual(detection_metrics([0,0],[True,False])['auroc'],.5)
        self.assertEqual(detection_metrics([0,0],[True,False])['fpr95'],1.)
        self.assertEqual(detection_metrics([1,0],[True,False])['fpr95'],0.)

    def test_missing_groups_are_na(self):
        s,c,h=calibrated()
        for y in ([0,0],[2,2],[9,9],[2,9]):
            r=evaluate(s,c,h,[[4.,0.],[0.,0.]],y,['e0','e1'],input_kind='raw_logits')
            self.assertIsNone(r['known_macro_f1']); self.assertIsNone(r['hos'])
            if 2 not in y or 9 not in y: self.assertIsNone(r['detection']['auroc'])
        r=evaluate(s,c,h,[[0.,0.],[0.,0.],[4.,0.]], [2,7,9],['e0','e1','e2'],input_kind='raw_logits')
        self.assertEqual(r['hos'],0.)

    def test_identity_and_overlap(self):
        s,c,h=calibrated(); c0=copy.deepcopy(c); s0=s.as_dict()
        for ids in (['cal/0','e1'],['e0','e0']):
            with self.assertRaises(ValueError): evaluate(s,c,h,[[1.,0.],[0.,1.]], [2,7],ids,input_kind='raw_logits')
        bad=copy.deepcopy(c);bad['threshold']+=1
        with self.assertRaises(ValueError): evaluate(s,bad,h,[[0.,0.]], [9],['e'],input_kind='raw_logits')
        other=scorer('energy')
        with self.assertRaises(ValueError): evaluate(other,c,h,[[0.,0.]], [9],['e'],input_kind='raw_logits')
        for label in ([99],[0.5],[-1],[True]):
            with self.assertRaises(ValueError): evaluate(s,c,h,[[0.,0.]],label,['e'],input_kind='raw_logits')
        bad=s.as_dict();bad['known'].reverse()
        with self.assertRaises(ValueError): OpenSetScorer(bad,s.sha256)
        self.assertEqual(c,c0);self.assertEqual(s.as_dict(),s0)

    def test_invalid_contracts_and_numerics(self):
        for t in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError): scorer(temperature=t)
        for known,unknown in ((KNOWN,KNOWN),([],UNKNOWN),([dict(id=0,name='background')],UNKNOWN),
                              (KNOWN,[dict(id=12,name=' crop ')])):
            with self.assertRaises(ValueError):
                OpenSetScorer.freeze('msp',known,unknown,model_sha256='a'*64,representation='x')
        s=scorer('energy')
        for a in ([[float('nan'),0]],[[float('inf'),0]],[],[[1]],[[True,False]],[[1+2j,0]]):
            with self.assertRaises(ValueError): s.score(a,input_kind='raw_logits')
        with self.assertRaises(FloatingPointError): s.score([[1e308,-1e308]],input_kind='raw_logits')

    def test_prototype_split_overlap(self):
        s=scorer('prototype_squared_distance',source_features=[[0.,0.],[4.,4.]],source_labels=[2,7],
                 source_ids=['s0','s1'],source_role='source_train',source_provenance='c'*64)
        with self.assertRaises(ValueError):
            calibrate(s,[[1.,0.],[4.,3.]],[2,7],['s0','c1'],input_kind='features',
                      role='source_validation',provenance_sha256='b'*64,known_coverage=.95)
        c,h=calibrate(s,[[1.,0.],[4.,3.]],[2,7],['c0','c1'],input_kind='features',
                      role='source_validation',provenance_sha256='b'*64,known_coverage=.95)
        with self.assertRaises(ValueError): evaluate(s,c,h,[[2.,2.]],[9],['s1'],input_kind='features')
        r=evaluate(s,c,h,[[0.,0.],[4.,4.],[20.,20.]],[2,7,9],['e0','e1','e2'],input_kind='features')
        self.assertEqual(r['hos'],1.);self.assertEqual(r['detection']['auroc'],1.)

    def test_existing_m8_raw_logits_without_model_changes(self):
        import torch
        from tests.test_fusion_m8 import model
        torch.manual_seed(512)
        net=model().eval(); state=copy.deepcopy(net.state_dict())
        x=torch.randn(6,7,14)
        rng=torch.random.get_rng_state().clone()
        with torch.no_grad():
            raw=net('a',x,superpixel_scale=9,return_logits=True)
            probability=net('a',x,superpixel_scale=9)
        known=[dict(id=i+1,name=f'class-{i+1}') for i in range(3)]
        unknown=[dict(id=4,name='held-out-semantic')]
        for head,(z,p) in enumerate(zip(raw,probability)):
            torch.testing.assert_close(z.softmax(-1),p,rtol=0,atol=0)
            for method in ('msp','energy'):
                s=OpenSetScorer.freeze(method,known,unknown,model_sha256='a'*64,
                                      representation=f'M8/head-{head}')
                values=z.detach().numpy(); score,pred=s.score(values,input_kind='raw_logits')
                expected=1-z.double().softmax(-1).max(-1).values if method=='msp' else -z.double().logsumexp(-1)
                np.testing.assert_allclose(score,expected.numpy(),rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(pred,z.argmax(-1).numpy()+1)
        for k,v in net.state_dict().items(): torch.testing.assert_close(v,state[k],rtol=0,atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
