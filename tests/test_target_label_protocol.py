import copy,json,unittest
import numpy as np
from model.research_config import digest
from loadData.target_label_protocol import make_target_family,phase_view,final_query,validate_family


class TargetLabelProtocolTests(unittest.TestCase):
    def fixture(self):
        labels=np.repeat(np.arange(1,4),24).reshape(8,9);labels[0,:]=0
        return labels,make_target_family(labels,'paviaU',64)

    def test_nested_fixed_query_and_only_released_phase_labels(self):
        labels,f=self.fixture();nested=[];queries=[]
        for k in (1,5,10):
            a=phase_view(f,k,'adapt',expected_hash=f['family_hash']);v=phase_view(f,k,'validation',expected_hash=f['family_hash'])
            self.assertEqual(len(a['indices']),3*k);self.assertEqual(len(v['indices']),15)
            self.assertEqual(a['total_annotation_labels'],3*(k+5))
            self.assertFalse(set(a['indices'])&set(v['indices']))
            for view in (a,v):
                self.assertNotIn('query_indices',view);self.assertNotIn('support_order',view)
                self.assertFalse(view['query_labels_visible']);self.assertFalse(view['unreleased_max_pool_labels_visible'])
                self.assertEqual(labels.ravel()[view['indices']].tolist(),view['labels'])
            nested.append(set(a['indices']));queries.append(a['query_indices_hash'])
        self.assertTrue(nested[0]<=nested[1]<=nested[2]);self.assertEqual(len(set(queries)),1)
        self.assertEqual(f,make_target_family(labels,'paviaU',64))

    def test_final_query_requires_checkpoint_and_immutable_ground_truth(self):
        labels,f=self.fixture()
        for key in (None,'x','a'*40,True):
            with self.assertRaises(ValueError):final_query(f,labels,expected_hash=f['family_hash'],checkpoint_sha256=key)
        q=final_query(f,labels,expected_hash=f['family_hash'],checkpoint_sha256='a'*64)
        self.assertEqual(q['indices'],f['query_indices']);self.assertEqual(q['labels'],labels.ravel()[q['indices']].tolist())
        poison=labels.copy();poison.ravel()[f['query_indices'][0]]=0
        with self.assertRaises(ValueError):final_query(f,poison,expected_hash=f['family_hash'],checkpoint_sha256='a'*64)
        with self.assertRaises(ValueError):phase_view(f,10,'final_query',expected_hash=f['family_hash'])

    def test_pinned_hash_tamper_roles_and_protocol_rejected(self):
        _,f=self.fixture()
        for field in ('context','support_order','validation','query_indices'):
            bad=copy.deepcopy(f)
            if field=='context':bad[field]['spatial_isolation']=True
            elif field=='support_order':bad[field]['1'][0]=bad['query_indices'][0]
            elif field=='validation':bad[field]['indices'][0]=bad['query_indices'][0]
            else:bad[field].append(bad[field][0])
            with self.assertRaises(ValueError):validate_family(bad,f['family_hash'])
            bad['family_hash']=digest({k:v for k,v in bad.items() if k!='family_hash'})
            if field!='support_order':
                with self.assertRaises(ValueError):validate_family(bad,bad['family_hash'])
        for k in (True,0,2,20):
            with self.assertRaises(ValueError):phase_view(f,k,'adapt',expected_hash=f['family_hash'])

    def test_insufficient_noncontiguous_invalid_labels_and_seed_fail_without_redraw(self):
        for labels in (np.ones((3,5)),np.array([[0,2]]),np.array([[-1,1]]),np.array([[1.5,2]])):
            with self.assertRaises(ValueError):make_target_family(labels,'paviaU',64)
        labels,_=self.fixture()
        for seed in (True,-1,2**32):
            with self.assertRaises(ValueError):make_target_family(labels,'paviaU',seed)
        with self.assertRaises(ValueError):make_target_family(labels,'unknown',64)


if __name__=='__main__':unittest.main()
