"""Role rejection, immutable original membership and bounded-array negative tests."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import bank_reader as reader
from batch import patch_batch
from spatial_plan import strict_plan


class OriginalRoleGates(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.width=12
        groups={'support':{'seed':512,'per_class':5,'indices':list(range(5))+list(range(12,17)),'labels':[1]*5+[2]*5},'validation':{'seed':3031,'per_class':5,'indices':list(range(60,65))+list(range(72,77)),'labels':[1]*5+[2]*5}}
        labels={'scene':'xuzhou','development_roles_hash':'synthetic','release_hash':'synthetic',**groups};raw=reader.canonical(labels);self.labels=self.root/'labels.json';self.labels.write_bytes(raw)
        rows=[]
        for i,(role,box) in enumerate([('devtrain',[0,3,0,6]),('devval',[5,8,0,6]),('devtrain',[0,3,8,12])]):
            key='support' if role=='devtrain' else 'validation';a,b,c,d=box;points=sorted([[j,y] for j,y in zip(groups[key]['indices'],groups[key]['labels']) if a<=j//12<b and c<=j%12<d])
            rows.append({'id':i,'role':role,'box':box,'parent_tile':i,'parent_box':box,'blocked':None,'origin':{'synthetic':True},'partition':'train' if role=='devtrain' else 'val','representation':'raw_fp32','released_samples':{'count':len(points),'sha256':reader.sha(reader.canonical(points))}})
        reg={'scene':'xuzhou','shape':[8,12,3],'classes':2,'custody':{'label_release':{'sha256':reader.sha(raw)}},'statistics':{'minimum':[0,0,0],'maximum':[10,10,10]},'rows':rows};reg_raw=reader.canonical(reg);(self.root/'registration.json').write_bytes(reg_raw);self.reg_sha=reader.sha(reg_raw);members=[]
        for row in rows:
            member=f"tile-{row['id']}.npy";a,b,c,d=row['box'];arr=np.arange((b-a)*(d-c)*3,dtype=np.float32).reshape(b-a,d-c,3);f=io.BytesIO();np.save(f,arr,allow_pickle=False);data=f.getvalue()
            meta={'row':row,'registration':self.reg_sha,'member':member,'dtype':'<f4','content':{'bytes':len(data),'npy_sha256':reader.sha(data),'array_sha256':'synthetic'}};raw=reader.canonical(meta);name=f"tile-{row['id']}.json";members.append({'id':row['id'],'role':row['role'],'name':name,'sha256':reader.sha(raw)})
            if row['released_samples']['count']:(self.root/name).write_bytes(raw);(self.root/member).write_bytes(data)
        raw=reader.canonical({'registration':self.reg_sha,'members':members});(self.root/'bank.json').write_bytes(raw)
        self.override=patch.dict(reader.PINS,{'xuzhou':{'bank':reader.sha(raw),'registration':self.reg_sha,'shape':(8,12,3),'classes':2}});self.override.start()
    def tearDown(self):self.override.stop();self.tmp.cleanup()
    def bank(self):return reader.ReleasedBank(self.root,'xuzhou',self.labels)
    def test_metadata_admits_labeled_subset_without_opening_any_array(self):
        with patch('bank_reader.np.load',side_effect=AssertionError('payload opened')):
            bank=self.bank();self.assertEqual(bank.payload_reads,0);self.assertEqual(set(bank.members),{0,1})
    def test_query_wrong_role_and_unlabeled_context_rejected_before_array_read(self):
        bank=self.bank()
        with patch('bank_reader.np.load',side_effect=AssertionError('payload opened')):
            for role,tile in [('query',0),('test',0),('devtrain',1),('devtrain',2),('devval',99)]:
                with self.assertRaises(ValueError):bank.read_tile(role,tile)
        self.assertEqual(bank.payload_reads,0)
    def test_original_memberships_and_authority_bytes_cannot_change(self):
        bank=self.bank();data=(self.root/'registration.json').read_bytes();(self.root/'registration.json').write_bytes(data+b' ')
        with self.assertRaisesRegex(ValueError,'metadata bytes'):bank.read_tile('devtrain',0)
    def test_corrupt_array_rejected_before_numpy_decoder(self):
        bank=self.bank();p=self.root/'tile-0.npy';raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with patch('bank_reader.np.load',side_effect=AssertionError('decoder opened')):
            with self.assertRaisesRegex(ValueError,'NPY bytes'):bank.read_tile('devtrain',0)
    def test_original_normalizer_and_readonly_columns_are_explicit(self):
        bank=self.bank();x=bank.read_tile('devtrain',0);self.assertEqual(x['spectral_raw_hwc'].shape,(3,6,3));self.assertTrue(np.array_equal(x['spectral_normalized_hwc'],x['spectral_raw_hwc']/10));self.assertFalse(x['native_physical_mapping_authenticated']);self.assertEqual(x['processed_column_ids'],(0,1,2));self.assertFalse(x['spectral_raw_hwc'].flags.writeable)
    def test_symlink_bank_member_rejected(self):
        p=self.root/'tile-0.npy';target=self.root/'other.npy';p.rename(target);p.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'immutable root'):self.bank().read_tile('devtrain',0)
    def test_patch_edges_never_read_an_external_halo(self):
        cube=np.arange(3*6*3,dtype=np.float32).reshape(3,6,3);actual,centers=patch_batch(cube,[0],[0,3,0,6],12)
        self.assertEqual(centers,((0,0),));self.assertTrue(np.array_equal(actual[0,:,0,0],cube[0,0]));self.assertTrue(np.array_equal(actual[0,:,6,6],cube[2,3]))
        with self.assertRaisesRegex(ValueError,'escapes tile'):patch_batch(cube,[7],[0,3,0,6],12)
    def test_strict_parent_subset_never_promotes_validation_or_reads_payload(self):
        bank=self.bank();bank.rows[1]['parent_tile']=0 # Synthetic metadata case: both roles share parent0.
        with patch('bank_reader.np.load',side_effect=AssertionError('payload opened')):result=strict_plan(bank)
        self.assertEqual(result['excluded_shared_parents'],[0]);self.assertEqual(result['common_K5_V5_feasible_original_classes'],[]);self.assertEqual(result['new_array_reads'],0)
        self.assertEqual(result['roles']['devtrain']['tiles'],[]);self.assertEqual(result['roles']['devval']['tiles'],[])


if __name__=='__main__':unittest.main()
