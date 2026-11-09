"""Finite-grid calibration must not consume confirmation to select a gate."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from inference.gate_calibration import GateProtocol,calibrate,confirm,save_freeze,load_freeze
from inference.tile_cascade import gate_score,cascade
from inference.light_cnn import LightCNN
from model.research_config import digest
from tests.test_tile_cascade import setup


def data_pin(records,blocks):
    byid={b['id']:b for b in blocks};canonical=[]
    for r in records:
        canonical.append(dict(id=r['id'],labels=np.asarray(r['labels'])[byid[r['id']]['label_indices']].astype(np.int64).tolist(),
            fast=r['fast'].tolist(),full=r['full'].tolist(),coverage_known=r['coverage_known'],coverage_fraction=r['coverage_fraction']))
    return digest(sorted(canonical,key=lambda r:r['id']))


def fixture():
    blocks=[]
    for i,(role,condition) in enumerate([('training','full'),('selection','full'),('calibration','full'),('calibration','gap'),('confirmation','full'),('confirmation','gap')]):
        blocks.append(dict(id=str(i),scene='registered_scene',condition=condition,role=role,context=[i*3,i*3+1,0,3],label_indices=[0,1]))
    full=np.array([[.9,.1],[.1,.9],[.5,.5]],dtype=np.float32)
    good=np.array([[.9,.1],[.1,.9],[.95,.05]],dtype=np.float32)
    bad=np.array([[.9,.1],[.6,.4],[.9,.1]],dtype=np.float32)
    records=[]
    for role in ('calibration','confirmation'):
        records.append([dict(id=b['id'],labels=np.array([1.,2.,np.nan]),fast=(good if b['condition']=='full' else bad).copy(),full=full.copy(),coverage_known=True,coverage_fraction=1.) for b in blocks if b['role']==role])
    spec=dict(schema='dy-gate-protocol-v1',name='synthetic_max_probability',full_state='1'*64,light_artifact='2'*64,criterion='max_probability',
        thresholds=[.5,.8,.95],min_coverage=.8,classes=2,min_class_support=1,max_drop=dict(oa=0.,aa=0.,recall=0.),blocks=blocks,
        data_hashes=dict(calibration=data_pin(records[0],blocks),confirmation=data_pin(records[1],blocks)))
    return spec,*records


def make(spec):return GateProtocol(spec,digest(spec))


class GateCalibrationTests(unittest.TestCase):
    def test_predeclared_candidate_fixed_confirmation_and_exact_class_metrics(self):
        spec,cal,val=fixture();protocol=make(spec);f=calibrate(protocol,cal)
        self.assertEqual(f['selected'],2);self.assertEqual(f['candidates'][1]['accepted_tiles'],2)
        self.assertFalse(f['candidates'][1]['passes_empirical_protection'])
        self.assertEqual(f['candidates'][1]['metrics']['confusion'],[[2,0],[1,1]])
        self.assertEqual(f['candidates'][1]['groups'][1]['metrics']['support'],[1,1])
        receipt=confirm(protocol,f,val,expected_freeze_hash=digest(f));self.assertEqual(receipt['status'],'empirical_pass')
        self.assertEqual(receipt['inference_kwargs']['threshold'],.8);self.assertEqual(receipt['tested']['accepted_tiles'],1)
        self.assertFalse(receipt['statistical_noninferiority']);self.assertFalse(receipt['deployment_speed_verified'])

    def test_confirmation_failure_keeps_full_without_reselection(self):
        spec,cal,val=fixture();val[0]['fast'][1]=[.95,.05];spec['data_hashes']['confirmation']=data_pin(val,spec['blocks']);p=make(spec)
        frozen=calibrate(p,cal);pin=digest(frozen);receipt=confirm(p,frozen,val,expected_freeze_hash=pin)
        self.assertEqual(receipt['status'],'keep_full');self.assertEqual(receipt['inference_kwargs'],dict(mode='full'))
        self.assertEqual(receipt['selected'],2);self.assertEqual(digest(frozen),pin)
        self.assertEqual(receipt['tested']['metrics']['recall'],[1.,.5])
        self.assertEqual(confirm(p,frozen,val,expected_freeze_hash=pin),receipt)

    def test_small_class_condition_protection_even_when_aggregate_tolerance_permits(self):
        spec,cal,val=fixture();spec['max_drop']=dict(oa=.5,aa=.5,recall=.02);p=make(spec);f=calibrate(p,cal)
        self.assertFalse(f['candidates'][1]['passes_empirical_protection']);self.assertEqual(f['selected'],2)
        # Missing one class in a condition cannot be hidden by other groups.
        for phase in (cal,val):phase[1]['labels'][1]=1
        spec['data_hashes']=dict(calibration=data_pin(cal,spec['blocks']),confirmation=data_pin(val,spec['blocks']));p=make(spec)
        f=calibrate(p,cal);self.assertEqual(f['selected'],0);self.assertFalse(f['candidates'][0]['passes_empirical_protection'])
        self.assertEqual(confirm(p,f,val,expected_freeze_hash=digest(f))['status'],'keep_full')

    def test_unlabeled_context_controls_whole_tile_gate_but_not_accuracy(self):
        spec,cal,val=fixture();cal[0]['fast'][2]=[.5,.5];spec['data_hashes']['calibration']=data_pin(cal,spec['blocks']);f=calibrate(make(spec),cal)
        self.assertEqual(f['selected'],0);self.assertEqual(f['candidates'][2]['accepted_tiles'],0)
        self.assertEqual(f['candidates'][0]['metrics']['count'],4)
        cal[0]['labels'][2]=999999;self.assertEqual(calibrate(make(spec),cal),f)

    def test_unknown_and_low_coverage_mandatory_full(self):
        for field,value in [('coverage_known',False),('coverage_fraction',.1)]:
            spec,cal,val=fixture();cal[0][field]=value;spec['data_hashes']['calibration']=data_pin(cal,spec['blocks'])
            f=calibrate(make(spec),cal);self.assertEqual(f['selected'],0)
            self.assertFalse(f['fast_endpoint']['routes'][0]['fast'])

    def test_spatial_selection_reuse_test_role_and_missing_phase_rejected(self):
        spec,_,_=fixture()
        for kind in ('same_context','halo_overlap','test','missing','duplicate','labels','grid','support'):
            s=copy.deepcopy(spec)
            if kind=='same_context':s['blocks'][2]['context']=s['blocks'][1]['context']
            elif kind=='halo_overlap':s['blocks'][2]['context']=[3,7,0,3]
            elif kind=='test':s['blocks'][2]['role']='test'
            elif kind=='missing':s['blocks']=[b for b in s['blocks'] if b['role']!='selection']
            elif kind=='duplicate':s['blocks'][2]['id']=s['blocks'][0]['id']
            elif kind=='labels':s['blocks'][2]['label_indices']=[0,0]
            elif kind=='grid':s['thresholds']=list(np.linspace(0,1,9))
            else:s['min_class_support']=0
            with self.assertRaises(ValueError):make(s)

    def test_data_roles_external_hash_models_and_protocol_mutation_rejected(self):
        spec,cal,val=fixture();p=make(spec)
        with self.assertRaises(ValueError):GateProtocol(spec,'0'*64)
        with self.assertRaises(ValueError):calibrate(p,val)
        for kind in ('label','probability','coverage','incomplete','duplicate'):
            rows=copy.deepcopy(cal)
            if kind=='label':rows[0]['labels'][0]=2
            elif kind=='probability':rows[0]['fast'][0]=[.5,.5]
            elif kind=='coverage':rows[0]['coverage_known']=False
            elif kind=='incomplete':rows.pop()
            else:rows[1]=rows[0]
            with self.assertRaises(ValueError):calibrate(p,rows)
        p._spec['light_artifact']='3'*64
        with self.assertRaises(ValueError):calibrate(p,cal)

    def test_freeze_roundtrip_external_hash_no_overwrite_no_reselection(self):
        spec,cal,val=fixture();p=make(spec);f=calibrate(p,cal);pin=digest(f)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frozen.json';filepin=save_freeze(p,f,path);restored=load_freeze(p,path,expected_file_sha256=filepin,expected_freeze_hash=pin)
            self.assertEqual(f,restored);self.assertEqual(confirm(p,f,val,expected_freeze_hash=pin),confirm(p,restored,val,expected_freeze_hash=pin))
            with self.assertRaises(FileExistsError):save_freeze(p,f,path)
            with patch('inference.gate_calibration.json.loads',side_effect=AssertionError('parsed before authentication')):
                with self.assertRaises(ValueError):load_freeze(p,path,expected_file_sha256='0'*64,expected_freeze_hash=pin)
        bad=copy.deepcopy(f);bad['selected']=1
        with self.assertRaises(ValueError):confirm(p,bad,val,expected_freeze_hash=digest(bad))
        other=copy.deepcopy(spec);other['light_artifact']='3'*64
        with self.assertRaises(ValueError):confirm(make(other),f,val,expected_freeze_hash=pin)

    def test_three_criteria_match_actual_cascade_whole_block_boundary(self):
        full,light,rows=setup();r=rows[0];probs=light(r['dataset'],r['x'],wavelengths=r['wavelengths'])
        for criterion in ('max_probability','margin','entropy'):
            score=gate_score(probs,criterion)
            actual=cascade(full,light,[r],mode='adaptive',criterion=criterion,threshold=score)
            self.assertEqual(actual['counts']['accepted_tiles'],1)
            self.assertEqual(actual['outputs'][0]['score'],score)
            threshold=np.nextafter(score,-np.inf if criterion=='entropy' else np.inf).item()
            self.assertEqual(cascade(full,light,[r],mode='adaptive',criterion=criterion,threshold=threshold)['counts']['accepted_tiles'],0)
            spec,cal,val=fixture();spec['criterion']=criterion;spec['thresholds']=[0.,1.] if criterion!='entropy' else [0.,10.]
            p=make(spec);f=calibrate(p,cal);confirm(p,f,val,expected_freeze_hash=digest(f))

    def test_reordering_is_deterministic_and_no_optimizer_or_rng(self):
        spec,cal,val=fixture();p=make(spec);rng=torch.get_rng_state().clone()
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('updated')):
            a=calibrate(p,cal);b=calibrate(p,list(reversed(cal)));self.assertEqual(a,b)
            self.assertEqual(confirm(p,a,val,expected_freeze_hash=digest(a)),confirm(p,a,list(reversed(val)),expected_freeze_hash=digest(a)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
