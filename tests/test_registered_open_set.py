import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation import registered_open_set as r
from evaluation.open_set import digest
from transfer.registered_development import _pin,_json_new


ROOT=Path(__file__).resolve().parents[1]


def fixture(base,seed=64):
    inputs=base/'inputs';inputs.mkdir();rng=np.random.default_rng(seed)
    known=[dict(id=i,name=f'known-{i}') for i in (2,7,13)]
    unknown=[dict(id=i,name=f'unknown-{i}') for i in (19,23)]
    dims=[3,4,5]
    p=inputs/'model.bin';p.write_bytes(b'synthetic opaque model fixture')
    model=_pin(p)
    evidence=dict(version=1,kind='open_set_producer_declaration_v1',model_sha256=model['sha256'],
        known=known,heads=r.HEADS,feature_dimensions=dims,representations=['fixture/'+h for h in r.HEADS],
        source_training_class_ids=[2,7,13],unknown_training_excluded=True,logits_kind='raw_logits')
    ep=_json_new(inputs/'producer.json',evidence)
    packets={}
    for role,n in [('source',9),('calibration',12),('evaluation',18)]:
        a={'sample_ids':np.array([f'{seed}/{role}/{i}' for i in range(n)])}
        for h,d in zip(r.HEADS,dims):
            a['features_'+h]=rng.normal(size=(n,d))
            if role!='source':a['logits_'+h]=rng.normal(size=(n,3))
        if role!='evaluation':a['labels']=np.resize([2,7,13],n)
        p=inputs/(role+'.npz')
        with p.open('xb') as f:np.savez(f,**a)
        packets[role]=_pin(p)
    labels=dict(version=1,role='development_evaluation',sample_ids=[f'{seed}/evaluation/{i}' for i in range(18)],
                labels=np.resize([2,7,13,19,23,0],18).tolist())
    raw=(json.dumps(labels,sort_keys=True)+'\n').encode()
    labelpin=dict(path=str(inputs/'released-labels.json'),sha256=hashlib.sha256(raw).hexdigest())
    v=dict(version=1,kind='open_set_development_arrays_v1',known=known,unknown=unknown,heads=r.HEADS,
        temperature=[1.,.5,2.],known_coverage=.8,model=model,model_evidence=ep,**packets,
        labels=labelpin,calibration_role='source_validation')
    reg=_json_new(inputs/'registration.json',v)
    return reg,v,raw


def changed(reg,value):
    p=Path(reg['path']);p.write_text(json.dumps(value));return _pin(p)


def changed_packet(reg,v,key,fn):
    p=Path(v[key]['path'])
    with np.load(p,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    fn(a)
    with p.open('wb') as f:np.savez(f,**a)
    v=copy.deepcopy(v);v[key]=_pin(p)
    return changed(reg,v)


class RegisteredOpenSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

    def cli(self,mode,pin,out,expected=0):
        key='registration' if mode=='freeze' else 'freeze'
        p=subprocess.run([sys.executable,str(ROOT/'scripts/export_open_set_development.py'),
            '--expected-commit',self.sha,mode,'--'+key,pin['path'],'--'+key+'-sha256',pin['sha256'],
            '--out',str(out)],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(p.returncode,expected,p.stderr)
        return json.loads(p.stdout) if expected==0 else None

    def test_real_cli_all_nine_and_deferred_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v,raw=fixture(base)
            f=self.cli('freeze',reg,base/'frozen')
            self.assertFalse(Path(v['labels']['path']).exists())
            frozen=json.loads(Path(f['path']).read_text());self.assertEqual(len(frozen['rows']),9)
            self.assertIs(frozen['evaluation_labels_read'],False)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'inputs').iterdir()}
            Path(v['labels']['path']).write_bytes(raw)
            out=self.cli('export',f,base/'export');receipt=json.loads(Path(out['path']).read_text())
            self.assertEqual(receipt['status'],'passed');self.assertEqual(len(receipt['reports']),9)
            self.assertEqual({(x['head'],x['method']) for x in receipt['reports']},
                             {(h,m) for h in r.HEADS for m in r.METHODS})
            for row in receipt['reports']:
                value=json.loads(Path(row['report']['path']).read_text())
                self.assertEqual(value['unlabeled_excluded'],3);self.assertEqual(value['labeled_count'],15)
            for name,h in before.items():self.assertEqual(hashlib.sha256((base/'inputs'/name).read_bytes()).hexdigest(),h)

    def test_no_early_label_read_and_missing_row_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v,raw=fixture(base);original=r._read;seen=[]
            def guarded(pin):
                seen.append(pin['path'])
                if pin['path']==v['labels']['path']:raise AssertionError('early labels')
                return original(pin)
            with patch.object(r,'_read',side_effect=guarded):
                f=r.freeze_open_set(reg,out=base/'frozen',expected_commit=self.sha)
                bad=json.loads(Path(f['path']).read_text());bad['rows'].pop();bad['freeze_hash']=digest({k:x for k,x in bad.items() if k!='freeze_hash'})
                fp=changed(f,bad)
                with self.assertRaises(ValueError):r.export_open_set(fp,out=base/'bad',expected_commit=self.sha)
            self.assertNotIn(v['labels']['path'],seen);self.assertFalse((base/'bad').exists())

    def test_changed_inputs_rejected_before_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v,raw=fixture(base);f=r.freeze_open_set(reg,out=base/'f',expected_commit=self.sha)
            Path(v['model']['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):r.export_open_set(f,out=base/'out',expected_commit=self.sha)
            self.assertFalse((base/'out').exists());self.assertFalse(Path(v['labels']['path']).exists())

    def test_labels_wrong_hash_role_order_keep_failure(self):
        for kind in ('missing','bytes','role','order','unknown'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v,raw=fixture(base);label=json.loads(raw)
                if kind=='role':label['role']='query'
                if kind=='order':label['sample_ids'].reverse()
                if kind=='unknown':label['labels'][0]=999
                if kind in ('role','order','unknown'):
                    raw=json.dumps(label).encode();v['labels']['sha256']=hashlib.sha256(raw).hexdigest();reg=changed(reg,v)
                f=r.freeze_open_set(reg,out=base/'frozen',expected_commit=self.sha)
                if kind!='missing':Path(v['labels']['path']).write_bytes(b'bad' if kind=='bytes' else raw)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    r.export_open_set(f,out=base/'export',expected_commit=self.sha)
                self.assertTrue((base/'export'/'failure.json').is_file());self.assertFalse((base/'export'/'receipt.json').exists())
                with self.assertRaises(FileExistsError):r.export_open_set(f,out=base/'export',expected_commit=self.sha)

    def test_wrong_roles_semantics_model_and_columns(self):
        changes=[lambda v:v.update(kind='query_arrays'),lambda v:v.update(heads=list(reversed(r.HEADS))),
                 lambda v:v.update(calibration_role='target_test'),lambda v:v.update(temperature=[1.,1.]),
                 lambda v:v['known'].reverse()]
        for fn in changes:
            with tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v,_=fixture(base);fn(v);reg=changed(reg,v)
                with self.assertRaises(ValueError):r.freeze_open_set(reg,out=base/'bad',expected_commit=self.sha)
                self.assertFalse((base/'bad').exists())

    def test_array_payload_shapes_overlap_and_labels(self):
        changes=[('evaluation',lambda a:a.update(labels=np.zeros(18,dtype=int))),
                 ('evaluation',lambda a:a.update(logits_cnn=np.zeros((18,2)))),
                 ('evaluation',lambda a:a['features_graph'].__setitem__((0,0),float('nan'))),
                 ('calibration',lambda a:a['labels'].__setitem__(0,19)),
                 ('source',lambda a:a['labels'].__setitem__(0,0)),
                 ('evaluation',lambda a:a['sample_ids'].__setitem__(0,'64/source/0'))]
        for key,fn in changes:
            with tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v,_=fixture(base);reg=changed_packet(reg,v,key,fn)
                with self.assertRaises(ValueError):r.freeze_open_set(reg,out=base/'bad',expected_commit=self.sha)
                self.assertFalse((base/'bad').exists())

    def test_outputs_code_and_repeated_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v,raw=fixture(base)
            with self.assertRaises(ValueError):r.freeze_open_set(reg,out=base/'inputs'/'out',expected_commit=self.sha)
            with self.assertRaises(ValueError):r.freeze_open_set(reg,out=base/'f',expected_commit='a'*40)
            f=r.freeze_open_set(reg,out=base/'f',expected_commit=self.sha)
            with self.assertRaises(FileExistsError):r.freeze_open_set(reg,out=base/'f',expected_commit=self.sha)
            Path(v['labels']['path']).write_bytes(raw)
            first=r.export_open_set(f,out=base/'a',expected_commit=self.sha)
            second=r.export_open_set(f,out=base/'b',expected_commit=self.sha)
            aa=json.loads(Path(first['path']).read_text());bb=json.loads(Path(second['path']).read_text())
            self.assertEqual([x['report']['sha256'] for x in aa['reports']], [x['report']['sha256'] for x in bb['reports']])
            with self.assertRaises(FileExistsError):r.export_open_set(f,out=base/'a',expected_commit=self.sha)


if __name__=='__main__':unittest.main()
