import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation import known_context_preparation as p
from evaluation.open_set_producer import _observation
from transfer.registered_development import _pin, _json_new

ROOT=Path(__file__).resolve().parents[1]


def fixture(base, seed=64):
    inputs=base/'inputs';inputs.mkdir();rng=np.random.default_rng(seed)
    observations=[]
    for n,role in enumerate(p.ROLES):
        x=rng.normal(size=(4,6,7)).astype(np.float32)+n*100
        x[:,:,0]=np.float32(n*100+3)  # Constant source band must not divide by zero.
        a=dict(spectral=x,wavelengths_nm=np.linspace(450,900,7,dtype=np.float32),indices=np.arange(0,24,2))
        if n<2:a['context_labels']=np.resize([2,7,13],24).reshape(4,6)
        path=inputs/(role+'.npz')
        with path.open('xb') as f:np.savez(f,**a)
        observations.append(dict(role=role,scene_uid='synthetic',scene_shape=[30,30],origin=[n*10,0],
                                 superpixel_scale=4,input=_pin(path)))
    v=dict(version=1,kind='known_full_context_preparation_v1',dataset='fixture',
        known=[dict(id=k,name=f'known-{k}') for k in (2,7,13)],unknown=[dict(id=19,name='unknown')],
        observations=observations,labels=dict(path=str(inputs/'unreleased.json'),sha256='0'*64))
    return _json_new(inputs/'registration.json',v),v


def change(reg,v):
    Path(reg['path']).write_text(json.dumps(v));return _pin(reg['path'])


def packet(reg,v,index,fn):
    path=Path(v['observations'][index]['input']['path'])
    with np.load(path) as f:a={k:f[k].copy() for k in f.files}
    fn(a)
    with path.open('wb') as f:np.savez(f,**a)
    v['observations'][index]['input']=_pin(path)
    return change(reg,v)


class KnownContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

    def prepare(self,reg,out):return p.prepare_known_contexts(reg,out=out,expected_commit=self.sha)

    def test_cli_and_independent_minmax_coordinates_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v=fixture(base);before={f:f.read_bytes() for f in (base/'inputs').iterdir()}
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_open_set_contexts.py'),
                '--expected-commit',self.sha,'--registration',reg['path'],'--registration-sha256',reg['sha256'],
                '--out',str(base/'out')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            r=json.loads(Path(json.loads(result.stdout)['path']).read_text())
            with np.load(v['observations'][0]['input']['path']) as f:source=f['spectral']
            low=np.array([min(source[:,:,b].ravel()) for b in range(7)],np.float32)
            high=np.array([max(source[:,:,b].ravel()) for b in range(7)],np.float32)
            self.assertEqual(r['statistics']['minimum'],low.tolist())
            for n,obs in enumerate(r['observations']):
                got=_observation(obs['input'],obs['role'],[2,7,13])
                with np.load(v['observations'][n]['input']['path']) as f:
                    expected=np.stack([(f['spectral'][:,:,b]-low[b])/(high[b]-low[b] if high[b]>low[b] else np.float32(1)) for b in range(7)],axis=-1)
                    np.testing.assert_array_equal(got['x'][:,:,:7],expected)
                    if n<2:np.testing.assert_array_equal(got['labels'],f['context_labels'].ravel()[f['indices']])
                for y in range(4):
                    for x in range(6):np.testing.assert_array_equal(got['x'][y,x,-2:],np.array([(n*10+y)/29,x/29],np.float32))
            self.assertFalse(r['training_history_verified']);self.assertFalse(r['evaluation_labels_read'])
            self.assertFalse(Path(v['labels']['path']).exists())
            for f,raw in before.items():self.assertEqual(f.read_bytes(),raw)

    def test_unknown_zero_outside_selected_indices_fail_before_output(self):
        for role in (0,1):
            for label in (0,19,999,-1):
                with self.subTest(role=role,label=label),tempfile.TemporaryDirectory() as tmp:
                    base=Path(tmp);reg,v=fixture(base)
                    reg=packet(reg,v,role,lambda a:a['context_labels'].__setitem__((0,1),label))
                    with self.assertRaisesRegex(ValueError,'Every visible'):self.prepare(reg,base/'out')
                    self.assertFalse((base/'out').exists())

    def test_roles_geometry_band_and_semantic_rejections(self):
        for mode in ('overlap','shape','outside','role','semantic','duplicate'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v=fixture(base)
                if mode=='overlap':v['observations'][1]['origin']=[1,1]
                if mode=='shape':v['observations'][1]['scene_shape']=[31,30]
                if mode=='outside':v['observations'][1]['origin']=[29,0]
                if mode=='role':v['observations'][2]['role']='query'
                if mode=='semantic':v['unknown'][0]['id']=2
                if mode=='duplicate':v['observations'][1]['input']=v['observations'][0]['input']
                reg=change(reg,v)
                with self.assertRaises(ValueError):self.prepare(reg,base/'out')
                self.assertFalse((base/'out').exists())

    def test_payload_rejections(self):
        variants=[(0,lambda a:a.pop('context_labels')),
            (0,lambda a:a.update(context_labels=np.ones((4,6),np.int64)*2)),
            (1,lambda a:a.update(wavelengths_nm=a['wavelengths_nm']+1)),
            (2,lambda a:a.update(labels=np.ones(12))),
            (2,lambda a:a.update(context_labels=np.ones((4,6)))),
            (2,lambda a:a['spectral'].__setitem__((0,0,0),np.nan)),
            (0,lambda a:a.update(indices=np.array([0,0,1])))]
        for n,fn in variants:
            with self.subTest(index=n),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v=fixture(base);reg=packet(reg,v,n,fn)
                with self.assertRaises(ValueError):self.prepare(reg,base/'out')
                self.assertFalse((base/'out').exists())

    def test_calibration_and_evaluation_cannot_change_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v=fixture(base);a=json.loads(Path(self.prepare(reg,base/'first')['path']).read_text())
            for n in (1,2):reg=packet(reg,v,n,lambda d:d.update(spectral=d['spectral']*1000))
            b=json.loads(Path(self.prepare(reg,base/'second')['path']).read_text())
            self.assertEqual(a['statistics'],b['statistics'])
            self.assertEqual(a['checks'][0],b['checks'][0])

    def test_tampered_hash_wrong_code_and_output_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v=fixture(base)
            with self.assertRaises(ValueError):p.prepare_known_contexts(reg,out=base/'out',expected_commit='0'*40)
            with self.assertRaises(ValueError):self.prepare(reg,base/'inputs'/'out')
            bad=copy.deepcopy(reg);bad['sha256']='0'*64
            with self.assertRaises(ValueError):self.prepare(bad,base/'out')
            self.assertFalse((base/'out').exists())

    def test_no_deferred_label_read_and_repeat_exact_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v=fixture(base);read=p._read
            def guard(pin):
                self.assertNotEqual(pin['path'],v['labels']['path']);return read(pin)
            with patch.object(p,'_read',side_effect=guard):
                a=json.loads(Path(self.prepare(reg,base/'a')['path']).read_text())
                b=json.loads(Path(self.prepare(reg,base/'b')['path']).read_text())
            self.assertEqual(a['checks'],b['checks'])
            with self.assertRaises(FileExistsError):self.prepare(reg,base/'a')


if __name__=='__main__':unittest.main()
