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
import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from transfer.registered_development import _pin, _json_new
from evaluation import open_set_producer as p
from evaluation.registered_open_set import freeze_open_set, export_open_set

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def fixture(base,seed=64):
    inputs=base/'inputs';inputs.mkdir();rng=np.random.default_rng(seed)
    known=[dict(id=i,name=f'known-{i}') for i in (2,7,13)]
    unknown=[dict(id=i,name=f'unknown-{i}') for i in (19,23)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=MultiDatasetVCASAHGCN({'a':3},latent_channels=8,adapter_embed_dim=8,
            locality_window=None,graph_mode='learned',norm_type='group_norm').eval()
        save_model_checkpoint(model,inputs/'model.pt')
    checkpoint=_pin(inputs/'model.pt');sidecar=_pin(inputs/'model.pt.json')
    decl=dict(version=1,kind='open_set_training_semantics_declaration_v1',
        checkpoint_sha256=checkpoint['sha256'],sidecar_sha256=sidecar['sha256'],dataset='a',known=known,
        source_training_class_ids=[2,7,13],unknown_training_excluded=True)
    declaration=_json_new(inputs/'declaration.json',decl)
    observations=[]
    for n,role in enumerate(p.ROLES):
        bands=7+n*3;ix=np.arange(0,36,2,dtype=np.int64)
        data=dict(x=rng.normal(size=(6,7,bands+2)).astype(np.float32),
            wavelengths_nm=np.linspace(450,900,bands,dtype=np.float32),indices=ix)
        if n<2:data['labels']=np.resize([2,7,13],len(ix))
        path=inputs/(role+'.npz')
        with path.open('xb') as f:np.savez(f,**data)
        observations.append(dict(role=role,scene_uid=f'synthetic-{seed}',origin=[n*10,0],superpixel_scale=9,input=_pin(path)))
    ids=[json.dumps([f'synthetic-{seed}',20+int(i)//7,int(i)%7],separators=(',',':')) for i in ix]
    raw=(json.dumps(dict(version=1,role='development_evaluation',sample_ids=ids,
                        labels=np.resize([2,7,13,19,23,0],len(ix)).tolist()))+'\n').encode()
    v=dict(version=1,kind='m8_open_set_development_production_v1',known=known,unknown=unknown,dataset='a',
        checkpoint=checkpoint,sidecar=sidecar,training_declaration=declaration,observations=observations,
        labels=dict(path=str(inputs/'eval-labels.json'),sha256=hashlib.sha256(raw).hexdigest()),
        temperature=[1.,.5,2.],known_coverage=.8)
    return _json_new(inputs/'registration.json',v),v,raw


def change(reg,v):
    path=Path(reg['path']);path.write_text(json.dumps(v));return _pin(path)


def packet(reg,v,index,fn):
    path=Path(v['observations'][index]['input']['path'])
    with np.load(path,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    fn(a)
    with path.open('wb') as f:np.savez(f,**a)
    v['observations'][index]['input']=_pin(path)
    return change(reg,v)


class OpenSetProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

    def run_producer(self,reg,out):return p.produce_open_set(reg,out=out,expected_commit=self.sha)

    def test_actual_cli_produce_freeze_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v,raw=fixture(base)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/produce_open_set_development.py'),
                '--expected-commit',self.sha,'--registration',reg['path'],'--registration-sha256',reg['sha256'],
                '--out',str(base/'produced')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads(Path(json.loads(result.stdout)['path']).read_text())
            self.assertTrue(receipt['network_output_verified']);self.assertFalse(receipt['training_history_verified'])
            self.assertFalse(Path(v['labels']['path']).exists());self.assertEqual(len(receipt['checks']),3)
            f=freeze_open_set(receipt['arrays_registration'],out=base/'freeze',expected_commit=self.sha)
            Path(v['labels']['path']).write_bytes(raw)
            out=export_open_set(f,out=base/'export',expected_commit=self.sha)
            self.assertEqual(len(json.loads(Path(out['path']).read_text())['reports']),9)

    def test_independent_native_outputs_and_readonly_three_seeds(self):
        for seed in (64,512,2048):
            with self.subTest(seed=seed),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v,_=fixture(base,seed);before={f.name:f.read_bytes() for f in (base/'inputs').iterdir()}
                rng=torch.get_rng_state().clone();out=self.run_producer(reg,base/'produced')
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                receipt=json.loads(Path(out['path']).read_text())
                with torch.random.fork_rng(devices=[]):model=load_model_checkpoint(v['checkpoint']['path'])
                for key,obs in zip(('source','calibration','evaluation'),v['observations']):
                    with np.load(obs['input']['path']) as a,np.load(receipt['outputs'][key]['path']) as got:
                        x=torch.from_numpy(a['x']);wave=torch.from_numpy(a['wavelengths_nm']);ix=a['indices']
                        with torch.no_grad():
                            f=model.backbone.forward_features(x,wavelengths=wave,wavelength_unit='nm',
                                dataset_name='a',superpixel_scale=9)
                            z=model('a',x,wavelengths=wave,wavelength_unit='nm',superpixel_scale=9,return_logits=True)
                        for h,features,logits in zip(p.HEADS,f,z):
                            np.testing.assert_array_equal(got['features_'+h],features[ix].numpy())
                            if key!='source':np.testing.assert_array_equal(got['logits_'+h],logits[ix].numpy())
                for name,raw in before.items():self.assertEqual((base/'inputs'/name).read_bytes(),raw)

    def test_full_context_overlap_and_roles(self):
        for kind in ('overlap','role','duplicate','columns','declaration','sidecar','hash'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v,_=fixture(base)
                if kind=='overlap':v['observations'][2]['origin']=[1,1]
                if kind=='role':v['observations'][2]['role']='query'
                if kind=='duplicate':v['observations'][2]['input']=v['observations'][1]['input']
                if kind=='columns':v['known']=v['known'][::-1]
                if kind=='declaration':Path(v['training_declaration']['path']).write_text('{}')
                if kind=='sidecar':v['sidecar']=v['training_declaration']
                if kind=='hash':v['checkpoint']['sha256']='0'*64
                reg=change(reg,v)
                with self.assertRaises(ValueError):self.run_producer(reg,base/'out')
                self.assertFalse((base/'out').exists())

    def test_observation_rejections(self):
        variants=[(2,lambda a:a.update(labels=np.ones(len(a['indices']),dtype=int))),
            (0,lambda a:a.update(labels=np.resize([2,7,19],len(a['indices'])))),
            (1,lambda a:a['x'].__setitem__((0,0,0),np.nan)),
            (1,lambda a:a.update(wavelengths_nm=a['wavelengths_nm'][::-1])),
            (2,lambda a:a.update(indices=np.array([0,0]))),
            (2,lambda a:a.update(indices=np.array([1000]))),
            (0,lambda a:a.update(x=a['x'].astype(np.float64)))]
        for i,(idx,fn) in enumerate(variants):
            with self.subTest(i=i),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v,_=fixture(base);reg=packet(reg,v,idx,fn)
                with self.assertRaises(ValueError):self.run_producer(reg,base/'out')

    def test_unsupported_checkpoint_preserves_failure(self):
        for setting in ('batch_norm','double'):
            with self.subTest(setting=setting),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);reg,v,_=fixture(base)
                model=MultiDatasetVCASAHGCN({'a':3},latent_channels=8,adapter_embed_dim=8,
                    locality_window=None,graph_mode='learned',norm_type='batch_norm' if setting=='batch_norm' else 'group_norm')
                if setting=='double':model=model.double()
                save_model_checkpoint(model,Path(v['checkpoint']['path']))
                v['checkpoint']=_pin(Path(v['checkpoint']['path']));v['sidecar']=_pin(Path(v['sidecar']['path']))
                dpath=Path(v['training_declaration']['path']);d=json.loads(dpath.read_text())
                d.update(checkpoint_sha256=v['checkpoint']['sha256'],sidecar_sha256=v['sidecar']['sha256'])
                dpath.write_text(json.dumps(d));v['training_declaration']=_pin(dpath);reg=change(reg,v)
                with self.assertRaises(ValueError):self.run_producer(reg,base/'out')
                self.assertTrue((base/'out'/'failure.json').exists());self.assertFalse((base/'out'/'receipt.json').exists())
                with self.assertRaises(FileExistsError):self.run_producer(reg,base/'out')

    def test_no_label_reads_and_output_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v,_=fixture(base);original=p._read;seen=[]
            def read(pin):
                self.assertNotEqual(pin,v['labels']);seen.append(pin);return original(pin)
            with patch.object(p,'_read',side_effect=read):self.run_producer(reg,base/'out')
            self.assertFalse(Path(v['labels']['path']).exists());self.assertGreater(len(seen),10)
            with self.assertRaises(FileExistsError):self.run_producer(reg,base/'out')
            with self.assertRaises(ValueError):self.run_producer(reg,base/'inputs'/'inside')
            with self.assertRaises(ValueError):p.produce_open_set(reg,out=base/'wrong',expected_commit='0'*40)

    def test_output_equivalence_failure_is_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reg,v,_=fixture(base);original=MultiDatasetVCASAHGCN.predict_features
            def corrupt(model,*args,**kwargs):
                z=original(model,*args,**kwargs)
                if kwargs.get('return_logits'):return tuple(a+1 for a in z)
                return z
            with patch.object(MultiDatasetVCASAHGCN,'predict_features',corrupt):
                with self.assertRaises(ValueError):self.run_producer(reg,base/'out')
            self.assertTrue((base/'out'/'failure.json').exists())


if __name__=='__main__':unittest.main()
