import copy,hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from evaluation.authenticated_probe_pack import CAPTURE_COMMIT,CAPTURE_SOURCE,export
from evaluation.frozen_representation_capture import POLICY,capture,authenticate
from evaluation.linear_probe import POLICY as PROBE_POLICY,advance,capture_pack,fingerprint
from loadData.dataset_registry import file_hash
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot,write_json
from target_bn_calibration import OBS_KEYS
from target_data_context import load_target_context
from tests.test_frozen_representation_capture import synthetic_boundary


def fixture(root):
    freeze,analysis,name,paths=synthetic_boundary(root)
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',CAPTURE_COMMIT],text=True).splitlines()
    files={n:hashlib.sha256(subprocess.check_output(['git','show',CAPTURE_COMMIT+':'+n]).replace(b'\r\n',b'\n')).hexdigest()
           for n in names if n.endswith('.py') or n.startswith('configs/') and n.endswith('.json')}
    assert digest(files)==CAPTURE_SOURCE
    producer={'commit':CAPTURE_COMMIT,'source_hash':CAPTURE_SOURCE,'files':files,'dirty':False,'branch':'synthetic-historical-capture-schema'}
    _,_,_,case,record,_,pins=authenticate(freeze,file_hash(freeze),analysis,file_hash(analysis),name,producer)
    items,_,_=load_target_context('xuzhou',paths[0],paths[1],file_hash(paths[1]),paths[2],file_hash(paths[2]),per_class=10,support_seed=64)
    item=items['xuzhou'];obs={k:item[k] for k in OBS_KEYS};model=load_model_checkpoint(Path(case['out'])/'primary.pt',device='cpu')
    stages,_,evidence=capture(model,'xuzhou',obs)
    out=root/'capture';out.mkdir()
    for stage,value in stages.items():
        torch.save(value,out/(stage+'-features.pt'));write_json(out/(stage+'-summary.json'),{'synthetic':True})
    receipt={'status':'passed','policy':POLICY,'code':producer,'case':name,'freeze_sha256':file_hash(freeze),
        'analysis_sha256':file_hash(analysis),'protected_files':pins,'outputs':{p.name:file_hash(p) for p in out.iterdir()},
        'model_forwards':1,'optimizer_updates':0,'bn_calibration_updates':0,'raw_labels_deserialized':False,
        'new_query_release':False,'channel_selection':False,'evidence':evidence}
    path=out/'receipt.json';write_json(path,receipt)
    direct=capture_pack(model,'xuzhou',obs,labels={r:item[r+'_labels'] for r in ('train','val')},row_ids={r:item[r+'_indices'] for r in ('train','val')})
    return path,freeze,analysis,direct,paths


class AuthenticatedProbeTests(unittest.TestCase):
    def test_actual_stage_arrays_cli_and_probe_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,freeze,analysis,direct,_=fixture(root);code=code_snapshot();out=root/'pack'
            original=torch.load;loads=[]
            def only_features(p,*args,**kwargs):
                self.assertIn(Path(p).name,('stage1-features.pt','stage2-features.pt'));loads.append(str(p));return original(p,*args,**kwargs)
            with patch('torch.load',side_effect=only_features),patch('torch.optim.Adam',side_effect=AssertionError('No optimizer')):
                result=export(path,file_hash(path),freeze,analysis,out,code)
            self.assertEqual(len(loads),2);self.assertEqual((result['train_rows'],result['val_rows']),(90,45))
            pack=original(out/'pack.pt',weights_only=True)
            for k in ('features','labels','row_ids'):self.assertEqual(fingerprint(pack[k]),fingerprint(direct[k]))
            settings=dict(policy=PROBE_POLICY,classes=9,width=4,seed=64,steps=8,lr=.1,l2=.01,scale_floor=1e-8)
            full,report=advance(pack,settings);part,_=advance(pack,settings,stop_after=3);restored,_=advance(pack,settings,state=part)
            self.assertEqual(fingerprint(full),fingerprint(restored));_,old=advance(direct,settings)
            self.assertEqual(report['layers'],old['layers'])
            proc=subprocess.run([sys.executable,'scripts/export_authenticated_probe_pack.py','--expected-commit',code['commit'],
                '--capture',str(path),'--capture-sha256',file_hash(path),'--freeze',str(freeze),'--analysis',str(analysis),'--out',str(root/'cli')],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(fingerprint(original(root/'cli/pack.pt',weights_only=True)),fingerprint(pack))

    def test_predecode_producer_query_hash_existing_output_and_late_file_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,freeze,analysis,_,paths=fixture(root);value=json.loads(path.read_text());code=code_snapshot()
            for change in (lambda v:v['code'].update(commit='wrong'),lambda v:v.update(new_query_release=True),
                           lambda v:v.update(model_forwards=54),lambda v:v['outputs'].pop('stage2-features.pt')):
                bad=copy.deepcopy(value);change(bad);write_json(path,bad)
                with patch('torch.load',side_effect=AssertionError('No decode')):
                    with self.assertRaises(ValueError):export(path,file_hash(path),freeze,analysis,root/'bad',code)
            write_json(path,value)
            existing=root/'existing';existing.mkdir()
            with patch('torch.load',side_effect=AssertionError('No decode')):
                with self.assertRaises(FileExistsError):export(path,file_hash(path),freeze,analysis,existing,code)
                with self.assertRaises(ValueError):export(path,'0'*64,freeze,analysis,root/'bad',code)
            for p in (path.parent/'stage2-features.pt',paths[2]):
                old=p.read_bytes();p.write_bytes(old+b'changed')
                with patch('torch.load',side_effect=AssertionError('No decode')):
                    with self.assertRaises(ValueError):export(path,file_hash(path),freeze,analysis,root/'bad',code)
                p.write_bytes(old)

    def test_rehashed_stage_wrong_Q_and_wrong_shape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,freeze,analysis,_,_=fixture(root);r=json.loads(path.read_text());f=path.parent/'stage1-features.pt'
            original=torch.load(f,weights_only=True);code=code_snapshot()
            for field in ('Q','cnn'):
                value=copy.deepcopy(original)
                if field=='Q':value['Q']=value['Q']*2
                else:value['cnn']=value['cnn'][:,:4]
                torch.save(value,f);r['outputs'][f.name]=file_hash(f);write_json(path,r)
                with self.assertRaises(ValueError):export(path,file_hash(path),freeze,analysis,root/'bad',code)


if __name__=='__main__':unittest.main()
