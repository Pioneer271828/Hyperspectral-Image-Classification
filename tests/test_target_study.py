import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from joint_baseline import NAMES
from joint_training import run_training,validation_epochs
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig
from research_runtime import write_json
from target_initialization import METHODS
from target_study import (EXPECTED,load_study,full_settings,TargetStudyRuntime,case_name,select_case,validate_source_case)
from target_training import target_settings
from test_target_training import fixture
from test_joint_training import SOURCE,PROVENANCE,checkpoints,hashes
from source_pretrain import full_settings as source_full_settings
import test_target_initialization as initial_tests

DEVICE=os.environ.get('G4H_DEVICE','cpu')


def launch_fixture(root):
    cases=[]
    for target in NAMES:
        for seed in (64,512,2048):
            for method in METHODS:
                name=case_name(target,seed,method)
                cases.append({'target':target,'seed':seed,'method':method,'name':name,
                    'settings':full_settings(target,method,seed),'out':str(root/'cases'/name),
                    'inputs':{'target_observation':target},'source_spec':{} if method==METHODS[0] else {'source':'fixture'},
                    'source_result':None if method==METHODS[0] else {'fixture':True}})
    return {'code':SOURCE,'study':copy.deepcopy(EXPECTED),'query_labels_accessed':False,
        'test_scoring':False,'expected_updates':40500,'out_dir':str(root),'cases':cases}


class TargetStudyTests(unittest.TestCase):
    def test_protocol_budgets_and_tamper_rejected(self):
        self.assertEqual(load_study(),EXPECTED)
        for target in NAMES:
            for seed in (64,512,2048):
                for method in METHODS:
                    opts=full_settings(target,method,seed);self.assertEqual(opts['optimizer_steps'],1500)
                    self.assertEqual(len(validation_epochs(opts)),51)
                    self.assertEqual(validation_epochs(opts)[-1],1499)
                    self.assertEqual(opts['config']['training']['primary_checkpoint'],'val_loss')
                    self.assertEqual(opts['head_seed'],seed+1000)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json';bad=copy.deepcopy(EXPECTED);bad['updates_per_case']=5000;write_json(path,bad)
            with self.assertRaises(ValueError):load_study(path)
            opts=full_settings('xuzhou',METHODS[0],64,device=DEVICE);opts['support_seed']=512
            with self.assertRaises(ValueError):TargetStudyRuntime(opts,{},Path(tmp)/'none')

    def test_launch_exact27_source_roles_paired_inputs_paths_and_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);launch=launch_fixture(root)
            for case in launch['cases']:
                self.assertIs(select_case(launch,SOURCE,case['target'],case['seed'],case['method']),case)
            for change in ('code','query','duplicate','path','settings','source','paired'):
                bad=copy.deepcopy(launch)
                if change=='code':bad['code']={}
                elif change=='query':bad['query_labels_accessed']=True
                elif change=='duplicate':bad['cases'][-1]=bad['cases'][0]
                elif change=='path':bad['cases'][0]['out']=str(root/'other')
                elif change=='settings':bad['cases'][0]['settings']['epochs']=5000
                elif change=='source':bad['cases'][0]['source_spec']={'source':'forbidden'}
                else:bad['cases'][1]['inputs']={'different':True}
                with self.assertRaises(ValueError):select_case(bad,SOURCE,'xuzhou',64,METHODS[0])

    def test_source_result_pairs_model_seed_and_producer_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=initial_tests.TargetInitializationTests().source(root)
            result=root/'result.json'
            source_opts=source_full_settings('xuzhou',64)
            value={'status':'completed','optimizer_steps':1002,'start_code':{'commit':EXPECTED['source_producer_commit']},
                'settings':source_opts,'selected_step':1002,
                'artifacts':{'primary.pt':source[1],'primary.pt.json':source[2]}}
            primary=root/'primary.pt';primary.write_bytes(source[0].read_bytes())
            Path(str(primary)+'.json').write_bytes(Path(str(source[0])+'.json').read_bytes())
            meta=json.loads(Path(str(primary)+'.json').read_text());training=meta['training_config']
            training.update(joint_runtime={**source_opts,'executed_optimizer_steps':1002,'selected_step':1002},
                checkpoint_optimizer_step=1002,research_config=source_opts['config'],
                training_code={'commit':EXPECTED['source_producer_commit']})
            write_json(Path(str(primary)+'.json'),meta);value['artifacts']['primary.pt.json']=file_hash(str(primary)+'.json')
            write_json(result,value)
            case={'target':'xuzhou','seed':64,'method':METHODS[1],
                'source_spec':dict(source_path=str(primary),source_weight_sha256=source[1],source_sidecar_sha256=value['artifacts']['primary.pt.json']),
                'source_result':{'path':str(result),'sha256':file_hash(result)}}
            validate_source_case(case)
            for field in ('seed','target'):
                bad=copy.deepcopy(case);bad[field]=512 if field=='seed' else 'paviaU'
                with self.assertRaises(ValueError):validate_source_case(bad)
            value['start_code']['commit']='wrong';write_json(result,value);case['source_result']['sha256']=file_hash(result)
            with self.assertRaises(ValueError):validate_source_case(case)
            with self.assertRaises(ValueError):validate_source_case(dict(case,method=METHODS[0]))

    def test_full_runtime_scaled_fixture_reuses_selected_resume_path(self):
        # Explicit engineering fixture, NOT1500updates or full real acceptance.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(data,provenance,guard),_=fixture(root)
            source=initial_tests.TargetInitializationTests().source(root)
            spec=dict(source_path=str(source[0]),source_weight_sha256=source[1],source_sidecar_sha256=source[2])
            for method in METHODS:
                opts=target_settings('xuzhou',method,device=DEVICE)
                opts.update(protocol=EXPECTED['protocol'],profile='full',epochs=2,val_interval=1,optimizer_steps=3)
                opts['config']['training'].update(epochs=2,val_interval=1);opts['target_study']=copy.deepcopy(EXPECTED)
                directory=root/method;kwargs={} if method==METHODS[0] else spec
                with patch('target_study.full_settings',return_value=copy.deepcopy(opts)):
                    runtime=TargetStudyRuntime(opts,data,directory,source_spec=kwargs)
                    run_training(opts,data,directory,provenance,SOURCE,runtime_adapter=runtime,stop_after=1,device=DEVICE)
                    result=run_training(opts,data,directory,provenance,SOURCE,resume=True,device=DEVICE,
                        runtime_adapter=TargetStudyRuntime(opts,data,directory,source_spec=kwargs))
                    self.assertEqual(result['updates_this_call'],2)
                    self.assertEqual(result['result']['optimizer_steps'],3)
                    before=hashes(directory)
                    self.assertEqual(run_training(opts,data,directory,provenance,SOURCE,resume=True,device=DEVICE,
                        runtime_adapter=TargetStudyRuntime(opts,data,directory,source_spec=kwargs))['updates_this_call'],0)
                    self.assertEqual(before,hashes(directory))
            guard()
        help_result=subprocess.run([sys.executable,'scripts/train_target_full.py','--help'],capture_output=True,text=True)
        self.assertEqual(help_result.returncode,0,help_result.stderr);self.assertIn('--launch-sha256',help_result.stdout)


if __name__=='__main__':unittest.main()
