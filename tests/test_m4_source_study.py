import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from joint_baseline import NAMES
from joint_training import run_training,validation_epochs,schedule_for,_tree_fingerprint
from m4_transfer import M4SourceRuntime,source_options,graph_binding
from m4_source_study import MODES,load_study,full_settings,select_case,case_name,M4SourceFullRuntime
from model.research_config import ResearchConfig
from target_initialization import state_id
from test_calibrated_selection import items
from test_joint_training import SOURCE,checkpoints,hashes

DEVICE=os.environ.get('M4G4B_DEVICE','cpu');torch.set_num_threads(1)


class M4SourceStudyTests(unittest.TestCase):
    def fixture(self):
        cases=[]
        for h in NAMES:
            for s in (64,512,2048):
                for m in MODES:
                    v=full_settings(h,s,m);binding={n:{'Q':'a'*64,'A':'b'*64} for n in v['source_tasks']}
                    ctx={'held_out':h,'sources':v['source_tasks'],'target_payload_accessed':False}
                    cases.append({'name':case_name(h,s,m),'held_out':h,'seed':s,'mode':m,'settings':v,
                        'out':str(Path('/study/cases')/case_name(h,s,m)),'graph_binding':binding,'source_context':ctx,
                        'provenance':{**ctx,'m4_mode':m,'m4_graph_binding':binding}})
        return {'code':SOURCE,'study':load_study(),'cases':cases,'out_dir':'/study','test_scoring':False,'expected_updates':18036}

    def test_full_grid_budget_native_schedule_and_old_short_unchanged(self):
        launch=self.fixture();self.assertEqual(sum(c['settings']['optimizer_steps'] for c in launch['cases']),18036)
        for c in launch['cases']:
            opt=c['settings'];M4SourceFullRuntime().validate(opt)
            self.assertEqual(len(validation_epochs(opt)),51);self.assertEqual(len(schedule_for(opt,task_names=tuple(opt['source_tasks']))),1002)
            self.assertEqual(select_case(launch,SOURCE,c['held_out'],c['seed'],c['mode']),c)
            short=source_options(c['held_out'],c['mode'],seed=c['seed']);M4SourceRuntime().validate(short)
            with self.assertRaises(ValueError):M4SourceRuntime().validate(opt)
            with self.assertRaises(ValueError):M4SourceFullRuntime().validate(short)
            self.assertEqual(short['optimizer_steps'],8)

    def test_launch_role_mode_path_config_and_producer_tamper(self):
        for field in ('case_count','mode','output','config','context','producer','budget','score','study'):
            bad=copy.deepcopy(self.fixture())
            if field=='case_count':bad['cases'].pop()
            elif field=='mode':bad['cases'][0]['mode']='x_region'
            elif field=='output':bad['cases'][1]['out']=bad['cases'][0]['out']
            elif field=='config':bad['cases'][0]['settings']['config']['model']['region_residual_mode']='none'
            elif field=='context':bad['cases'][0]['source_context']['target_payload_accessed']=True
            elif field=='producer':bad['code']={'commit':'wrong'}
            elif field=='budget':bad['expected_updates']=9018
            elif field=='score':bad['test_scoring']=True
            else:bad['study']['epochs']=3
            with self.assertRaises(ValueError):select_case(bad,SOURCE,'paviaU',64,MODES[0])
        for key,value in [('epochs',3),('optimizer_steps',8),('m4_mode','none'),('source_tasks',list(NAMES))]:
            settings=full_settings('xuzhou',64,MODES[0]);settings[key]=value
            with self.assertRaises(ValueError):M4SourceFullRuntime().validate(settings)

    def test_scaled_full_runtime_continuation_gradient_binding_and_native_finalization(self):
        # This mock is confined to a synthetic unit fixture; no CLI accepts shortened full budgets.
        for mode in MODES:
            tiny=full_settings('xuzhou',64,mode);tiny.update(epochs=3,val_interval=1,optimizer_steps=8)
            tiny['config']['training'].update(epochs=3,val_interval=1)
            data={n:i for n,i in items().items() if n in tiny['source_tasks']}
            prov={'held_out':'xuzhou','sources':tiny['source_tasks'],'target_payload_accessed':False,
                'm4_mode':mode,'m4_graph_binding':graph_binding(data)}
            with self.assertRaises(ValueError):M4SourceFullRuntime().validate(tiny)
            with patch('m4_source_study.full_settings',return_value=tiny),tempfile.TemporaryDirectory() as tmp:
                a,b=Path(tmp)/'a',Path(tmp)/'b';runtime=M4SourceFullRuntime()
                run_training(tiny,data,a,prov,SOURCE,device=DEVICE,runtime_adapter=runtime)
                run_training(tiny,data,b,prov,SOURCE,device=DEVICE,runtime_adapter=runtime,stop_after=2)
                result=run_training(tiny,data,b,prov,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)
                self.assertEqual(result['updates_this_call'],6)
                if DEVICE=='cpu':
                    for k in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities'):
                        self.assertEqual(_tree_fingerprint(checkpoints(a)[k]),_tree_fingerprint(checkpoints(b)[k]),k)
                trace=json.loads((b/'trace.json').read_text());self.assertGreater(sum(x['m4_changed_elements'] for x in trace),0)
                self.assertTrue(all(x['m4_parameter_gradient_norm']>0 for x in trace))
                before=hashes(b)
                self.assertEqual(run_training(tiny,data,b,prov,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)['updates_this_call'],0)
                self.assertEqual(hashes(b),before)
                step=result['result']['selected_step'];raw=b/f'raw-validation-{step:06d}.pt'
                wrong=copy.deepcopy(data);wrong['paviaU']['A']=wrong['paviaU']['A']*2
                with self.assertRaises(ValueError):runtime.load_selected(b/'primary.pt',wrong,raw,device=DEVICE)

    def test_initial_state_binding_and_two_remote_cli_surfaces(self):
        cfg=ResearchConfig.from_dict(full_settings('xuzhou',64,MODES[1])['config']);counts={n:9 if n=='paviaU' else 22 for n in cfg.data.datasets}
        model=M4SourceFullRuntime().build_model(cfg,counts);pin=state_id(model)
        with self.assertRaises(ValueError):M4SourceFullRuntime('0'*64).build_model(cfg,counts)
        with patch.object(M4SourceRuntime,'build_model',return_value=model):
            self.assertIs(M4SourceFullRuntime(pin).build_model(cfg,counts),model)
        for entry in ('scripts/train_m4_source_full.py','research/reproduction/m4g4b/supervise.py'):
            r=subprocess.run([sys.executable,entry,'--help'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--expected-commit',r.stdout)


if __name__=='__main__':unittest.main()
