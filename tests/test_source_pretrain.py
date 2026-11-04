import copy,json,tempfile,unittest,subprocess,sys
from pathlib import Path
from unittest.mock import patch
from joint_baseline import NAMES
from joint_training import run_training,schedule_for,validation_epochs,_tree_fingerprint
from source_training import source_settings,SourceRuntime
from source_pretrain import load_study,full_settings,select_case,SourcePretrainRuntime
from test_calibrated_selection import items
from test_joint_training import SOURCE,PROVENANCE,checkpoints,hashes


class SourcePretrainTests(unittest.TestCase):
    def fixture(self):
        return {'code':SOURCE,'study':load_study(),'test_scoring':False,'cases':[
            {'held_out':h,'seed':s,'out':f'/{h}/{s}','settings':full_settings(h,s)} for h in NAMES for s in (64,512,2048)]}

    def test_nine_grid_budget_source_only_and_old_short_unchanged(self):
        launch=self.fixture();self.assertEqual(sum(c['settings']['optimizer_steps'] for c in launch['cases']),9018)
        for c in launch['cases']:
            opts=c['settings'];SourcePretrainRuntime().validate(opts);self.assertEqual(len(validation_epochs(opts)),51)
            self.assertEqual(len(schedule_for(opts,task_names=tuple(opts['source_tasks']))),1002)
            self.assertNotIn(c['held_out'],opts['source_tasks']);self.assertEqual(select_case(launch,SOURCE,c['held_out'],c['seed']),c)
            short=source_settings(c['held_out'],seed=c['seed']);SourceRuntime().validate(short)
            self.assertEqual(short['optimizer_steps'],8)
        result=subprocess.run([sys.executable,'scripts/train_source_full.py','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--stop-after',result.stdout)

    def test_launch_and_config_tamper_rejected(self):
        launch=self.fixture();bads=[]
        for field in ('cases','study','code'):
            bad=copy.deepcopy(launch)
            if field=='cases':bad[field].pop()
            elif field=='study':bad[field]['epochs']=999
            else:bad[field]['commit']='wrong'
            bads.append(bad)
        bad=copy.deepcopy(launch);bad['cases'][1]['out']=bad['cases'][0]['out'];bads.append(bad)
        bad=copy.deepcopy(launch);bad['cases'][0]['settings']['config']['data']['datasets']=list(NAMES);bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):select_case(bad,SOURCE,'paviaU',64)
        for field,value in (('epochs',3),('optimizer_steps',12),('source_tasks',list(NAMES)),('test_scoring',True)):
            opts=full_settings('xuzhou',64);opts[field]=value
            with self.assertRaises(ValueError):SourcePretrainRuntime().validate(opts)

    def test_full_runtime_routing_resume_on_explicit_scaled_unit_fixture_only(self):
        # No CLI accepts this budget: mock only the expected factory inside this unit fixture.
        tiny=full_settings('xuzhou',64);tiny.update(epochs=3,val_interval=1,optimizer_steps=8)
        tiny['config']['training'].update(epochs=3,val_interval=1)
        data={n:i for n,i in items().items() if n in tiny['source_tasks']};runtime=SourcePretrainRuntime()
        with self.assertRaises(ValueError):runtime.validate(tiny)
        with patch('source_pretrain.full_settings',return_value=tiny),tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            run_training(tiny,data,a,PROVENANCE,SOURCE,device='cpu',runtime_adapter=runtime)
            run_training(tiny,data,b,PROVENANCE,SOURCE,device='cpu',runtime_adapter=runtime,stop_after=2)
            result=run_training(tiny,data,b,PROVENANCE,SOURCE,device='cpu',runtime_adapter=runtime,resume=True)
            self.assertEqual(result['updates_this_call'],6)
            for key in ('model','optimizer','scheduler','rng','history','trace'):
                self.assertEqual(_tree_fingerprint(checkpoints(a)[key]),_tree_fingerprint(checkpoints(b)[key]))
            before=hashes(b);self.assertEqual(run_training(tiny,data,b,PROVENANCE,SOURCE,device='cpu',runtime_adapter=runtime,resume=True)['updates_this_call'],0);self.assertEqual(before,hashes(b))


if __name__=='__main__':unittest.main()
