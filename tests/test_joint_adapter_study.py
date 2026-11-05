import copy,json,tempfile,unittest,subprocess,sys
from pathlib import Path
from joint_adapter_study import load_study,case_settings,select_case
from joint_adapter_modes import AdapterModeRuntime
from joint_training import load_plan,schedule_for
from joint_baseline import load_config


class AdapterStudyTests(unittest.TestCase):
    def test_full_entry_help_uses_adapter_modes(self):
        result=subprocess.run([sys.executable,'scripts/train_joint_adapter_full.py','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('wavelength,index,uniform',result.stdout)
        self.assertNotIn('--supervision',result.stdout)
        result=subprocess.run([sys.executable,'research/reproduction/m2j2/supervise.py','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--gpus',result.stdout)

    def fixture(self):
        study=load_study();plan=load_plan('configs/experiments/base97_training.json');base=load_config('configs/experiments/base97_joint.json')
        cases=[{'mode':m,'seed':s,'out':f'/{m}/{s}','settings':case_settings(study,plan,base,m,s)}
               for m in study['modes'] for s in study['training_seeds']]
        return {'code':{'commit':'fixture'},'study':study,'cases':cases},plan,base

    def test_full_grid_budget_and_only_mode_differs(self):
        launch,_,_=self.fixture();self.assertEqual(sum(c['settings']['optimizer_steps'] for c in launch['cases']),13527)
        for seed in (64,512,2048):
            comparable=[]
            for c in [c for c in launch['cases'] if c['seed']==seed]:
                self.assertEqual(select_case(launch,launch['code'],load_study(),c['mode'],seed),c)
                opts=copy.deepcopy(c['settings']);AdapterModeRuntime(c['mode']).validate(opts)
                self.assertEqual(len(schedule_for(opts)),1503)
                opts.pop('adapter_comparison');opts['config']['model'].pop('adapter_attention_mode');comparable.append(opts)
            self.assertTrue(all(o==comparable[0] for o in comparable))

    def test_missing_duplicate_output_source_and_policy_rejected(self):
        launch,_,_=self.fixture();bads=[]
        bad=copy.deepcopy(launch);bad['cases'].pop();bads.append(bad)
        bad=copy.deepcopy(launch);bad['cases'][-1]=bad['cases'][0];bads.append(bad)
        bad=copy.deepcopy(launch);bad['cases'][-1]['out']=bad['cases'][0]['out'];bads.append(bad)
        bad=copy.deepcopy(launch);bad['code']['commit']='changed';bads.append(bad)
        bad=copy.deepcopy(launch);bad['study']['metadata_quality']='measured';bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):select_case(bad,launch['code'],load_study(),'wavelength',64)

    def test_changed_seed_budget_and_profile_rejected(self):
        launch,plan,base=self.fixture();study=load_study()
        for seed in (65,True,64.0):
            with self.assertRaises(ValueError):case_settings(study,plan,base,'wavelength',seed)
        bad=copy.deepcopy(study);bad['updates_per_case']=5000
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.json';p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load_study(p)
        with self.assertRaises(ValueError):case_settings(bad,plan,base,'wavelength',64)


if __name__=='__main__':unittest.main()
