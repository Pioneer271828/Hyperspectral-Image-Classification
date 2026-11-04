import copy,json,os,tempfile,unittest
from pathlib import Path
import numpy as np
from joint_baseline import NAMES
from joint_training import run_training,schedule_for,_tree_fingerprint
from source_training import SourceRuntime,source_settings
from test_joint_training import settings,SOURCE,PROVENANCE,checkpoints,hashes
from test_calibrated_selection import items

DEVICE=os.environ.get('G4B_DEVICE','cpu')


class SourceTrainingTests(unittest.TestCase):
    def test_legacy_schedule_unchanged_and_source_roles_policies_rejected(self):
        for seed in (64,512,2048):
            old=settings(seed);rng=np.random.default_rng(seed)
            expected=[str(n) for _ in range(4) for n in rng.choice(NAMES,size=3,replace=True)]
            self.assertEqual(schedule_for(old),expected)
            for held in NAMES:
                opts=source_settings(held,seed=seed);runtime=SourceRuntime();tasks=runtime.task_names(opts)
                self.assertNotIn(held,tasks);self.assertEqual(len(schedule_for(opts,task_names=tasks)),8)
        for change in ('epochs','source_tasks','config','validation_policy','test_scoring'):
            bad=source_settings('xuzhou')
            if change=='config':bad['config']['training']['lr']=1
            elif change=='source_tasks':bad[change]=list(NAMES)
            elif change=='validation_policy':bad[change]={}
            elif change=='test_scoring':bad[change]=True
            else:bad[change]=500
            with self.assertRaises(ValueError):SourceRuntime().validate(bad)
        self.assertEqual(source_settings('xuzhou',('paviaU',))['optimizer_steps'],4)

    def test_source_only_resume_and_readonly_completion(self):
        opts=source_settings('xuzhou');data={n:i for n,i in items().items() if n in opts['source_tasks']}
        runtime=SourceRuntime()
        with tempfile.TemporaryDirectory() as tmp:
            full,split=Path(tmp)/'full',Path(tmp)/'split'
            result=run_training(opts,data,full,PROVENANCE,SOURCE,device=DEVICE,runtime_adapter=runtime)
            run_training(opts,data,split,PROVENANCE,SOURCE,device=DEVICE,runtime_adapter=runtime,stop_after=3)
            resumed=run_training(opts,data,split,PROVENANCE,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)
            self.assertEqual(resumed['updates_this_call'],5)
            for key in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities'):
                self.assertEqual(_tree_fingerprint(checkpoints(full)[key]),_tree_fingerprint(checkpoints(split)[key]),key)
            self.assertEqual(result['result']['optimizer_steps'],8)
            self.assertEqual(set(result['result']['per_dataset_updates']),set(data))
            for row in checkpoints(full)['trace']:
                self.assertGreater(row['gradient_norms']['graph'],0)
                self.assertGreater(row['graph_changed_elements'],0)
            saved=hashes(split)
            self.assertEqual(run_training(opts,data,split,PROVENANCE,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)['updates_this_call'],0)
            self.assertEqual(saved,hashes(split))
            step=resumed['result']['selected_step']
            model=runtime.load_selected(split/'primary.pt',data,split/f'raw-validation-{step:06d}.pt',device=DEVICE)
            self.assertEqual(tuple(model.heads),tuple(data))
            self.assertFalse(any('xuzhou' in n.split('.') for n in model.state_dict()))
            self.assertEqual(set(result['result']['selected_validation']['metrics']),set(data))
            bad=copy.deepcopy(opts);bad['source_tasks']=['paviaU']
            with self.assertRaises(ValueError):run_training(bad,data,split,PROVENANCE,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                run_training(opts,data,split,{**PROVENANCE,'changed':True},SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)

    def test_single_source_and_task_order_or_test_payload_rejected(self):
        opts=source_settings('xuzhou',('paviaU',));data={'paviaU':items()['paviaU']};runtime=SourceRuntime()
        with tempfile.TemporaryDirectory() as tmp:
            result=run_training(opts,data,Path(tmp)/'single',PROVENANCE,SOURCE,device=DEVICE,runtime_adapter=runtime)
            self.assertEqual(result['result']['per_dataset_updates'],{'paviaU':4})
            with self.assertRaises(ValueError):run_training(opts,items(),Path(tmp)/'bad',PROVENANCE,SOURCE,device=DEVICE,runtime_adapter=runtime)
            data['paviaU']['test_gt']=None
            with self.assertRaises(ValueError):run_training(opts,data,Path(tmp)/'test',PROVENANCE,SOURCE,device=DEVICE,runtime_adapter=runtime)


if __name__=='__main__':unittest.main()
