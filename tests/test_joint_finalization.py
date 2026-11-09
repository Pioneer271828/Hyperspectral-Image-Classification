"""The revised inference bound cannot hide changed weights, classes or incomplete training."""
import copy,json,os
from pathlib import Path
import shutil,tempfile,unittest
import torch
from evaluation.joint_finalization import finalize,verify_probability_pair,POLICY
from scripts.train_joint_m4 import m4_settings,graph_binding,M4JointRuntime
from joint_training import run_training
from research_runtime import write_json
from model.research_config import digest
from test_calibrated_selection import items
from test_joint_training import settings,PROVENANCE,hashes

DEVICE=os.environ.get('FINALIZE_DEVICE','cpu')

class FinalizationTests(unittest.TestCase):
    def test_bounded_roundoff_and_class_change_rejection(self):
        old=torch.tensor([[.8,.2],[.5000001,.4999999]])
        new=old.clone();new[0]+=torch.tensor([3e-6,-3e-6])
        self.assertGreater(verify_probability_pair(old,new)['max_abs_error'],0)
        swapped=old.clone();swapped[1]=swapped[1].flip(0)
        with self.assertRaisesRegex(ValueError,'classifications'):verify_probability_pair(old,swapped)
        large=old.clone();large[0]+=torch.tensor([1e-4,-1e-4])
        with self.assertRaisesRegex(ValueError,'bound'):verify_probability_pair(old,large)
        invalid=old.clone();invalid[0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid'):verify_probability_pair(old,invalid)
        with self.assertRaises(ValueError):verify_probability_pair(old,old.double())

    def setup_run(self,temp):
        data=items();opts=m4_settings(settings(),'x_residual')
        files={'fixture.py':'fixture-source'}
        source={'commit':'a'*40,'branch':'fixture','files':files,'source_hash':digest(files),'dirty':False}
        consumer={**source,'commit':'b'*40,'files':{**files,'verifier.py':'additive'}}
        consumer['source_hash']=digest(consumer['files'])
        provenance={**PROVENANCE,'m4_graph_binding':graph_binding(data)}
        folder=Path(temp)/'case'
        run_training(opts,data,folder,provenance,source,device=DEVICE,runtime_adapter=M4JointRuntime())
        (folder/'result.json').unlink()
        # Simulate interrupted finalization after weights are written, before summary.
        (folder/'trace.json').unlink();(folder/'history.json').unlink()
        progress=json.loads((folder/'progress.json').read_text());progress['status']='running';write_json(folder/'progress.json',progress)
        return folder,opts,data,provenance,source,consumer

    def test_full_state_completion_without_update_and_immutable_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.setup_run(temp);folder=args[0]
            before={n:h for n,h in hashes(folder).items() if n.endswith(('.pt','.pt.json'))}
            result=finalize(*args,M4JointRuntime(),device=DEVICE)
            self.assertEqual(result['operation'],'finalized_without_training')
            self.assertEqual(result['updates_this_call'],0);self.assertEqual(result['result']['optimizer_steps'],12)
            self.assertEqual(result['result']['finalization_policy'],POLICY)
            self.assertEqual(before,{n:h for n,h in hashes(folder).items() if n in before})
            all_before=hashes(folder)
            self.assertEqual(finalize(*args,M4JointRuntime(),device=DEVICE)['operation'],'verified_completed')
            self.assertEqual(all_before,hashes(folder))

    def test_incomplete_or_changed_source_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.setup_run(temp);folder=args[0]
            bad=copy.deepcopy(args[-1]);bad['files']['fixture.py']='changed'
            before=hashes(folder)
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                finalize(*args[:-1],bad,M4JointRuntime(),device=DEVICE)
            self.assertEqual(before,hashes(folder))
            progress=json.loads((folder/'progress.json').read_text());progress['cursor']=11;write_json(folder/'progress.json',progress)
            before=hashes(folder)
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                finalize(*args,M4JointRuntime(),device=DEVICE)
            self.assertEqual(before,hashes(folder))

    def test_changed_weights_and_probabilities_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.setup_run(temp);folder=args[0]
            pristine=Path(temp)/'saved';shutil.copytree(folder,pristine)
            weights=torch.load(folder/'primary.pt',weights_only=True)
            name=next(n for n,v in weights.items() if v.is_floating_point())
            weights[name].reshape(-1)[0]+=1
            torch.save(weights,folder/'primary.pt')
            before=hashes(folder)
            with self.assertRaises(ValueError):finalize(*args,M4JointRuntime(),device=DEVICE)
            self.assertEqual(before,hashes(folder));self.assertFalse((folder/'result.json').exists())
            shutil.copyfile(pristine/'primary.pt',folder/'primary.pt')
            values=torch.load(folder/'selected-probabilities.pt',weights_only=True)
            values['paviaU']['train']['fused'][0,0]+=1e-4
            torch.save(values,folder/'selected-probabilities.pt')
            before=hashes(folder)
            with self.assertRaisesRegex(ValueError,'probabilities changed'):
                finalize(*args,M4JointRuntime(),device=DEVICE)
            self.assertEqual(before,hashes(folder))

if __name__=='__main__':unittest.main()
