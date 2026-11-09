"""E1-C frozen plans, five actual training arms and conservative paired decisions."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_runtime import code_snapshot,hash_file,write_json
from model.research_config import digest
from transfer import development_training as dev
from transfer.e1_ablation import (MODES,SEEDS,spec,freeze_plan,prepare_policies,verify_plan,case_base,case_path,
    policy_for,run_case,verified_case,summarize,decision)


class E1CampaignTests(unittest.TestCase):
    def fixture(self,root):
        from test_development_training import DevelopmentTrainingTests
        tiles,record,cube,bands=DevelopmentTrainingTests().fixture();scene=dev.prepare_scene(cube,bands,tiles,record)
        marker=root/'input.json';marker.write_text('fixed input\n')
        inputs={'development_hash':record['validation_hash'],'files':{str(marker):hash_file(marker)}}
        plan=prepare_policies(freeze_plan(code_snapshot(),inputs,root/'campaign',smoke=True),scene,record)
        write_json(Path(plan['output'])/'plan.json',plan);return plan,scene,record

    def test_registration_exact_budget_shared_policy_and_training_unseen_masks(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d));policy=policy_for(plan,64);value=policy.to_dict()
            self.assertEqual(len(value['training']),5);self.assertEqual(len(value['validation']),9)
            self.assertEqual(value['training'][0]['selected_positions'],value['training'][2]['selected_positions'])
            for n in plan['spec']['primary_perturbations']:
                self.assertTrue(all(value['validation'][n]['selected_positions']!=t['selected_positions'] for t in value['training']))
            self.assertEqual(len({case_base(plan,c)['training']['seed'] for c in MODES}),1)
            raw=freeze_plan(code_snapshot(),{'development_hash':spec()['development_hash'],'files':{}},Path(d)/'full')
            self.assertEqual(raw['seeds'],list(SEEDS));self.assertEqual(len(MODES)*len(SEEDS)*raw['updates'],75000)
            self.assertFalse((Path(raw['output'])/'cases').exists())

    def test_five_actual_arms_review_pairing_and_byte_preserving_resume(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d))
            for mode in MODES:run_case(plan,scene,record,64,mode,device=os.environ.get('E1_DEVICE','cpu'))
            result=summarize(plan);self.assertEqual(result['target_updates'],25);self.assertEqual(result['decision'],'engineering_preflight_only')
            for mode in MODES:
                path=case_path(plan,64,mode);before={str(p):hash_file(p) for p in path.rglob('*') if p.is_file()}
                review=json.loads((path/'review.json').read_text());self.assertEqual(len(review['conditions']),9)
                self.assertFalse(review['test_evaluated']);cov=review['participation']['coverage']
                self.assertEqual(cov['parameters'],0 if mode=='none' else 194)
                self.assertEqual(cov['train_calls'],0 if mode=='none' else 5)
                with patch.object(dev,'adapt',side_effect=AssertionError('resume must not train')):
                    resumed=run_case(plan,scene,record,64,mode,resume=True)
                self.assertEqual(resumed['operation'],'verified_existing');self.assertEqual(resumed['status'],'completed')
                self.assertEqual(before,{str(p):hash_file(p) for p in path.rglob('*') if p.is_file()})
            path=case_path(plan,64,'continuous')/'review.json';path.write_text('{}')
            with self.assertRaises(ValueError):verified_case(plan,64,'continuous')

    def test_plan_input_and_policy_tamper_rejected_even_if_plan_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d))
            for key,value in [('updates',4),('seeds',[512]),('modes',['continuous'])]:
                bad=copy.deepcopy(plan);bad[key]=value;bad['plan_hash']=digest({k:v for k,v in bad.items() if k!='plan_hash'})
                with self.assertRaises(ValueError):verify_plan(bad)
            Path(plan['policies']['64']['path']).write_text('{}')
            with self.assertRaises(ValueError):verify_plan(plan)
            self.assertFalse((Path(plan['output'])/'cases').exists())

    def test_wrong_roles_and_existing_failures_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d));bad=copy.deepcopy(record);bad['validation_hash']='bad'
            with self.assertRaises(ValueError):run_case(plan,scene,bad,64,'none')
            self.assertFalse((Path(plan['output'])/'cases').exists())
            path=case_path(plan,64,'none');path.mkdir(parents=True);write_json(path/'case.json',{'status':'failed'})
            before=(path/'case.json').read_bytes()
            with self.assertRaises((ValueError,KeyError)):run_case(plan,scene,record,64,'none',resume=True)
            self.assertEqual(before,(path/'case.json').read_bytes())

    def test_decision_requires_full_protection_and_beating_every_control(self):
        summary={c:{'full_oa':{'mean':.8},'full_aa':{'mean':.8},'range_oa':{'mean':.6}} for c in MODES}
        summary['continuous']['range_oa']['mean']=.7
        pairs=[{'full_oa_vs_none':0,**{'range_oa_vs_'+c:.1 for c in MODES if c!='continuous'}} for _ in SEEDS]
        self.assertEqual(decision(summary,pairs,'registered_development_screen'),'development_coverage_candidate_pending_confirmation')
        summary['shuffled']['range_oa']['mean']=.7
        self.assertEqual(decision(summary,pairs,'registered_development_screen'),'no_joint_coverage_gain_keep_e1_off')
        summary['shuffled']['range_oa']['mean']=.6;summary['continuous']['full_aa']['mean']=.79
        self.assertEqual(decision(summary,pairs,'registered_development_screen'),'no_joint_coverage_gain_keep_e1_off')
        self.assertEqual(decision(summary,pairs,'five_step_preflight'),'engineering_preflight_only')


if __name__=='__main__':unittest.main()
