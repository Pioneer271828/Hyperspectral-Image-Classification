"""Actual three-arm training, historical zeroing, frozen pairing and review safety."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from loadData.spatial_tiles import observe_tile, tile_slice
from model.research_config import digest
from research_runtime import code_snapshot,hash_file,write_json
from train_vca_multidataset import make_train_input
from transfer import development_training as dev
from transfer.observation_policy import ObservationPolicy
from transfer.m1_ablation import (freeze_plan,prepare_policies,verify_plan,run_case,summarize,verified_case,
                                 case_path,MODES,SEEDS,decision,spec)


class M1CampaignTests(unittest.TestCase):
    def fixture(self,root):
        from test_development_training import DevelopmentTrainingTests
        tiles,record,cube,bands=DevelopmentTrainingTests().fixture()
        scene=dev.prepare_scene(cube,bands,tiles,record);marker=root/'input.json';marker.write_text('fixed input\n')
        inputs={'development_hash':record['validation_hash'],'files':{str(marker):hash_file(marker)}}
        plan=prepare_policies(freeze_plan(code_snapshot(),inputs,root/'campaign',smoke=True),scene,record)
        write_json(Path(plan['output'])/'plan.json',plan)
        return plan,scene,record

    def test_support_zero_matches_actual_legacy_function_without_touching_context(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d));value=json.loads(Path(plan['policies']['64']['legacy_support_zero']['path']).read_text())
            policy=ObservationPolicy(value);tile=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devtrain')
            obs=observe_tile(scene.cube,scene.plan,scene.stats,tile)
            local=dev.sparse_labels(record,scene.plan,'train_support')[tile_slice(tile)].reshape(-1)
            source=torch.tensor(np.concatenate((obs.spectral,obs.coordinates),axis=-1))
            retained=value['training'][1]['selected_positions'];dropped=[i for i in range(source.shape[-1]-2) if i not in retained]
            permutation=torch.tensor(dropped+retained)
            item={'net_input':source,'bands':len(scene.bands.band_ids),'height':source.shape[0],'width':source.shape[1],
                  'train_samples_gt':torch.tensor(local)}
            with patch('train_vca_multidataset.torch.randperm',return_value=permutation):
                old=make_train_input(item,(len(dropped)+.01)/len(scene.bands.band_ids))
            view=policy.view(scene,record,tile,phase='training',step=2)
            np.testing.assert_array_equal(old.numpy(),np.concatenate((view.observation.spectral,view.observation.coordinates),axis=-1))
            np.testing.assert_array_equal(view.observation.spectral.reshape(-1,source.shape[-1]-2)[local==0],obs.spectral.reshape(-1,source.shape[-1]-2)[local==0])
            self.assertEqual(len(view.bands.wavelengths),scene.cube.shape[-1])
            self.assertEqual(policy.training_record(2)['selected_band_count'],scene.cube.shape[-1])
            self.assertEqual(policy.training_record(2)['retained_band_count'],len(retained))
            with self.assertRaises(ValueError):policy.selected_statistics(scene,step=2)
            with self.assertRaises(ValueError):policy.view(scene,record,next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devval'),phase='training',step=1)

    def test_policies_share_bank_and_perturbations_and_default_old_hash(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d))
            values={m:json.loads(Path(plan['policies']['64'][m]['path']).read_text()) for m in MODES}
            self.assertEqual(values['legacy_support_zero']['training'],values['real_subset']['training'])
            self.assertEqual(len({digest(v['validation']) for v in values.values()}),1)
            self.assertNotIn('training_application',values['real_subset'])
            zero=values['legacy_support_zero'];zero.pop('policy_hash');zero.pop('training_application')
            self.assertEqual(digest(zero),values['real_subset']['policy_hash'])
            with self.assertRaises(ValueError):ObservationPolicy.freeze(scene,record,updates=2,training_configs=[{'mode':'none'}],train_seed=64,validation_seed=512,training_application='unknown')
            scene.bands=replace(scene.bands,valid=(False,)+(True,)*(len(scene.bands.band_ids)-1))
            with self.assertRaises(ValueError):ObservationPolicy.freeze(scene,record,updates=2,training_configs=[{'mode':'none'}],train_seed=64,validation_seed=512,training_application='support_zero')

    def test_actual_three_arm_training_review_and_byte_preserving_resume(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d))
            for mode in MODES:run_case(plan,scene,record,64,mode,device=os.environ.get('M1_DEVICE','cpu'))
            summary=summarize(plan)
            self.assertEqual(summary['target_updates'],6);self.assertEqual(summary['decision'],'engineering_preflight_only')
            for mode in MODES:
                path=case_path(plan,64,mode);before={str(p):hash_file(p) for p in path.rglob('*') if p.is_file()}
                review=json.loads((path/'review.json').read_text());self.assertEqual(len(review['conditions']),6)
                self.assertEqual(review['participation']['train_calls'],2)
                self.assertEqual(review['conditions']['full']['fused']['count'],45)
                if mode=='legacy_support_zero':self.assertEqual(list(review['participation']['train_width_counts']),[str(scene.cube.shape[-1])])
                with patch.object(dev,'adapt',side_effect=AssertionError('resume must not train')):
                    r=run_case(plan,scene,record,64,mode,resume=True)
                self.assertEqual(r['operation'],'verified_existing');self.assertEqual(r['status'],'completed')
                self.assertEqual(before,{str(p):hash_file(p) for p in path.rglob('*') if p.is_file()})
            checkpoint=case_path(plan,64,'real_subset')/'training/model.pt'
            with checkpoint.open('ab') as s:s.write(b'corrupt test')
            with self.assertRaises(ValueError):verified_case(plan,64,'real_subset')

    def test_frozen_policy_input_plan_tamper_and_full_budget(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d))
            raw=freeze_plan(code_snapshot(),{'development_hash':spec()['development_hash'],'files':{}},Path(d)/'full')
            self.assertEqual(raw['seeds'],list(SEEDS));self.assertEqual(len(raw['modes'])*len(raw['seeds'])*raw['updates'],45000)
            for key,value in [('updates',3),('seeds',[512]),('modes',['full'])]:
                wrong=copy.deepcopy(plan);wrong[key]=value;wrong['plan_hash']=digest({k:v for k,v in wrong.items() if k!='plan_hash'})
                with self.assertRaises(ValueError):verify_plan(wrong)
            path=Path(plan['policies']['64']['full']['path']);path.write_text('{}')
            with self.assertRaises(ValueError):verify_plan(plan)
            self.assertFalse((Path(plan['output'])/'cases').exists())

    def test_failed_and_inprogress_and_wrong_case_never_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            plan,scene,record=self.fixture(Path(d))
            with self.assertRaises(ValueError):run_case(plan,scene,record,65,'full')
            wrong=copy.deepcopy(record);wrong['validation_hash']='wrong'
            with self.assertRaises(ValueError):run_case(plan,scene,wrong,64,'full')
            self.assertFalse((Path(plan['output'])/'cases').exists())
            for mode,status in [('full','failed'),('real_subset','running')]:
                path=case_path(plan,64,mode);path.mkdir(parents=True);write_json(path/'case.json',{'status':status})
                before=(path/'case.json').read_bytes()
                with self.assertRaises((ValueError,KeyError)):run_case(plan,scene,record,64,mode,resume=True)
                self.assertEqual(before,(path/'case.json').read_bytes())

    def test_registered_decision_protects_full_accuracy_and_both_controls(self):
        def summaries(full=.8,real=.8,zero=.7,robust=.6):
            return {'full':{'full_oa':{'mean':full},'full_aa':{'mean':full},'perturbed_oa':{'mean':.4}},
                    'legacy_support_zero':{'perturbed_oa':{'mean':zero}},
                    'real_subset':{'full_oa':{'mean':real},'full_aa':{'mean':real},'perturbed_oa':{'mean':robust}}}
        pairs=[{'full_oa_vs_full':0.,'perturbed_oa_vs_full':.2,'perturbed_oa_vs_legacy_support_zero':.1}]*3
        self.assertEqual(decision(summaries(zero=.5),pairs,'registered_development_screen'),'development_robustness_candidate_pending_confirmation')
        self.assertEqual(decision(summaries(real=.79,zero=.5),pairs,'registered_development_screen'),'no_joint_accuracy_robustness_gain_keep_full')
        self.assertEqual(decision(summaries(),pairs,'registered_development_screen'),'no_joint_accuracy_robustness_gain_keep_full')
        self.assertEqual(decision(summaries(zero=.5),pairs,'two_step_preflight'),'engineering_preflight_only')

    def test_summary_uses_sample_sd_and_rejects_broken_pair_identity(self):
        with tempfile.TemporaryDirectory() as d:
            plan=freeze_plan(code_snapshot(),{'development_hash':spec()['development_hash'],'files':{}},Path(d)/'full')
            plan['plan_hash']=digest(plan)  # aggregation-only fixture; file/plan validation is tested separately
            for i,seed in enumerate(SEEDS):
                for mode in MODES:
                    path=case_path(plan,seed,mode);path.mkdir(parents=True)
                    full=.6+.1*i;perturbed=full+({'full':-.2,'legacy_support_zero':-.1,'real_subset':.1}[mode])
                    scores={c:{'fused':{'oa':full if c=='full' else perturbed,'aa':full,'kappa':full,'macro_f1':full}} for c in ('full','gap','random_25','random_50','random_75','stride')}
                    write_json(path/'review.json',{'conditions':scores,'selected_step':100})
            def verified(p,s,m):
                return {k:str(s) for k in ('initial_backbone_hash','initial_head_hash','sample_trace_hash','statistics_hash','evaluation_bank_hash','perturbation_sequence_hash')}|{'started_unix':0.,'finished_unix':1.}
            with patch('transfer.m1_ablation.verify_plan'),patch('transfer.m1_ablation.verified_case',side_effect=verified):
                report=summarize(plan)
                self.assertAlmostEqual(report['summary']['full']['full_oa']['sample_sd'],.1)
                self.assertEqual(report['decision'],'development_robustness_candidate_pending_confirmation')
            def broken(p,s,m):return verified(p,s,m)|({'evaluation_bank_hash':'changed'} if m=='real_subset' else {})
            with patch('transfer.m1_ablation.verify_plan'),patch('transfer.m1_ablation.verified_case',side_effect=broken),self.assertRaisesRegex(ValueError,'paired mismatch'):
                summarize(plan)
