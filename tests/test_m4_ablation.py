"""M4 campaign fairness, actual four-arm training, review and immutable resume."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from model.research_config import digest
from research_runtime import code_snapshot, hash_file, write_json
from transfer import development_training
from transfer.m4_ablation import (freeze_plan, verify_plan, run_case, verified_case, summarize,
                                  case_path, spec, MODES)
from scripts.run_m4_ablation import schedule, execute


class M4CampaignTests(unittest.TestCase):
    def fixture(self, root):
        from test_development_training import DevelopmentTrainingTests
        plan, record, cube, bands = DevelopmentTrainingTests().fixture()
        scene = development_training.prepare_scene(cube, bands, plan, record)
        marker = root / 'frozen-input.json'; marker.write_text('frozen input\n')
        inputs = {'development_hash': record['validation_hash'], 'files': {str(marker): hash_file(marker)}}
        frozen = freeze_plan(code_snapshot(), inputs, root / 'campaign', smoke=True)
        Path(frozen['output']).mkdir(); write_json(Path(frozen['output']) / 'plan.json', frozen)
        return frozen, scene, record

    def test_four_arm_short_training_pairs_metrics_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, scene, record = self.fixture(Path(directory))
            for mode in MODES:
                result = run_case(plan, scene, record, 64, mode, device=os.environ.get('M4_DEVICE', 'cpu'))
                self.assertEqual(result['status'], 'completed')
            summary = summarize(plan)
            self.assertEqual(summary['target_updates'], 8)
            self.assertEqual(summary['decision'], 'engineering_preflight_only')
            self.assertEqual(len(summary['rows']), 4)
            for row in summary['rows']:
                self.assertEqual(row['count'], 45)
                self.assertEqual([c['support'] for c in row['per_class']], [5] * 9)
            for mode in MODES:
                path = case_path(plan, 64, mode); original = (path / 'case.json').read_bytes()
                artifact = json.loads(original)['artifact_dir']
                with patch.object(development_training, 'adapt', side_effect=AssertionError('verified resume must not train')):
                    resumed = run_case(plan, scene, record, 64, mode, resume=True)
                self.assertEqual(resumed['status'], 'completed')
                self.assertEqual(resumed['operation'], 'verified_existing')
                self.assertEqual(resumed['artifact_dir'], artifact)
                self.assertEqual(original, (path / 'case.json').read_bytes())
            receipt = verified_case(plan, 64, 'x_residual')
            path = case_path(plan, 64, 'x_residual')
            review = json.loads((path / 'review.json').read_text())
            self.assertEqual(review['participation']['train_calls'], 2)
            self.assertIn('residual_hint_off', review['interventions'])
            self.assertFalse(review['test_evaluated'])
            # Corruption of a real checkpoint must reject reuse, preserving it for audit.
            with (path / 'training/model.pt').open('ab') as stream: stream.write(b'intentional probe')
            with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
                verified_case(plan, 64, 'x_residual')
            self.assertEqual(receipt['artifact_dir'], str(path / 'training'))

    def test_plan_identity_budget_scope_and_input_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _, _ = self.fixture(Path(directory))
            for field, value in [('updates', 3), ('seeds', [512]), ('modes', ['none'])]:
                wrong = copy.deepcopy(plan); wrong[field] = value
                wrong['plan_hash'] = digest({k: v for k, v in wrong.items() if k != 'plan_hash'})
                with self.assertRaises(ValueError): verify_plan(wrong)
            wrong = copy.deepcopy(plan); wrong['plan_hash'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): verify_plan(wrong)
            marker = Path(next(iter(plan['inputs']['files'])))
            marker.write_text('changed input\n')
            with self.assertRaisesRegex(ValueError, 'input changed'): verify_plan(plan)

    def test_registered_full_budget_and_development_identity_are_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _, _ = self.fixture(Path(directory))
            inputs = copy.deepcopy(plan['inputs']); inputs['development_hash'] = spec()['development_hash']
            full = freeze_plan(code_snapshot(), inputs, Path(directory) / 'full')
            self.assertEqual(full['updates'] * len(full['seeds']) * len(full['modes']), 60000)
            self.assertEqual(full['seeds'], [64, 512, 2048])
            inputs['development_hash'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'protocol'): freeze_plan(code_snapshot(), inputs, Path(directory) / 'bad')

    def test_failed_and_in_progress_cases_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, scene, record = self.fixture(Path(directory))
            with patch.object(development_training, 'adapt', side_effect=RuntimeError('owned failure probe')):
                with self.assertRaisesRegex(RuntimeError, 'owned failure'):
                    run_case(plan, scene, record, 64, 'none', device='cpu')
            path = case_path(plan, 64, 'none'); before = (path / 'case.json').read_bytes()
            with self.assertRaises(ValueError): run_case(plan, scene, record, 64, 'none', resume=True)
            self.assertEqual(before, (path / 'case.json').read_bytes())
            active = case_path(plan, 64, 'x_residual'); active.mkdir()
            write_json(active / 'case.json', {'status': 'running', 'plan_hash': plan['plan_hash'], 'seed': 64, 'mode': 'x_residual'})
            before = (active / 'case.json').read_bytes()
            with self.assertRaises(ValueError): run_case(plan, scene, record, 64, 'x_residual', resume=True)
            self.assertEqual(before, (active / 'case.json').read_bytes())

    def test_unregistered_case_and_scene_are_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, scene, record = self.fixture(Path(directory))
            with self.assertRaises(ValueError): run_case(plan, scene, record, 65, 'none')
            with self.assertRaises(ValueError): run_case(plan, scene, record, 64, 'x_region')
            wrong = copy.deepcopy(record); wrong['validation_hash'] = 'wrong'
            with self.assertRaises(ValueError): run_case(plan, scene, wrong, 64, 'none')
            self.assertFalse((Path(plan['output']) / 'cases').exists())

    def test_scheduler_keeps_seed_groups_unique_and_one_assignment_per_gpu(self):
        jobs = schedule([64, 512, 2048], ['GPU-a', 'GPU-b'])
        self.assertEqual(jobs, [{'seed':64, 'gpu':'GPU-a'}, {'seed':512, 'gpu':'GPU-b'}, {'seed':2048, 'gpu':'GPU-a'}])
        for seeds, gpus in [([64,64], ['GPU-a']), ([64,512], ['GPU-a','GPU-a']), ([64], [])]:
            with self.assertRaises(ValueError): schedule(seeds, gpus)

    def test_worker_failure_stops_pending_seed_groups_and_preserves_execution_record(self):
        with tempfile.TemporaryDirectory() as directory:
            smoke, _, _ = self.fixture(Path(directory))
            inputs=copy.deepcopy(smoke['inputs']); inputs['development_hash']=spec()['development_hash']
            plan=freeze_plan(code_snapshot(),inputs,Path(directory)/'full')
            Path(plan['output']).mkdir(); write_json(Path(plan['output'])/'plan.json',plan)
            real_popen,real_output=subprocess.Popen,subprocess.check_output
            launched=[]
            class FakeProcess:
                def __init__(self,seed): self.seed=seed; self.pid=40000+seed
                def poll(self): return 1 if self.seed==64 else 0
                def wait(self): return self.poll()
            def popen(command,*args,**kwargs):
                if 'worker' in command:
                    seed=int(command[command.index('--seed')+1]); launched.append(seed); return FakeProcess(seed)
                return real_popen(command,*args,**kwargs)
            def output(command,*args,**kwargs):
                if command[0]=='nvidia-smi': return '2, GPU-a, 20000 MiB, 0 %\n6, GPU-b, 20000 MiB, 0 %\n'
                return real_output(command,*args,**kwargs)
            with patch('scripts.run_m4_ablation.subprocess.Popen',side_effect=popen), \
                 patch('scripts.run_m4_ablation.subprocess.check_output',side_effect=output), \
                 self.assertRaisesRegex(ValueError,'worker failed'):
                execute(plan,['GPU-a','GPU-b'])
            self.assertEqual(launched,[64,512])
            manifest=next((Path(plan['output'])/'executions').glob('*/campaign.json'))
            saved=json.loads(manifest.read_text()); self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['jobs'][2]['status'],'not_started_after_failure')
            self.assertFalse((Path(plan['output'])/'.controller.lock').exists())

    def test_candidate_rule_requires_capacity_gain_and_uses_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            smoke,_,_=self.fixture(Path(directory)); inputs=copy.deepcopy(smoke['inputs'])
            inputs['development_hash']=spec()['development_hash']
            plan=freeze_plan(code_snapshot(),inputs,Path(directory)/'full')
            values={'none':[.6,.7,.8],'capacity_control':[.7,.6,.8],
                    'x_residual':[.8,.7,.9],'residual_only':[.4,.5,.6]}
            for i,seed in enumerate(plan['seeds']):
                for mode in MODES:
                    path=case_path(plan,seed,mode); path.mkdir(parents=True)
                    score=values[mode][i]
                    write_json(path/'review.json',{'selected_step':100,'metrics':{'fused':
                        {'oa':score,'aa':score,'kappa':score,'macro_f1':score}}})
            def completed(plan,seed,mode):
                return {k:str(seed) for k in ('shared_initialization_hash','initial_head_hash',
                        'sample_trace_hash','statistics_hash')}|{'started_unix':0.,'finished_unix':1.}
            with patch('transfer.m4_ablation.verified_case',side_effect=completed):
                summary=summarize(plan)
                self.assertEqual(summary['decision'],'development_candidate_pending_confirmation')
                self.assertAlmostEqual(summary['summary']['none']['oa']['sample_sd'],.1)
                for seed in plan['seeds']:
                    write_json(case_path(plan,seed,'capacity_control')/'review.json',
                        {'selected_step':100,'metrics':{'fused':{'oa':.95,'aa':.95,'kappa':.95,'macro_f1':.95}}})
                self.assertEqual(summarize(plan)['decision'],'no_confirmed_residual_gain_keep_baseline')


if __name__ == '__main__':
    unittest.main()
