"""External CLI, exact continuation, exposure and failure-preserving boundaries."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from model.research_config import canonical_json, digest
from research_runtime import ROOT, code_snapshot
from transfer.curriculum_cli import register_curriculum, execute_curriculum, load_registration
from transfer.curriculum_training import CurriculumRun
from transfer.curriculum_exposure import ExposurePlan
from transfer.g3_training import state_hash
from transfer.observation_curriculum import CurriculumPolicy
from transfer.observation_policy import ObservationPolicy
from transfer.registered_development import export_development
from transfer.sample_curriculum import SampleCurriculum, score_training_support
from tests.test_multiclass_development import fixture


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepared(root, classes=9, arm='sample', updates=None):
    # The original 3-class/24-step synthetic fixture is an exposure rejection
    # control. Its separate positive fixture has a fixed 48-step budget.
    updates = (48 if classes == 3 else 24) if updates is None else updates
    routed, scene, record, _, _ = fixture(classes)
    kwargs = copy.deepcopy(routed.backbone_kwargs)
    kwargs.pop('graph_route')
    source = MultiDatasetVCASAHGCN({scene.dataset: classes}, **kwargs).eval()
    source.load_state_dict({k: v for k, v in routed.state_dict().items() if not k.startswith('backbone.graph_router.')})
    source.backbone.adapter.cache_attention = False
    inputs = root/'inputs'
    inputs.mkdir()
    development = export_development(scene, record, inputs/'data')
    weight = inputs/'source.pt'
    save_model_checkpoint(source, weight)
    plan = ExposurePlan.freeze(scene, record, updates=updates, seed=64,
        minimum_class_blocks={str(c): 1 for c in record['classes']}, minimum_support_visits=1)
    exposure = inputs/'exposure.json'
    exposure.write_text(canonical_json(plan.to_dict()))
    sample, observation = None, None
    if arm == 'sample':
        scores = score_training_support(source, scene, record, expected_teacher_state=state_hash(source))
        value = SampleCurriculum.freeze(scores, updates=updates, order='easy_to_hard', seed=512,
                                       start_fraction=.2, full_at=updates, minimum_weight=.1)
        path = inputs/'sample.json'
        path.write_text(canonical_json(value.to_dict()))
        sample = pin(path)
    if arm == 'observation':
        configs = [{'mode': 'none'}, {'mode': 'random_subset', 'keep_fraction': .5}]
        base = ObservationPolicy.freeze(scene, record, updates=updates, training_configs=configs,
                                        train_seed=64, validation_seed=512)
        value = CurriculumPolicy.freeze(base, order='easy_to_hard', seed=2048,
                                        severity={digest(c): i for i, c in enumerate(configs)})
        path = inputs/'observation.json'
        path.write_text(canonical_json(value.to_dict()))
        observation = pin(path)
    return dict(source=pin(weight), metadata=pin(Path(str(weight)+'.json')), development=development,
                exposure=pin(exposure), sample_policy=sample, observation_policy=observation)


def command(sha, action, **values):
    args = [sys.executable, str(ROOT/'scripts/run_curriculum.py'), '--expected-commit', sha, action]
    for name, value in values.items():
        if value is None:
            continue
        if isinstance(value, dict):
            args += ['--'+name.replace('_', '-'), value['path'], '--'+name.replace('_', '-')+'-sha256', value['sha256']]
        else:
            args += ['--'+name.replace('_', '-'), str(value)]
    return args


class CurriculumCLITests(unittest.TestCase):
    def setUp(self):
        self.sha = code_snapshot()['commit']

    def register(self, root, inputs):
        return register_curriculum(inputs, out=root/'inputs/registration.json', lr=.0001,
                                   torch_seed=512, expected_commit=self.sha)

    def test_cli_three_classes_three_arms_exact_restore_and_completed_zero(self):
        for classes in (3, 9, 22):
            for arm in ('off', 'sample', 'observation'):
                with self.subTest(classes=classes, arm=arm), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    inputs = prepared(root, classes, arm)
                    budget = 48 if classes == 3 else 24
                    def cli(action, **values):
                        p = subprocess.run(command(self.sha, action, **values), cwd=ROOT, text=True,
                            capture_output=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
                        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
                        return json.loads(p.stdout)
                    reg = cli('register', **inputs, out=root/'inputs/registration.json', lr=.0001, torch_seed=512)
                    part = cli('run', registration=reg, out=root/'part', to_step=2)
                    done = cli('run', registration=reg, out=root/'done', to_step=budget, resume=part['checkpoint'])
                    with patch('torch.optim.Adam.step', side_effect=AssertionError('completed run updated')):
                        zero = execute_curriculum(reg, out=root/'zero', to_step=budget,
                                                  expected_commit=self.sha, resume=done['checkpoint'])
                    full = execute_curriculum(reg, out=root/'full', to_step=budget, expected_commit=self.sha)
                    self.assertEqual((part['new_updates'], done['new_updates'], zero['new_updates']), (2, budget-2, 0))
                    a = torch.load(done['checkpoint']['path'], weights_only=True)
                    b = torch.load(full['checkpoint']['path'], weights_only=True)
                    for key in ('history', 'selected_step', 'progress', 'rng', 'contract_hash'):
                        self.assertEqual(a[key], b[key])
                    self.assertTrue(torch.equal(a['torch_rng'], b['torch_rng']))
                    for key in ('state', 'selected_state'):
                        for name, value in a[key].items():
                            self.assertTrue(torch.equal(value, b[key][name]))
                    for i, slot in a['optimizer']['state'].items():
                        for name, value in slot.items():
                            self.assertTrue(torch.equal(value, b['optimizer']['state'][i][name]))
                    self.assertEqual(a['history'][-1]['validation']['support'], [5]*classes)
                    self.assertTrue(full['acceptance']['exposure']['exposure']['meets_minimum'])
                    self.assertFalse(full['independent_acceptance'])
                    for item in zero['inputs']:
                        self.assertEqual(pin(Path(item['path'])), item)

    def test_changed_file_rejected_before_weight_deserialization(self):
        for name in ('source', 'metadata', 'development', 'exposure', 'sample_policy'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                inputs = prepared(root)
                bad = copy.deepcopy(inputs)
                bad[name]['sha256'] = '0'*64
                with patch('torch.load', side_effect=AssertionError('loaded before hash authentication')):
                    with self.assertRaises(ValueError):
                        self.register(root, bad)
                self.assertFalse((root/'inputs/registration.json').exists())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reg = self.register(root, prepared(root))
            with patch('torch.load', side_effect=AssertionError('loaded before resume authentication')):
                with self.assertRaises(ValueError):
                    execute_curriculum(reg, out=root/'bad', to_step=1, expected_commit=self.sha,
                                       resume={**reg, 'sha256': '0'*64})

    def test_semantic_contract_tamper_budget_and_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = prepared(root)
            reg = self.register(root, inputs)
            original = json.loads(Path(reg['path']).read_text())
            with patch('torch.optim.Adam.step', side_effect=AssertionError('invalid execution updated')):
                for key, value in (('lr', .002), ('expected_commit', '0'*40), ('query', 1), ('extra', 'invalid')):
                    v = copy.deepcopy(original)
                    v[key] = value
                    path = root/'inputs'/f'bad-{key}.json'
                    path.write_text(canonical_json(v))
                    with self.assertRaises(ValueError):
                        execute_curriculum(pin(path), out=root/'bad', to_step=1, expected_commit=self.sha)
                for target in (-1, 25, True):
                    with self.assertRaises(ValueError):
                        execute_curriculum(reg, out=root/'bad', to_step=target, expected_commit=self.sha)
                with self.assertRaises(ValueError):
                    execute_curriculum(reg, out=root/'inputs/inside', to_step=1, expected_commit=self.sha)
                with patch('transfer.curriculum_cli.code_snapshot', return_value={**code_snapshot(), 'dirty': True}):
                    with self.assertRaises(ValueError):
                        execute_curriculum(reg, out=root/'bad', to_step=1, expected_commit=self.sha)
            good = execute_curriculum(reg, out=root/'part', to_step=2, expected_commit=self.sha)
            with self.assertRaises(FileExistsError):
                execute_curriculum(reg, out=root/'part', to_step=2, expected_commit=self.sha)
            for out, target in ((root/'bad', 1), (root/'part/nested', 3)):
                with self.assertRaises(ValueError):
                    execute_curriculum(reg, out=out, to_step=target, expected_commit=self.sha, resume=good['checkpoint'])

    def test_exposure_and_policy_budget_tamper_and_combination_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = prepared(root, classes=3, updates=24)
            with patch('torch.optim.Adam.step', side_effect=AssertionError('underexposed registration updated')):
                with self.assertRaisesRegex(ValueError, 'exposure'):
                    self.register(root, original)
            self.assertFalse((root/'inputs/registration.json').exists())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = prepared(root)
            with patch('torch.optim.Adam.step', side_effect=AssertionError('registration updated')):
                dual = copy.deepcopy(inputs)
                dual['observation_policy'] = inputs['sample_policy']
                with self.assertRaisesRegex(ValueError, 'separate'):
                    self.register(root, dual)
                path = Path(inputs['exposure']['path'])
                v = json.loads(path.read_text())
                v['events'][0]['support_ids'].append(999)
                v['plan_hash'] = digest({k: value for k, value in v.items() if k != 'plan_hash'})
                path.write_text(canonical_json(v))
                inputs['exposure'] = pin(path)
                with self.assertRaises(ValueError):
                    self.register(root, inputs)

    def test_failed_attempt_preserved_and_not_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reg = self.register(root, prepared(root))
            with patch.object(CurriculumRun, '_evaluate', side_effect=RuntimeError('injected after optimizer')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    execute_curriculum(reg, out=root/'failed', to_step=2, expected_commit=self.sha)
            status = json.loads((root/'failed/status.json').read_text())
            packet = torch.load(status['failure']['path'], weights_only=True)
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['completed_updates'], 0)
            self.assertTrue(packet['optimizer']['state'])
            self.assertFalse(packet['resumable'])
            with self.assertRaises(ValueError):
                execute_curriculum(reg, out=root/'bad', to_step=2, expected_commit=self.sha, resume=status['failure'])

    def test_transport_does_not_read_heldout_and_cli_missing_pin_rejected(self):
        from loadData.spatial_tiles import tile_slice
        _, scene, record, _, _ = fixture(22)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = export_development(scene, record, root/'a')
            for tile in record['roles']:
                if tile['role'] not in ('devtrain', 'devval'):
                    scene.cube[tile_slice(tile)] = np.nan
            b = export_development(scene, record, root/'b')
            self.assertEqual(a['sha256'], b['sha256'])
            p = subprocess.run(command(self.sha, 'run', registration=a, to_step=0, out=root/'bad', resume='missing'),
                               cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('must be paired', p.stderr)
            self.assertFalse((root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
