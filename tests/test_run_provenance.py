"""Actual G3 run settings, read-only corruption detection and numerical neutrality."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

import torch

from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import hash_file
from train_spatial import SpatialConfig
from transfer import development_training as development
from transfer import g3_training as g3
from transfer.run_provenance import verify_run_artifacts
import test_development_training as development_tests
from test_transfer_training import fixture


ROOT = Path(__file__).resolve().parents[1]
PARENT = '2d8e15e7d76d38d9f76a94351283eaa606d3e48f'
torch.set_num_threads(1)


def parent_module(name):
    module = types.ModuleType(name)
    source = subprocess.check_output(['git', 'show', PARENT + ':transfer/' + name + '.py'], cwd=ROOT, text=True)
    exec(compile(source, name + '_parent', 'exec'), module.__dict__)
    return module


class RunProvenanceTests(unittest.TestCase):
    def device(self):
        return os.environ.get('G3_DEVICE', 'cpu')

    def development_fixture(self):
        plan, record, cube, bands = development_tests.DevelopmentTrainingTests().fixture()
        scene = development.prepare_scene(cube, bands, plan, record)
        _, base, _ = fixture()
        return scene, record, base

    def assert_record(self, folder, result, *, seed, steps, phase, interval):
        restored = load_model_checkpoint(folder / 'model.pt', self.device())
        config = SpatialConfig(result['resolved_config'])
        self.assertEqual(config.training.seed, seed)
        self.assertEqual(config.budget['optimizer_steps'], steps)
        for key in ('resolved_config', 'runtime_config', 'runtime_config_hash'):
            self.assertEqual(result[key], restored.training_config[key])
        self.assertEqual(result['runtime_config']['seeds']['model'], seed)
        self.assertEqual(result['runtime_config']['phase'], phase)
        self.assertEqual(result['runtime_config']['validation_every_updates'], interval)
        before = {p.name: hash_file(p) for p in folder.iterdir() if p.is_file()}
        checked = verify_run_artifacts(folder)
        self.assertEqual(checked['optimizer_steps'], steps)
        self.assertEqual(before, {p.name: hash_file(p) for p in folder.iterdir() if p.is_file()})

    def test_development_overrides_and_two_sampling_modes(self):
        scene, record, base = self.development_fixture()
        template = copy.deepcopy(base)
        with tempfile.TemporaryDirectory() as directory:
            for index, mode in enumerate(('uniform', 'class_first')):
                folder = Path(directory) / mode
                seed = (64, 2048)[index]
                result = development.adapt(scene, record, base, folder, steps=3, val_every=2,
                                           seed=seed, sampling=mode, device=self.device())
                self.assert_record(folder, result, seed=seed, steps=3, phase='development', interval=2)
                self.assertEqual(result['runtime_config']['seeds'],
                                 {'model': seed, 'head': seed + 2, 'sampler': seed + 4, 'training_torch': seed})
                self.assertEqual(result['runtime_config']['sampling']['mode'], mode)
                self.assertEqual([v['step'] for v in result['validation']], [2, 3])
                self.assertEqual(result['resolved_config']['training']['primary_checkpoint'], 'val_loss')
        self.assertEqual(base, template)

    def test_source_and_all_fixed_final_target_methods(self):
        _, base, (scene, labels, target, _, support) = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, result = g3.train_source(scene, labels, base, root / 'source', steps=3, val_every=2,
                                            model_seed=512, sampler_seed=519, device=self.device(), task_hash='source')
            self.assert_record(root / 'source', result, seed=512, steps=3, phase='source', interval=2)
            self.assertEqual(result['runtime_config']['scheduler']['name'], 'ReduceLROnPlateau')
            self.assertEqual(result['runtime_config']['seeds']['sampler'], 519)
            for method in g3.METHODS:
                folder = root / method
                result = g3.adapt_target(target, support, base, folder, method=method, source=source,
                                         steps=3, model_seed=2048, head_seed=2050, sampler_seed=2052,
                                         device=self.device(), task_hash='target')
                self.assert_record(folder, result, seed=2048, steps=3, phase='target', interval=None)
                self.assertEqual(result['selected_step'], 3)
                self.assertIsNone(result['runtime_config']['scheduler'])
                self.assertIn('training.primary_checkpoint', result['runtime_config']['inactive_resolved_fields'])
                self.assertEqual(result['runtime_config']['frozen_backbone'], method == 'source_pretrained_head_only')

    def test_build_model_reports_actual_seed_without_mutating_template(self):
        _, base, _ = fixture()
        before = copy.deepcopy(base)
        first, config = g3.build_model(base, 'paviaU', 9, 512, self.device())
        second, other = g3.build_model(base, 'paviaU', 9, 2048, self.device())
        self.assertEqual(config.training.seed, 512)
        self.assertEqual(other.training.seed, 2048)
        self.assertNotEqual(g3.state_hash(first), g3.state_hash(second))
        self.assertEqual(base, before)

    def test_development_metadata_change_preserves_parent_training(self):
        scene, record, base = self.development_fixture()
        old = parent_module('development_training')
        # The parent's imported build_target is also replaced with its actual parent.
        old.build_target = parent_module('g3_training').build_target
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = old.adapt(scene, record, base, root / 'parent', seed=512, steps=3, val_every=2,
                                 sampling='class_first', device=self.device())
            current = development.adapt(scene, record, base, root / 'current', seed=512, steps=3, val_every=2,
                                        sampling='class_first', device=self.device())
            self.assertNotEqual(previous['resolved_config']['training']['seed'], 512)
            for key in ('initial_backbone_hash', 'initial_head_hash', 'selected_step', 'sampler'):
                self.assertEqual(previous[key], current[key], key)
            self.assertEqual([t['tile_id'] for t in previous['trace']], [t['tile_id'] for t in current['trace']])
            for before, after in zip(previous['trace'], current['trace']):
                for key in before['loss']:
                    self.assertAlmostEqual(before['loss'][key], after['loss'][key], delta=2e-5)
            before = load_model_checkpoint(root / 'parent/model.pt', self.device())
            after = load_model_checkpoint(root / 'current/model.pt', self.device())
            for key, value in before.state_dict().items():
                torch.testing.assert_close(after.state_dict()[key], value, rtol=2e-5, atol=2e-6, msg=key)

    def test_readonly_verifier_rejects_conflicting_records_and_trace(self):
        scene, record, base = self.development_fixture()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'run'
            result = development.adapt(scene, record, base, folder, seed=64, steps=3, val_every=2, device=self.device())
            result_path = folder / 'result.json'; sidecar_path = folder / 'model.pt.json'
            sidecar = json.loads(sidecar_path.read_text())
            for change in ('budget', 'seed', 'validation', 'selection', 'sampling', 'trace', 'checkpoint', 'legacy'):
                changed = copy.deepcopy(result)
                if change == 'budget': changed['resolved_config']['training']['steps_per_epoch'] += 1
                elif change == 'seed': changed['runtime_config']['seeds']['model'] += 1
                elif change == 'validation': changed['validation'][0]['step'] = 1
                elif change == 'selection': changed['selected_step'] = 99
                elif change == 'sampling': changed['runtime_config']['sampling']['mode'] = 'unknown'
                elif change == 'trace': changed['trace'][0]['tile_id'] = -1
                elif change == 'checkpoint': changed['checkpoint_sha256'] = '0' * 64
                else:
                    for key in ('runtime_config', 'runtime_config_hash', 'resolved_config'): changed.pop(key)
                result_path.write_text(json.dumps(changed))
                fingerprint = hash_file(result_path)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verify_run_artifacts(folder)
                self.assertEqual(hash_file(result_path), fingerprint)
                self.assertEqual(json.loads(sidecar_path.read_text()), sidecar)
            result_path.write_text(json.dumps(result))
            verify_run_artifacts(folder)

    def test_rehashed_conflicts_on_both_sides_are_still_rejected(self):
        scene, record, base = self.development_fixture()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'run'
            result = development.adapt(scene, record, base, folder, seed=64, steps=3, val_every=2, device=self.device())
            sidecar = json.loads((folder / 'model.pt.json').read_text())
            for change in ('seed', 'optimizer', 'sampler_seed', 'validation', 'selected_step'):
                changed = copy.deepcopy(result); altered = copy.deepcopy(sidecar)
                runtime = changed['runtime_config']
                if change == 'seed': runtime['seeds']['model'] += 1
                elif change == 'optimizer': runtime['optimizer']['initial_groups'][0]['lr'] *= 2
                elif change == 'sampler_seed': runtime['seeds']['sampler'] += 1
                elif change == 'validation': runtime['validation_every_updates'] = 1
                else:
                    changed['selected_step'] = 1
                    altered['training_config']['selected_step'] = 1
                    # A tie-free forced contradiction with the recorded minimum.
                    changed['validation'][0]['nll'] = 1
                    changed['validation'][1]['nll'] = 2
                changed['runtime_config_hash'] = digest(runtime)
                for key in ('resolved_config', 'runtime_config', 'runtime_config_hash'):
                    altered['training_config'][key] = changed[key]
                (folder / 'model.pt.json').write_text(json.dumps(altered))
                changed['sidecar_sha256'] = hash_file(folder / 'model.pt.json')
                (folder / 'result.json').write_text(json.dumps(changed))
                with self.subTest(change=change), self.assertRaises(ValueError): verify_run_artifacts(folder)


if __name__ == '__main__':
    unittest.main()
