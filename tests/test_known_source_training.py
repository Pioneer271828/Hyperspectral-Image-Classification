import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from evaluation.known_context_preparation import prepare_known_contexts
from evaluation import known_source_training as training
from model.checkpoint_io import load_model_checkpoint
from test_known_context_preparation import fixture
from transfer.registered_development import _json_new, _pin


ROOT = Path(__file__).resolve().parents[1]


def _prepared_trial(base, sha):
    reg, raw = fixture(base)
    raw['dataset'] = 'xuzhou'
    Path(reg['path']).write_text(json.dumps(raw))
    reg = _pin(Path(reg['path']))
    prepared = prepare_known_contexts(reg, out=base/'prepared', expected_commit=sha)
    # Neither reserved role is needed for training or checkpoint selection.
    for role in ('source_validation', 'development_evaluation'):
        for directory in ('inputs', 'prepared'):
            (base/directory/(role+'.npz')).unlink()
    config = json.loads((ROOT/'configs/experiments/screen_m6_none.json').read_text())
    config['model'].update(latent_channels=8, adapter_embed_dim=8)
    config = _json_new(base/'inputs'/'config.json', config)
    trial = _json_new(base/'inputs'/'train.json', dict(version=1, kind='known_source_fixed_budget_v1',
                                             prepared=prepared, config=config, seed=64, updates=2))
    return trial, raw


class KnownSourceTrainingTests(unittest.TestCase):
    def test_fresh_known_class_training_uses_only_source_and_writes_producer_declaration(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with TemporaryDirectory() as folder:
            base = Path(folder)
            trial, raw = _prepared_trial(base, sha)
            result = subprocess.run([sys.executable, str(ROOT/'scripts/train_known_source.py'),
                                     '--expected-commit', sha, '--registration', trial['path'],
                                     '--registration-sha256', trial['sha256'], '--out', str(base/'trained')],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(Path(json.loads(result.stdout)['path']).read_text())
            self.assertEqual(receipt['optimizer_updates'], 2)
            self.assertEqual(receipt['selected_step'], 2)
            self.assertEqual(receipt['source_annotated_pixels'], 24)
            self.assertEqual(receipt['source_supervised_pixels'], 12)
            self.assertNotEqual(receipt['initial_state_sha256'], receipt['final_state_sha256'])
            declaration = json.loads(Path(receipt['training_declaration']['path']).read_text())
            self.assertEqual(declaration['source_training_class_ids'], [2, 7, 13])
            self.assertEqual(declaration['checkpoint_sha256'], receipt['checkpoint']['sha256'])
            model = load_model_checkpoint(receipt['checkpoint']['path'])
            self.assertEqual(model.heads['xuzhou'].out_features, 3)
            self.assertEqual(model.training_config['selected_step'], 2)
            self.assertFalse(Path(raw['labels']['path']).exists())

    def test_finite_loss_with_nonfinite_gradient_or_last_step_state_is_not_published(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        normal_build = training.build_model
        normal_step = torch.optim.Adam.step
        with TemporaryDirectory() as folder:
            base = Path(folder)
            trial, raw = _prepared_trial(base, sha)

            def bad_backward(*args, **kwargs):
                model, config = normal_build(*args, **kwargs)
                model.heads['xuzhou'].weight.register_hook(lambda g: torch.full_like(g, float('inf')))
                return model, config

            def verify_failure(stage, prior, attempted):
                out = base / stage
                with self.assertRaisesRegex(ValueError, f'Nonfinite source training {stage}'):
                    training.train_known_source(trial, out=out, expected_commit=sha)
                self.assertEqual({p.name for p in out.iterdir()}, {'failure.json'})
                failure = json.loads((out/'failure.json').read_text())
                self.assertEqual((failure['status'], failure['stage'], failure['failed_step']),
                                 ('failed', stage, prior+1))
                self.assertEqual((failure['accepted_optimizer_updates'], failure['optimizer_steps_attempted']),
                                 (prior, attempted))
                self.assertTrue(torch.isfinite(torch.tensor(failure['finite_loss'])))
                self.assertEqual(len(failure['accepted_loss']), prior)
                self.assertEqual(failure['query'], 0)
                self.assertFalse(Path(raw['labels']['path']).exists())

            with patch.object(training, 'build_model', side_effect=bad_backward):
                with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Adam must not run')):
                    verify_failure('gradient', 0, 0)

            for stage in ('model_state', 'optimizer_state'):
                with self.subTest(stage=stage):
                    steps = 0

                    def bad_last_step(optimizer, *args, **kwargs):
                        nonlocal steps
                        result = normal_step(optimizer, *args, **kwargs)
                        steps += 1
                        if steps == 2:
                            parameter, state = next(iter(optimizer.state.items()))
                            with torch.no_grad():
                                target = parameter if stage == 'model_state' else state['exp_avg_sq']
                                target.fill_(float('nan'))
                        return result

                    with patch.object(torch.optim.Adam, 'step', new=bad_last_step):
                        verify_failure(stage, 1, 2)
                    self.assertEqual(steps, 2)
