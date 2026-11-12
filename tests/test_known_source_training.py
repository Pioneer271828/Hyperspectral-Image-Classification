import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from evaluation.known_context_preparation import prepare_known_contexts
from model.checkpoint_io import load_model_checkpoint
from test_known_context_preparation import fixture
from transfer.registered_development import _json_new, _pin


ROOT = Path(__file__).resolve().parents[1]


class KnownSourceTrainingTests(unittest.TestCase):
    def test_fresh_known_class_training_uses_only_source_and_writes_producer_declaration(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with TemporaryDirectory() as folder:
            base = Path(folder)
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
