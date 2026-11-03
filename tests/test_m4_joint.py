"""M4 joint settings, real gradient and safe continuation contracts."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
import torch

from joint_calibrated_selection import calibrated_settings, load_selected
from joint_training import run_training, _tree_fingerprint
from model.checkpoint_io import load_model_checkpoint
from model.research_config import ResearchConfig
from scripts.train_joint_m4 import m4_settings
from test_calibrated_selection import items
from test_joint_training import SOURCE, PROVENANCE, checkpoints, hashes, settings

DEVICE = os.environ.get('M4J1_DEVICE', 'cpu')
torch.set_num_threads(1)

class M4JointTests(unittest.TestCase):
    def test_default_settings_are_exact_and_modes_are_bound(self):
        original = settings(); saved = copy.deepcopy(original)
        self.assertEqual(m4_settings(original, 'none'), calibrated_settings(original))
        self.assertEqual(original, saved)
        for mode in ('capacity_control', 'x_residual', 'residual_only', 'x_region'):
            self.assertEqual(m4_settings(original, mode)['config']['model']['region_residual_mode'], mode)
        with self.assertRaises(ValueError): m4_settings(original, 'bad')

    def test_active_module_updates_and_completed_resume_is_immutable(self):
        data = items(); opts = m4_settings(settings(), 'x_residual')
        torch.manual_seed(opts['model_seed'])
        initial = ResearchConfig.from_dict(opts['config']).build_model({n: i['class_num'] for n, i in data.items()})
        initial_weight = initial.backbone.region_residual.projection.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            first = run_training(opts, data, path, PROVENANCE, SOURCE, device=DEVICE, stop_after=5)
            self.assertEqual(first['cursor'], 5)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                run_training(m4_settings(settings(), 'none'), data, path, PROVENANCE, SOURCE, device=DEVICE, resume=True)
            result = run_training(opts, data, path, PROVENANCE, SOURCE, device=DEVICE, resume=True)
            self.assertEqual(result['updates_this_call'], 7)
            last = load_model_checkpoint(path / 'last.pt', device=DEVICE)
            self.assertFalse(torch.equal(last.backbone.region_residual.projection.weight.detach().cpu(), initial_weight))
            selected = result['result']['selected_step']
            primary = load_selected(path / 'primary.pt', data, path / f'raw-validation-{selected:06d}.pt', device=DEVICE)
            self.assertEqual(primary.backbone.region_residual_mode, 'x_residual')
            before = hashes(path)
            self.assertEqual(run_training(opts, data, path, PROVENANCE, SOURCE, device=DEVICE, resume=True)['updates_this_call'], 0)
            self.assertEqual(before, hashes(path))

if __name__ == '__main__': unittest.main()
