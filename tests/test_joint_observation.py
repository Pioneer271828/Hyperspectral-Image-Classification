"""M1 joint views: unchanged full control, replay, restart and identity guards."""
import copy
from dataclasses import replace
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from joint_calibrated_selection import calibrated_settings
from joint_observation import freeze_settings, JointObservationRuntime
from joint_training import run_training, _tree_fingerprint, schedule_for
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from test_calibrated_selection import items
from test_joint_training import settings, SOURCE, PROVENANCE, checkpoints, hashes

DEVICE = os.environ.get('M1J1_DEVICE', 'cpu')
torch.set_num_threads(1)


def metadata(data):
    return {n: BandMetadata('synthetic:'+n, tuple(range(i['bands'])),
                            tuple(i['wavelengths'].tolist()), 'nm', 'approximate', (True,)*i['bands'],
                            None if i['fwhm'] is None else tuple(i['fwhm'].tolist()),
                            None if i['fwhm'] is None else 'nm') for n, i in data.items()}


def frozen(data, preset='alternate75', seed=8128):
    return freeze_settings(settings(), data, metadata(data), preset=preset, sampling_seed=seed, plan_hash='fixture')


def run(opts, data, path, **kwargs):
    return run_training(opts, data, path, PROVENANCE, SOURCE, device=DEVICE,
                        runtime_adapter=JointObservationRuntime(opts), **kwargs)


class JointObservationTests(unittest.TestCase):
    def test_freezing_is_repeatable_rng_independent_and_labels_do_not_select_bands(self):
        data = items()
        before = _tree_fingerprint((torch.get_rng_state(), np.random.get_state()[1], random.getstate()))
        opts = frozen(data)
        self.assertEqual(before, _tree_fingerprint((torch.get_rng_state(), np.random.get_state()[1], random.getstate())))
        self.assertEqual(opts, frozen(data))
        self.assertNotEqual(opts['joint_observation']['records'], frozen(data, seed=8129)['joint_observation']['records'])
        other = copy.deepcopy(data)
        other['paviaU']['train_samples_gt'].fill_(0)
        self.assertEqual(opts['joint_observation']['records'], frozen(other)['joint_observation']['records'])
        self.assertNotEqual(opts['joint_observation']['policy_hash'], frozen(other)['joint_observation']['policy_hash'])
        self.assertEqual(schedule_for(opts), schedule_for(settings()))

    def test_real_view_synchronizes_channels_metadata_coordinates_and_preserves_graph(self):
        data = items()
        for item in data.values(): item['fwhm'] = torch.full((item['bands'],), 20.)
        opts = frozen(data); runtime = JointObservationRuntime(opts)
        for index, name in enumerate(schedule_for(opts)):
            item = data[name]; before = _tree_fingerprint(item)
            view, meta, record = runtime.view(index, name, item)
            positions = record['selected_positions']
            if index % 2 == 0: self.assertIs(view, item)
            else: self.assertLess(view['bands'], item['bands'])
            torch.testing.assert_close(view['net_input'][..., -2:], item['net_input'][..., -2:], rtol=0, atol=0)
            torch.testing.assert_close(view['net_input'][..., :-2], item['net_input'][..., positions], rtol=0, atol=0)
            self.assertEqual(tuple(view['wavelengths'].tolist()), meta.wavelengths)
            self.assertEqual(tuple(view['fwhm'].tolist()), meta.fwhm)
            self.assertEqual(meta.wavelength_quality, 'approximate')
            self.assertIs(view['Q'], item['Q']); self.assertIs(view['A'], item['A'])
            self.assertEqual(before, _tree_fingerprint(item))

    def test_reject_corruption_changed_inputs_budget_and_ambiguous_calibration_units(self):
        data = items(); opts = frozen(data)
        bad = copy.deepcopy(opts); bad['joint_observation']['records'][1]['observation']['selected_positions'].reverse()
        with self.assertRaises(ValueError): JointObservationRuntime(bad)
        bad = copy.deepcopy(opts); bad['epochs'] += 1
        with self.assertRaises(ValueError): JointObservationRuntime(bad)
        runtime = JointObservationRuntime(opts)
        altered = copy.deepcopy(data); altered['paviaU']['A'].add_(1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad'
            with self.assertRaises(ValueError): run(opts, altered, path)
            self.assertFalse(path.exists())
        with self.assertRaises(ValueError): runtime.view(-1, 'paviaU', data['paviaU'])
        for item in data.values(): item['fwhm'] = torch.full((item['bands'],), 2.)
        with self.assertRaisesRegex(ValueError, 'ambiguous'): frozen(data)
        bad = copy.deepcopy(opts)
        bad['joint_observation']['records'][0]['dataset'] = 'bad'
        bad['joint_observation']['policy_hash'] = digest({k:v for k,v in bad['joint_observation'].items() if k!='policy_hash'})
        with self.assertRaisesRegex(ValueError, 'sequence'): JointObservationRuntime(bad)

    def test_full_control_preserves_training_state_history_rng_and_checkpoint(self):
        data = items(); opts = frozen(data, 'full')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_training(calibrated_settings(settings()), data, root/'old', PROVENANCE, SOURCE, device=DEVICE)
            run(opts, data, root/'full')
            old, full = checkpoints(root/'old'), checkpoints(root/'full')
            for key in ('model', 'optimizer', 'scheduler', 'history', 'rng', 'best_probabilities'):
                self.assertEqual(_tree_fingerprint(old[key]), _tree_fingerprint(full[key]), key)
            cleaned = [{k:v for k,v in row.items() if k!='observation'} for row in full['trace']]
            self.assertEqual(old['trace'], cleaned)

    def test_interrupted_resume_matches_uninterrupted_and_binds_policy(self):
        data = items(); opts = frozen(data)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run(opts, data, root/'whole')
            result = run(opts, data, root/'split', stop_after=5)
            self.assertEqual(result['updates_this_call'], 5)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                run(frozen(data, seed=8129), data, root/'split', resume=True)
            result = run(opts, data, root/'split', resume=True)
            self.assertEqual(result['updates_this_call'], 7)
            a, b = checkpoints(root/'whole'), checkpoints(root/'split')
            for key in ('model', 'optimizer', 'scheduler', 'history', 'rng', 'trace', 'best_probabilities'):
                self.assertEqual(_tree_fingerprint(a[key]), _tree_fingerprint(b[key]), key)
            before = hashes(root/'split')
            self.assertEqual(run(opts, data, root/'split', resume=True)['updates_this_call'], 0)
            self.assertEqual(before, hashes(root/'split'))
            runtime = JointObservationRuntime(frozen(data, 'full'))
            step = result['result']['selected_step']
            with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                runtime.load_selected(root/'split'/'primary.pt', data, root/'split'/f'raw-validation-{step:06d}.pt', device=DEVICE)
            # Participation plus difference from the otherwise identical full control.
            run(frozen(data, 'full'), data, root/'full')
            self.assertNotEqual(_tree_fingerprint(a['model']), _tree_fingerprint(checkpoints(root/'full')['model']))
            self.assertTrue(all(row['gradient_norms']['adapter'] > 0 for row in a['trace']))
            self.assertTrue(any(row['observation']['bands'] < row['observation']['source_bands'] for row in a['trace']))


if __name__ == '__main__': unittest.main()
