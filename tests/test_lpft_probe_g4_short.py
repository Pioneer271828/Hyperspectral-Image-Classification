"""Bounded real-model integration on synthetic source/context artifacts."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lpft_probe import LPFTPlan
from lpft_probe.g4_short import G4ShortCase
from joint_training import _tree_fingerprint
from loadData.dataset_registry import file_hash
from test_g4_light_full_target import registration_fixture


class G4ShortTests(unittest.TestCase):
    def test_full_source_loss_calibration_endpoint_and_interior_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            _, registration, code = registration_fixture(Path(directory))
            ref = registration['reference_launch']
            with patch('lpft_probe.g4_short.REFERENCE_LAUNCH_SHA256', ref['sha256']):
                case = G4ShortCase(ref['path'], ref['sha256'], expected_commit=code['commit'],
                                   target='xuzhou', seed=64, mode='none',
                                   plan=LPFTPlan(6, 2, .001, .0001))
            before = case.endpoint_metrics()
            self.assertEqual(before['completed_steps'], 0)
            self.assertEqual(set(before['metrics']), {'train', 'val'})
            self.assertFalse(before['case']['query_labels_accessed'])
            self.assertEqual(case.step()['update_phase'], 'probe')
            self.assertEqual(case.step()['update_phase'], 'probe')
            self.assertEqual(case.step()['update_phase'], 'finetune')
            saved = case.snapshot()
            rows = [case.step() for _ in range(3)]
            final = _tree_fingerprint(case.snapshot())
            metrics = case.endpoint_metrics()
            self.assertEqual(metrics['completed_steps'], 6)
            self.assertEqual(_tree_fingerprint(case.snapshot()), final)
            self.assertNotEqual(before['model_state_sha256'], metrics['model_state_sha256'])
            self.assertFalse(metrics['accuracy_gain_established'])
            case.restore_snapshot(saved)
            self.assertEqual(rows, [case.step() for _ in range(3)])
            self.assertEqual(final, _tree_fingerprint(case.snapshot()))
            self.assertEqual(metrics, case.endpoint_metrics())
            case.guard()

    def test_short_budget_wrong_launch_and_changed_input_rejected(self):
        with self.assertRaisesRegex(ValueError, 'twelve'):
            G4ShortCase('/unused', 'a' * 64, expected_commit='unused', target='xuzhou',
                        seed=64, mode='none', plan=LPFTPlan(1500, 100, .001, .0001))
        with tempfile.TemporaryDirectory() as directory:
            _, registration, code = registration_fixture(Path(directory))
            ref = registration['reference_launch']
            args = dict(expected_commit=code['commit'], target='xuzhou', seed=64,
                        mode='none', plan=LPFTPlan(6, 2, .001, .0001))
            with patch('lpft_probe.g4_short.REFERENCE_LAUNCH_SHA256', ref['sha256']):
                with self.assertRaises(ValueError):
                    G4ShortCase(ref['path'], '0' * 64, **args)
                case = G4ShortCase(ref['path'], ref['sha256'], **args)
                saved = case.snapshot()
                wrong = copy.deepcopy(saved)
                wrong['case']['mode'] = 'x_residual'
                with self.assertRaisesRegex(ValueError, 'another run'):
                    case.restore_snapshot(wrong)
                self.assertEqual(_tree_fingerprint(saved), _tree_fingerprint(case.snapshot()))
                with Path(ref['path']).open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaises(ValueError):
                    case.step()
                self.assertEqual(case.run.completed_steps, 0)
                self.assertNotEqual(file_hash(ref['path']), ref['sha256'])


if __name__ == '__main__':
    unittest.main()
