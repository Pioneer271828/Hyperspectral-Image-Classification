import unittest
from dataclasses import replace

import numpy as np
import torch

from loadData.data_read import read_band_metadata
from model.research_config import digest
from transfer.g3_training import Scene
from transfer.p4_frozen_band_probe import compare, freeze_conditions, selected_view


class FrozenBandProbeTests(unittest.TestCase):
    def setUp(self):
        shape = (4, 8)
        self.tile = {'id': 1, 'partition': 'val', 'box': [0, 4, 0, 4]}
        cube = np.arange(4 * 8 * 270, dtype=np.float32).reshape(4, 8, 270)
        stats = {'plan_hash': 'guarded', 'fit_partition': 'train',
                 'minimum': [0.] * 270, 'maximum': [1.] * 270}
        stats['statistics_hash'] = digest(stats)
        self.scene = Scene('whu-honghu', cube, {'shape': shape, 'plan_hash': 'guarded'},
                           read_band_metadata('whu-honghu', 270), 22, stats)
        self.conditions = freeze_conditions(self.scene)

    def test_retained_columns_only_and_reject_unreleased_tile(self):
        stride = self.conditions['stride']
        allowed = {1: self.tile}
        original = selected_view(self.scene, self.tile, allowed, stride)
        poisoned = self.scene.cube.copy()
        poisoned[:, :, 1::2] = np.nan
        poisoned[:, 4:] = np.nan
        scene = replace(self.scene, cube=poisoned)
        probed = selected_view(scene, self.tile, allowed, stride)
        np.testing.assert_array_equal(probed.observation.spectral, original.observation.spectral)
        np.testing.assert_array_equal(probed.observation.global_ids, original.observation.global_ids)
        np.testing.assert_array_equal(original.observation.spectral, self.scene.cube[:, :4, ::2])
        with self.assertRaisesRegex(ValueError, 'Nonfinite retained'):
            selected_view(scene, self.tile, allowed, self.conditions['full'])
        with self.assertRaisesRegex(ValueError, 'released validation'):
            selected_view(scene, dict(self.tile, box=[0, 4, 4, 8]), allowed, stride)

    def test_metadata_only_conditions_keep_every_observed_column(self):
        self.assertEqual(freeze_conditions(self.scene), self.conditions)
        views = {name: selected_view(self.scene, self.tile, {1: self.tile}, self.conditions[name])
                 for name in ('stride', 'stride_missing', 'stride_shift10nm')}
        for name in ('stride_missing', 'stride_shift10nm'):
            self.assertEqual(self.conditions[name]['reference'], 'stride')
            np.testing.assert_array_equal(views[name].observation.spectral, views['stride'].observation.spectral)
            self.assertEqual(views[name].bands.band_ids, views['stride'].bands.band_ids)
        np.testing.assert_allclose(np.array(views['stride_shift10nm'].bands.wavelengths)
                                   - np.array(views['stride'].bands.wavelengths), 10.)

    def test_predictions_and_class_changes_share_reference_labels(self):
        target = torch.tensor([0, 1, 1])
        reference = torch.tensor([[.8, .2], [.1, .9], [.6, .4]])
        actual = torch.tensor([[.6, .4], [.2, .8], [.4, .6]])
        record = compare(target, actual, reference, 2)
        self.assertEqual(record['confusion_matrix'], [[1, 0], [0, 2]])
        self.assertEqual(record['changed_by_class'], [0, 1])
        self.assertEqual(record['changed_predictions'], 1)
