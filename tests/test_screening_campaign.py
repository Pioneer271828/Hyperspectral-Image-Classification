"""Paired initialization, observational progress and campaign identity protection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.checkpoint_io import load_model_checkpoint
from research_runtime import code_snapshot, tensor_hash
from scripts.run_m6_screening import initial_state, paired_initializer, state_digest, prepare, campaign_lock
from train_spatial import SpatialConfig, execute

torch.set_num_threads(1)


def fixture(mode='none'):
    config = SpatialConfig({'name': 'paired_fixture_' + mode, 'dataset': 'paviaU',
        'model': {'graph_mode': 'learned', 'latent_channels': 8, 'adapter_embed_dim': 8, 'norm_type': 'group_norm'},
        'interaction': {'interaction_mode': mode},
        'training': {'epochs': 0, 'steps_per_epoch': 2, 'primary_checkpoint': 'val_loss'}, 'tile_size': 12, 'gap': 1})
    cube = np.random.default_rng(9).normal(size=(60, 60, 6)).astype(np.float32)
    labels = np.tile(np.arange(1, 10), 400).reshape(60, 60)
    return config, cube, labels


class ScreeningCampaignTests(unittest.TestCase):
    def test_common_state_all_modes_and_training_rng_are_identical(self):
        base, _, _ = fixture()
        before = torch.get_rng_state().clone()
        state = initial_state(base)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        expected = state_digest(state)
        samples = []
        for mode in ('none', 'add', 'gated'):
            config, _, _ = fixture(mode)
            torch.manual_seed(9)
            model = config.research.build_model({'paviaU': 9})
            info = paired_initializer(state, expected, 2026)(model)
            self.assertEqual(info['state_digest'], expected)
            self.assertEqual(state_digest({key: model.state_dict()[key] for key in state}), expected)
            samples.append(torch.rand(8))
        for value in samples[1:]: self.assertTrue(torch.equal(value, samples[0]))
        broken = {key: value.clone() for key, value in state.items()}
        first = next(iter(broken)); broken[first].add_(1)
        with self.assertRaisesRegex(ValueError, 'changed'):
            paired_initializer(broken, expected, 2026)

    def test_progress_callback_does_not_change_weights_or_selection(self):
        config, cube, labels = fixture()
        code = code_snapshot(); events = []; results = []
        loader = lambda: (cube, labels, {'cube': tensor_hash(cube)}, None)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for root, observer in ((first, None), (second, events.append)):
                result = execute(config, root, code, loader, progress_callback=observer)
                folder = Path(root) / result['run_id'] / 'attempts' / result['attempt']
                results.append((torch.load(folder / 'model.pt', weights_only=True), json.loads((folder / 'metrics.json').read_text())))
            for key, value in results[0][0].items(): self.assertTrue(torch.equal(value, results[1][0][key]), key)
            self.assertEqual(results[0][1], results[1][1])
        self.assertEqual([event['phase'] for event in events], ['train', 'train', 'validation'])
        self.assertEqual([event['optimizer_steps'] for event in events], [1, 2, 2])
        self.assertNotIn('test', events[-1])

    def test_initializer_provenance_restored_and_success_skips_callback(self):
        config, cube, labels = fixture('gated')
        state = initial_state(config); expected = state_digest(state)
        loader = lambda: (cube, labels, {}, None)
        code = code_snapshot()
        with tempfile.TemporaryDirectory() as root:
            result = execute(config, root, code, loader, model_initializer=paired_initializer(state, expected, 2026))
            folder = Path(root) / result['run_id'] / 'attempts' / result['attempt']
            self.assertEqual(result['initialization']['state_digest'], expected)
            restored = load_model_checkpoint(folder / 'model.pt')
            self.assertEqual(restored.training_config['initialization'], result['initialization'])
            with patch('torch.optim.Adam', side_effect=AssertionError('Must not retrain')):
                skipped = execute(config, root, code, loader, resume=True,
                    model_initializer=lambda model: self.fail('Must skip initialization'))
            self.assertEqual(skipped['status'], 'skipped')

    def test_prepare_is_reproducible_rejects_changes_and_lock_is_exclusive(self):
        code = code_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, state, expected = prepare(root, code)
            _, other, again = prepare(root, code)
            self.assertEqual(expected, again); self.assertEqual(state_digest(state), state_digest(other))
            with campaign_lock(root):
                with self.assertRaises(FileExistsError):
                    with campaign_lock(root): self.fail('Acquired a duplicate campaign lock')
            self.assertFalse((root / '.campaign.lock').exists())
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                prepare(root, {**code, 'commit': '0' * 40})
            (root / 'shared_initial_state.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                prepare(root, code)


if __name__ == '__main__': unittest.main()
