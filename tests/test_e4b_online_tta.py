import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from evaluate_tta import HEADS, MODES, run_online
from model.checkpoint_io import save_model_checkpoint
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from research_runtime import hash_file, tensor_hash, write_json


def model(norm='domain_bn'):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(22)
        return MultiDatasetVCASAHGCN({'shared': 2, 'other': 2}, graph_mode='learned', norm_type=norm).eval()


def frame(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:8, :8]
    return dict(spectral=rng.random((8, 8, 6), dtype=np.float32) + np.float32(seed / 10),
                coordinates=np.stack((yy/7, xx/7), axis=-1).astype(np.float32),
                wavelengths_nm=np.linspace(440, 830, 6, dtype=np.float32))


def execute(source, mode, reset, ordered, callback=None):
    predictions = []
    def sink(scene, frame_id, outputs):
        predictions.append(outputs)
        return None if callback is None else callback(scene, frame_id, outputs)
    stream = [(scene, str(i), payload) for i, (scene, payload) in enumerate(ordered)]
    result = run_online(source, stream, task='shared', mode=mode, reset=reset, scale=4,
                        lr=.05, rank=4, seed=21, on_prediction=sink)
    return result, predictions


class OnlineTTATests(unittest.TestCase):
    def test_all_four_paths_change_only_their_allowlist_and_first_predictions_match(self):
        source = model()
        before = {key: tensor_hash(value) for key, value in source.state_dict().items()}
        data = [('a', frame(1)), ('a', frame(2))]
        frozen, predictions = execute(source, 'frozen', 'continual', data)
        for mode in MODES:
            result, outputs = execute(source, mode, 'continual', data)
            self.assertTrue(all(np.array_equal(a,b) for a,b in zip(outputs[0], predictions[0])))
            self.assertEqual(result['optimizer_updates'], 2 if mode in ('affine','adapter') else 0)
            self.assertEqual(sum(row['statistics_updates'] for row in result['events']), 2 if mode == 'statistics' else 0)
            changes = {key for row in result['events'] for key in row['changed']}
            self.assertEqual(bool(changes), mode != 'frozen')
            self.assertTrue(changes <= set(result['allowed_state']))
            self.assertFalse(any('norms.other' in key or key.startswith('heads.') for key in changes))
            if mode == 'statistics':
                self.assertTrue(all(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in changes))
            if mode == 'adapter':
                self.assertTrue(all(k.startswith('adapter.') for k in changes))
                self.assertTrue(all(np.array_equal(a,b) for a,b in zip(outputs[1][1:], predictions[1][1:])))
        self.assertEqual(before, {key: tensor_hash(value) for key, value in source.state_dict().items()})

    def test_episodic_restores_and_continual_preserves_order_effect(self):
        source = model()
        a, b = frame(1), frame(3)
        for mode in ('statistics','affine','adapter'):
            episodic, p = execute(source, mode, 'episodic', [('a',a),('b',b),('a',a)])
            self.assertEqual([row['reset'] for row in episodic['events']], [True,True,True])
            self.assertTrue(all(np.array_equal(x,y) for x,y in zip(p[0], p[2])))
            continual, p = execute(source, mode, 'continual', [('a',a),('b',b),('a',a)])
            self.assertEqual([row['reset'] for row in continual['events']], [True,False,False])
            self.assertFalse(np.array_equal(p[0][0], p[2][0]))
            _, reversed_predictions = execute(source, mode, 'continual', [('b',b),('a',a)])
            self.assertFalse(np.array_equal(p[0][0], reversed_predictions[1][0]))

    def test_scoring_is_pre_update_and_cannot_influence_update(self):
        source = model()
        events = []
        data = [('a',frame(1)),('a',frame(2))]
        _, expected = execute(source, 'affine', 'continual', data)
        def adversarial_score(scene, frame_id, probabilities):
            events.append(frame_id)
            for value, original in zip(probabilities, expected[int(frame_id)]):
                np.testing.assert_array_equal(value, original)
                value.fill(-1)
            return {'score': int(frame_id)}
        result, _ = execute(source, 'affine', 'continual', data, adversarial_score)
        self.assertEqual(events, ['0','1'])
        self.assertEqual([row['prediction']['score'] for row in result['events']], [0,1])

    def test_incompatible_statistics_and_label_injection_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no supported mutable state'):
            execute(model('group_norm'), 'statistics', 'continual', [('a',frame(1))])
        calls = []
        with self.assertRaisesRegex(ValueError, 'accepts only'):
            execute(model(), 'affine', 'episodic', [('a', {**frame(1), 'labels': np.ones(64)})],
                    lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_cli_freezes_prediction_files_and_rejects_label_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/'model.pt'
            save_model_checkpoint(model(), checkpoint)
            obs = root/'frame.npz'
            np.savez(obs, **frame(2))
            registration = root/'registration.json'
            record = dict(policy='e4b_online_predict_then_adapt_cpu_v1', checkpoint=str(checkpoint),
                          checkpoint_sha256=hash_file(checkpoint),
                          sidecar_sha256=hash_file(checkpoint.with_suffix('.pt.json')),
                          task='shared', mode='adapter', reset='continual', scale=4, lr=.05, rank=4, seed=21,
                          stream=[dict(scene='a', frame_id='x', path=str(obs), sha256=hash_file(obs))])
            commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
            def command(out):
                write_json(registration, record)
                return [sys.executable,'-m','evaluate_tta','--registration',str(registration),
                        '--registration-sha256',hash_file(registration),'--expected-commit',commit,'--out',str(out)]
            output = root/'accepted'
            subprocess.run(command(output), check=True)
            result = json.loads((output/'receipt.json').read_text())
            self.assertFalse(result['target_labels_accessed'])
            self.assertEqual(result['optimizer_updates'], 1)
            artifact = result['events'][0]['prediction']
            self.assertEqual(hash_file(output/artifact['path']), artifact['sha256'])
            with np.load(output/artifact['path'], allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(HEADS))
            np.savez(obs, **frame(2), labels=np.ones(64))
            record['stream'][0]['sha256'] = hash_file(obs)
            rejected = root/'rejected'
            run = subprocess.run(command(rejected), capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('unexpected fields', run.stderr)
            self.assertEqual(list(rejected.iterdir()), [])
