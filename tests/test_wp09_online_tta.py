"""Registered online adaptation, legacy oracle, crash custody and offline scores."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluate_tta import HEADS, MODES, _Session, run_online
from evaluation.online_tta import score
from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from research_runtime import code_snapshot, hash_file
from test_e4b_online_tta import frame
from transfer.online_tta_replay import replay
from transfer.online_tta_run import (_rng_hash, _rng, committed_history, execute, load_checkpoint,
                                     read_predictions, runtime_identity)
from transfer.online_tta_stream import VERSION, ORDERS, StreamRegistration, frame_budget, parse_json


def pin(path):
    return dict(path=str(path), sha256=hash_file(path))


def write(path, payload):
    path.write_text(json.dumps(payload))
    return pin(path)


def fixture(root, classes, norm='domain_bn'):
    root.mkdir()
    source = root / 'source.pt'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        model = MultiDatasetVCASAHGCN({'shared': classes, 'other': 2}, graph_mode='learned',
                                      norm_type=norm).eval()
    save_model_checkpoint(model, source)
    checkpoint = dict(weights=pin(source), sidecar=pin(source.with_suffix('.pt.json')))
    payloads = dict(A=frame(1), B=frame(3))
    observations = {}
    for name, arrays in payloads.items():
        path = root / f'{name}.npz'
        np.savez(path, **arrays)
        observations[name] = pin(path)
    bands = BandMetadata('synthetic', tuple(range(6)), tuple(map(float, payloads['A']['wavelengths_nm'])),
                         'nm', 'approximate', (True,) * 6)
    snapshot = code_snapshot()
    class_ids = list(range(1, classes + 1))
    provenance = write(root / 'provenance.json', dict(kind='synthetic', source=checkpoint,
        producer_commit=snapshot['commit'], exporter_commit=snapshot['commit'], origin_hash='a' * 64,
        selected_step=0, class_ids=class_ids, bands_hash=bands.fingerprint,
        observations=observations, baseline_hashes={}))
    frames = [dict(frame_id=f'{obs}{i}', scene='synthetic', segment=obs, observation=obs)
              for i, obs in enumerate('AABBBA', 1)]
    reg = dict(version=VERSION, commit=snapshot['commit'], source_hash=snapshot['source_hash'],
               runtime=runtime_identity(), source=checkpoint, provenance=provenance, class_ids=class_ids,
               bands=asdict(bands), observations=observations, frames=frames, order='original',
               settings=dict(task='shared', mode='frozen', reset='continual', scale=4, lr=.05, rank=4, seed=21),
               budget=frame_budget('frozen', len(frames)))
    return model, payloads, reg


def registration(root, spec, mode, reset='continual', order='original'):
    settings = dict(spec['settings'], mode=mode, reset=reset)
    return write(root / f'{mode}-{reset}-{order}.json',
                 dict(spec, settings=settings, order=order, budget=frame_budget(mode, len(spec['frames']))))


def child(reg, out, stop=None):
    command = [sys.executable, '-m', 'run_online_tta', '--registration', reg['path'],
               '--registration-sha256', reg['sha256'], '--out', str(out)]
    if stop is not None:
        command += ['--to-frame', str(stop)]
    output = subprocess.run(command, capture_output=True, text=True)
    if output.returncode:
        raise AssertionError(output.stderr)
    return json.loads(output.stdout)


class PersistentOnlineTTATests(unittest.TestCase):
    def test_four_modes_legacy_oracle_and_multiboundary_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            for classes in (9, 22):
                root = Path(tmp) / str(classes)
                model, payloads, spec = fixture(root, classes)
                for mode in MODES:
                    for reset in ('episodic', 'continual'):
                        with self.subTest(classes=classes, mode=mode, reset=reset):
                            reg_pin = registration(root, spec, mode, reset)
                            reg = StreamRegistration(reg_pin)
                            stem = f'{mode}-{reset}'
                            full, split = root / (stem + '-full'), root / (stem + '-split')
                            rng_before = _rng_hash(_rng())
                            execute(reg_pin, full)
                            if classes == 9 and reset == 'episodic':
                                for stop in (1, 2, 4, 6):
                                    child(reg_pin, split, stop)
                            else:
                                for stop in (1, 2, 4, 6):
                                    execute(reg_pin, split, to_frame=stop)
                            with patch.object(_Session, 'forward', side_effect=AssertionError('completed forward')):
                                self.assertEqual(execute(reg_pin, split)['actual'], frame_budget(mode, 0))
                            self.assertEqual(_rng_hash(_rng()), rng_before)
                            full_rows, full_tip = committed_history(reg, full, complete=True)
                            split_rows, split_tip = committed_history(reg, split, complete=True)
                            self.assertEqual(full_rows, split_rows)
                            self.assertEqual(full_tip, split_tip)
                            predicted = {}

                            def oracle_sink(_, frame_id, probs):
                                predicted[frame_id] = dict(zip(HEADS, probs))

                            stream = [(str(f['episode']), f['frame_id'], payloads[f['observation']])
                                      for f in reg.frames]
                            oracle = run_online(model, stream, on_prediction=oracle_sink, **reg.settings)
                            for i, (event, expected) in enumerate(zip(full_rows, oracle['events']), start=1):
                                packets = read_predictions(full / f'frame-{i:06d}' / 'prediction.npz')
                                self.assertEqual(reg.fingerprints(packets),
                                                 reg.fingerprints(predicted[expected['frame_id']]))
                                for key in ('reset', 'entropy', 'changed'):
                                    self.assertEqual(event[key], expected[key])
                                self.assertFalse(any('norms.other' in k or not k.startswith(('backbone.', 'adapter.'))
                                                     for k in event['changed']))
                            self.assertEqual([r['frame']['episode'] for r in full_rows], [1, 1, 2, 2, 2, 3])
                            if mode == 'adapter':
                                for i in (1, 2):
                                    path = full / f'frame-{i:06d}' / 'gradients.pt'
                                    grads = load_checkpoint(path, full_rows[i - 1]['gradients'])
                                    self.assertEqual(bool(grads['adapter.0.weight'].count_nonzero()), i == 2)
                                    self.assertGreater(int(grads['adapter.2.weight'].count_nonzero()), 0)
                                    self.assertEqual('adapter.0.weight' in full_rows[i - 1]['changed'], i == 2)

    def test_faults_after_prediction_and_after_atomic_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture'
            _, _, spec = fixture(root, 9)
            reg = registration(root, spec, 'affine')
            out = root / 'after-prediction'

            def fail_update(session, _):
                self.assertTrue((out / '.pending-000001' / 'prediction.npz').is_file())
                raise RuntimeError('injected before first SGD')

            with patch.object(_Session, 'adapt', fail_update), self.assertRaisesRegex(RuntimeError, 'injected'):
                execute(reg, out)
            attempt = parse_json((out / '.pending-000001' / 'attempt.json').read_bytes())
            self.assertEqual(attempt['completed']['predictions'], 1)
            self.assertEqual(attempt['started']['optimizer_updates'], 1)
            self.assertEqual(attempt['completed']['optimizer_updates'], 0)
            with self.assertRaisesRegex(RuntimeError, 'cannot be resumed'):
                execute(reg, out)
            self.assertFalse((out / 'frame-000001').exists())

            from transfer import online_tta_run as runner
            out = root / 'after-update'
            write_payload = runner._write

            def lose_checkpoint(path, writer):
                if path.name == 'checkpoint.pt':
                    raise RuntimeError('injected after first SGD')
                write_payload(path, writer)

            with patch.object(runner, '_write', lose_checkpoint), self.assertRaisesRegex(RuntimeError, 'injected'):
                execute(reg, out)
            attempt = parse_json((out / '.pending-000001' / 'attempt.json').read_bytes())
            self.assertEqual(attempt['completed']['optimizer_updates'], 1)
            self.assertFalse((out / 'frame-000001').exists())
            with self.assertRaisesRegex(RuntimeError, 'cannot be resumed'):
                execute(reg, out)

            out = root / 'after-commit'
            real = runner._journal

            def lose_progress(path, value):
                if path.name == 'progress.json' and value['cursor'] == 1 and not value['in_flight']:
                    raise RuntimeError('injected after rename')
                real(path, value)

            with patch.object(runner, '_journal', lose_progress), self.assertRaisesRegex(RuntimeError, 'injected'):
                execute(reg, out)
            before = hash_file(out / 'frame-000001' / 'event.json')
            self.assertEqual(execute(reg, out)['added'], 5)
            self.assertEqual(hash_file(out / 'frame-000001' / 'event.json'), before)
            self.assertEqual(execute(reg, out)['added'], 0)

    def test_registration_and_history_reject_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture'
            _, _, spec = fixture(root, 9)
            reg = registration(root, spec, 'frozen')
            bad_path = root / 'bad.json'
            edits = [
                dict(spec, labels='/external/labels'),
                dict(spec, frames=[*spec['frames'], spec['frames'][0]]),
                dict(spec, frames=[dict(spec['frames'][0], target=1), *spec['frames'][1:]]),
                dict(spec, budget=frame_budget('affine', 6)),
                dict(spec, class_ids=list(reversed(spec['class_ids']))),
                dict(spec, bands=dict(spec['bands'], wavelength_quality='verified')),
                dict(spec, source=dict(spec['source'], weights=dict(spec['source']['weights'], sha256='0' * 64))),
                dict(spec, runtime=dict(spec['runtime'], threads=999)),
                dict(spec, settings=dict(spec['settings'], task='other')),
            ]
            for edit in edits:
                with self.subTest(fields=edit.keys()), self.assertRaises(ValueError):
                    execute(write(bad_path, edit), root / 'bad')
            bad_path.write_text(Path(reg['path']).read_text().replace('"version":', '"version": 1, "version":', 1))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                StreamRegistration(pin(bad_path))
            out = root / 'run'
            execute(reg, out, to_frame=2)
            for changed in (registration(root, spec, 'affine'), registration(root, spec, 'frozen', order='reverse')):
                with self.assertRaisesRegex(ValueError, 'different registration'):
                    execute(changed, out)
            packet = out / 'frame-000001' / 'prediction.npz'
            raw = packet.read_bytes()
            packet.write_bytes(raw + b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                execute(reg, out)
            packet.write_bytes(raw)
            shutil.rmtree(out / 'frame-000002')
            with self.assertRaisesRegex(ValueError, 'Progress marker'):
                execute(reg, out)

    def test_observation_schema_and_groupnorm_statistics_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture'
            _, payloads, spec = fixture(root, 9, norm='group_norm')
            with self.assertRaisesRegex(ValueError, 'no supported mutable state'):
                execute(registration(root, spec, 'statistics'), root / 'gn')
            for name, data in (('labels', dict(payloads['A'], labels=np.zeros((8, 8)))),
                               ('physical', dict(payloads['A'], wavelengths_nm=payloads['A']['wavelengths_nm'] + 1))):
                path = root / f'{name}.npz'
                np.savez(path, **data)
                spec['observations']['A'] = pin(path)
                provenance = parse_json(Path(spec['provenance']['path']).read_bytes())
                spec['provenance'] = write(root / 'provenance.json', dict(provenance, observations=spec['observations']))
                reg = registration(root, spec, 'frozen')
                with self.assertRaisesRegex(ValueError, 'observation arrays|wavelength identity'):
                    execute(reg, root / name)

    def test_offline_order_pairing_and_zero_update_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture'
            _, _, spec = fixture(root, 9)
            runs, labels = [], []
            for mode in ('frozen', 'adapter'):
                for order in ORDERS:
                    tag = f'{mode}-{order}'
                    reg_pin = registration(root, spec, mode, order=order)
                    out = root / tag
                    execute(reg_pin, out)
                    replay_out = root / f'replay-{tag}'
                    with patch.object(_Session, 'adapt', side_effect=AssertionError('replay update')):
                        replay(reg_pin, out, replay_out)
                    runs.append(dict(registration=reg_pin, directory=str(out), replay=pin(replay_out / 'replay.json')))
            reg = StreamRegistration(reg_pin)
            for index, item in enumerate(reversed(reg.frames)):
                path = root / f'truth-{index}.npy'
                data = np.zeros((8, 8), dtype=np.int64)
                data[:4, :4] = 1 + int(item['frame_id'][1:]) % 3
                np.save(path, data, allow_pickle=False)
                labels.append(dict(frame_id=item['frame_id'], observation_sha256=reg.observation_identity(item),
                                   labels=pin(path)))
            release = write(root / 'labels.json', dict(kind='synthetic', population=reg.population,
                class_ids=spec['class_ids'], release_receipt_sha256='b' * 64,
                training_class_counts=[30, 12, 6, 1, 1, 1, 1, 1, 1], frames=labels))
            plan = write(root / 'plan.json', dict(version=1, label_release=release, runs=runs))
            result = score(plan)
            summary = root / 'scored.json'
            subprocess.run([sys.executable, '-m', 'evaluation.online_tta', '--plan', plan['path'],
                            '--plan-sha256', plan['sha256'], '--out', str(summary)], check=True)
            self.assertEqual(result, parse_json(summary.read_bytes()))
            online = result['online_pre_update']
            self.assertEqual(len(online), 6)
            self.assertEqual(online['frozen/continual/reverse']['versus_original_order']['fused']['metrics']['nll'], 0)
            for value in online.values():
                for metrics in value['metrics'].values():
                    self.assertEqual(metrics['count'], 96)
                    self.assertEqual(metrics['missing_class_ids'], [4, 5, 6, 7, 8, 9])
                    self.assertIsNotNone(metrics['nll'])
                self.assertIsNone(value['versus_frozen']['cnn']['small_class_protected_at_2pp']['9'])
            forgetting = result['zero_update_anchor_replay']['adapter/continual/original']
            self.assertEqual(forgetting['initial']['fused']['count'], 32)
            self.assertNotEqual(forgetting['final_minus_initial']['fused']['metrics']['nll'], 0)
            (Path(runs[-1]['directory']) / 'completion.json').unlink()
            Path(release['path']).unlink()
            with self.assertRaisesRegex(ValueError, 'before scoring'):
                score(plan)


if __name__ == '__main__':
    unittest.main()
