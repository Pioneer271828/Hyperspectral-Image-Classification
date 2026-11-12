"""Seven matched precision arms, strict resume and packed endpoints on 9/22 classes."""
from pathlib import Path
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compression.int8 import CalibratedMatrix
from compression.int8_model import calibration_contract
from compression.precision_fallback import LAYER_ORDER
from compression.precision_recovery import ARMS, PrecisionRecoveryRun, frozen_buffers, new_run
from compression.precision_recovery_audit import audit, select_families
from compression.precision_recovery_cost import cost
from compression.registered_precision_recovery import complete, execute, initial_model, load_registration, register
from loadData.observation_sampling import BandMetadata
from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan, tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from research_runtime import ROOT, code_snapshot
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.development_training import prepare_scene


def fixture(path, classes):
    """One available support block so every four-step exposure visits every support point."""
    path.mkdir()
    plan = make_tile_plan((84, 84), 28, 1, 2026)
    labels = np.zeros((84, 84), np.int64)
    tiles = [t for t in plan['tiles'] if t['partition'] == 'train'][:2]
    for tile in tiles:
        view = labels[tile_slice(tile)]
        view[...] = np.resize(np.arange(1, classes + 1), view.shape)
    record = make_development(labels, plan, list(range(1, classes + 1)))
    waves = tuple(np.linspace(460, 850, 8))
    bands = BandMetadata('synthetic', tuple(range(8)), waves, 'nm', 'approximate', (True,) * 8)
    cube = np.random.default_rng(22).normal(size=(84, 84, 8)).astype(np.float32)
    scene = prepare_scene(cube, bands, plan, record)
    torch.manual_seed(64)
    model = MultiDatasetVCASAHGCN({scene.dataset: classes}, latent_channels=8, adapter_embed_dim=8,
                                 locality_window=None, graph_mode='learned', norm_type='group_norm').eval()
    model.backbone.adapter.cache_attention = False
    weights = path / 'source.pt'
    save_model_checkpoint(model, weights)
    development = dev.export_development(scene, record, path / 'development')
    return dev._json_new(path / 'request.json', dict(scope='synthetic', development=development,
        weights=dev._pin(weights), sidecar=dev._pin(str(weights) + '.json'), budget=4, prefix=2))


def _cli(code, action, pin, *args, log=None):
    cmd = [sys.executable, str(ROOT / 'scripts' / 'run_precision_recovery.py'), '--expected-commit', code['commit'],
           action, '--registration', pin['path'], pin['sha256'], *args]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if log is not None:
        log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class PrecisionRecoveryTests(unittest.TestCase):
    def test_matched_9_and_22_class_cli_resume_and_packages(self):
        evidence = os.environ.get('WP10_SYNTHETIC_EVIDENCE')
        if evidence:
            path = Path(evidence)
            path.mkdir(parents=True, exist_ok=False)
            self._matrix(path)
        else:
            with TemporaryDirectory() as name:
                self._matrix(Path(name))

    def _matrix(self, path):
        code = code_snapshot()
        for classes in (9, 22):
            with self.subTest(classes=classes):
                req = fixture(path / f'inputs-{classes}', classes)
                out = path / f'run-{classes}'
                pin = register(req, out, code)
                value = admission.read(pin)
                notice = dev._json_new(out / 'notice.json', dict(registration=pin, maximum_updates=56, cpu_workers=1))
                self.assertEqual(value['validation_events'], [1, 2, 3, 4])
                self.assertEqual(len(value['calibration_tiles']), 1)
                dense = value['initial']['fp32']['evidence']['dense_tensor_hashes']
                for row in value['initial'].values():
                    self.assertEqual(row['evidence']['dense_tensor_hashes'], dense)
                for arm in ARMS:
                    full = execute(pin, code, notice, arm, 'full', None)
                    if classes == 9:
                        args = ['--notice', notice['path'], notice['sha256'], '--arm', arm]
                        prefix = _cli(code, 'stage', pin, *args, '--stage', 'prefix',
                                      log=out / f'{arm}-prefix.log')
                        resumed = _cli(code, 'stage', pin, *args, '--stage', 'resume', '--prior',
                                       prefix['path'], prefix['sha256'], log=out / f'{arm}-resume.log')
                    else:
                        prefix = execute(pin, code, notice, arm, 'prefix', None)
                        with patch.object(CalibratedMatrix, 'observe', side_effect=AssertionError('recalibration on resume')):
                            resumed = execute(pin, code, notice, arm, 'resume', prefix)
                    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('completed update')):
                        end = execute(pin, code, notice, arm, 'completed', resumed)
                    self.assertEqual([admission.read(item)['new_updates'] for item in (full, prefix, resumed, end)],
                                     [4, 2, 2, 0])
                completed = complete(pin, code)
                audit_pin = audit(pin, completed, code)
                result = admission.read(audit_pin)
                self.assertEqual(result['actual_updates'], 56)
                self.assertEqual(result['independent_updates'], 0)
                self.assertEqual(len(result['report']), 7)
                for arm, row in result['report'].items():
                    self.assertEqual(len(row['history']), 4)
                    self.assertEqual(row['updates'], 8)
                    self.assertEqual(row['selected']['metrics']['fused']['count'], classes * 5)
                    self.assertEqual(row['selected']['package']['format'],
                                     {'fp32': 'fp32', 'qat': 'qat_integer', 'ternary': 'ternary_packed'}[ARMS[arm][0]])
                    for layer in ARMS[arm][1]:
                        self.assertGreater(row['parameter_activity'][layer + '.layer.weight']['grad_steps'], 0)
                        self.assertGreater(row['parameter_activity'][layer + '.layer.weight']['changed_steps'], 0)
                        self.assertEqual(row['deploy_kernel_calls'][layer], 4)
                if classes == 9:
                    measured = admission.read(cost(pin, audit_pin, code))
                    self.assertEqual(sum(len(rows) for rows in measured['workers'].values()), 9)
                    self.assertEqual(measured['actual_kernel_evidence']['ptq']['kind'], 'int8_convolution')
                    self.assertEqual(measured['actual_kernel_evidence']['ternary']['kind'], 'packed_decode_fp32')
                    self._denials(pin, code, notice)

    def _denials(self, pin, code, notice):
        root, value, base, scene, record, plan = load_registration(pin, code)
        seed = initial_model(root, value, base, scene, record, 'qat-first')
        run = new_run(seed, scene, record, plan, 'qat-first')
        with self.assertRaisesRegex(ValueError, 'training tiles only'):
            validation = value['validation']['tiles'][0]['id']
            calibration_contract(base, scene, record, LAYER_ORDER, [validation], qat=True)
        key = LAYER_ORDER[0]
        run.model.get_submodule(key).input_quant.observer_enabled.fill_(1)
        with self.assertRaisesRegex(ValueError, 'ranges are not frozen'):
            run.step()
        self.assertEqual(run.completed, 0)
        self.assertNotEqual(frozen_buffers(run.model, 'qat-first'), frozen_buffers(seed, 'qat-first'))
        with self.assertRaisesRegex(FileExistsError, 'already consumed'):
            execute(pin, code, notice, 'qat-first', 'full', None)
        engine = torch.backends.quantized.engine
        try:
            torch.backends.quantized.engine = 'qnnpack'
            with self.assertRaisesRegex(ValueError, 'INT8 engine'):
                load_registration(pin, code)
        finally:
            torch.backends.quantized.engine = engine
        status = admission.read(dev._pin(root / 'attempts' / 'qat-first' / 'full' / 'status.json'))
        packet = torch.load(status['checkpoint']['path'], map_location='cpu', weights_only=True)
        alterations = {
            'optimizer': lambda v: v['optimizer']['param_groups'][0].update(lr=.004),
            'cursor': lambda v: v['progress'].update(completed_updates=3),
            'observer': lambda v: v['state'][key + '.input_quant.observer_enabled'].fill_(1),
        }
        for name, change in alterations.items():
            data = torch.load(status['checkpoint']['path'], map_location='cpu', weights_only=True)
            change(data)
            target = root / f'bad-{name}.pt'
            torch.save(data, target)
            with self.assertRaises(ValueError):
                PrecisionRecoveryRun.load(seed, scene, record, target, plan=plan, trial='qat-first', selector_seed=64,
                    expected_sha256=dev._pin(target)['sha256'], expected_contract_hash=packet['contract_hash'])
        with self.assertRaisesRegex(ValueError, 'registered QAT|contract changed'):
            PrecisionRecoveryRun.load(seed, scene, record, status['checkpoint']['path'], plan=plan, trial='qat-second',
                selector_seed=64, expected_sha256=status['checkpoint']['sha256'], expected_contract_hash=packet['contract_hash'])
        registration_path = Path(pin['path'])
        original = registration_path.read_bytes()
        for edit in (dict(budget=5), dict(lr=.004), dict(calibration_tiles=[]), dict(assignments={}),
                     dict(source=dict(value['source'], weights=dict(value['source']['weights'], sha256='0' * 64))),
                     dict(runtime=dict(value['runtime'], torch='unknown'))):
            try:
                registration_path.write_text(json.dumps(dict(value, **edit)))
                with self.assertRaises(ValueError):
                    load_registration(dev._pin(registration_path), code)
            finally:
                registration_path.write_bytes(original)
        dataset = Path(value['source']['development']['path'])
        raw = dataset.read_bytes()
        try:
            dataset.write_bytes(raw + b'changed')
            with self.assertRaisesRegex(ValueError, 'Input file identity changed'):
                load_registration(pin, code)
        finally:
            dataset.write_bytes(raw)

    def test_fixed_fallback_and_joint_gate(self):
        def flags(ok):
            return {'fused': {'passes_metric_guards': ok}}
        result = select_families({arm: flags(arm in ('qat-second', 'qat-both', 'ternary-first', 'ternary-second'))
                                  for arm in ARMS if arm != 'fp32'})
        self.assertEqual(result['qat']['arm'], 'qat-second')
        self.assertEqual(result['qat']['trials'][-1]['failed_single_layers'], ['qat-first'])
        self.assertEqual(result['ternary']['arm'], 'ternary-first')
        result = select_families({arm: flags(False) for arm in ARMS if arm != 'fp32'})
        self.assertTrue(all(item['arm'] == 'fp32' for item in result.values()))


if __name__ == '__main__':
    unittest.main()
