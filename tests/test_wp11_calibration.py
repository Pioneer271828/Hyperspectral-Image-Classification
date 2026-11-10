"""Calibration uses a committed confirmation identity without opening the package."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from inference.gate_calibration import SPARSE_PROTOCOL
from inference.route_controls import request_hash
from model.research_config import digest
from research_runtime import code_snapshot, write_json
from scripts import run_wp11_calibration as calibration
from scripts.run_wp11_confirmation import run as confirm
from tests.test_wp11_confirmation import file_pin, synthetic_package


def read(pin):
    return json.loads(Path(pin['path']).read_text())


def save(root, name, value):
    path = root / f'{name}.json'
    write_json(path, value)
    return file_pin(path)


def package(root, *, min_support=1):
    original, (_, _, _, _, _, admission, _, requests) = synthetic_package(root)
    confirmation = read(original)
    gate = read(confirmation['gate_protocol'])
    blocks = [b for b in admission['blocks'] if b['role'] == 'calibration']
    labels = {b['id']: list(range(1, 10)) for b in blocks}
    gate.update(schema=SPARSE_PROTOCOL, admission_hash=digest(admission), min_class_support=min_support,
                calibration_request_hash=request_hash([r for r in requests if r['id'] in labels]),
                data_hashes=dict(calibration=digest(labels), confirmation=admission['labels_sha256']))
    cal_labels = save(root, 'calibration-labels', labels)
    cal_blocks = save(root, 'calibration-blocks', blocks)
    receipt = save(root, 'custody-receipt', dict(role='calibration', status='packaged', raw_labels_read=False,
                   query=0, labels_sha256=cal_labels['sha256'], blocks_sha256=cal_blocks['sha256'],
                   labels_content_sha256=digest(labels)))
    confirmation['gate_protocol'] = save(root, 'sparse-gate', gate)
    spec = {k: confirmation[k] for k in ('development', 'full_weights', 'full_metadata', 'full_history',
            'history_evidence', 'light', 'light_training_sha256', 'light_contract_hash', 'admission', 'gate_protocol')}
    spec.update(calibration_receipt=receipt, calibration_blocks=cal_blocks, calibration_labels=cal_labels,
                calibration_tiles=[r for r in confirmation['tiles'] if r['tile_id'] in {b['tile_id'] for b in blocks}])
    return spec, confirmation, gate


class CalibrationTests(unittest.TestCase):
    def test_freeze_without_confirmation_files_then_run_actual_confirmation(self):
        affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, {min(affinity)})
            with TemporaryDirectory() as directory:
                root = Path(directory)
                spec, confirmation, gate = package(root)
                protected = [Path(confirmation['confirmation_labels']['path'])]
                protected.extend(Path(row['file']['path']) for row in confirmation['tiles'] if row not in spec['calibration_tiles'])
                saved = {path: path.read_bytes() for path in protected}
                for path in protected:
                    path.unlink()
                out = root / 'calibrated'
                receipt = calibration.run(save(root, 'calibration-spec', spec), out, code_snapshot())
                self.assertEqual(receipt['status'], 'adaptive_frozen')
                self.assertEqual((receipt['calibration_tiles'], receipt['confirmation_observations'], receipt['query']), (1, 0, 0))
                for path, raw in saved.items():
                    path.write_bytes(raw)
                confirmation['gate_freeze'] = file_pin(out / 'gate-freeze.json')
                curve = read(confirmation['curve_protocol'])
                curve['gate_protocol'] = digest(gate)
                next(p for p in curve['points'] if p['id'] == 'adaptive')['policy_hash'] = receipt['freeze_hash']
                confirmation['curve_protocol'] = save(root, 'sparse-curve', curve)
                pairs = read(confirmation['comparison'])
                pairs['curve_protocol_hash'] = digest(curve)
                confirmation['comparison'] = save(root, 'sparse-pairs', pairs)
                check = confirm(save(root, 'confirmation-spec', confirmation), root / 'confirmed', code_snapshot())
                self.assertEqual(check['support'], [1]*9)
                self.assertEqual(check['confirmation']['confirmation_hash'], gate['data_hashes']['confirmation'])
        finally:
            os.sched_setaffinity(0, affinity)

    def test_wrong_role_and_request_pin_are_rejected_before_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _, gate = package(root)
            custody = dict(read(spec['calibration_receipt']), role='confirmation')
            changed_gate = dict(gate, calibration_request_hash='0'*64)
            cases = [dict(spec, calibration_receipt=save(root, 'wrong-role', custody)),
                     dict(spec, gate_protocol=save(root, 'wrong-request', changed_gate))]
            for i, case in enumerate(cases):
                with patch.object(calibration, '_read', wraps=calibration._read) as reader:
                    with self.assertRaisesRegex(ValueError, 'custody receipt|input bytes'):
                        calibration.run(save(root, f'bad-{i}', case), root / 'bad', code_snapshot())
                    self.assertNotIn(spec['calibration_labels'], [call.args[0] for call in reader.call_args_list])
            self.assertFalse((root / 'bad').exists())

    def test_insufficient_support_freezes_full(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _, _ = package(root, min_support=2)
            receipt = calibration.run(save(root, 'calibration-spec', spec), root / 'result', code_snapshot())
            self.assertEqual((receipt['status'], receipt['selected'], receipt['accepted_tiles']), ('keep_full', 0, 0))
