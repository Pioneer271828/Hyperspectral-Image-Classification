"""Exercise a precommitted comparison through real synthetic inference."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research_runtime import code_snapshot
from scripts import prepare_wp11_confirmation as prepare
from scripts import run_wp11_confirmation as confirm
from scripts.prepare_wp11_gate import run as register
from scripts.run_wp11_calibration import run as calibrate
from tests.test_wp11_calibration import read, save
from tests.test_wp11_confirmation import file_pin
from tests.test_wp11_gate_registration import synthetic_registration


def frozen_package(root, code, *, min_support=1):
    spec = synthetic_registration(root, code, with_design=True)
    spec['settings']['min_class_support'] = min_support
    labels_path = root / 'confirmation/labels.json'
    labels = file_pin(labels_path)
    raw = labels_path.read_bytes()
    labels_path.unlink()
    early = register(save(root, 'early', spec), root / 'early-output', code)
    frozen_dir = root / 'calibrated'
    receipt = calibrate(early['calibration'], frozen_dir, code)
    registration = dict(handoff=early['confirmation'], freeze=file_pin(frozen_dir / 'gate-freeze.json'),
                        freeze_receipt=file_pin(frozen_dir / 'receipt.json'))
    return spec, receipt, registration, labels, raw


class ConfirmationRegistrationTests(unittest.TestCase):
    def test_committed_controls_generate_confirmation_without_labels_and_execute(self):
        affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, {min(affinity)})
            with TemporaryDirectory() as directory:
                root, code = Path(directory), code_snapshot()
                early, receipt, release, labels, raw = frozen_package(root, code)
                self.assertEqual(receipt['status'], 'adaptive_frozen')
                self.assertEqual(receipt['confirmation_design_sha256'], early['confirmation_design']['sha256'])
                release['confirmation_labels'] = labels
                ready = prepare.run(save(root, 'release', release), root / 'ready', code)
                self.assertEqual(ready['status'], 'registered')
                self.assertFalse(ready['held_aside_labels_read'])
                self.assertFalse(Path(labels['path']).exists())
                final = read(ready['confirmation'])
                self.assertEqual(final['fixed'], read(early['confirmation_design'])['fixed'])
                changed_curve = dict(read(final['curve_protocol']), ece_bins=5)
                invalid = dict(final, curve_protocol=save(root, 'changed-curve', changed_curve))
                missing_binding = {key: val for key, val in final.items() if key != 'confirmation_binding'}
                for index, tampered in enumerate((invalid, missing_binding)):
                    with patch.object(confirm, 'load_development', side_effect=AssertionError('too early')):
                        with self.assertRaisesRegex(ValueError, 'precommitted design|freeze binding'):
                            confirm.run(save(root, f'tampered-{index}', tampered), root / 'bad', code)
                Path(labels['path']).write_bytes(raw)
                result = confirm.run(ready['confirmation'], root / 'confirmed', code)
                self.assertEqual(result['support'], [1]*9)
                self.assertEqual(result['query'], 0)
        finally:
            os.sched_setaffinity(0, affinity)

    def test_full_fallback_requires_no_confirmation_release(self):
        with TemporaryDirectory() as directory:
            root, code = Path(directory), code_snapshot()
            _, receipt, release, labels, _ = frozen_package(root, code, min_support=2)
            self.assertEqual(receipt['status'], 'keep_full')
            result = prepare.run(save(root, 'release', release), root / 'ready', code)
            self.assertEqual(result['status'], 'keep_full')
            self.assertFalse(Path(labels['path']).exists())
            self.assertFalse((root / 'ready/confirmation-registration.json').exists())

    def test_rejects_swapped_design_and_deferred_label_hash(self):
        with TemporaryDirectory() as directory:
            root, code = Path(directory), code_snapshot()
            early, _, release, labels, _ = frozen_package(root, code)
            changed = dict(read(early['confirmation_design']), ece_bins=3)
            handoff = dict(read(release['handoff']), confirmation_design=save(root, 'changed-design', changed))
            bad = dict(release, handoff=save(root, 'changed-handoff', handoff), confirmation_labels=labels)
            with self.assertRaisesRegex(ValueError, 'differs from calibration registration'):
                prepare.run(save(root, 'bad-release', bad), root / 'bad', code)
            bad = dict(release, confirmation_labels=dict(labels, sha256='0'*64))
            with self.assertRaisesRegex(ValueError, 'differs from custody'):
                prepare.run(save(root, 'bad-label', bad), root / 'bad', code)
            self.assertFalse((root / 'bad').exists())
            self.assertFalse(Path(labels['path']).exists())
