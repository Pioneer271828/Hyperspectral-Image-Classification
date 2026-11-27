"""LXC-only behavioral tests with generated spectra and synthetic labels."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from .fixture import make
from .protocol import METHODS, admit, sha, OPTIMIZER
from .tasks import Task, dependencies
from .run import ROOT, execute, paired_plan, seed_all

ADAPTER = AUTHOR = None


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class Protocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.manifest = make(self.root/'roles'); self.original = json.loads(self.manifest.read_text())

    def tearDown(self):
        self.temp.cleanup()

    def change(self, mutate):
        value = copy.deepcopy(self.original); mutate(value)
        self.manifest.write_text(json.dumps(value)); return self.manifest

    def test_normalizer_uses_only_training(self):
        a = admit(self.manifest, synthetic=True)
        p = self.root/'roles/devval.npz'
        with np.load(p) as z:
            x, y = z['spectra']*100+9, z['labels']
        np.savez(p, spectra=x, labels=y)
        self.change(lambda v: v['scenes']['toy']['roles']['devval'].update(sha256=sha(p)))
        b = admit(self.manifest, synthetic=True)
        self.assertEqual(a['data']['toy']['normalizer_sha256'], b['data']['toy']['normalizer_sha256'])
        self.assertTrue(np.array_equal(a['data']['toy']['roles']['train']['spectra'], b['data']['toy']['roles']['train']['spectra']))
        self.assertGreater(b['data']['toy']['roles']['devval']['spectra'].max(), 1)

    def test_metadata_rejections(self):
        def scene(v): return v['scenes']['toy']
        changes = [
            lambda v: v.update(query_access=True),
            lambda v: v.update(confirmation_access=True),
            lambda v: v.update(kind='future_development'),
            lambda v: scene(v)['roles'].update(confirmation={'path': 'must-never-open'}),
            lambda v: scene(v)['roles']['devval'].update(label_access='sealed'),
            lambda v: scene(v)['roles']['devval']['samples'][0].update(group='train-0'),
            lambda v: scene(v)['groups']['devval-0'].update(original_parent='train-0'),
            lambda v: scene(v)['groups']['devval-0'].update(box=[0, 110, 0, 10]),
            lambda v: scene(v)['groups']['devval-0'].update(provenance='unknown'),
            lambda v: scene(v)['groups']['devval-0'].update(grid='unbound'),
            lambda v: scene(v)['roles']['train']['samples'][0].update(box=[0, 7, 0, 7]),
            lambda v: scene(v)['roles']['train'].update(path='../escape.npz'),
            lambda v: scene(v)['roles']['train'].update(sha256='0'*64),
            lambda v: scene(v).update(wavelengths_nm=[500]*8),
            lambda v: scene(v).update(geolocation_error_pixels=float('nan')),
        ]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                self.change(change)
                with self.assertRaises(ValueError): admit(self.manifest, synthetic=True)

    def test_parent_buffer_not_center_distance(self):
        # Small safe patch centers do not excuse overlapping parent uncertainty.
        self.change(lambda v: v['scenes']['toy']['groups']['devval-0'].update(box=[22, 110, 0, 10]))
        with self.assertRaisesRegex(ValueError, 'Expanded parent'):
            admit(self.manifest, synthetic=True)

    def test_constant_training_band_is_rejected(self):
        p = self.root/'roles/train.npz'
        with np.load(p) as z: x, y = z['spectra'].copy(), z['labels'].copy()
        x[:, :, :, 0] = 1
        np.savez(p, spectra=x, labels=y)
        self.change(lambda v: v['scenes']['toy']['roles']['train'].update(sha256=sha(p)))
        with self.assertRaisesRegex(ValueError, 'constant'): admit(self.manifest, synthetic=True)

    def test_query_tensor_is_not_a_supported_input(self):
        self.change(lambda v: v['scenes']['toy']['roles'].update(query={'path': '/forbidden/query.npz'}))
        with self.assertRaisesRegex(ValueError, 'Only released'): admit(self.manifest, synthetic=True)


class Models(Protocol):
    # Inherit protocol tests intentionally? No: unittest inheritance would repeat
    # them. Keep only the common setup and teardown below.
    test_normalizer_uses_only_training = None
    test_metadata_rejections = None
    test_parent_buffer_not_center_distance = None
    test_constant_training_band_is_rejected = None
    test_query_tensor_is_not_a_supported_input = None

    def task(self, method):
        data = admit(self.manifest, synthetic=True)['data']['toy']
        return Task('toy', data, method, ADAPTER, AUTHOR)

    def test_shared_initial_weights_and_predictions(self):
        states = []; outputs = []
        for method in METHODS[:-1]:
            seed_all(64); t = self.task(method); t.model.eval()
            states.append(copy.deepcopy(t.model.state_dict()))
            with torch.no_grad(): outputs.append(t.forward('devval', 0)[:3])
        for state, out in zip(states[1:], outputs[1:]):
            self.assertTrue(all(torch.equal(v, state[k]) for k, v in states[0].items()))
            self.assertTrue(all(torch.equal(a, b) for a,b in zip(outputs[0], out)))

    def test_physical_inputs_and_unmodified_author_forward(self):
        native = self.task('none'); dsnet = self.task('dsnet')
        x = dsnet.inputs('train', 0)
        self.assertTrue(torch.equal(x, native.inputs('train', 0)))
        dsnet.model.eval()
        with torch.no_grad():
            actual = dsnet.forward('train', 0)
            direct = dsnet.model.model(x.permute(2,0,1).unsqueeze(0).contiguous())
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual, direct)))
        self.assertEqual(native.inputs('train', 0).shape, (7, 7, 8))

    def test_all_methods_gradients_and_eval_immutability(self):
        for method in METHODS:
            with self.subTest(method=method):
                seed_all(64); t = self.task(method); t.model.train(); loss = t.loss(0); loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in t.model.parameters()))
                t.model.eval(); before = copy.deepcopy(t.model.state_dict())
                with torch.no_grad(): m = t.evaluate()
                self.assertTrue(equal(before, t.model.state_dict())); self.assertEqual(m['points'], 2)

    def contract(self):
        admitted = admit(self.manifest, synthetic=True)
        return {'roster': [0,1], 'sweeps': 2, 'seed': 64, 'plan': paired_plan([0,1], 2, 64),
                'updates': 4, 'selection_steps': [0,2,4], 'device': 'cpu', 'query_reads': 0,
                'confirmation': False, 'optimizer': OPTIMIZER, 'protected': admitted['protected']}

    def test_full_state_resume_all_four_methods(self):
        for method in METHODS:
            with self.subTest(method=method):
                c = self.contract(); full = self.root/f'{method}-full'; split = self.root/f'{method}-split'
                factory = lambda: self.task(method)
                execute(full, c, factory)
                execute(split, c, factory, stop_after=1)
                execute(split, c, factory, resume=True)
                a = torch.load(full/'stages/000004/state.pt', weights_only=True)
                b = torch.load(split/'stages/000004/state.pt', weights_only=True)
                self.assertTrue(equal(a,b), 'Model/Adam/RNG/modes/history differ')
                def forbidden(): raise AssertionError('Completed reentry constructed a model')
                receipt = execute(split, c, forbidden, resume=True)
                self.assertEqual(receipt['new_neural_calls'], 0)

    def test_uncertain_optimizer_update_never_replayed(self):
        c = self.contract(); destination = self.root/'uncertain'
        def fault(phase, step):
            if phase == 'after_optimizer' and step == 1: raise RuntimeError('synthetic injected interruption')
        with self.assertRaises(RuntimeError): execute(destination, c, lambda: self.task('dsnet'), fault=fault)
        def forbidden(): raise AssertionError('Unsafe resume constructed model')
        with self.assertRaisesRegex(ValueError, 'Pending'): execute(destination, c, forbidden, resume=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--native', required=True); p.add_argument('--author', required=True)
    a, rest = p.parse_known_args(); torch.set_num_threads(1)
    ADAPTER, AUTHOR = dependencies(ROOT, a.native, a.author)
    unittest.main(argv=[sys.argv[0]]+rest)
