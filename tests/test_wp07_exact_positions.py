from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from scripts.run_native_released_points import arithmetic
from transfer import wp07_exact_positions as position
from transfer import wp07_position_run as run
from transfer import wp07_condition_run as previous
from transfer import wp07_graph_range as graph
from transfer import wp07_observed_conditions as conditions
from transfer.native_scale_observation import forward
from transfer.xuzhou_multiscale_audit import _exact


class ExactPositionTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.provider = run.FixtureProvider(9, 'exact_mask')
        cls.entries = run.cache(cls.provider, 'fixture')

    def test_collision_is_preserved_as_summary_negative_and_mask_separates(self):
        a, b = (position.POSITION_SETS[n] for n in ('interior_a', 'interior_b'))
        self.assertTrue(torch.equal(torch.tensor(conditions.index_features(a, 20)),
                                    torch.tensor(conditions.index_features(b, 20))))
        va, vb = position.mask_features(a, 20), position.mask_features(b, 20)
        self.assertNotEqual(va, vb)
        self.assertEqual(sum(va[1:]), sum(vb[1:]))
        for bad, count in [((1, 0), 20), ((0, 0), 20), ((0, 1), 437)]:
            with self.assertRaises(ValueError):
                position.mask_features(bad, count)

    def test_exact_slice_keeps_parent_cells_role_and_fixed_metadata(self):
        events = self.provider.events()[1]
        for family in ('shift', 'interior'):
            a, b = [next(e for e in events if e['band'] == family + '_' + side) for side in ('a', 'b')]
            left, labels, points = self.provider.materialize('devval', a, 'fixture')
            right, labels2, points2 = self.provider.materialize('devval', b, 'fixture')
            self.assertTrue(np.array_equal(labels, labels2))
            self.assertEqual(points.summary(), points2.summary())
            self.assertEqual(left.receipt['source']['parent_observation'], right.receipt['source']['parent_observation'])
            self.assertEqual(left.lattice, right.lattice)
            self.assertEqual(left.view.bands.band_ids, position.POSITION_SETS[a['band']])
            self.assertFalse(left.view.observation.spectral.flags.writeable)
            self.assertNotEqual(left.receipt['spectral'], right.receipt['spectral'])

    @torch.no_grad()
    def test_same_content_controls_missing_and_branch_isolation(self):
        model = position.initialize(self.provider.baseline(64), 7385)
        events = self.provider.events()[1]
        for family in ('shift', 'interior'):
            event = next(e for e in events if e['band'] == family + '_a')
            actual, _, _ = self.provider.materialize('devval', event, 'fixture')
            entry = self.entries[previous.key('devval', event)]

            def call(rep, intervention, cnn='joint', gnn='joint'):
                with position.representation(model, rep), conditions.selection_policy(model, cnn, gnn), graph.frozen_support(model, entry, 'content'):
                    return forward(model, actual, 'fixture', 'devval', condition=position.bind(actual, intervention), graph_scale=16)

            for rep in position.REPRESENTATIONS:
                a, ta = call(rep, 'matched')
                b, tb = call(rep, 'paired')
                self.assertEqual((ta['q'], ta['a0']), (tb['q'], tb['a0']))
                if family == 'interior' and rep == 'summary8':
                    self.assertTrue(all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])))
                else:
                    self.assertNotEqual(ta['weights'], tb['weights'])
                    self.assertFalse(torch.equal(a[0], b[0]))
                for cnn, gnn, head in [('joint', 'mean', 1), ('mean', 'joint', 2)]:
                    a, _ = call(rep, 'matched', cnn, gnn)
                    b, _ = call(rep, 'paired', cnn, gnn)
                    self.assertTrue(torch.equal(a[head], b[head]))
                for mode in ('mean', 'content'):
                    a, _ = call(rep, 'matched', mode, mode)
                    b, _ = call(rep, 'paired', mode, mode)
                    self.assertTrue(all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])))
                for mode, fallback in [('metadata', 'mean'), ('joint', 'content')]:
                    a, _ = call(rep, 'missing', mode, mode)
                    b, _ = call(rep, 'matched', fallback, fallback)
                    self.assertTrue(all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])))

    def test_extension_preserves_old_tensors_and_rng_and_equal_initial_state(self):
        old = conditions.initialize(self.provider.baseline(64), 7385)
        before = deepcopy(old.state_dict())
        rng = torch.get_rng_state().clone()
        with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('No GPU reseeding')):
            position.extend(old, 1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, old.state_dict()[name]) for name, value in before.items()))
        initial = position.initialize(self.provider.baseline(64), 7385).state_dict()
        for rep in position.REPRESENTATIONS:
            with position.representation(old, rep):
                self.assertEqual(set(old.state_dict()), set(initial))

    def test_registration_cache_and_restore_are_bound_and_real_is_rejected(self):
        with TemporaryDirectory() as directory, arithmetic(Path(directory), None) as execution:
            spec = run.contract(self.provider, 64, 'both.joint', 'fixture', execution.numeric, {'cache': 'fixture'}, {'synthetic': True})
            engine = run.PositionRun(spec, self.provider, self.entries, 'fixture', execution)
            engine.start()
            engine.update(spec['training'][0])
            state = engine.pack()
            replay = run.PositionRun(spec, self.provider, self.entries, 'fixture', execution)
            replay.restore(state, 1)
            self.assertTrue(_exact(state, replay.pack()))
            for mutation in ({'representation': 'summary8'}, {'steps': 25}):
                with self.assertRaisesRegex(ValueError, 'registration'):
                    run.PositionRun({**spec, **mutation}, self.provider, self.entries, 'fixture', execution)
            binding = position.bind(self.provider.materialize('devval', self.provider.events()[1][0], 'fixture')[0])
            with self.assertRaises(ValueError):
                replace(binding, master_count=21).validate(engine._materialize('devval', self.provider.events()[1][0])[0], {})
            self.provider.synthetic = False
            try:
                with self.assertRaisesRegex(ValueError, 'synthetic'):
                    run.contract(self.provider, 64, 'both.joint', 'fixture', execution.numeric, {}, {})
            finally:
                self.provider.synthetic = True
