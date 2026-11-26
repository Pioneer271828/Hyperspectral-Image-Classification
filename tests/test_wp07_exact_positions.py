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
from transfer.wp07_position_real import RegisteredPositionProvider


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

            for rep in position.SUPPORTED_REPRESENTATIONS:
                a, ta = call(rep, 'matched')
                b, tb = call(rep, 'paired')
                self.assertEqual((ta['q'], ta['a0']), (tb['q'], tb['a0']))
                if family == 'interior' and rep in ('summary8', 'balanced_summary'):
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
            changed_buffer = deepcopy(state)
            name = next(iter(spec['fixed_buffers']))
            changed_buffer['model'][name].add_(.01)
            with self.assertRaisesRegex(ValueError, 'fixed position buffer'):
                replay.restore(changed_buffer, 1)
            changed_entries = deepcopy(self.entries)
            k = previous.key('devtrain', spec['training'][0])
            q = changed_entries[k]['q'].coalesce()
            changed_entries[k]['q'] = torch.sparse_coo_tensor(q.indices(), q.values()*.99, q.shape).coalesce()
            with self.assertRaisesRegex(ValueError, 'geometry changed'):
                run.PositionRun(spec, self.provider, changed_entries, 'fixture', execution).restore(state, 1)
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

    def test_real_view_rejection_precedes_payload_access(self):
        gate = RegisteredPositionProvider.__new__(RegisteredPositionProvider)
        event = dict(tile=3, factor=1, band='full')
        gate._registry = {('devval', 3, 'full', 1): {'source': {'positions': [0, 1, 2, 3]}}}
        from unittest.mock import Mock
        gate.provider = Mock()
        for role, changed, values in [('devval', event, [0, 2]),
                                      ('devtrain', event, None),
                                      ('devval', {**event, 'band': 'odd'}, None)]:
            with self.assertRaises(ValueError):
                gate.request(role, changed, 'fixture', positions=values)
        gate.provider.materialize.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'optimizer release'):
            gate.contract()

    def test_metadata_only_real_proposal_matches_count_span_and_collision(self):
        for master in (103, 436):
            views = position.proposed_real_positions(master)
            a, shifted, interior = (views[k] for k in ('shared_a', 'shift_b', 'interior_b'))
            self.assertEqual(len(a), (master + 1)//2)
            self.assertEqual((len(a), a[0], a[-1]), (len(interior), interior[0], interior[-1]))
            self.assertEqual(tuple(v + 1 for v in a), shifted)
            self.assertTrue(torch.equal(torch.tensor(conditions.index_features(a, master)),
                                        torch.tensor(conditions.index_features(interior, master))))
            self.assertNotEqual(position.mask_features(a, master), position.mask_features(interior, master))

    def test_balanced_representations_share_projection_and_unit_norm_input(self):
        old = conditions.IndexConditionEncoder()
        encoder = position.PositionEncoder(old)
        frame = encoder.position_rotation.double()
        self.assertTrue(torch.allclose(frame.T @ frame, torch.eye(437, dtype=torch.float64), atol=1e-6, rtol=0))
        ref = encoder.exact_mask[0].weight
        descriptor = conditions.IndexPacket(None, position.POSITION_SETS['interior_a'], 20)
        for representation in position.BALANCED_REPRESENTATIONS:
            encoder.zero_grad(set_to_none=True)
            inputs = []
            hook = encoder.exact_mask.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
            encoder.representation = representation
            encoder(descriptor, None, None, ref).square().sum().backward()
            hook.remove()
            self.assertEqual(len(inputs), 1)
            self.assertEqual(tuple(inputs[0].shape), (437,))
            self.assertAlmostEqual(float(inputs[0].norm()), 1., places=6)
            self.assertTrue(all(p.grad is not None for p in encoder.exact_mask.parameters()))
            self.assertTrue(all(p.grad is None for p in encoder.indices.parameters()))
