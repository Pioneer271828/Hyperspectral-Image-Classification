"""Guarded synthetic producer, consumer CLI and independent numerical checks."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat
import torch

from loadData.dataset_registry import DATASETS
from loadData.observation_sampling import BandMetadata, sample
from model.research_config import digest
from research_runtime import hash_file, write_json
from tests import test_p4_coverage_batch as fixture
from train_spatial import SpatialConfig
from transfer import p4_coverage_review as old_review
from transfer import p4_frozen_diagnostic_metrics as independent, p4_frozen_diagnostics as diag
from transfer.g3_training import Scene, state_hash
from transfer.p4_spatial import initialize, read


class FrozenDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = fixture.CoverageBatchTests
        cls.addClassCleanup(source.doClassCleanups)
        source.setUpClass()
        cls.root, cls.code = source.root, source.code
        audit = old_review.review(source.scope_path, source.scope_sha, source.execution_path, source.execution_sha,
                                  cls.root / 'prior-audit', cls.root / 'prior-metadata', cls.code)
        parents = {'scope': source.scope_path, 'execution': source.execution_path, 'audit': audit,
                   'metadata': cls.root / 'prior-metadata' / 'metadata.json'}
        cls.links = {key: {'path': str(path), 'sha256': hash_file(path)} for key, path in parents.items()}
        args = [part for key, row in cls.links.items() for part in
                ('--' + key, row['path'], '--' + key + '-sha256', row['sha256'])]
        cls.scope = cls.cli('p4_frozen_diagnostics', 'prepare', '--synthetic', *args, '--out', cls.root / 'diagnosis')
        cls.receipt = cls.cli('p4_frozen_diagnostics', 'run', '--scope', cls.scope, '--scope-sha256',
                              hash_file(cls.scope), '--out', cls.root / 'replay')
        cls.verification = cls.cli('p4_frozen_diagnostic_metrics', '--scope', cls.scope, '--scope-sha256',
            hash_file(cls.scope), '--receipt', cls.receipt, '--receipt-sha256', hash_file(cls.receipt),
            '--out', cls.root / 'independent')

    @classmethod
    def cli(cls, module, *args):
        command = [sys.executable, '-m', 'transfer.' + module, *map(str, args), '--expected-commit', cls.code['commit']]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f'{module} CLI failed: {result.stderr}')
        payload = json.loads(result.stdout)
        path = Path(payload['path'])
        if hash_file(path) != payload['sha256']:
            raise ValueError('CLI did not authenticate its output')
        return path

    def test_complete_cli_and_independent_readouts(self):
        run, audit = read(self.receipt), read(self.verification)
        for key, count in (('fit_heads', 720), ('observation_heads', 840), ('observation_class_recalls', 18480),
                           ('bypass_heads', 96), ('updates', 0)):
            self.assertEqual(run[key], count)
            self.assertEqual(audit[key], count)
        self.assertEqual(run['bitwise_selected_reference_heads'], 120)
        self.assertEqual(audit['comparison_heads'], 2256)
        self.assertFalse(audit['query_accessed'])
        self.assertFalse(audit['raw_gt_loaded'])
        self.assertEqual(audit['cases'], 40)
        self.assertGreater(audit['actual_tile_forwards']['fit'], 0)
        self.assertTrue(all(len(read(self.scope)['cases'][index]['unique_exposed_train_labels_by_class']) == 22
                            for index in (0, 20)))

    def test_producer_runtime_and_consumer_code_are_separate(self):
        self.assertNotEqual(diag.PRODUCER, self.code['commit'])
        pinned = diag.runtime_blobs(diag.PRODUCER, self.code['commit'])
        self.assertIn('model/VCASAHGCN.py', pinned)
        self.assertNotIn('transfer/p4_frozen_diagnostics.py', pinned)
        with self.assertRaisesRegex(ValueError, 'Producer runtime blobs differ'):
            diag.runtime_blobs(self.code['commit'], diag.PRODUCER)
        with self.assertRaisesRegex(ValueError, 'Unaccepted real producer'):
            diag.parent_documents(self.links, self.code, synthetic=False)
        bad = deepcopy(self.links)
        bad['metadata'] = bad['audit']
        with self.assertRaises((ValueError, KeyError)):
            diag.parent_documents(bad, self.code, synthetic=True)

    def test_rejects_changed_case_conditions_and_stages(self):
        baseline = read(self.scope)
        for name in ('cases', 'seed', 'selected', 'last'):
            scope = deepcopy(baseline)
            if name == 'cases':
                scope['cases'].pop()
            elif name == 'seed':
                scope['conditions']['random50']['seed'] = 3133
            else:
                scope['cases'][0]['state'][name] = 'a' * 64
            scope['scope_hash'] = digest({key: v for key, v in scope.items() if key != 'scope_hash'})
            path = self.root / f'altered-{name}.json'
            write_json(path, scope)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'case, label, condition or seed'):
                diag.load_context(path, hash_file(path), self.code)

    def test_rejects_bad_run_binding_and_independent_metric_tamper(self):
        receipt = read(self.receipt)
        changed = deepcopy(receipt)
        changed['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            independent.inventory(read(self.scope), changed, hash_file(self.scope))
        with self.assertRaisesRegex(ValueError, 'Pinned document changed'):
            independent.verify(self.scope, hash_file(self.scope), self.receipt, 'a' * 64,
                               self.root / 'verify-invalid', self.code)
        case_id = receipt['rows'][0]['case']['id']
        path = self.receipt.parent / f'{case_id}.json'
        original = path.read_bytes()
        amended = self.receipt.parent / 'amended.json'
        try:
            result = read(path)
            result['readouts']['fit__initial__train']['heads']['fused']['nll'] += .1
            write_json(path, result)
            receipt['artifacts'][path.name] = hash_file(path)
            write_json(amended, receipt)
            with self.assertRaisesRegex(ValueError, 'Independent metrics differ'):
                independent.verify(self.scope, hash_file(self.scope), amended, hash_file(amended),
                                   self.root / 'verify-amended', self.code)
        finally:
            path.write_bytes(original)

    def test_half_range_exact_boundary_empty_and_duplicates(self):
        bands = BandMetadata('synthetic', (0, 2, 5), (400., 500., 600.), 'nm', 'approximate', (True,) * 3)
        _, record = sample(diag.stub(bands, 'geometry'), bands, {'mode': 'uniform_stride', 'stride': 2, 'offset': 0})
        conditions = diag.freeze_conditions(bands, 'geometry', record)
        self.assertEqual(conditions['low_half']['sampling']['selected_band_ids'], [0, 2])
        self.assertEqual(conditions['low_half']['same_bands_as'], diag.REFERENCE)
        self.assertEqual(conditions['high_half']['sampling']['selected_band_ids'], [5])
        self.assertEqual(conditions['high_half']['same_bands_as'], 'odd_stride')
        self.assertEqual(conditions['random50']['seed'], 3032)
        same = replace(bands, wavelengths=(500.,) * 3)
        _, record = sample(diag.stub(same, 'geometry'), same, {'mode': 'uniform_stride', 'stride': 2, 'offset': 0})
        self.assertEqual(diag.freeze_conditions(same, 'geometry', record)['high_half']['status'], 'unavailable')

    def test_real_graph_on_poisoned_deleted_columns_and_role_denial(self):
        scope, reg, plan, allowed, bands, stats, populations, *_ = diag.load_context(
            self.scope, hash_file(self.scope), self.code)
        row = scope['cases'][1]
        labels = populations[row['release']]
        self.assertEqual(labels['train'][2]['labels'], 22)
        self.assertEqual(labels['val'][2]['labels'], 110)
        wrong_tile = labels['val'][0][0][0]
        for role in ('query', 'original_gt', 'train'):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'roles|escape'):
                diag.fixed_population([(wrong_tile, np.zeros((1, 1)))], allowed, role, plan['shape'][1], 1)
        _, variable, classes, _ = DATASETS[diag.guarded.SCENE]
        cube = loadmat(reg['image'], variable_names=[variable])[variable]
        scene = Scene(diag.guarded.SCENE, cube, plan, bands, classes, stats)
        condition = scope['conditions']['odd_stride']
        kept = condition['sampling']['selected_positions']
        poisoned = np.full_like(cube, np.nan)
        poisoned[:, :, kept] = cube[:, :, kept]
        model, _ = initialize(SpatialConfig(reg['config']), coverage_arm=row['case']['arm'])
        model.eval()
        before = state_hash(model)
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Unexpected optimizer')):
            expected, graph = diag.forward_rows(model, scene, diag.fixed_views(scene, allowed, labels['val'][0],
                                                          condition, 'val'), row['case']['arm'])
            alternative = replace(scene, cube=poisoned)
            actual, other = diag.forward_rows(model, alternative, diag.fixed_views(alternative, allowed,
                                           labels['val'][0], condition, 'val'), row['case']['arm'])
        self.assertEqual(graph, other)
        self.assertTrue(all(torch.equal(expected[h], actual[h]) for h in diag.HEADS))
        self.assertEqual(state_hash(model), before)

    def test_val_class_recall_has_twenty_point_resolution(self):
        labels = np.repeat(np.arange(1, 23), 5)
        p = np.zeros((110, 22), dtype=np.float32)
        p[:, 0] = 1
        p[-1] = 0
        p[-1, -1] = 1
        result = independent.recompute(p, labels)
        self.assertEqual(result['recall'][-1], .2)
        self.assertEqual(result['recall'][1], 0)
        self.assertEqual(result['prediction_histogram'][-1], 1)
        self.assertTrue(independent.equivalent(result, diag.score(torch.from_numpy(p), labels)))
