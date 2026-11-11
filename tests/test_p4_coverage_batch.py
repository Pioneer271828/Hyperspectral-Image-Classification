"""Exercise the fixed coverage batch and independent review on guarded synthetic data."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from model.research_config import digest
from research_runtime import code_snapshot, hash_file, write_json
from tests.test_p4_coverage import fixture
from transfer import p4_coverage_batch as batch, p4_coverage_review as audit, p4_spatial as spatial


class CoverageBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.code = code_snapshot()
        fixture_root = cls.root / 'fixture'
        fixture_root.mkdir()
        parent, _, _ = fixture(fixture_root)
        prior = fixture_root / 'prior.json'
        write_json(prior, {'status': 'passed', 'protected_files': {str(parent): hash_file(parent)},
                           'query_accessed': False})
        cls.scope_path = batch.prepare(parent, hash_file(parent), prior, hash_file(prior),
                                       cls.root / 'scope', cls.code, synthetic=True)
        cls.scope_sha = hash_file(cls.scope_path)
        cls.scope, cls.reg = batch.load_scope(cls.scope_path, cls.scope_sha, cls.code)

        def in_process(command, log):
            values = dict(zip(command[3::2], command[4::2]))
            result = spatial.run_case(cls.reg, int(values['--case']), Path(values['--out']),
                                      int(values['--until']), cls.code, coverage_arm=values['--coverage-arm'])
            log.write_text(json.dumps(result) + '\n', encoding='utf-8')
            return 0

        # Keep one queue because in-process optimization uses shared RNG. The
        # production CLI launches a fresh process per path and resume.
        cls.execution_path = batch.run(cls.scope_path, cls.scope_sha, cls.root / 'execution', cls.code,
                                       workers=1, runner=in_process)
        cls.execution_sha = hash_file(cls.execution_path)
        cls.execution = spatial.read(cls.execution_path)

    def run_review(self, execution=None, sha=None):
        return audit.review(self.scope_path, self.scope_sha, execution or self.execution_path,
                            sha or self.execution_sha, self.root / ('review-' + self._testMethodName),
                            self.root / ('diag-' + self._testMethodName), self.code)

    def test_actual_forty_cases_complete_and_replay_three_modes(self):
        receipt = spatial.read(self.run_review())
        self.assertEqual(receipt['status'], 'passed')
        self.assertEqual(receipt['cases'], 40)
        self.assertEqual(receipt['exact_full_split_pairs'], 40)
        self.assertEqual(receipt['validation_heads'], 120)
        self.assertEqual(receipt['class_recalls'], 2640)
        self.assertEqual(receipt['updates'], 0)
        self.assertFalse(receipt['query_accessed'])
        self.assertEqual(len({row['validation_population_sha256'] for row in receipt['rows']}), 1)
        self.assertEqual(len({r['pair_sha256'] for r in receipt['rows']}), 8)
        self.assertEqual(self.execution['optimizer_updates'], 320)
        for row in self.execution['rows']:
            self.assertEqual([e['result']['updates_this_call'] for e in row['calls']], [4, 2, 2, 0])
            self.assertEqual(row['calls'][1]['partial'], {'cursor': 2, 'validation_events': 0,
                                                         'has_selected_state': False})
        diagnostics = spatial.read(self.root / ('diag-' + self._testMethodName) / 'metadata.json')
        self.assertEqual(diagnostics['validation_heads'], 240)
        self.assertEqual(diagnostics['retained_band_ids'], receipt['retained_band_ids'])
        for row in diagnostics['rows']:
            if row['case']['arm'] != 'none':
                self.assertTrue(all(x['fallback'] == 'missing_metadata' and x['usable_anchors'] == 0
                                    for x in row['conditions']['missing']['tile_metadata']))

    def test_manifest_rejects_missing_or_duplicate_cases_and_wrong_budget(self):
        keys = [r['id'] for r in self.scope['cases']]
        self.assertEqual(len(keys), len(set(keys)))
        for change in ('missing', 'duplicate', 'wrong_arm'):
            reg = deepcopy(self.reg)
            if change == 'missing':
                reg['cases'].pop()
            elif change == 'duplicate':
                reg['cases'][-1] = reg['cases'][0]
            else:
                reg['coverage']['arms'][-1] = reg['coverage']['arms'][0]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, '40 unique'):
                batch.manifest(reg, synthetic=True)
        with self.assertRaisesRegex(ValueError, 'registered update budget'):
            batch.manifest(self.reg, synthetic=False)

    def test_failure_stops_before_next_case_and_outputs_are_isolated(self):
        calls = []

        def fail(command, log):
            calls.append(command)
            log.write_text('synthetic failure\n', encoding='utf-8')
            return 12

        folder = self.root / 'failed'
        with self.assertRaisesRegex(RuntimeError, 'Batch stopped'):
            batch.run(self.scope_path, self.scope_sha, folder, self.code, workers=1, runner=fail)
        self.assertEqual(len(calls), 1)
        self.assertFalse((folder / 'execution.json').exists())
        self.assertEqual(spatial.read(folder / 'failure.json')['completed_cases'], [])
        with self.assertRaises(FileExistsError):
            batch.run(self.scope_path, self.scope_sha, folder, self.code, workers=1, runner=fail)
        with self.assertRaisesRegex(ValueError, 'sibling'):
            batch.run(self.scope_path, self.scope_sha, folder / 'nested', self.code, workers=1, runner=fail)
        self.assertEqual(len(calls), 1)

    def test_review_rejects_bad_sha_incomplete_and_wrong_arm(self):
        with self.assertRaisesRegex(ValueError, 'Pinned document changed'):
            self.run_review(sha='1' * 64)
        with self.assertRaisesRegex(ValueError, 'Pinned document changed'):
            batch.load_scope(self.scope_path, '2' * 64, self.code)
        for change in ('missing', 'wrong_arm'):
            execution = deepcopy(self.execution)
            if change == 'missing':
                execution['rows'].pop()
            else:
                execution['rows'][0]['case']['arm'] = 'continuous'
            path = self.root / (change + '.json')
            write_json(path, execution)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, '40-case execution'):
                self.run_review(path, hash_file(path))

    def test_review_rejects_tampered_artifact(self):
        row = self.execution['rows'][0]
        path = self.execution_path.parent / row['case']['id'] / 'full' / 'result.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'Pinned input or artifact changed'):
                self.run_review()
        finally:
            path.write_bytes(original)

    def test_review_rejects_wrong_exposure_selection_and_unauthorized_labels(self):
        row = self.execution['rows'][0]
        case = row['case']
        folder = self.execution_path.parent / case['id'] / 'full'
        state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=True)
        result = spatial.read(folder / 'result.json')
        wrong = dict(result, best_step=2)
        with self.assertRaisesRegex(ValueError, 'first minimum'):
            audit.chosen_event(state, wrong, self.scope['validation_steps'])
        exposure = spatial.read(self.reg['exposure']['plan'])
        cohort = exposure['cohorts'][0]
        plan = spatial.read(self.reg['plan'])
        geometry, allowed, _, _ = audit.inputs(self.reg, plan, exposure)
        release = audit.released_roles(self.reg, plan, geometry, allowed, case)
        changed = deepcopy(state['trace'])
        changed[0]['tile'] = -999
        with self.assertRaisesRegex(ValueError, 'class exposure'):
            audit.check_exposure(dict(state, trace=changed), result, release, cohort, exposure)

        path = self.root / 'unauthorized.json'
        reg = deepcopy(self.reg)
        entry = reg['cases'][case['index']]
        released = spatial.read(entry['release'])
        tile = next(t for t in plan['tiles'] if t['partition'] == 'test')
        y, x = tile['box'][0], tile['box'][2]
        released['validation']['indices'][0] = y * plan['shape'][1] + x
        released['validation']['indices'].sort()
        released['release_hash'] = digest({k: v for k, v in released.items() if k != 'release_hash'})
        write_json(path, released)
        entry['release'] = str(path)
        reg['pins'][str(path)] = hash_file(path)
        with self.assertRaisesRegex(ValueError, 'escapes the registered spatial role'):
            audit.released_roles(reg, plan, geometry, allowed, case)
