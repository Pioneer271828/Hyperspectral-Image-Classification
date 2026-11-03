"""Reject misleading delivery claims using real temporary Git histories."""
import copy
import subprocess
import tempfile
from pathlib import Path
import unittest

from scripts.verify_delivery_status import validate_status


class DeliveryStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Delivery Test')
        self.git('config', 'user.email', 'delivery@example.invalid')
        self.git('config', 'commit.gpgsign', 'false')
        (self.root / 'evidence.md').write_text('declared acceptance scope\n')
        (self.root / 'model.py').write_text('initial\n')
        self.main = self.commit('initial')
        (self.root / 'model.py').write_text('first module\n')
        first = self.commit('first module')
        (self.root / 'model.py').write_text('second module\n')
        second = self.commit('second module')
        self.data = {
            'schema_version': 1,
            'snapshot': {'date': '2026-11-02', 'commits_ahead_of_main': 2, 'pull_request_count': 2,
                         'open_pull_request_count': 2, 'issue_count': 2},
            'main': {'commit': self.main},
            'baseline': {'commit': second, 'tested_code_commit': first, 'is_default_branch': False, 'evidence': ['evidence.md']},
            'tracking': {'main_integration_state': 'not_merged', 'next_module': 'M4', 'main_merge_required_before_m4': False},
            'pull_requests': [
                {'number': 2, 'base_pr': None, 'base_ref': 'main', 'base_sha': self.main, 'head_ref': 'first',
                 'head_sha': first, 'included_in_baseline': True, 'state': 'open', 'merged': False, 'issues': [1]},
                {'number': 4, 'base_pr': 2, 'base_ref': 'first', 'base_sha': first, 'head_ref': 'second',
                 'head_sha': second, 'included_in_baseline': True, 'state': 'open', 'merged': False, 'issues': [3]},
            ],
            'bridge_commits': [],
            'issues': [{'number': 1, 'state': 'open', 'linked_prs': [2], 'bridge_commits': []},
                       {'number': 3, 'state': 'open', 'linked_prs': [4], 'bridge_commits': []}],
            'modules': [{'id': 'M5', 'implementation': 'delivered', 'engineering': 'accepted',
                         'empirical': 'pending', 'integration': 'pending_main_review',
                         'issues': [1, 3], 'prs': [2, 4], 'evidence': ['evidence.md']}],
        }

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True, stderr=subprocess.STDOUT).strip()

    def commit(self, message):
        self.git('add', '.')
        self.git('commit', '-qm', message)
        return self.git('rev-parse', 'HEAD')

    def test_valid_snapshot_with_open_issues_is_readonly(self):
        before = copy.deepcopy(self.data); head = self.git('rev-parse', 'HEAD')
        result = validate_status(self.data, self.root)
        self.assertEqual(result['pull_requests'], 2)
        self.assertEqual(result['ahead_of_main'], 2)
        self.assertEqual(self.data, before)
        self.assertEqual(self.git('rev-parse', 'HEAD'), head)
        self.assertEqual(self.git('status', '--porcelain'), '')

    def test_rejects_duplicate_missing_evidence_and_false_merge(self):
        for change in ('duplicate_pr', 'missing_evidence', 'false_merge', 'false_default', 'issue_reference', 'acceptance_without_implementation'):
            data = copy.deepcopy(self.data)
            if change == 'duplicate_pr': data['pull_requests'][1]['number'] = 2
            elif change == 'missing_evidence': data['modules'][0]['evidence'] = ['missing.md']
            elif change == 'false_merge': data['pull_requests'][0]['merged'] = True
            elif change == 'false_default': data['baseline']['is_default_branch'] = True
            elif change == 'issue_reference': data['issues'][0]['linked_prs'] = []
            else: data['modules'][0]['implementation'] = 'planned'
            with self.subTest(change=change), self.assertRaises(ValueError): validate_status(data, self.root)

    def test_rejects_dependency_cycle_and_unincluded_head(self):
        data = copy.deepcopy(self.data)
        data['pull_requests'][0]['base_pr'] = 4
        with self.assertRaisesRegex(ValueError, 'cycle'): validate_status(data, self.root)
        self.git('checkout', '-qb', 'unrelated', self.main)
        (self.root / 'model.py').write_text('side branch\n')
        side = self.commit('unrelated')
        self.git('checkout', '--detach', self.data['baseline']['commit'])
        data = copy.deepcopy(self.data); data['pull_requests'][1]['head_sha'] = side
        with self.assertRaisesRegex(ValueError, 'inclusion'): validate_status(data, self.root)

    def test_bridge_commits_must_cover_exactly_the_unreviewed_gap(self):
        (self.root / 'evidence.md').write_text('bridge evidence\n')
        bridge = self.commit('bridge without independent PR')
        data = copy.deepcopy(self.data); data['baseline']['commit'] = bridge
        data['snapshot']['commits_ahead_of_main'] = 3
        data['bridge_commits'] = [{'sha': bridge, 'issues': [], 'evidence': ['evidence.md']}]
        self.assertEqual(validate_status(data, self.root)['bridge_commits'], 1)
        data['bridge_commits'] = []
        with self.assertRaisesRegex(ValueError, 'bridge coverage'): validate_status(data, self.root)

    def test_rejects_silent_algorithm_change_after_reference(self):
        (self.root / 'model.py').write_text('unreviewed model mutation\n')
        self.commit('unreviewed change')
        with self.assertRaisesRegex(ValueError, 'algorithm/training'): validate_status(self.data, self.root)


if __name__ == '__main__':
    unittest.main()
