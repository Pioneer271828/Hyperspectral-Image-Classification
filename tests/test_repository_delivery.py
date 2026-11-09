import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from repository_delivery import STATUS_PATH, SUMMARY_PATH, render_status, validate_status, check_repository

ROOT = Path(__file__).resolve().parents[1]


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.status = json.loads((ROOT / STATUS_PATH).read_text(encoding='utf-8'))

    def test_current_status_matches_and_retains_uncompleted_research(self):
        validate_status(self.status, ROOT)
        self.assertEqual((ROOT / SUMMARY_PATH).read_text(encoding='utf-8'), render_status(self.status))
        self.assertIn('未完成完整研究闭环', render_status(self.status))
        self.assertEqual(self.status['main_merge'], 'pending_review')

    def test_unsafe_or_unreviewed_delivery_claim_rejected(self):
        for key, value in [('main_merge', 'merged'), ('real_training_started_by_this_task', True),
                           ('main_snapshot', 'main'), ('ci_runner', 'passed')]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.status)
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_status(bad, ROOT)
        bad = copy.deepcopy(self.status)
        bad['required_files'].append('../outside.py')
        with self.assertRaises(ValueError):
            validate_status(bad, ROOT)

    def test_missing_repair_and_runtime_change_block_acceptance(self):
        def git_output(args, **kwargs):
            if 'rev-parse' in args:
                return 'false\n' if '--is-shallow-repository' in args else 'a' * 40 + '\n'
            if 'status' in args:
                return ''
            if 'diff' in args:
                return 'model/VCASAHGCN.py\n'
            raise AssertionError(args)
        with patch('repository_delivery.subprocess.check_output', side_effect=git_output):
            with patch('repository_delivery.subprocess.run') as run:
                run.return_value.returncode = 1
                with self.assertRaisesRegex(ValueError, 'not included'):
                    check_repository(ROOT)
                run.return_value.returncode = 0
                with self.assertRaisesRegex(ValueError, 'defaults changed'):
                    check_repository(ROOT)

    def test_wrong_sha_and_shallow_checkout_rejected(self):
        with patch('repository_delivery.subprocess.check_output', return_value='a' * 40):
            with self.assertRaisesRegex(ValueError, 'Unexpected checkout SHA'):
                check_repository(ROOT, 'b' * 40)
        def shallow(args, **kwargs):
            if 'status' in args:
                return ''
            return 'true' if '--is-shallow-repository' in args else 'a' * 40
        with patch('repository_delivery.subprocess.check_output', side_effect=shallow):
            with self.assertRaisesRegex(ValueError, 'Full history required'):
                check_repository(ROOT)

    def test_stale_projection_rejected_before_acceptance(self):
        def git_output(args, **kwargs):
            if 'rev-parse' in args:
                return 'false' if '--is-shallow-repository' in args else 'a' * 40
            return ''
        with patch('repository_delivery.subprocess.check_output', side_effect=git_output), \
                patch('repository_delivery.subprocess.run') as run, \
                patch('repository_delivery.render_status', return_value='stale'):
            run.return_value.returncode = 0
            with self.assertRaisesRegex(ValueError, 'Markdown is stale'):
                check_repository(ROOT)


if __name__ == '__main__':
    unittest.main()
