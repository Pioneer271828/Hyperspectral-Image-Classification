"""Exercise the fitting worker with actual guarded synthetic labels and Adam steps."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from model.research_config import digest
from research_runtime import code_snapshot, hash_file, write_json
from test_p4_coverage import fixture
from tests.test_p4_stable_cuda import synthetic_schema
from transfer import p4_exposure
from transfer import p4_full_budget_audit as audit
from transfer import p4_full_budget_campaign as campaign, p4_full_budget_registration as registration
from transfer import p4_runtime_device as device
from transfer import p4_three_arm_contract as contract, p4_three_arm_fit as fit
from transfer import p4_three_arm_review as review

QUALIFICATION = os.environ.get('WP05_FIT_QUALIFICATION')
GPU = os.environ.get('WP05_FIT_GPU')
BASE_PROCESS_ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        PYTHONDONTWRITEBYTECODE='1')


@contextmanager
def fitting_schema(classes):
    with synthetic_schema(classes), patch.object(p4_exposure, 'CLASSES', classes):
        yield


def setup(root, classes, code, variant, *, gpus=(), qualifications=()):
    source = root / 'inputs'
    source.mkdir()
    parent, _, _ = fixture(source, classes=classes)
    prior = source / 'prior.json'
    write_json(prior, {'status': 'passed', 'query_accessed': False,
                       'protected_files': {str(parent): hash_file(parent)}})
    output = root / 'execution'
    scope = fit.prepare(parent, prior, root / 'registration', output, code, variant=variant, classes=classes,
                        gpus=gpus, qualifications=qualifications)
    return scope, output


def isolated_worker(classes, direct):
    def launch(command, logs, case, operation, locks):
        if direct:
            args = dict(zip(command[4::2], command[5::2]))
            return campaign.worker(Path(args['--scope']), args['--scope-sha256'], Path(args['--out']), case['id'],
                operation, int(args['--until']) if '--until' in args else None, code_snapshot(),
                lease_fds=tuple(stream.fileno() for stream in locks))
        wrapped = [*command[:2], 'test_p4_three_arm_fit', '--classes', str(classes), *command[3:]]
        return campaign.invoke(wrapped, logs, case, operation, locks)
    return launch


def run_fixture(root, classes, variant, *, gpu=None, qualification=None, direct=False):
    code = code_snapshot()
    with fitting_schema(classes):
        pins = {} if gpu is None else {'gpus': [gpu], 'qualifications': [qualification]}
        scope, execution = setup(root, classes, code, variant, **pins)
        checksum = hash_file(scope)
        launcher = isolated_worker(classes, direct)
        try:
            result = campaign.run(scope, checksum, execution, code, launcher=launcher)
        except RuntimeError as error:
            status = campaign.read(execution / 'status.json')
            raise RuntimeError(status['errors']) from error
        if result['status'] != 'completed':
            raise RuntimeError(result)
        receipt = campaign.accept(scope, checksum, execution, code)
        initial = {str(p): hash_file(p) for p in execution.rglob('*') if p.is_file()}
        def unexpected_dispatch(*args):
            raise AssertionError('Completed reentry dispatched a worker')

        repeat = campaign.run(scope, checksum, execution, code, resume=True, launcher=unexpected_dispatch)
        assert repeat['optimizer_updates_this_invocation'] == 0
        assert initial == {str(p): hash_file(p) for p in execution.rglob('*') if p.is_file()}
        analysis = None
        if variant != 'long_synthetic':
            path = review.review(scope, checksum, execution, receipt, hash_file(receipt), root / 'review', code)
            analysis = {'file': str(path), 'sha256': hash_file(path)}
        return {'scope': str(scope), 'scope_sha256': checksum, 'execution': str(execution),
                'acceptance': str(receipt), 'acceptance_sha256': hash_file(receipt),
                'review': analysis, 'updates': result['optimizer_updates'],
                'completed_resume_updates': repeat['optimizer_updates_this_invocation']}


class HongHuFittingTests(unittest.TestCase):
    def test_two_class_counts_complete_matrix_independent_and_readonly(self):
        for classes in (9, 22):
            with self.subTest(classes=classes), TemporaryDirectory() as directory:
                output = run_fixture(Path(directory), classes, 'bounded_synthetic', direct=True)
                evidence = campaign.read(Path(output['review']['file']))
                self.assertEqual((evidence['classes'], evidence['verified_training_updates']), (classes, 54))
                self.assertEqual((len(evidence['rows']), evidence['optimizer_updates']), (9, 0))
                self.assertIsNone(evidence['hypothesis_supported'])
                self.assertEqual(set(evidence['comparisons']), {'none', 'anchor_only'})
                for row in evidence['rows']:
                    self.assertEqual(set(row['observations']), set(contract.CONDITIONS))
                    self.assertEqual([(p['start'], p['until']) for p in row['fitting']['resume']],
                                     [(0, 2), (2, 3), (3, 6)])
                    self.assertEqual(row['fitting']['resume'][0]['after'], row['fitting']['resume'][1]['before'])
                    for condition in row['observations'].values():
                        for head in condition['heads'].values():
                            self.assertEqual(head['class_support'], [5]*classes)
                    self.assertEqual(row['parameter_audit']['schema'], 'p4_adam_named_step_evidence_v1')

    def test_scope_rejects_tamper_incomplete_dispatch_and_reused_output(self):
        code = code_snapshot()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scope, output = setup(root, 22, code, 'bounded_synthetic')
            checksum = hash_file(scope)
            for change in (lambda v: v['cases'].pop(),
                           lambda v: v['cases'][0].update(model_seed=31),
                           lambda v: v['cases'][0].update(arm='binary'),
                           lambda v: v['cases'][0].update(physical_uuid='GPU-11111111-1111-1111-1111-111111111111'),
                           lambda v: v['budget'].update(planned_optimizer_updates=180000)):
                with self.subTest(change=change):
                    bad = campaign.read(scope)
                    change(bad)
                    bad['scope_hash'] = digest({k: v for k, v in bad.items() if k != 'scope_hash'})
                    path = scope.parent / 'bad.json'
                    write_json(path, bad)
                    with self.assertRaises(ValueError):
                        campaign.run(path, hash_file(path), output, code)
            with self.assertRaisesRegex(ValueError, 'different registered execution'):
                campaign.run(scope, checksum, root / 'second-execution', code)
            with self.assertRaisesRegex(ValueError, 'fixed registered worker stops'):
                campaign.run(scope, checksum, output, code, segment=4)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                campaign.run(scope, checksum, output, code)
            with self.assertRaises(ValueError):
                registration.load(scope, '0'*64, code)
            with self.assertRaises(FileNotFoundError):
                review.review(scope, checksum, output, output/'acceptance.json', '0'*64, root/'review', code)
            self.assertFalse((root/'review').exists())
            prior_path = Path(campaign.read(scope)['prior_receipt'])
            command = [sys.executable, '-m', 'transfer.p4_full_budget_campaign', 'prepare',
                '--parent', str(root/'inputs/parent.json'), '--prior-receipt', str(prior_path),
                '--purpose', contract.PURPOSE, '--protocol', device.CPU, '--expected-commit', code['commit'],
                '--out', str(root/'legacy-unowned')]
            result = subprocess.run(command, capture_output=True, text=True, env=BASE_PROCESS_ENV)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn('invalid choice', result.stderr)
            self.assertFalse((root/'legacy-unowned').exists())

    def test_review_preserves_failed_primary_and_per_condition_protection(self):
        def observation(oa, recall):
            n = len(recall)
            return {name: {'heads': {'fused': {'oa': oa, 'aa': sum(recall)/n, 'recall': recall}}}
                    for name in contract.CONDITIONS}

        rows = []
        for seed in contract.SEEDS:
            rows.extend([
                {'case': {'model_seed': seed, 'arm': 'none'}, 'observations': observation(.50, [.5]*22)},
                {'case': {'model_seed': seed, 'arm': 'continuous'}, 'observations': observation(.55, [.5]*22)},
                {'case': {'model_seed': seed, 'arm': 'anchor_only'}, 'observations': observation(.55, [.5]*22)},
            ])
        value = review.paired_outcomes(rows)
        self.assertTrue(value['none']['passed'])
        self.assertFalse(value['anchor_only']['primary_passed'])
        rows[1]['observations']['full']['heads']['fused']['recall'][20] = 0.3
        value = review.paired_outcomes(rows)
        self.assertTrue(value['none']['primary_passed'])
        self.assertFalse(value['none']['passed'])
        self.assertIn(21, value['none']['conditions']['full']['small_classes_below_threshold'])
        with self.assertRaisesRegex(ValueError, 'complete nine-case'):
            review.paired_outcomes(rows[:-1])

    def test_resume_oracle_rejects_wrong_rng_and_checkpoint_chain(self):
        state = {'fingerprints': {'rng': '1'*64}}
        sequence = [0, 1, 2, 3]
        first = {'cursor': 0, 'rng_sha256': '0'*64, 'state_sha256': None, 'next_tile': 0}
        middle = {'cursor': 2, 'rng_sha256': '6'*64, 'state_sha256': '5'*64, 'next_tile': 2}
        last = {'cursor': 4, 'rng_sha256': '1'*64, 'state_sha256': 'a'*64, 'next_tile': None}
        receipts = [{'start': 0, 'until': 2, 'state_sha256': '5'*64, 'resume': {'before': first, 'after': middle}},
                    {'start': 2, 'until': 4, 'state_sha256': 'a'*64, 'resume': {'before': {**middle}, 'after': last}}]
        self.assertEqual(len(audit.verify_fit_resume(state, receipts, sequence)), 2)
        receipts[1]['resume']['before']['rng_sha256'] = '7'*64
        with self.assertRaisesRegex(ValueError, 'Broken fitting resume'):
            audit.verify_fit_resume(state, receipts, sequence)


def main():
    if sys.argv[1:2] == ['--classes']:
        classes = int(sys.argv[2])
        del sys.argv[1:3]
        with fitting_schema(classes):
            campaign.main()
        return
    if sys.argv[1:2] != ['fixture']:
        unittest.main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('fixture')
    parser.add_argument('--classes', type=int, choices=(9, 22), required=True)
    parser.add_argument('--variant', choices=('bounded_synthetic', 'long_synthetic'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    if code_snapshot()['commit'] != args.expected_commit or bool(GPU) != bool(QUALIFICATION):
        raise ValueError('Synthetic fixture requires the exact checkout and complete device assignment')
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    result = run_fixture(out, args.classes, args.variant, gpu=GPU, qualification=QUALIFICATION)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
