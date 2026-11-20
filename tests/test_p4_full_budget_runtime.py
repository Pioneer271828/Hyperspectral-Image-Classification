"""Exercise the registered P4 optimizer and acceptance with synthetic guarded labels."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from research_runtime import code_snapshot, hash_file, write_json
from tests.test_p4_coverage import fixture
from train_spatial import SpatialConfig
from transfer import p4_full_budget_audit as audit, p4_full_budget_campaign as campaign
from transfer import p4_full_budget_registration as registration, p4_runtime_device as device
from transfer.p4_spatial import initialize, run_case


GPU_A = 'GPU-11111111-1111-1111-1111-111111111111'
GPU_B = 'GPU-22222222-2222-2222-2222-222222222222'
GPU_C = 'GPU-33333333-3333-3333-3333-333333333333'


def parent_fixture(root):
    path = root / 'fixture'
    path.mkdir()
    parent, _, _ = fixture(path)
    prior = path / 'accepted.json'
    write_json(prior, {'status': 'passed', 'query_accessed': False,
                       'protected_files': {str(parent): hash_file(parent)}})
    return parent, prior


def prepare(root, parent, prior, code, *, protocol=device.CPU, gpus=(), steps=6, interval=3,
            purpose=registration.LEGACY):
    return registration.prepare(parent, hash_file(parent), prior, hash_file(prior), root, code,
                                protocol=protocol, gpus=gpus, synthetic=True, steps=steps, interval=interval,
                                purpose=purpose)


def own(path, scope):
    path.mkdir()
    for name in ('cases', 'logs'):
        (path / name).mkdir()
    write_json(path / 'owner.json', campaign.owner(path, scope, hash_file(scope)))


class FullBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.code = code_snapshot()
        cls.parent, cls.prior = parent_fixture(cls.root)
        cls.small = prepare(cls.root / 'small-scope', cls.parent, cls.prior, cls.code)
        cls.scope, cls.reg = registration.load(cls.small, hash_file(cls.small), cls.code)

    def test_model_seeds_pair_arms_and_keep_legacy_initialization(self):
        rows = self.scope['cases']
        self.assertEqual(len({row['id'] for row in rows}), 120)
        self.assertEqual(len({row['pair_sha256'] for row in rows}), 24)
        self.assertEqual({row['sampler_seed'] for row in rows}, {64})
        self.assertEqual({row['model_seed'] for row in rows}, {64, 512, 2048})
        self.assertEqual({row['method'] for row in rows}, {'scratch', 'shared_transfer'})
        for row in rows[:15]:
            _, rebuilt = registration.initialize_case(self.reg, row)
            self.assertEqual(rebuilt, row['initialization'])
        _, default = initialize(SpatialConfig(self.reg['config']), coverage_arm='none')
        self.assertEqual(default, rows[0]['initialization'])
        self.assertEqual(len({rows[5 * i]['initialization']['base_initial_state_sha256'] for i in range(3)}), 3)
        for cohort in self.scope['exposure'].values():
            self.assertLessEqual(cohort['first_all_classes_step'], 3)
            self.assertEqual(len(cohort['per_label_counts_at_events']['6']), cohort['support_label_count'])

    def test_cpu_full_1500_actual_updates_and_strict_multi_boundary_resume(self):
        path = prepare(self.root / 'long-scope', self.parent, self.prior, self.code, steps=1500, interval=30,
                       purpose=registration.AUDITED)
        sha = hash_file(path)
        scope, _ = registration.load(path, sha, self.code)
        case = next(row for row in scope['cases'] if row['method'] == 'shared_transfer'
                    and row['model_seed'] == 512 and row['arm'] == 'continuous')
        self.assertEqual(scope['budget']['planned_optimizer_updates'], 180000)
        self.assertEqual(scope['budget']['validation_steps'], list(range(30, 1501, 30)))
        roots = [self.root / name for name in ('long-full', 'long-split')]
        for out, stops in zip(roots, ((1500,), (2, 30, 660, 1500))):
            own(out, path)
            cursor = 0
            for stop in stops:
                proof = campaign.worker(path, sha, out, case['id'], 'advance', stop, self.code)
                self.assertEqual(proof['updates'], stop - cursor)
                cursor = stop
                folder = campaign.bound_folder(out, case)
                state = torch.load(folder / 'state.pt', weights_only=True)
                self.assertEqual(state['scheduler']['last_epoch'], stop // 30)
                self.assertEqual(state['best'] is None, stop < 30)
            self.assertEqual(campaign.worker(path, sha, out, case['id'], 'completed', None, self.code)['updates'], 0)
        states = [torch.load(campaign.bound_folder(out, case) / 'state.pt', weights_only=True) for out in roots]
        self.assertEqual(states[0]['fingerprints'], states[1]['fingerprints'])
        self.assertEqual(states[0]['runtime'], states[1]['runtime'])
        proof = campaign.worker(path, sha, roots[1], case['id'], 'audit', None, self.code)
        self.assertEqual(proof, {'status': 'accepted', 'case': case['id'], 'updates': 0})
        row = campaign.accepted(campaign.bound_folder(roots[1], case), scope, case, sha)
        self.assertEqual(row['optimizer_updates'], 1500)
        self.assertEqual(len(row['validation_steps']), 50)
        self.assertEqual(set(row['fit']), {'initial', 'selected', 'last'})
        self.assertEqual(len(row['observations']), 7)
        self.assertEqual(row['nonselected_event_probability_replays'], 0)
        self.assertTrue(all(parameter['adam_steps'] == 1500 for parameter in row['parameter_audit']['parameters']))

    def test_cuda_graph_operator_math_on_cpu(self):
        image = torch.linspace(-2, 3, 70, dtype=torch.float64).reshape(1, 2, 5, 7).requires_grad_()
        for size in ((2, 3), (1, 1), (5, 7)):
            self.compare_operator(device.adaptive_bin_mean(image, size),
                                  torch.nn.functional.adaptive_avg_pool2d(image, size), (image,))
        coordinates = torch.tensor([[0, 0, 2, 2, 3], [0, 1, 1, 1, 2]])
        weights = torch.linspace(-1, 1, 5, dtype=torch.float64).requires_grad_()
        features = torch.linspace(-2, 2, 12, dtype=torch.float64).reshape(3, 4).requires_grad_()
        graph = torch.sparse_coo_tensor(coordinates, weights, (4, 3))
        self.compare_operator(device.sparse_row_product(graph, features), torch.sparse.mm(graph, features),
                              (weights, features))

    def compare_operator(self, got, expected, inputs):
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
        sensitivity = torch.linspace(.1, 1.3, got.numel(), dtype=got.dtype).reshape_as(got)
        got_grad = torch.autograd.grad(got, inputs, sensitivity, retain_graph=True)
        expected_grad = torch.autograd.grad(expected, inputs, sensitivity)
        torch.testing.assert_close(got_grad, expected_grad, rtol=1e-12, atol=1e-12)

    def test_small_matrix_actual_acceptance_and_no_replay_after_completion(self):
        out = self.root / 'all-cases'
        calls = []

        def launch(command, logs, case, operation, locks):
            self.assertEqual(len(locks), 2)
            self.assertTrue(all(not stream.closed for stream in locks))
            args = dict(zip(command[4::2], command[5::2]))
            calls.append((case['id'], operation))
            return campaign.worker(self.small, hash_file(self.small), out, case['id'], operation,
                                   int(args['--until']) if '--until' in args else None, self.code,
                                   lease_fds=tuple(lock.fileno() for lock in locks))

        result = campaign.run(self.small, hash_file(self.small), out, self.code, launcher=launch)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['optimizer_updates'], 720)
        self.assertEqual(result['optimizer_updates_this_invocation'], 720)
        self.assertEqual(len(calls), 360)
        receipt = campaign.accept(self.small, hash_file(self.small), out, self.code)
        proof = campaign.read(receipt)
        self.assertEqual((proof['cases'], proof['pairs'], proof['optimizer_updates']), (120, 24, 720))
        old = hash_file(receipt)
        result = campaign.run(self.small, hash_file(self.small), out, self.code, resume=True, launcher=launch)
        self.assertEqual(result['optimizer_updates_this_invocation'], 0)
        self.assertEqual(len(calls), 360)
        self.assertEqual(hash_file(campaign.accept(self.small, hash_file(self.small), out, self.code)), old)

    def test_worker_cli_and_identity_isolation(self):
        out = self.root / 'cli-case'
        own(out, self.small)
        case = self.scope['cases'][0]
        command = [sys.executable, '-m', 'transfer.p4_full_budget_campaign', 'worker',
                   '--expected-commit', self.code['commit'], '--scope', str(self.small),
                   '--scope-sha256', hash_file(self.small), '--out', str(out),
                   '--case', case['id'], '--operation', 'advance', '--until', '2']
        rejected = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('--coordinator-fd', rejected.stderr)
        with device.lease(out / '.coordinator.lock') as coordinator, device.lease(out / '.cpu.lock') as slot:
            with self.assertRaisesRegex(ValueError, 'Worker lease does not match'):
                campaign.worker(self.small, hash_file(self.small), out, case['id'], 'advance', 2, self.code,
                                lease_fds=(coordinator.fileno(), coordinator.fileno()))
            result = campaign.invoke_operation(self.small, hash_file(self.small), out, self.code, case,
                                               'advance', 2, (coordinator, slot), campaign.invoke)
        self.assertEqual(result['updates'], 2)
        self.assertEqual(audit.ledger(campaign.bound_folder(out, case), case, hash_file(self.small))[0], 2)
        with self.assertRaisesRegex(ValueError, 'explicit runtime'):
            run_case(self.reg, 0, out / 'unowned', 2, self.code, coverage_arm='none')
        folder = campaign.bound_folder(out, case)
        _, calls = audit.ledger(folder, case, hash_file(self.small))
        runtime = {'scope_sha256': hash_file(self.small), 'case': case, 'numeric': calls[0]['numeric']}
        wrong = dict(runtime, numeric={**runtime['numeric'], 'protocol': device.CUDA})
        with self.assertRaisesRegex(ValueError, 'Resume identity'):
            run_case(self.reg, 0, folder, 3, self.code, coverage_arm='none', runtime=wrong)
        wrong = dict(runtime, case={**case, 'model_seed': 512})
        with self.assertRaisesRegex(ValueError, 'initialization'):
            run_case(self.reg, 0, folder, 3, self.code, coverage_arm='none', runtime=wrong)
        with self.assertRaisesRegex(ValueError, 'positive registered'):
            campaign.worker(self.small, hash_file(self.small), out, case['id'], 'advance', 7, self.code)
        with self.assertRaisesRegex(ValueError, 'completed full budget'):
            campaign.worker(self.small, hash_file(self.small), out, case['id'], 'audit', None, self.code)
        self.assertEqual(audit.ledger(folder, case, hash_file(self.small))[0], 2)

    def test_gpu_gate_wait_resume_failure_and_lease(self):
        for gpus in ((GPU_A, GPU_B), (GPU_A, GPU_B, GPU_C)):
            keys = registration.matrix(self.reg, device.CUDA, gpus)
            self.assertEqual({sum(row['physical_uuid'] == gpu for row in keys) for gpu in gpus}, {120 // len(gpus)})
            self.assertTrue(all(len({r['physical_uuid'] for r in keys[i:i+5]}) == 1 for i in range(0, 120, 5)))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            registration.matrix(self.reg, device.CUDA, (GPU_A, GPU_A))
        path = prepare(self.root / 'gpu-scope', self.parent, self.prior, self.code,
                       protocol=device.CUDA, gpus=(GPU_A, GPU_B))
        out = self.root / 'gpu-wait'
        sha = hash_file(path)
        calls = []

        def fail(*args):
            calls.append(args)
            raise RuntimeError('Synthetic worker failure')

        with patch.object(device, 'snapshot', return_value={'free_mib': 12287, 'utilization': 60}):
            wait = campaign.run(path, sha, out, self.code, launcher=fail)
        self.assertEqual(wait['status'], 'waiting_for_resources')
        self.assertEqual(wait['optimizer_updates'], 0)
        self.assertFalse(calls)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            campaign.accept(path, sha, out, self.code)
        with device.lease(out / '.coordinator.lock'):
            with self.assertRaises(device.ResourceUnavailable):
                campaign.run(path, sha, out, self.code, resume=True, launcher=fail)
        with patch.object(device, 'snapshot', side_effect=lambda gpu:
                          {'free_mib': 12288, 'utilization': 60 if gpu == GPU_A else 61}):
            with self.assertRaisesRegex(RuntimeError, 'stopped dispatch'):
                campaign.run(path, sha, out, self.code, resume=True, launcher=fail)
        self.assertEqual(len(calls), 1)
        self.assertEqual(campaign.read(out / 'status.json')['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'requires diagnosis'):
            campaign.run(path, sha, out, self.code, resume=True, launcher=fail)

    def test_uncommitted_or_tampered_state_fails_closed(self):
        out = self.root / 'uncommitted'
        own(out, self.small)
        case = self.scope['cases'][0]
        folder = campaign.bound_folder(out, case)
        folder.mkdir()
        (folder / 'state.pt').write_bytes(b'never accepted')
        with self.assertRaisesRegex(ValueError, 'Uncommitted'):
            campaign.worker(self.small, hash_file(self.small), out, case['id'], 'advance', 2, self.code)
        self.assertFalse((folder / 'result.json').exists())


@unittest.skipUnless(os.environ.get('P4_TEST_GPU_UUID') and os.environ.get('P4_TEST_PEER_UUID'),
                     'Native CUDA test requires a resource-admitted physical UUID')
class FullBudgetCudaTests(unittest.TestCase):
    def test_native_cuda_worker_reload_and_split(self):
        code = code_snapshot()
        gpu, peer = os.environ['P4_TEST_GPU_UUID'], os.environ['P4_TEST_PEER_UUID']
        with TemporaryDirectory() as temp:
            root = Path(temp)
            parent, prior = parent_fixture(root)
            scope_path = prepare(root / 'scope', parent, prior, code, protocol=device.CUDA_AUDITED, gpus=(gpu, peer),
                                 purpose=registration.AUDITED)
            sha = hash_file(scope_path)
            scope, _ = registration.load(scope_path, sha, code)
            case = next(r for r in scope['cases'] if r['physical_uuid'] == gpu and r['arm'] == 'continuous'
                        and r['method'] == 'shared_transfer' and r['model_seed'] == 512)
            roots = [root / 'full', root / 'split']
            campaign.slot_lock(roots[0], gpu).parent.mkdir()
            for out, stops in zip(roots, ((6,), (2, 3, 6))):
                own(out, scope_path)
                operations = [('advance', stop) for stop in stops] + [('completed', None)]
                if out == roots[1]:
                    operations.append(('audit', None))
                with device.lease(out / '.coordinator.lock') as coordinator, device.lease(campaign.slot_lock(out, gpu)) as slot:
                    cursor = 0
                    for operation, stop in operations:
                        try:
                            result = campaign.invoke_operation(scope_path, sha, out, code, case, operation, stop,
                                                               (coordinator, slot), campaign.invoke)
                        except device.ResourceUnavailable:
                            raise
                        except RuntimeError as error:
                            logs = '\n'.join(path.read_text() for path in sorted((out / 'logs').glob('*.log')))
                            self.fail(f'{error}\n{logs}')
                        self.assertEqual(result['updates'], stop - cursor if stop is not None else 0)
                        if stop is not None:
                            cursor = stop
            states = [torch.load(campaign.bound_folder(out, case) / 'state.pt', map_location='cpu', weights_only=True)
                      for out in roots]
            self.assertIsNotNone(states[0]['rng']['cuda'])
            self.assertEqual(states[0]['fingerprints'], states[1]['fingerprints'])
            receipt = campaign.accepted(campaign.bound_folder(roots[1], case), scope, case, sha)
            self.assertEqual(receipt['audit_optimizer_updates'], 0)
            self.assertIn('parameter_audit', receipt)
