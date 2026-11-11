"""Test named Adam evidence and the isolated, paired HongHu admission contract."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.research_config import digest
from research_runtime import code_snapshot, hash_file
from tests.test_p4_full_budget_runtime import GPU_A, GPU_B, own, parent_fixture, prepare
from transfer import p4_cuda_qualification as qualification, p4_full_budget_campaign as campaign
from transfer import p4_full_budget_registration as registration, p4_runtime_device as device
from transfer import p4_update_evidence as updates


class AdamEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Module()
        self.model.backbone = nn.Module()
        self.model.backbone.coverage_modulation = nn.Linear(2, 1, bias=False)
        self.model.backbone.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
        self.model.heads = nn.ParameterDict({'one': nn.Parameter(torch.ones(2)), 'two': nn.Parameter(torch.ones(2))})
        optimizer = self.optimizer()
        self.rows = updates.layout(self.model, optimizer)
        self.layout_sha = digest(self.rows)
        updates.bind_names(optimizer, self.rows)
        recorder = updates.Recorder(self.model, optimizer, self.rows)
        trace = []
        coverage = self.model.backbone.coverage_modulation.weight
        for step, grad in enumerate((torch.zeros_like(coverage), None, torch.ones_like(coverage)), 1):
            optimizer.zero_grad(set_to_none=True)
            coverage.grad = grad
            self.model.heads.one.grad = torch.full_like(self.model.heads.one, .1 * step)
            self.model.heads.two.grad = torch.full_like(self.model.heads.two, -.2 * step)
            before = coverage.detach().clone()
            optimizer.step()
            trace.append({'step': step, 'parameter_flags': recorder.take_step(),
                          'coverage': {'usable_anchors': int(grad is not None), 'parameters': {
                              'backbone.coverage_modulation.weight': {
                                  'gradient_l1': None if grad is None else float(grad.abs().sum()),
                                  'changed_elements': int((before != coverage.detach()).sum())}}}})
        self.state = {'cursor': 3, 'trace': trace, 'optimizer': optimizer.state_dict(),
                      'parameter_audit': recorder.snapshot()}
        self.coverage_index = next(row['index'] for row in self.rows if row['category'] == 'coverage')
        self.head_indices = [row['index'] for row in self.rows if row['category'] == 'target_heads']

    def optimizer(self):
        return torch.optim.Adam([{'params': self.model.backbone.coverage_modulation.parameters()},
                                 {'params': self.model.heads.parameters(), 'lr': .03}], lr=.01, weight_decay=.1)

    def check(self, state):
        return updates.check(self.model, self.optimizer(), state, self.layout_sha)

    def test_none_and_zero_gradient_have_different_adam_participation(self):
        result = self.check(self.state)
        self.assertEqual(result['by_category']['coverage'], {
            'parameters': 1, 'gradient_present': 2, 'gradient_zero': 1, 'gradient_none': 1, 'parameter_changed': 2})
        frozen, = [row for row in result['parameters'] if not row['trainable']]
        self.assertEqual((frozen['participation'], frozen['adam_steps']), ('frozen_by_configuration', 0))

    def test_missing_and_wrong_number_of_moments_fail_independently(self):
        state = deepcopy(self.state)
        del state['optimizer']['state'][self.head_indices[0]]
        with self.assertRaisesRegex(ValueError, 'Missing or unexpected'):
            self.check(state)
        for step in (1, 4):
            state = deepcopy(self.state)
            state['optimizer']['state'][self.head_indices[0]]['step'].fill_(step)
            with self.assertRaisesRegex(ValueError, 'Adam step differs'):
                self.check(state)

    def test_swapped_parameter_moments_names_indices_and_groups_fail(self):
        a, b = self.head_indices
        state = deepcopy(self.state)
        moments = state['optimizer']['state']
        moments[a], moments[b] = moments[b], moments[a]
        with self.assertRaisesRegex(ValueError, 'Named Adam moment binding'):
            self.check(state)
        for field in ('param_names', 'params'):
            state = deepcopy(self.state)
            state['optimizer']['param_groups'][1][field].reverse()
            with self.assertRaisesRegex(ValueError, 'group, parameter index or name'):
                self.check(state)
        state = deepcopy(self.state)
        state['optimizer']['param_groups'].reverse()
        with self.assertRaisesRegex(ValueError, 'group, parameter index or name'):
            self.check(state)

    def test_negative_second_moment_and_resume_counter_drift_fail(self):
        state = deepcopy(self.state)
        state['optimizer']['state'][self.coverage_index]['exp_avg_sq'].fill_(-1)
        with self.assertRaisesRegex(ValueError, 'Negative Adam second moment'):
            self.check(state)
        state = deepcopy(self.state)
        state['parameter_audit']['counts'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'Cumulative Adam participation'):
            self.check(state)
        state = deepcopy(self.state)
        state['trace'][1]['step'] = 1
        with self.assertRaisesRegex(ValueError, 'aligned to the optimizer cursor'):
            self.check(state)


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.code = code_snapshot()
        cls.parent, cls.prior = parent_fixture(cls.root)
        cls.scope_path = prepare(cls.root / 'paired', cls.parent, cls.prior, cls.code, steps=60, interval=30,
                                 purpose=registration.ADMISSION)
        cls.sha = hash_file(cls.scope_path)
        cls.scope, cls.reg = registration.load(cls.scope_path, cls.sha, cls.code)

    def test_registered_roles_and_both_gpus_before_first_dispatch(self):
        cases = self.scope['cases']
        self.assertEqual((len(cases), self.scope['budget']['planned_optimizer_updates']), (20, 2400))
        self.assertEqual({row['support_seed'] for row in cases}, {2048})
        self.assertEqual({row['model_seed'] for row in cases}, {512})
        self.assertEqual(self.scope['paths'], {'full': [60], 'split': [2, 30, 60]})
        with self.assertRaisesRegex(ValueError, 'two registered CUDA'):
            registration.check_budget(registration.ADMISSION, device.CPU, [], False, 60, 30)
        rows = registration.matrix(self.reg, device.CUDA_V2, [GPU_A, GPU_B], registration.ADMISSION)
        for gpu in (GPU_A, GPU_B):
            assigned = [row for row in rows if row['physical_uuid'] == gpu]
            self.assertEqual(len(assigned), 10)
            self.assertEqual({row['k'] for row in assigned}, {1, 5})
            self.assertEqual({row['method'] for row in assigned}, {'scratch', 'shared_transfer'})
        path = prepare(self.root / 'synthetic-cuda-pair', self.parent, self.prior, self.code, steps=60, interval=30,
                       purpose=registration.ADMISSION, protocol=device.CUDA_V2, gpus=(GPU_A, GPU_B))
        sha = hash_file(path)
        scope, reg = registration.load(path, sha, self.code)
        reported = {GPU_A: {'free_mib': 12288, 'utilization': 60}, GPU_B: {'free_mib': 12287, 'utilization': 0}}
        with (patch.object(registration, 'load', return_value=({**scope, 'synthetic': False}, reg)),
              patch.object(device, 'simultaneous_inventory', return_value=reported)):
            status = campaign.run(path, sha, self.root / 'blocked', self.code,
                                  launcher=lambda *args: self.fail('Worker dispatched before both GPUs qualified'))
        self.assertEqual((status['status'], status['optimizer_updates']), ('waiting_for_resources', 0))

    def test_actual_paired_paths_hold_exact_checkpoint_and_nominal_audit(self):
        out = self.root / 'paired-execution'
        own(out, self.scope_path)
        case = next(row for row in self.scope['cases'] if row['arm'] == 'continuous')
        with self.assertRaisesRegex(ValueError, 'registered paired stop'):
            campaign.worker(self.scope_path, self.sha, out, case['id'], 'advance', 3, self.code, path_name='split')
        for name, stops in self.scope['paths'].items():
            before = 0
            for stop in stops:
                result = campaign.worker(self.scope_path, self.sha, out, case['id'], 'advance', stop,
                                         self.code, path_name=name)
                self.assertEqual(result['updates'], stop-before)
                before = stop
            result = campaign.worker(self.scope_path, self.sha, out, case['id'], 'completed', None,
                                     self.code, path_name=name)
            self.assertEqual(result['updates'], 0)
        self.assertEqual(campaign.worker(self.scope_path, self.sha, out, case['id'], 'audit', None,
                                         self.code)['updates'], 0)
        row = campaign.accepted(campaign.bound_folder(out, case), self.scope, case, self.sha)
        self.assertEqual(row['optimizer_updates'], 120)
        self.assertEqual(row['exact_path_fingerprints']['full'], row['exact_path_fingerprints']['split'])
        self.assertEqual(len(row['observations']), 1)
        self.assertEqual({name: len(phases['advance']) for name, phases in row['synchronized_timing'].items()},
                         {'full': 1, 'split': 3})
        with self.assertRaisesRegex(ValueError, 'Accepted case cannot be rerun'):
            campaign.worker(self.scope_path, self.sha, out, case['id'], 'advance', 2, self.code, path_name='split')

    def test_real_geometry_math_references_and_tolerances(self):
        config = {'tile_size': 32, 'model': {'learned_assignment_radius': 1, 'learned_graph_radius': 1}}
        plan = {'shape': [940, 475], 'plan_hash': 'synthetic_geometry_contract'}
        roles = {'train': [{'box': [0, 32, 0, 32]}, {'box': [32, 46, 0, 32]}],
                 'val': [{'box': [48, 80, 0, 32]}]}
        contract = qualification.geometry_contract(plan, roles, config)
        self.assertIn([12, 27], contract['shapes'])
        self.assertIn([14, 32], contract['shapes'])
        with device.arithmetic(device.CPU):
            results, matrices = qualification.run_fixtures(contract, 'cpu')
        self.assertEqual(list(results), qualification.fixture_specs(contract))
        self.assertEqual(matrices['32x32/assignment']['shape'], [1024, 16])
        self.assertEqual(matrices['32x32/adjacency128']['channels'], 128)
        self.assertTrue(all(proof['compatible'] for row in results.values()
                            for output in row.values() for proof in output.values()))
