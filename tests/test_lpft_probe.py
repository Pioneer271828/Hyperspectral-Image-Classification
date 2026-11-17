"""CPU-only short probes. Synthetic correctness is not HSI accuracy evidence."""
import copy
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from lpft_probe import LPFTPlan, phase_binding, start_phase


class TinyThreeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Tanh())
        self.heads = nn.Linear(6, 3)
        self.hgnn_heads = nn.Linear(6, 3)
        self.cnn_heads = nn.Linear(6, 3)

    def forward(self, inputs):
        value = self.backbone(inputs)
        return [head(value) for head in (self.heads, self.hgnn_heads, self.cnn_heads)]


class PhaseProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(251)
        self.model = TinyThreeHead().double()
        self.x = torch.randn(18, 4, dtype=torch.float64)
        self.y = torch.arange(18) % 3
        self.plan = LPFTPlan(12, 4, head_lr=0.03, backbone_lr=0.003)

    def loss(self, model):
        return sum(nn.functional.cross_entropy(o, self.y) for o in model(self.x)) / 3

    def update(self, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = self.loss(model)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def assert_tensors_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), name)

    def test_plan_rejects_invalid_budgets_and_rates(self):
        for total, probe in ((True, 1), (12, True), (0, 0), (12, 0), (12, 12), (12, 13)):
            with self.assertRaises(ValueError):
                LPFTPlan(total, probe, 0.03, 0.003)
        for head, body, decay in ((float('nan'), .1, 0), (.1, float('inf'), 0),
                                  (.1, .2, 0), (.1, 0, 0), (.1, .01, -1), (True, .01, 0)):
            with self.assertRaises(ValueError):
                LPFTPlan(12, 4, head, body, decay)

    def test_cursor_and_budget_include_both_phases(self):
        self.assertEqual([self.plan.phase(k) for k in (0, 3, 4, 11, 12)],
                         ['probe', 'probe', 'finetune', 'finetune', 'complete'])
        for cursor in (-1, 13, True, 1.0):
            with self.assertRaises(ValueError):
                self.plan.phase(cursor)
        self.assertEqual(phase_binding(self.model, self.plan, 12)['trainable_names'], [])

    def test_probe_changes_heads_and_preserves_backbone_and_buffers(self):
        before = copy.deepcopy(self.model.state_dict())
        optimizer, binding = start_phase(self.model, self.plan, 0)
        for _ in range(4):
            self.update(self.model, optimizer)
        after = self.model.state_dict()
        for name in before:
            if name.startswith('backbone.'):
                self.assertTrue(torch.equal(before[name], after[name]), name)
        for prefix in ('heads.', 'hgnn_heads.', 'cnn_heads.'):
            self.assertTrue(any(not torch.equal(before[n], after[n]) for n in before if n.startswith(prefix)))
        self.assertFalse(any(n.startswith('backbone.') for n in binding['trainable_names']))
        self.assertFalse(any(p.grad is not None for p in self.model.backbone.parameters()))

    def test_transition_preserves_weights_resets_adam_and_sets_small_backbone_lr(self):
        optimizer, _ = start_phase(self.model, self.plan, 0)
        for _ in range(4):
            self.update(self.model, optimizer)
        before = copy.deepcopy(self.model.state_dict())
        self.assertTrue(optimizer.state)
        fine, _ = start_phase(self.model, self.plan, 4)
        self.assertFalse(fine.state)
        self.assert_tensors_equal(before, self.model.state_dict())
        self.assertEqual([g['lr'] for g in fine.param_groups], [.03, .003])
        self.update(self.model, fine)
        self.assertFalse(torch.equal(before['backbone.0.weight'], self.model.backbone[0].weight))
        for name, buffer in self.model.named_buffers():
            self.assertTrue(torch.equal(before[name], buffer), name)

    def test_twelve_updates_match_independent_manual_policy(self):
        reference = copy.deepcopy(self.model)
        actual_optimizer = reference_optimizer = None
        for cursor in range(12):
            if cursor in (0, 4):
                actual_optimizer, _ = start_phase(self.model, self.plan, cursor)
                reference.eval()
                for name, parameter in reference.named_parameters():
                    parameter.grad = None
                    parameter.requires_grad_(cursor == 4 or not name.startswith('backbone.'))
                head_params = list(reference.heads.parameters()) + list(reference.hgnn_heads.parameters()) + list(reference.cnn_heads.parameters())
                groups = [{'params': head_params, 'lr': .03}]
                if cursor == 4:
                    groups.append({'params': list(reference.backbone.parameters()), 'lr': .003})
                reference_optimizer = torch.optim.Adam(groups, weight_decay=0, betas=(.9, .999), eps=1e-8, amsgrad=False)
            self.assertEqual(self.update(self.model, actual_optimizer), self.update(reference, reference_optimizer))
        self.assert_tensors_equal(self.model.state_dict(), reference.state_dict())

    def test_boundary_restart_reconstructs_same_finetune_trajectory(self):
        optimizer, binding = start_phase(self.model, self.plan, 0)
        for _ in range(4):
            self.update(self.model, optimizer)
        stream = io.BytesIO()
        torch.save({'model': self.model.state_dict(), 'completed_steps': 4}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        resumed = TinyThreeHead().double()
        resumed.load_state_dict(saved['model'], strict=True)
        left, left_binding = start_phase(self.model, self.plan, 4)
        right, right_binding = start_phase(resumed, self.plan, saved['completed_steps'])
        self.assertEqual(left_binding, right_binding)
        self.assertNotEqual(binding['sha256'], left_binding['sha256'])
        for _ in range(8):
            self.assertEqual(self.update(self.model, left), self.update(resumed, right))
        self.assert_tensors_equal(self.model.state_dict(), resumed.state_dict())

    def test_invalid_phase_entry_does_not_mutate_model(self):
        before = copy.deepcopy(self.model.state_dict())
        flags = [p.requires_grad for p in self.model.parameters()]
        for cursor in (1, 3, 5, 12, -1, True):
            with self.assertRaises(ValueError):
                start_phase(self.model, self.plan, cursor)
            self.assert_tensors_equal(before, self.model.state_dict())
            self.assertTrue(self.model.training)
            self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])

    def test_unknown_and_aliased_parameters_are_rejected_before_mutation(self):
        self.model.extra = nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError, 'Unknown parameter'):
            start_phase(self.model, self.plan, 0)
        self.assertTrue(self.model.training)
        del self.model.extra
        self.model.heads.weight = self.model.cnn_heads.weight
        with self.assertRaisesRegex(ValueError, 'Aliased'):
            start_phase(self.model, self.plan, 0)
        self.assertTrue(self.model.training)

    def test_binding_is_readonly_deterministic_and_policy_sensitive(self):
        before = copy.deepcopy(self.model.state_dict())
        binding = phase_binding(self.model, self.plan, 0)
        self.assertEqual(binding, phase_binding(self.model, self.plan, 3))
        other = phase_binding(self.model, LPFTPlan(12, 4, .03, .001), 0)
        self.assertNotEqual(binding['sha256'], other['sha256'])
        self.assert_tensors_equal(before, self.model.state_dict())
        self.assertTrue(self.model.training)

    def test_existing_g4_network_accepts_two_stage_policy_on_synthetic_support(self):
        from model.research_config import ResearchConfig
        from target_initialization import initialize_target, METHODS
        from target_training import loss_terms, selected_outputs, target_settings
        from test_target_training import fixture
        with tempfile.TemporaryDirectory() as directory:
            (items, _, guard), _ = fixture(Path(directory))
            model, _ = initialize_target('xuzhou', METHODS[0], model_seed=64, head_seed=1064)
            config = ResearchConfig.from_dict(target_settings('xuzhou', METHODS[0])['config'])
            plan = LPFTPlan(6, 2, .001, .0001)
            initial = copy.deepcopy(model.state_dict())
            optimizer, _ = start_phase(model, plan, 0)
            for cursor in range(6):
                if cursor == 2:
                    for name, value in model.state_dict().items():
                        if name.startswith('backbone.'):
                            self.assertTrue(torch.equal(initial[name], value), name)
                    optimizer, _ = start_phase(model, plan, cursor)
                optimizer.zero_grad(set_to_none=True)
                outputs = selected_outputs(model, 'xuzhou', items['xuzhou'], 'train')
                loss = loss_terms(outputs, items['xuzhou']['train_labels'], config)['total']
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
            final = model.state_dict()
            for prefix in ('backbone.', 'heads.', 'hgnn_heads.', 'cnn_heads.'):
                self.assertTrue(any(not torch.equal(initial[n], final[n])
                                    for n, _ in model.named_parameters() if n.startswith(prefix)), prefix)
            for name, buffer in model.named_buffers():
                self.assertTrue(torch.equal(initial[name], buffer), name)
            guard()


if __name__ == '__main__':
    unittest.main()
