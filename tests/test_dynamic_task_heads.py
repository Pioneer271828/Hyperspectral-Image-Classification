"""Dynamic task contracts; set DYNAMIC_HEAD_DEVICE=cuda for remote GPU checks."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from model.VCASAHGCN import (
    DomainSpecificBatchNorm1d, DomainSpecificBatchNorm2d, MultiDatasetVCASAHGCN,
)
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint


torch.set_num_threads(1)
FAMILIES = ('heads', 'hgnn_heads', 'cnn_heads')
DOMAIN_NORMS = (DomainSpecificBatchNorm1d, DomainSpecificBatchNorm2d)


class DynamicTaskTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.empty(0, device=os.environ.get('DYNAMIC_HEAD_DEVICE', 'cpu')).device
        torch.manual_seed(29)

    def model(self, norm='domain_bn', graph='learned', counts=None):
        return MultiDatasetVCASAHGCN(
            {'source': 3} if counts is None else counts,
            latent_channels=8, adapter_embed_dim=8, locality_window=None,
            norm_type=norm, graph_mode=graph,
        ).to(self.device)

    def forward(self, model, task='target'):
        x = torch.randn(4, 5, 8, device=self.device)
        kwargs = {'wavelengths': torch.linspace(450, 950, 6, device=self.device),
                  'superpixel_scale': 4, 'return_logits': True}
        if model.backbone.graph_mode == 'static':
            q = nn.functional.one_hot(torch.arange(20, device=self.device) % 5, 5).float()
            kwargs.update(Q=q.to_sparse().coalesce(),
                          A=torch.eye(5, device=self.device).to_sparse().coalesce())
        return model(task, x, **kwargs)

    def assert_predictions_close(self, actual, expected, graph):
        # CUDA learned-graph reductions can vary on an unchanged model. State
        # tensors are still compared bitwise; only these predictions use a
        # float32 tolerance (unchanged-model probe: max absolute delta 1.5e-8).
        tolerant = self.device.type == 'cuda' and graph == 'learned'
        torch.testing.assert_close(actual, expected, rtol=1e-6 if tolerant else 0,
                                   atol=1e-7 if tolerant else 0)

    def test_heads_inherit_placement_dtype_and_model_mode(self):
        for norm in ('domain_bn', 'batch_norm', 'group_norm'):
            for dtype in (torch.float32, torch.float64):
                for training in (True, False):
                    with self.subTest(norm=norm, dtype=dtype, training=training):
                        model = self.model(norm=norm).to(dtype=dtype).train(training)
                        model.add_dataset_head('target', 4)
                        features = torch.randn(7, 64, device=self.device, dtype=dtype)
                        for name in FAMILIES:
                            head = getattr(model, name)['target']
                            self.assertEqual(head.weight.device, self.device)
                            self.assertEqual(head.weight.dtype, dtype)
                            self.assertEqual(head.training, training)
                        outputs = model.predict_features('target', features, features, features, True)
                        for output in outputs:
                            self.assertEqual(output.shape, (7, 4))
                            self.assertTrue(torch.isfinite(output).all())

    def test_domain_norms_inherit_state_and_keep_fresh_statistics(self):
        for kind, shape in ((DomainSpecificBatchNorm1d, (7, 4)),
                            (DomainSpecificBatchNorm2d, (2, 4, 3, 3))):
            for dtype in (torch.float32, torch.float64):
                for training in (True, False):
                    with self.subTest(kind=kind.__name__, dtype=dtype, training=training):
                        wrapper = kind(4, ['source']).to(device=self.device, dtype=dtype).train(training)
                        old = wrapper.norms['source']
                        with torch.no_grad():
                            old.weight.fill_(1.7)
                            old.bias.fill_(-.5)
                            old.running_mean.fill_(3.)
                            old.running_var.fill_(2.)
                            old.num_batches_tracked.fill_(9)
                        saved = copy.deepcopy(old.state_dict())
                        wrapper.add_dataset('target')
                        added = wrapper.norms['target']
                        self.assertEqual(added.training, training)
                        self.assertEqual(added.weight.device, self.device)
                        self.assertEqual(added.weight.dtype, dtype)
                        reference_kind = nn.BatchNorm1d if kind is DomainSpecificBatchNorm1d else nn.BatchNorm2d
                        reference = reference_kind(4).to(device=self.device, dtype=dtype).train(training)
                        for key, expected in reference.state_dict().items():
                            torch.testing.assert_close(added.state_dict()[key], expected, rtol=0, atol=0)
                        x = torch.randn(*shape, device=self.device, dtype=dtype, requires_grad=True)
                        actual = wrapper(x, 'target')
                        torch.testing.assert_close(actual, reference(x))
                        (actual * torch.randn_like(actual)).sum().backward()
                        self.assertGreater(added.weight.grad.abs().sum().item(), 0)
                        self.assertIsNone(old.weight.grad)
                        self.assertEqual(int(added.num_batches_tracked), int(training))
                        for key, expected in saved.items():
                            torch.testing.assert_close(old.state_dict()[key], expected, rtol=0, atol=0)
                        wrapper.add_dataset('target')
                        self.assertIs(wrapper.norms['target'], added)

    def test_mixed_training_modes_and_existing_state_are_preserved(self):
        model = self.model().train()
        model.backbone.eval()
        model.backbone.CNN_denoise[0].norm.norm.train()
        before = copy.deepcopy(model.state_dict())
        modes = {name: module.training for name, module in model.named_modules()}
        model.add_dataset_head('target', 4)
        for name in FAMILIES:
            self.assertTrue(getattr(model, name)['target'].training)
        for name, module in model.named_modules():
            if isinstance(module, DOMAIN_NORMS):
                self.assertEqual(module.norms['target'].training, modes[name])
            if name in modes:
                self.assertEqual(module.training, modes[name], name)
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0, msg=name)

    def test_empty_head_families_use_backbone_placement(self):
        model = self.model(counts={}).double().eval()
        model.add_dataset_head('target', 4)
        features = torch.randn(5, 64, device=self.device, dtype=torch.float64)
        outputs = model.predict_features('target', features, features, features)
        for name, output in zip(FAMILIES, outputs):
            head = getattr(model, name)['target']
            self.assertEqual(head.weight.device, self.device)
            self.assertEqual(head.weight.dtype, torch.float64)
            self.assertFalse(head.training)
            torch.testing.assert_close(output.sum(1), torch.ones(5, device=self.device, dtype=torch.float64))

    def test_each_head_family_uses_its_own_placement(self):
        model = self.model().eval()
        model.hgnn_heads.double()
        model.add_dataset_head('target', 4)
        for name in FAMILIES:
            family = getattr(model, name)
            self.assertEqual(family['target'].weight.device, family['source'].weight.device)
            self.assertEqual(family['target'].weight.dtype, family['source'].weight.dtype)

    def test_eval_full_forward_keeps_source_outputs_and_bn_statistics(self):
        for graph in ('static', 'learned'):
            with self.subTest(graph=graph):
                model = self.model(graph=graph).eval()
                # Repeat the same generated input before/after registration.
                torch.manual_seed(89)
                with torch.no_grad():
                    before_output = self.forward(model, 'source')
                before = copy.deepcopy(model.state_dict())
                model.add_dataset_head('target', 4)
                torch.manual_seed(89)
                with torch.no_grad():
                    after_output = self.forward(model, 'source')
                for actual, expected in zip(after_output, before_output):
                    self.assert_predictions_close(actual, expected, graph)
                for name, value in before.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0, msg=name)
                target_state = {name: value.clone() for name, value in model.named_buffers()
                                if '.target.' in name}
                outputs = self.forward(model)
                self.assertTrue(all(output.shape == (20, 4) for output in outputs))
                self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
                for name, expected in target_state.items():
                    torch.testing.assert_close(dict(model.named_buffers())[name], expected, rtol=0, atol=0, msg=name)

    def test_explicit_optimizer_enrollment_updates_new_heads_and_norms(self):
        for graph in ('static', 'learned'):
            with self.subTest(graph=graph):
                model = self.model(graph=graph).train()
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                enrolled = {id(p) for group in optimizer.param_groups for p in group['params']}
                model.add_dataset_head('target', 4)
                added = {name: p for name, p in model.named_parameters() if id(p) not in enrolled}
                self.assertTrue(any('norms.target.' in name for name in added))
                for name in FAMILIES:
                    self.assertIn(name + '.target.weight', added)
                optimizer.add_param_group({'params': list(added.values())})
                before = {name: p.detach().clone() for name, p in added.items()}
                source = {name: p.detach().clone() for name, p in model.named_parameters() if '.source.' in name}
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.forward(model)
                    labels = torch.arange(20, device=self.device) % 4
                    sum(nn.functional.cross_entropy(output, labels) for output in outputs).backward()
                    for name, p in added.items():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    optimizer.step()
                for name in FAMILIES:
                    key = name + '.target.weight'
                    self.assertGreater((added[key] - before[key]).abs().max().item(), 0, key)
                self.assertTrue(any((p - before[name]).abs().max().item() > 0
                                    for name, p in added.items() if 'norms.target.' in name))
                for name, expected in source.items():
                    torch.testing.assert_close(dict(model.named_parameters())[name], expected, rtol=0, atol=0, msg=name)

    def test_duplicate_task_is_rejected_without_mutation(self):
        model = self.model().eval()
        model.add_dataset_head('target', 4)
        before = copy.deepcopy(model.state_dict())
        config = copy.deepcopy(model.export_config())
        with self.assertRaises(ValueError):
            model.add_dataset_head('target', 6)
        self.assertEqual(config, model.export_config())
        for name, expected in before.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0, msg=name)

    def test_dynamic_checkpoint_restores_both_tasks_and_all_norms(self):
        for graph in ('static', 'learned'):
            with self.subTest(graph=graph), tempfile.TemporaryDirectory() as directory:
                model = self.model(graph=graph).eval()
                model.add_dataset_head('target', 4)
                path = Path(directory) / 'dynamic.pt'
                save_model_checkpoint(model, path)
                restored = load_model_checkpoint(path, device=self.device)
                self.assertEqual(restored.export_config(), model.export_config())
                self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0, msg=name)
                for task in ('source', 'target'):
                    torch.manual_seed(91)
                    with torch.no_grad():
                        expected = self.forward(model, task)
                    torch.manual_seed(91)
                    with torch.no_grad():
                        actual = self.forward(restored, task)
                    for a, b in zip(actual, expected):
                        self.assert_predictions_close(a, b, graph)


if __name__ == '__main__':
    unittest.main()
