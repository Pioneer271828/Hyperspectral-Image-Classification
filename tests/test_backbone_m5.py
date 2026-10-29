"""M5 stage scheduling and compatibility with the frozen pre-refactor model."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCN
from model.learnable_superpixel_graph import LearnableSuperpixelGraph
from model.region_pixel_alignment import pool_regions
from model.research_config import ResearchConfig
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint

BASE = 'b9ef4c98eea3a1df6f4ac4deebc2767fe94050cd'
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def make_model(**kwargs):
    return MultiDatasetVCASAHGCN({'a': 3, 'b': 5}, latent_channels=8, adapter_embed_dim=8,
                                locality_window=None, **kwargs)


def static_graph():
    Q = torch.nn.functional.one_hot(torch.arange(42) % 6, 6).float().to_sparse().coalesce()
    A = (torch.eye(6) + torch.roll(torch.eye(6), 1, 1)) / 2
    return Q, A


def legacy_module():
    module = types.ModuleType('legacy_backbone_m5')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':model/VCASAHGCN.py'], text=True)
    exec(compile(source, 'legacy_backbone_m5', 'exec'), module.__dict__)
    return module


class BackboneStageTests(unittest.TestCase):
    def test_legacy_outputs_gradients_parameters_and_bn_buffers(self):
        legacy = legacy_module()
        presets = ({}, {'interaction_mode': 'gated'}, {'graph_update_mode': 'graph'},
                   {'interaction_mode': 'gated', 'graph_update_mode': 'joint'},
                   {'interaction_mode': 'gated', 'interaction_strength': 0.})
        for norm in ('group_norm', 'domain_bn', 'batch_norm'):
            for graph in ('static', 'learned'):
                for options in presets:
                    with self.subTest(norm=norm, graph=graph, options=options):
                        torch.manual_seed(2026)
                        reference = legacy.MultiDatasetVCASAHGCN(
                            {'a': 3, 'b': 5}, latent_channels=8, adapter_embed_dim=8,
                            locality_window=None, norm_type=norm, graph_mode=graph, **options)
                        current = make_model(norm_type=norm, graph_mode=graph, **options)
                        current.load_state_dict(reference.state_dict(), strict=True)
                        self.assertEqual(set(reference.state_dict()), set(current.state_dict()))
                        self.assertEqual(sum(p.numel() for p in reference.parameters()), sum(p.numel() for p in current.parameters()))
                        for dataset, bands in (('a', 12), ('b', 17)):
                            original_x = torch.randn(6, 7, bands + 2, requires_grad=True)
                            current_x = original_x.detach().clone().requires_grad_()
                            wave = torch.linspace(450, 950, bands)
                            q, a = static_graph() if graph == 'static' else (None, None)
                            reference.zero_grad(set_to_none=True)
                            current.zero_grad(set_to_none=True)
                            expected = reference(dataset, original_x, q, a, wavelengths=wave, superpixel_scale=9, return_graph_info=True)
                            actual = current(dataset, current_x, q, a, wavelengths=wave, superpixel_scale=9, return_graph_info=True)
                            for old, new in zip(expected[:3], actual[:3]):
                                torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-6)
                            for outputs in (expected, actual):
                                loss = sum(-value[:, 0].clamp_min(1e-8).log().mean() for value in outputs[:3])
                                loss += sum(value for key, value in outputs[3].items() if key.endswith('_loss')) * .001
                                loss.backward()
                            torch.testing.assert_close(original_x.grad, current_x.grad, rtol=2e-4, atol=2e-6)
                            for (name, old), (_, new) in zip(reference.named_parameters(), current.named_parameters()):
                                if old.grad is None:
                                    self.assertIsNone(new.grad, name)
                                else:
                                    torch.testing.assert_close(old.grad, new.grad, rtol=2e-4, atol=2e-6, msg=name)
                            for name, old in reference.named_buffers():
                                torch.testing.assert_close(old, dict(current.named_buffers())[name], rtol=0, atol=0, msg=name)

    def test_all_paths_have_one_stage_order_and_expected_shapes(self):
        for options in ({}, {'interaction_mode': 'gated', 'graph_update_mode': 'joint'}):
            model = make_model(graph_mode='learned', norm_type='group_norm', **options)
            backbone = model.backbone
            trace = []
            with contextlib.ExitStack() as stack:
                for name in ('cnn_stage1', 'graph_stage1', 'cnn_stage2', 'graph_stage2'):
                    original = getattr(backbone, name)
                    def call(*args, _name=name, _original=original, **kwargs):
                        result = _original(*args, **kwargs)
                        trace.append((_name, tuple(result.shape)))
                        return result
                    stack.enter_context(patch.object(backbone, name, side_effect=call))
                model('a', torch.randn(6, 7, 14), wavelengths=torch.linspace(450, 950, 12), superpixel_scale=9)
            self.assertEqual(trace, [('cnn_stage1', (42, 128)), ('graph_stage1', (6, 128)),
                                     ('cnn_stage2', (42, 64)), ('graph_stage2', (6, 64))])

    def test_independent_cnn_input_applies_with_and_without_messages(self):
        Q, A = static_graph()
        x = torch.randn(6, 7, 64)
        regions = pool_regions(Q, x.reshape(-1, 64))
        changed = x + torch.randn_like(x) * .3
        for mode in ('none', 'gated'):
            backbone = make_model(graph_mode='static', norm_type='group_norm', interaction_mode=mode).backbone.eval()
            first = backbone.forward_stages(x, regions, Q, A)
            second = backbone.forward_stages(changed, regions, Q, A)
            self.assertGreater((first[0] - second[0]).abs().max().item(), 1e-5)
            if mode == 'none':
                torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)

    def test_graph_state_injection_bypasses_builder_and_keeps_losses_live(self):
        state = LearnableSuperpixelGraph(64).build_graph(torch.randn(6, 7, 64), 9)
        for mode in ('static', 'learned'):
            model = make_model(graph_mode=mode, norm_type='group_norm')
            with patch.object(LearnableSuperpixelGraph, 'build_graph', side_effect=AssertionError('must use injected graph')):
                outputs = model('a', torch.randn(6, 7, 14), graph_state=state, return_graph_info=True)
            self.assertEqual(outputs[0].shape, (42, 3))
            self.assertIs(outputs[3]['compactness_loss'], state.regularizers['compactness_loss'])
            with self.assertRaises(ValueError):
                model('a', torch.randn(6, 7, 14), *static_graph(), graph_state=state)
        wrapper = VCASAHGCN(6, 7, 14, 3, *static_graph(), .8, latent_channels=8,
                           adapter_embed_dim=8, norm_type='group_norm')
        self.assertEqual(wrapper(torch.randn(6, 7, 14), graph_state=state)[0].shape, (42, 3))
        with self.assertRaises(ValueError):
            wrapper(torch.randn(5, 7, 14), graph_state=state)

    def test_fixed_graph_controls_reach_only_expected_scorers(self):
        for q_mode, a_mode in (('fixed_grid', 'fixed_local'), ('fixed_grid', 'learned'), ('learned', 'learned')):
            model = make_model(graph_mode='learned', norm_type='group_norm', graph_assignment_mode=q_mode,
                               graph_adjacency_mode=a_mode)
            outputs = model('a', torch.randn(6, 7, 14), superpixel_scale=9, return_graph_info=True)
            sum(-y[:, 0].clamp_min(1e-8).log().mean() for y in outputs[:3]).backward()
            self.assertEqual(outputs[3]['assignment_mode'], q_mode)
            for name in ('pixel_query', 'seed_key', 'graph_query', 'graph_key'):
                expected = q_mode == 'learned' if name in ('pixel_query', 'seed_key') else a_mode == 'learned'
                grad = getattr(model.backbone.learnable_graph, name).weight.grad
                if expected:
                    self.assertIsNotNone(grad, name)
                    self.assertGreater(grad.abs().sum().item(), 0, name)
                else:
                    self.assertIsNone(grad, name)

    def test_config_cli_and_checkpoint_control_identity(self):
        import train_vca_multidataset as trainer
        base = ResearchConfig.from_dict({'model': {'graph_mode': 'learned'}})
        control = ResearchConfig.from_dict({'model': {'graph_mode': 'learned', 'graph_assignment_mode': 'fixed_grid'}})
        self.assertNotEqual(base.config_hash, control.config_hash)
        self.assertEqual(control.trainer_args('unused.pt', 'unused.json').graph_assignment_mode, 'fixed_grid')
        # Check the actual legacy CLI parser through its public parse_args entry.
        args = trainer.build_parser().parse_args(['--graph-mode', 'learned', '--graph-assignment-mode', 'fixed_grid'])
        self.assertEqual(ResearchConfig.from_legacy(args).model.graph_assignment_mode, 'fixed_grid')
        for invalid in ({'graph_assignment_mode': 'fixed_grid'}, {'graph_mode': 'learned', 'graph_adjacency_mode': 'bad'}):
            with self.assertRaises(ValueError): ResearchConfig.from_dict({'model': invalid})
        with tempfile.TemporaryDirectory() as directory:
            model = make_model(graph_mode='learned', norm_type='group_norm', graph_assignment_mode='fixed_grid',
                               graph_adjacency_mode='fixed_local').eval()
            path = Path(directory) / 'model.pt'
            save_model_checkpoint(model, path)
            restored = load_model_checkpoint(path)
            self.assertEqual(restored.backbone.graph_assignment_mode, 'fixed_grid')
            x = torch.randn(6, 7, 14)
            with torch.no_grad():
                torch.testing.assert_close(model('a', x)[0], restored('a', x)[0], rtol=0, atol=0)
            # Sidecars predating M5 omit both fields and must restore old defaults.
            model = make_model(graph_mode='learned', norm_type='group_norm').eval()
            save_model_checkpoint(model, path)
            sidecar = Path(str(path) + '.json')
            metadata = json.loads(sidecar.read_text())
            for name in ('graph_assignment_mode', 'graph_adjacency_mode'):
                metadata['backbone_kwargs'].pop(name)
            sidecar.write_text(json.dumps(metadata))
            restored = load_model_checkpoint(path)
            self.assertEqual(restored.backbone.graph_assignment_mode, 'learned')
            with torch.no_grad():
                torch.testing.assert_close(model('a', x)[0], restored('a', x)[0], rtol=0, atol=0)

    def test_legacy_static_non_stochastic_adjacency_is_preserved(self):
        model = make_model(graph_mode='static', norm_type='group_norm')
        Q, A = static_graph()
        # Historical static inputs need not obey the new manual GraphState contract.
        outputs = model('a', torch.randn(6, 7, 14), Q, A * 2)
        self.assertTrue(all(torch.isfinite(y).all() for y in outputs))


if __name__ == '__main__':
    unittest.main()
