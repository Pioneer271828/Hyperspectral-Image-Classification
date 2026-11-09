"""M6 numerical, direction, sparse-Q and checkpoint compatibility contracts."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from model.cross_branch_interaction import BidirectionalSelectiveInteraction, SelectiveMessage
from model.region_pixel_alignment import RegionPixelAlignment, pool_regions, broadcast_regions
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from train_spatial import SpatialConfig

ROOT = Path(__file__).resolve().parents[1]
BASE = '637e21b23481150bc36fb8006c659cc0dc83d021'
VARIANTS = ('none', 'add', 'gated', 'cnn_to_graph', 'graph_to_cnn', 'global')
torch.set_num_threads(1)


def example():
    torch.manual_seed(21)
    pixels = torch.randn(7, 4, dtype=torch.double)
    regions = torch.randn(3, 4, dtype=torch.double)
    values = torch.rand(14, dtype=torch.double) + .1
    indices = torch.stack((torch.arange(7).repeat_interleave(2),
                           torch.stack((torch.arange(7) % 3, (torch.arange(7) + 1) % 3), 1).flatten()))
    return pixels, regions, indices, values


def legacy_model():
    old = types.ModuleType('e0a_backbone_m6')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':model/VCASAHGCN.py'], text=True)
    exec(compile(source, 'e0a_backbone_m6', 'exec'), old.__dict__)
    return old.MultiDatasetVCASAHGCN


class InteractionM6Tests(unittest.TestCase):
    def test_soft_pool_broadcast_hand_values_and_empty_region(self):
        q = torch.tensor([[1., 0., 0.], [.5, .5, 0.], [0., 1., 0.]], dtype=torch.double)
        x = torch.tensor([[2., 4.], [8., 10.], [14., 16.]], dtype=torch.double)
        expected = torch.tensor([[4., 6.], [12., 14.], [0., 0.]], dtype=torch.double)
        broadcast = torch.tensor([[4., 6.], [8., 10.], [12., 14.]], dtype=torch.double)
        for matrix in (q, q.to_sparse()):
            torch.testing.assert_close(pool_regions(matrix, x), expected, rtol=0, atol=1e-14)
            torch.testing.assert_close(broadcast_regions(matrix, expected), broadcast, rtol=0, atol=1e-14)

    def test_dense_sparse_alignment_input_and_assignment_gradients(self):
        p, r, indices, values = example()
        results = []
        for sparse in (False, True):
            x, g, v = p.clone().requires_grad_(), r.clone().requires_grad_(), values.clone().requires_grad_()
            q = torch.sparse_coo_tensor(indices, v, (7, 3)).coalesce()
            if not sparse:
                q = q.to_dense()
            a, b = RegionPixelAlignment()(x, g, q)
            (a.square().sum() + b.square().sum()).backward()
            results.append((a, b, x.grad, g.grad, v.grad))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        self.assertGreater(results[1][-1].abs().sum().item(), 0)

    def test_both_directions_read_original_snapshot_without_mutation(self):
        p, r, indices, values = example()
        q = torch.sparse_coo_tensor(indices, values, (7, 3)).coalesce()
        original = [p.clone(), r.clone(), q.values().clone()]
        layer = BidirectionalSelectiveInteraction(4, 'gated', chunk_size=0).double()
        align = RegionPixelAlignment()
        expected_p, _ = layer.graph_to_cnn(p, align.to_pixels(r, q))
        expected_r, _ = layer.cnn_to_graph(r, align.to_regions(p, q))
        wrong_r, _ = layer.cnn_to_graph(r, align.to_regions(expected_p, q))
        a, b, _ = layer.forward_aligned(p, r, q, align)
        torch.testing.assert_close(a, expected_p, rtol=0, atol=0)
        torch.testing.assert_close(b, expected_r, rtol=0, atol=0)
        self.assertGreater((b - wrong_r).abs().max().item(), 1e-8)
        for before, after in zip(original, (p, r, q.values())):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_disabled_directions_skip_alignment_and_preserve_source(self):
        p, r, indices, values = example()
        q = torch.sparse_coo_tensor(indices, values, (7, 3)).coalesce()
        for direction, skipped, active in (('cnn_to_graph', 'to_pixels', 'to_regions'),
                                            ('graph_to_cnn', 'to_regions', 'to_pixels')):
            layer = BidirectionalSelectiveInteraction(4, direction=direction, chunk_size=0).double()
            align = RegionPixelAlignment()
            with patch.object(align, skipped, side_effect=AssertionError('Disabled direction aligned')):
                with patch.object(align, active, wraps=getattr(align, active)) as called:
                    a, b, info = layer.forward_aligned(p, r, q, align, True, True)
                    self.assertEqual(called.call_count, 1)
            self.assertNotIn(('graph_to_cnn' if direction == 'cnn_to_graph' else 'cnn_to_graph') + '_gate', info)
            torch.testing.assert_close(a if direction == 'cnn_to_graph' else b,
                                       p if direction == 'cnn_to_graph' else r, rtol=0, atol=0)

    def test_none_skips_both_and_zero_strength_is_exact_identity(self):
        p, r, indices, values = example()
        q = torch.sparse_coo_tensor(indices, values, (7, 3)).coalesce()
        align = RegionPixelAlignment()
        none = BidirectionalSelectiveInteraction(4, 'none').double()
        with patch.object(align, 'to_pixels', side_effect=AssertionError('none aligned')):
            with patch.object(align, 'to_regions', side_effect=AssertionError('none aligned')):
                a, b, info = none.forward_aligned(p, r, q, align, True, True)
        self.assertIs(a, p); self.assertIs(b, r); self.assertEqual(info, {})
        zero = BidirectionalSelectiveInteraction(4, strength=0, chunk_size=2).double()
        a, b, _ = zero.forward_aligned(p, r, q, align)
        torch.testing.assert_close(a, p, rtol=0, atol=0)
        torch.testing.assert_close(b, r, rtol=0, atol=0)

    def test_chunked_outputs_input_sparse_q_and_all_parameter_gradients(self):
        p, r, indices, values = example()
        for mode in ('add', 'gated'):
            for alignment in ('region', 'global'):
                for direction in ('both', 'cnn_to_graph', 'graph_to_cnn'):
                    with self.subTest(mode=mode, alignment=alignment, direction=direction):
                        full = BidirectionalSelectiveInteraction(4, mode, direction, chunk_size=0).double().train()
                        chunked = BidirectionalSelectiveInteraction(4, mode, direction, chunk_size=2).double().train()
                        chunked.load_state_dict(full.state_dict(), strict=True)
                        results = []
                        for layer in (full, chunked):
                            x, g, v = p.clone().requires_grad_(), r.clone().requires_grad_(), values.clone().requires_grad_()
                            q = torch.sparse_coo_tensor(indices, v, (7, 3)).coalesce()
                            a, b, info = layer.forward_aligned(x, g, q, RegionPixelAlignment(alignment), True, True)
                            (a.square().sum() + .7 * b.square().sum()).backward()
                            results.append((a, b, x.grad, g.grad, v.grad))
                            for key, gate in info.items():
                                self.assertFalse(gate.requires_grad, key)
                                if key.endswith('_gate'):
                                    self.assertTrue(((gate >= 0) & (gate <= 1)).all())
                        for a, b in zip(*results):
                            if a is None:
                                self.assertIsNone(b)
                            else:
                                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
                        if alignment == 'region':
                            self.assertGreater(results[0][-1].abs().sum().item(), 0)
                        else:
                            self.assertIsNone(results[0][-1])
                        for (name, a), (_, b) in zip(full.named_parameters(), chunked.named_parameters()):
                            self.assertIsNotNone(a.grad, name)
                            self.assertTrue(torch.isfinite(a.grad).all(), name)
                            torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-11, msg=name)

    def test_global_control_ignores_correspondence_only_in_messages(self):
        p, r, indices, values = example()
        q = torch.sparse_coo_tensor(indices, values, (7, 3)).coalesce().to_dense()
        other = q.roll(1, 1)
        layer = BidirectionalSelectiveInteraction(4, chunk_size=0).double()
        a = layer.forward_aligned(p, r, q, RegionPixelAlignment('global'))
        b = layer.forward_aligned(p, r, other, RegionPixelAlignment('global'))
        for x, y in zip(a[:2], b[:2]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertGreater((pool_regions(q, p) - pool_regions(other, p)).abs().max().item(), 0)

    def test_invalid_shapes_nonfinite_strength_and_empty_message(self):
        for strength in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                BidirectionalSelectiveInteraction(strength=strength)
        layer = BidirectionalSelectiveInteraction(4)
        with self.assertRaises(ValueError):
            layer.forward_aligned(torch.ones(3, 4), torch.ones(2, 4), torch.ones(2, 2), RegionPixelAlignment())
        for shape in ((4,), (2, 2, 4)):
            with self.assertRaises(ValueError):
                SelectiveMessage(4, 3, True, .1, 0)(torch.ones(shape), torch.ones(shape))
        for size in (0, 2):
            target = torch.empty(0, 4, requires_grad=True)
            result, gate = SelectiveMessage(4, 3, True, .1, size)(target, target)
            self.assertIs(result, target); self.assertEqual(tuple(gate.shape), (0, 1))

    def test_model_legacy_state_outputs_inputs_and_parameter_gradients(self):
        old_model = legacy_model()
        for graph_mode in ('static', 'learned'):
            for variant in VARIANTS:
                with self.subTest(graph_mode=graph_mode, variant=variant):
                    options = json.loads((ROOT / 'configs/experiments' / ('m6_spatial_' + variant + '.json')).read_text())['interaction']
                    kwargs = dict(graph_mode=graph_mode, norm_type='group_norm', latent_channels=8,
                                  adapter_embed_dim=8, locality_window=None, **options)
                    torch.manual_seed(2026)
                    old = old_model({'a': 3}, **kwargs)
                    new = MultiDatasetVCASAHGCN({'a': 3}, **kwargs)
                    new.load_state_dict(old.state_dict(), strict=True)
                    self.assertEqual(list(old.state_dict()), list(new.state_dict()))
                    x = torch.randn(6, 7, 14, requires_grad=True)
                    z = x.detach().clone().requires_grad_()
                    q = torch.nn.functional.one_hot(torch.arange(42) % 6, 6).float().to_sparse()
                    a = torch.eye(6)
                    if graph_mode == 'learned': q, a = None, None
                    wave = torch.linspace(450, 950, 12)
                    expected = old('a', x, q, a, wavelengths=wave, superpixel_scale=9)
                    actual = new('a', z, q, a, wavelengths=wave, superpixel_scale=9)
                    for p0, p1 in zip(expected, actual):
                        torch.testing.assert_close(p0, p1, rtol=0, atol=0)
                    for output in (expected, actual):
                        sum(-item[:, 0].clamp_min(1e-8).log().mean() for item in output).backward()
                    torch.testing.assert_close(x.grad, z.grad, rtol=0, atol=0)
                    for (name, p0), (_, p1) in zip(old.named_parameters(), new.named_parameters()):
                        if p0.grad is None: self.assertIsNone(p1.grad, name)
                        else: torch.testing.assert_close(p0.grad, p1.grad, rtol=0, atol=0, msg=name)

    def test_checkpoint_restores_every_variant_and_nondefault_controls(self):
        for variant in VARIANTS:
            options = json.loads((ROOT / 'configs/experiments' / ('m6_spatial_' + variant + '.json')).read_text())['interaction']
            options.update(interaction_strength=.23, interaction_chunk_pixels=3, interaction_hidden_dim=7)
            model = MultiDatasetVCASAHGCN({'a': 3}, graph_mode='learned', norm_type='group_norm',
                                         latent_channels=8, adapter_embed_dim=8, **options).eval()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'model.pt'
                save_model_checkpoint(model, path)
                restored = load_model_checkpoint(path)
                self.assertEqual(restored.backbone.interaction_config, model.backbone.interaction_config)
                x, wave = torch.randn(6, 7, 14), torch.linspace(450, 950, 12)
                with torch.no_grad():
                    a = model('a', x, wavelengths=wave, superpixel_scale=9)
                    b = restored('a', x, wavelengths=wave, superpixel_scale=9)
                for before, after in zip(a, b):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_six_configs_change_only_interaction_and_have_equal_budgets(self):
        reference = None
        for variant in VARIANTS:
            value = json.loads((ROOT / 'configs/experiments' / ('m6_spatial_' + variant + '.json')).read_text())
            config = SpatialConfig(value)
            self.assertEqual(config.research.interaction.graph_update_mode, 'none')
            self.assertEqual(config.research.interaction.interaction_strength, .1)
            self.assertEqual(config.research.interaction.interaction_hidden_dim, 32)
            self.assertEqual(config.budget['optimizer_steps'], 2)
            comparable = copy.deepcopy(config.to_dict())
            comparable.pop('name'); comparable.pop('interaction')
            if reference is None: reference = comparable
            self.assertEqual(comparable, reference)


if __name__ == '__main__':
    unittest.main()
