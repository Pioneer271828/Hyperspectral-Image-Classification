import unittest

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN
from transfer.region_message_attention import RegionMessageAttention, RegionMessageVCASAHGCN


def identity_attention(mode, *, top_k=2, chunk=1):
    layer = RegionMessageAttention(2, 2, mode, (1, 2), top_k, chunk).double()
    with torch.no_grad():
        for projection in (layer.query, layer.key, layer.value, layer.output):
            projection.weight.copy_(torch.eye(2))
    return layer


class RegionMessageTests(unittest.TestCase):
    def test_combined_one_softmax_matches_explicit_reference_and_chunked_gradients(self):
        torch.manual_seed(11)
        target = torch.tensor([[1., .2], [.9, .1], [-.8, .3], [9., 9.]], dtype=torch.double, requires_grad=True)
        source = torch.tensor([[1., .6], [.5, -.1], [-1., .3], [7., 7.]], dtype=torch.double, requires_grad=True)
        positions = torch.tensor([[.2, .1], [.5, .4], [.6, .9], [.9, .9]], dtype=torch.double)
        mask = torch.tensor([True, True, True, False])
        landmarks = torch.stack((target[:2].mean(0), target[2]))
        probed = landmarks @ source[:3].T / 2**.5
        compressed = probed.softmax(dim=1) @ source[:3]
        expected = []
        for query in target[:3]:
            assigned = (landmarks @ query).argmax()
            ids = probed[assigned].argsort(descending=True, stable=True)[:2]
            keys = torch.cat((landmarks, source[ids]))
            vals = torch.cat((compressed, source[ids]))
            expected.append((keys @ query / 2**.5).softmax(dim=0) @ vals)
        reference = torch.stack(expected)
        layer = identity_attention('combined')
        result = layer(target, source, positions, mask)
        torch.testing.assert_close(result[:3], reference)
        self.assertTrue(torch.equal(result[3], torch.zeros_like(result[3])))
        grad = torch.autograd.grad(result.square().sum(), (target, source), retain_graph=True)
        ref_grad = torch.autograd.grad(reference.square().sum(), (target, source))
        for actual, expected_grad in zip(grad, ref_grad, strict=True):
            torch.testing.assert_close(actual, expected_grad)
        torch.testing.assert_close(result, identity_attention('combined', chunk=16)(target, source, positions, mask))

    def test_route_all_candidates_matches_dense_and_small_masks_are_safe(self):
        target = torch.tensor([[1., 2.], [1., 2.], [2., 1.]], dtype=torch.double)
        source = torch.tensor([[.2, .6], [.3, -.2], [-1., 1.]], dtype=torch.double)
        pos = torch.tensor([[.1, .1], [.8, .8], [.9, .9]], dtype=torch.double)
        full = torch.ones(3, dtype=torch.bool)
        dense = identity_attention('dense')
        route = identity_attention('route', top_k=12)
        torch.testing.assert_close(route(target, source, pos, full), dense(target, source, pos, full))
        for mode in ('dense', 'compress', 'route', 'combined'):
            layer = identity_attention(mode, top_k=12)
            torch.testing.assert_close(layer(target, source, pos, ~full), torch.zeros_like(source))
            single = torch.tensor([False, True, False])
            torch.testing.assert_close(layer(target, source, pos, single), source * single[:, None])

    def test_three_heads_and_both_directions_train_and_restore(self):
        torch.manual_seed(71)
        classes = {'fixture': 3}
        architecture = dict(latent_channels=8, adapter_embed_dim=8, locality_window=None,
                            norm_type='group_norm', graph_mode='learned', interaction_mode='gated')
        control = MultiDatasetVCASAHGCN(classes, **architecture)
        torch.manual_seed(71)
        config = dict(mode='combined', width=8, landmark_grid=[2, 2], top_k=3, query_chunk=4)
        model = RegionMessageVCASAHGCN(classes, region_attention=config, **architecture)
        x = torch.randn(9, 9, 6)
        waves = torch.tensor([430., 525., 670., 810.])
        options = dict(wavelengths=waves, superpixel_scale=9, return_graph_info=True, return_logits=True)
        baseline = control('fixture', x, **options)
        output = model('fixture', x, **options)
        self.assertEqual(baseline[3]['graph_nnz'], output[3]['graph_nnz'])
        self.assertEqual(baseline[3]['assignment_nnz'], output[3]['assignment_nnz'])
        self.assertFalse(torch.equal(baseline[0], output[0]))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        exchange = model.backbone.interaction
        weights = [exchange.to_pixels.value.weight, exchange.to_regions.value.weight]
        before = [weight.detach().clone() for weight in weights]
        sum(head.square().mean() for head in output[:3]).backward()
        optimizer.step()
        for a, b in zip(before, weights, strict=True):
            self.assertTrue(torch.isfinite(b.grad).all())
            self.assertFalse(torch.equal(a, b))
        saved = model.export_config()
        restored = RegionMessageVCASAHGCN(saved['class_counts'], **saved['backbone_kwargs']).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        actual = model.eval()('fixture', x, **options)
        reload = restored('fixture', x, **options)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual[:3], reload[:3], strict=True)))

    def test_zero_strength_matches_original_exchange(self):
        torch.manual_seed(50)
        classes = {'fixture': 2}
        options = dict(latent_channels=8, adapter_embed_dim=8, graph_mode='learned', norm_type='group_norm',
                       interaction_mode='gated', interaction_strength=0.)
        control = MultiDatasetVCASAHGCN(classes, **options).eval()
        torch.manual_seed(50)
        config = dict(mode='combined', width=8, landmark_grid=(2, 2), top_k=2, query_chunk=4)
        model = RegionMessageVCASAHGCN(classes, region_attention=config, **options).eval()
        x = torch.randn(9, 9, 6)
        wavelengths = torch.tensor([430., 525., 670., 810.])
        expected = control('fixture', x, wavelengths=wavelengths, superpixel_scale=9)
        actual = model('fixture', x, wavelengths=wavelengths, superpixel_scale=9)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)))
