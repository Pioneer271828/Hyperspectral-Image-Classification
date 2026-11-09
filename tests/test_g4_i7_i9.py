"""I7/I9 bind existing message and edge controls; no new network arithmetic."""
from dataclasses import asdict, replace
import unittest

import torch
from torch.nn import functional as F

from g4_edge_target import configuration, initialize, verify_initialization_binding
from model.interaction_config import InteractionConfig
from target_initialization import METHODS


class JointControlsTests(unittest.TestCase):
    def test_legacy_six_configuration_payloads_unchanged(self):
        for condition in ("none", "gated", "graph", "gated_graph", "joint", "gated_joint"):
            old = asdict(replace(InteractionConfig(),
                interaction_mode="gated" if condition.startswith("gated") else "none",
                graph_update_mode="graph" if condition.endswith("graph") else "joint" if condition.endswith("joint") else "none"))
            self.assertEqual(configuration(condition), old)

    def test_global_alignment_same_capacity_same_initial_tensors_and_guard(self):
        regional, binding = initialize("xuzhou", METHODS[0], "gated_joint", model_seed=64, head_seed=1064)
        global_model, global_binding = initialize("xuzhou", METHODS[0], "global_joint", model_seed=64, head_seed=1064)
        self.assertEqual(set(regional.state_dict()), set(global_model.state_dict()))
        for name, value in regional.state_dict().items():
            self.assertTrue(torch.equal(value, global_model.state_dict()[name]), name)
        self.assertEqual(binding["added_parameter_count"], global_binding["added_parameter_count"])
        with self.assertRaises(ValueError):
            verify_initialization_binding(global_model, binding)
        global_model.backbone.alignment.mode = "region"
        with self.assertRaises(ValueError):
            verify_initialization_binding(global_model, global_binding)

    def test_explicit_global_mean_and_direct_message_formula_and_gradients(self):
        torch.manual_seed(341)
        q = torch.tensor([[1., 0., 0.], [.7, .3, 0.], [0., 1., 0.], [0., .2, .8], [0., 0., 1.]], dtype=torch.float64)
        for condition in ("add_joint", "global_joint"):
            model, _ = initialize("xuzhou", METHODS[0], condition, model_seed=512, head_seed=1512)
            interaction, alignment = model.backbone.interaction.double(), model.backbone.alignment
            pixels = torch.randn(5, 128, dtype=torch.float64, requires_grad=True)
            regions = torch.randn(3, 128, dtype=torch.float64, requires_grad=True)
            pixel_ref, region_ref = pixels.detach().clone().requires_grad_(), regions.detach().clone().requires_grad_()
            if condition == "global_joint":
                to_pixels = region_ref.mean(0).repeat(5, 1)
                to_regions = pixel_ref.mean(0).repeat(3, 1)
                self.assertTrue(torch.equal(alignment.to_regions(pixels, q), alignment.to_regions(pixels, q.flip(1))))
                self.assertTrue(torch.equal(alignment.to_pixels(regions, q), alignment.to_pixels(regions, q.flip(1))))
            else:
                to_pixels = q @ region_ref
                to_regions = (q / q.sum(0)).T @ pixel_ref
            def reference(module, target, source):
                norm = lambda x, layer: F.layer_norm(x, (128,), layer.weight, layer.bias, layer.eps)
                source = norm(source, module.source_norm)
                message = F.linear(source, module.projection.weight)
                if module.gate is None:
                    gate = 1.
                else:
                    target_norm = norm(target, module.target_norm)
                    features = torch.cat((target_norm, source, (target_norm - source).abs()), -1)
                    gate = torch.sigmoid(F.linear(F.gelu(F.linear(features, module.gate[0].weight,
                        module.gate[0].bias)), module.gate[2].weight, module.gate[2].bias))
                return target + .1 * gate * message
            expected = (reference(interaction.graph_to_cnn, pixel_ref, to_pixels),
                        reference(interaction.cnn_to_graph, region_ref, to_regions))
            actual = interaction.forward_aligned(pixels, regions, q, alignment)[:2]
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
            weights = [torch.randn_like(x) for x in actual]
            grad = torch.autograd.grad(sum((x*w).sum() for x,w in zip(actual, weights)),
                                       (pixels, regions, *interaction.parameters()))
            ref_grad = torch.autograd.grad(sum((x*w).sum() for x,w in zip(expected, weights)),
                                           (pixel_ref, region_ref, *interaction.parameters()))
            for a, b in zip(grad, ref_grad):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
