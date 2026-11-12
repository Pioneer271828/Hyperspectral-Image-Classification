"""Verify the WP07 CNN operator through a full three-head VCA forward."""

import unittest
from unittest.mock import patch

import torch

from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN, ObservationConditionEncoder


class ConditionalMultiscaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.architecture = dict(latent_channels=8, adapter_embed_dim=8, locality_window=None,
                                 norm_type="group_norm", graph_mode="learned")
        self.classes = {"xuzhou": 3}
        self.base = MultiDatasetVCASAHGCN(self.classes, **self.architecture).eval()
        self.x = torch.randn(9, 9, 6)
        self.wavelengths = torch.tensor([430., 525., 670., 810.])
        self.bands = BandMetadata("fixture", (0, 1, 2, 3), tuple(self.wavelengths.tolist()), "nm", "verified",
                                  (True,)*4, (8., 12., 8., 16.), "nm")

    def forward(self, model, **metadata):
        return model("xuzhou", self.x, wavelengths=self.wavelengths, superpixel_scale=12,
                     return_logits=True, **metadata)

    def test_fixed_dilation_preserves_baseline_and_changes_only_cnn_receptive_field(self):
        baseline = self.forward(self.base)
        for dilation in (1, 2, 3):
            model = ConditionalMultiscaleVCASAHGCN(self.classes, cnn_multiscale="fixed", cnn_dilation=dilation,
                                                  **self.architecture).eval()
            model.load_state_dict(self.base.state_dict(), strict=True)
            actual = self.forward(model)
            self.assertTrue(torch.equal(baseline[1], actual[1]))
            if dilation == 1:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, actual, strict=True)))
            else:
                self.assertFalse(torch.equal(baseline[2], actual[2]))

    def test_metadata_missing_fallback_and_joint_gradients_and_restore(self):
        metadata_only = ConditionalMultiscaleVCASAHGCN(self.classes, cnn_multiscale="metadata", **self.architecture).eval()
        mean = ConditionalMultiscaleVCASAHGCN(self.classes, cnn_multiscale="mean", **self.architecture).eval()
        mean.load_state_dict(self.base.state_dict(), strict=True)
        metadata_only.load_state_dict(self.base.state_dict(), strict=False)
        self.assertTrue(torch.equal(self.forward(metadata_only)[2], self.forward(mean)[2]))

        joint = ConditionalMultiscaleVCASAHGCN(self.classes, cnn_multiscale="joint", **self.architecture).eval()
        joint.load_state_dict(self.base.state_dict(), strict=False)
        options = dict(band_metadata=self.bands, coverage_fraction=.75, gsd_m=1.5)
        result = self.forward(joint, **options)
        self.assertFalse(torch.equal(result[2], self.forward(joint)[2]))
        result[2].square().mean().backward()
        layer = joint.backbone.CNN_Branch[0]
        for parameter in (layer.depth_conv.weight, layer.content_selector.weight, layer.condition_selector.weight,
                          layer.condition_encoder.band_encoder[0].weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        cfg = joint.export_config()
        restored = ConditionalMultiscaleVCASAHGCN(cfg["class_counts"], **cfg["backbone_kwargs"]).eval()
        restored.load_state_dict(joint.state_dict(), strict=True)
        self.assertTrue(torch.equal(result[2], self.forward(restored, **options)[2]))

    def test_condition_uses_valid_verified_bands_and_missing_values(self):
        encoder = ObservationConditionEncoder()
        ref = encoder.band_encoder[0].weight
        self.assertTrue(torch.equal(encoder(self.bands, None, None, ref),
                                    encoder(self.bands.subset((3, 1, 0, 2)), None, None, ref)))
        approximate = BandMetadata("fixture", (0,), (490.,), "nm", "approximate", (True,))
        self.assertIsNone(encoder(approximate, None, None, ref))
        self.assertEqual(encoder(None, None, 2., ref).shape, (32,))
        with self.assertRaisesRegex(ValueError, "GSD"):
            encoder(self.bands, None, 0., ref)

    def graph_model(self, mode, radius=1, **options):
        model = ConditionalMultiscaleVCASAHGCN(self.classes, graph_multiscale=mode, graph_radius=radius,
                                              **{**self.architecture, **options}).eval()
        missing, unexpected = model.load_state_dict(self.base.state_dict(), strict=mode in ("fixed", "mean"))
        self.assertFalse(unexpected)
        self.assertTrue(all(key.startswith("backbone.range_") for key in missing))
        return model

    def test_graph_ranges_build_one_assignment_and_mix_full_message_paths(self):
        baseline = self.forward(self.base)
        for adjacency in ("learned", "fixed_local"):
            with self.subTest(adjacency=adjacency):
                options = dict(graph_adjacency_mode=adjacency)
                narrow = self.graph_model("fixed", **options)
                wider = self.graph_model("fixed", radius=2, **options)
                mean = self.graph_model("mean", **options)
                a = self.forward(narrow, return_graph_info=True)
                b = self.forward(wider, return_graph_info=True)
                builder = mean.backbone.learnable_graph
                with patch.object(builder, "_build_assignment", wraps=builder._build_assignment) as assignment:
                    c = self.forward(mean, return_graph_info=True)
                    assignment.assert_called_once()
                if adjacency == "learned":
                    self.assertTrue(all(torch.equal(u, v) for u, v in zip(baseline, a[:3], strict=True)))
                self.assertEqual(c[3]["assignment_nnz"], a[3]["assignment_nnz"])
                self.assertGreater(b[3]["graph_nnz"], a[3]["graph_nnz"])
                self.assertEqual(c[3]["multiscale_graph"]["nnz"], (a[3]["graph_nnz"], b[3]["graph_nnz"]))
                self.assertFalse(torch.allclose(a[1], b[1]))
                self.assertTrue(torch.equal(a[2], c[2]))
                torch.testing.assert_close(c[1], (a[1] + b[1]) / 2)
                torch.testing.assert_close(c[0], (a[0] + b[0]) / 2)

    def test_graph_selection_fallback_gradient_and_exact_reload(self):
        metadata = self.graph_model("metadata")
        mean = self.graph_model("mean")
        self.assertTrue(torch.equal(self.forward(metadata)[1], self.forward(mean)[1]))

        joint = self.graph_model("joint")
        options = dict(band_metadata=self.bands, coverage_fraction=.75, gsd_m=1.5, return_graph_info=True)
        result = self.forward(joint, **options)
        perturbed = self.forward(joint, **{**options, "gsd_m": 5.})
        self.assertFalse(torch.equal(result[3]["multiscale_graph"]["weights"], perturbed[3]["multiscale_graph"]["weights"]))
        result[1].square().mean().backward()
        backbone = joint.backbone
        for parameter in (backbone.range_content.weight, backbone.range_metadata.weight,
                          backbone.range_encoder.band_encoder[0].weight, backbone.learnable_graph.graph_key.weight,
                          backbone.learnable_graph.pixel_query.weight, backbone.HGNN_Branch.hgc1.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        cfg = joint.export_config()
        restored = ConditionalMultiscaleVCASAHGCN(cfg["class_counts"], **cfg["backbone_kwargs"]).eval()
        restored.load_state_dict(joint.state_dict(), strict=True)
        self.assertTrue(torch.equal(result[1], self.forward(restored, **options)[1]))
        with self.assertRaisesRegex(ValueError, "internally"):
            self.forward(joint, Q=torch.eye(81))
