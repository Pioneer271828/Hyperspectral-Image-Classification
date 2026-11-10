"""Verify the WP07 CNN operator through a full three-head VCA forward."""

import unittest

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
