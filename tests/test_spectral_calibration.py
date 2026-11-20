"""Exercise trusted deliveries and fail-closed spatial checkpoint replay."""

from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from loadData.data_read import read_band_metadata
from loadData.observation_sampling import sample
from loadData.spatial_protocol import PartitionObservation, forward_partition
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from research_runtime import hash_file
from transfer.spectral_calibration import load_calibration
from transfer.spectral_checkpoint import (adapt_spectrum, forward_spectral_partition, load_registered_checkpoint,
                                          save_registered_checkpoint)
from transfer.spectral_registration import register


DEVICE = os.getenv("WP05_META_DEVICE", "cpu")


class SpectralCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.table = self.root / "bands.json"
        self.image = self.root / "release.bytes"
        self.image.write_bytes(b"synthetic fixed delivery")
        self.identity = dict(dataset="fixture", image_sha256=hash_file(self.image),
                             authority="synthetic fixture", version="1")
        self.entries = [
            dict(original_band_id=i, center=center, fwhm=width, valid=valid)
            for i, center, width, valid in [(1, .4, 6, True), (3, .83, 7, False), (4, .6, 5, True),
                                            (5, .6, 8, True), (8, 2.508, 9, True)]
        ]
        self.obs = PartitionObservation("train", np.random.default_rng(33).random((7, 8, 5), dtype=np.float32),
                                        np.zeros((7, 8, 2), dtype=np.float32), np.arange(56).reshape(7, 8), "plan")

    def load(self, *, entries=None, **approval_changes):
        self.table.write_text(json.dumps(dict(schema="delivered-spectral-calibration-v1", **self.identity,
                                              center_unit="um", fwhm_unit="nm",
                                              bands=self.entries if entries is None else entries)))
        approval = dict(**self.identity, evidence_url="fixture:source", evidence_sha256="2" * 64,
                        retained_band_ids=[1, 3, 4, 5, 8], license="synthetic fixture", table_sha256=hash_file(self.table))
        approval.update(approval_changes)
        return load_calibration(self.table, dataset="fixture", bands=5, image_path=self.image, approval=approval)

    @staticmethod
    def model():
        torch.manual_seed(17)
        return MultiDatasetVCASAHGCN({"fixture": 3}, latent_channels=8, adapter_embed_dim=8,
                                     graph_mode="learned", norm_type="group_norm", coverage_mode="affine").to(DEVICE).eval()

    def test_calibration_requires_the_reviewed_bytes_units_and_mapping(self):
        bands = self.load()
        self.assertEqual(bands.wavelengths, (400, 830, 600, 600, 2508))
        self.assertEqual(bands.fwhm, (6, 7, 5, 8, 9))
        self.assertEqual((bands.wavelength_unit, bands.wavelength_quality), ("nm", "verified"))
        for field, val in (("valid", 1), ("center", float("nan")), ("center", True), ("fwhm", None)):
            changed = [dict(self.entries[0], **{field: val}), *self.entries[1:]]
            with self.subTest(field=field, val=val), self.assertRaises(ValueError):
                self.load(entries=changed)
        with self.assertRaisesRegex(ValueError, "mapping differs"):
            self.load(entries=list(reversed(self.entries)))
        with self.assertRaisesRegex(ValueError, "table bytes differ"):
            self.load(table_sha256="0" * 64)
        self.image.write_bytes(b"same name, different bytes")
        with self.assertRaisesRegex(ValueError, "image bytes differ"):
            self.load()

    def test_opt_in_preserves_three_heads_and_uses_the_same_retained_coverage(self):
        bands, model = self.load(), self.model()
        kept, record = sample(self.obs, bands, {"mode": "none"})
        adapter = model.backbone.adapter
        hook = adapter._get_attention
        calls = []

        def legacy_two_argument_hook(wavelengths, widths):
            calls.append(True)
            return hook(wavelengths, widths)

        with torch.no_grad(), patch.object(adapter, "_get_attention", side_effect=legacy_two_argument_hook):
            actual = forward_spectral_partition(model, "fixture", self.obs, bands)
            expected = forward_partition(model, "fixture", kept.observation, wavelengths=kept.bands.wavelengths,
                                         fwhm=kept.bands.fwhm, wavelength_unit="nm", fwhm_unit="nm",
                                         coverage_metadata=kept.bands)
            _, details = adapt_spectrum(adapter, torch.as_tensor(self.obs.spectral, device=DEVICE), bands)
            other = replace(bands, source_id="another-calibration")
            _, other_details = adapt_spectrum(adapter, torch.as_tensor(self.obs.spectral, device=DEVICE), other)
        self.assertEqual(len(calls), 4)
        for a, b in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(actual[-1]["coverage"]["metadata_hash"], record["selected_metadata_hash"])
        self.assertEqual(actual[-1]["spectral_metadata"]["original_band_ids"], (1, 4, 5, 8))
        self.assertNotEqual(details["metadata_hash"], other_details["metadata_hash"])
        self.assertEqual(read_band_metadata("xuzhou", 436).wavelength_quality, "approximate")

    def test_reopened_checkpoint_rejects_same_tensor_with_changed_spatial_identity(self):
        bands, model = self.load(), self.model()
        cropped, record = sample(self.obs, bands, {"mode": "uniform_stride", "stride": 2, "offset": 0})
        registered = register("fixture", self.obs, bands, record)
        path = self.root / "model.spectral.pt"
        save_registered_checkpoint(model, path, registration=registered, superpixel_scale=85)
        with self.assertRaises(FileExistsError):
            save_registered_checkpoint(model, path, registration=registered)
        sha = hash_file(path)
        loaded = load_registered_checkpoint(path, expected_sha256=sha, registration=registered, device=DEVICE)
        actual = loaded.forward_partition("fixture", self.obs, bands)
        with torch.no_grad():
            expected = forward_spectral_partition(model, "fixture", cropped.observation, cropped.bands,
                                                  superpixel_scale=85)
        for a, b in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        changes = (dict(global_ids=self.obs.global_ids + 1), dict(plan_hash="other-plan"), dict(partition="val"),
                   dict(spectral=self.obs.spectral + 1), dict(coordinates=self.obs.coordinates + 1))
        with patch("transfer.spectral_checkpoint._forward_view", side_effect=AssertionError("Premature model call")):
            for fields in changes:
                with self.subTest(field=next(iter(fields))), self.assertRaisesRegex(ValueError, "sampling record|observation differs"):
                    loaded.forward_partition("fixture", replace(self.obs, **fields), bands)
            with self.assertRaisesRegex(ValueError, "registration identity"):
                loaded.forward_partition("other-scene", self.obs, bands)
            with self.assertRaisesRegex(ValueError, "sampling record identity"):
                loaded.forward_partition("fixture", self.obs, replace(bands, source_id="wrong"))
            for changed in ({"superpixel_scale": 86}, {"coverage_reference_nm": [400, 800]}):
                with self.assertRaisesRegex(TypeError, next(iter(changed))):
                    loaded.forward_partition("fixture", self.obs, bands, **changed)
        _, other = sample(self.obs, bands, {"mode": "none"})
        with self.assertRaisesRegex(ValueError, "spectral registration differs"):
            load_registered_checkpoint(path, expected_sha256=sha, registration=register("fixture", self.obs, bands, other))
        with self.assertRaisesRegex(ValueError, "checkpoint bytes differ"):
            load_registered_checkpoint(path, expected_sha256="0" * 64, registration=registered)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            load_model_checkpoint(path, config={}, class_counts={"fixture": 3})
        legacy = self.root / "legacy.pt"
        save_model_checkpoint(model, legacy)
        self.assertNotIn("spectral_registration", load_model_checkpoint(legacy).training_config)
        with self.assertRaisesRegex(ValueError, "legacy state dictionary"):
            load_registered_checkpoint(legacy, expected_sha256=hash_file(legacy), registration=registered)

    def test_checkpoint_rejects_non_fp32_weights_without_creating_an_artifact(self):
        bands = self.load()
        _, record = sample(self.obs, bands, {"mode": "none"})
        registered = register("fixture", self.obs, bands, record)
        path = self.root / "unsupported.pt"
        mixed = self.model()
        mixed.heads["fixture"].double()
        for model in (self.model().double(), self.model().half(), mixed):
            with self.assertRaisesRegex(ValueError, "requires FP32"):
                save_registered_checkpoint(model, path, registration=registered)
            self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, "original MultiDatasetVCASAHGCN"):
            save_registered_checkpoint(torch.nn.Linear(1, 1), path, registration=registered)
        self.assertFalse(path.exists())

        original = self.root / "original.pt"
        save_registered_checkpoint(self.model(), original, registration=registered)
        payload = torch.load(original, weights_only=True, map_location="cpu")
        name = next(k for k, v in payload["weights"].items() if v.is_floating_point())
        payload["weights"][name] = payload["weights"][name].double()
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "requires FP32"):
            load_registered_checkpoint(path, expected_sha256=hash_file(path), registration=registered, device=DEVICE)
