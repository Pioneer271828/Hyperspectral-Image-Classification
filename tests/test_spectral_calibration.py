import copy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from loadData.data_read import read_band_metadata
from loadData.observation_sampling import sample
from loadData.spatial_protocol import PartitionObservation, forward_partition
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from transfer.spectral_registration import register, restore


DEVICE = os.environ.get("WP05_META_DEVICE", "cpu")


class SpectralCalibrationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "calibration.json"
        self.image = "1" * 64
        self.identity = dict(dataset="fixture", image_sha256=self.image, authority="synthetic fixture", version="1")
        self.table = dict(schema="delivered-spectral-calibration-v1", **self.identity,
                          center_unit="um", fwhm_unit="nm", bands=[
                              dict(original_band_id=i, center=center, fwhm=width, valid=valid)
                              for i, center, width, valid in [(1, .4, 6, True), (3, .83, 7, False),
                                  (4, .6, 5, True), (5, .6, 8, True), (8, 2.508, 9, True)]])
        self.approval = dict(**self.identity, evidence_url="fixture:source", evidence_sha256="2" * 64,
                             retained_band_ids=[1, 3, 4, 5, 8], license="synthetic fixture")

    def load(self, table=None, **changes):
        raw = json.dumps(self.table if table is None else table).encode()
        self.path.write_bytes(raw)
        approval = dict(self.approval, table_sha256=sha256(raw).hexdigest(), **changes)
        return read_band_metadata("fixture", 5, calibration=self.path, approval=approval, image_sha256=self.image)

    def observation(self):
        values = np.random.default_rng(33).random((7, 8, 5), dtype=np.float32)
        return PartitionObservation("train", values, np.zeros((7, 8, 2), dtype=np.float32),
                                    np.arange(56).reshape(7, 8), "fixture-plan")

    def model(self, coverage):
        torch.manual_seed(17)
        return MultiDatasetVCASAHGCN({"fixture": 3}, latent_channels=8, adapter_embed_dim=8,
                                     graph_mode="learned", norm_type="group_norm", coverage_mode=coverage).to(DEVICE).eval()

    def test_units_identity_and_bad_tables(self):
        bands = self.load()
        self.assertEqual(bands.wavelengths, (400, 830, 600, 600, 2508))
        self.assertEqual((bands.wavelength_quality, bands.wavelength_unit, bands.fwhm_unit), ("verified", "nm", "nm"))
        self.assertEqual(bands.fwhm, (6, 7, 5, 8, 9))
        table = copy.deepcopy(self.table)
        table["center_unit"], table["fwhm_unit"] = "nm", "µm"
        for row in table["bands"]:
            row["center"] *= 1000
            row["fwhm"] /= 1000
        same = self.load(table)
        self.assertEqual((same.wavelengths, same.fwhm), (bands.wavelengths, bands.fwhm))
        self.assertNotEqual(bands.source_id, same.source_id)
        table["fwhm_unit"] = None
        for row in table["bands"]:
            row["fwhm"] = None
        self.assertIsNone(self.load(table).fwhm)
        for field, value in (("center", float("nan")), ("center", -1), ("center", True),
                             ("fwhm", None), ("fwhm", 0), ("original_band_id", 1), ("valid", 1)):
            bad = copy.deepcopy(self.table)
            bad["bands"][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load(bad)
        for bad in (dict(self.table, center_unit="auto"), dict(self.table, version="2"),
                    dict(self.table, bands=self.table["bands"][:-1]),
                    dict(self.table, bands=list(reversed(self.table["bands"])))):
            with self.assertRaises(ValueError):
                self.load(bad)
        with self.assertRaisesRegex(ValueError, "table bytes"):
            read_band_metadata("fixture", 5, calibration=self.path, image_sha256=self.image,
                               approval=dict(self.approval, table_sha256="0" * 64))
        self.assertEqual(read_band_metadata("xuzhou", 436).wavelength_quality, "approximate")
        self.assertEqual(read_band_metadata("unknown", 3).wavelength_quality, "index_only")

    def test_real_subsets_and_three_heads_preserve_source_identity(self):
        bands = self.load()
        obs = self.observation()
        full, full_record = sample(obs, bands, {"mode": "none"})
        crop, _ = sample(obs, bands, {"mode": "range_crop", "lower": 400, "upper": 600, "unit": "nm"})
        sparse, _ = sample(obs, bands, {"mode": "uniform_stride", "stride": 2, "offset": 0})
        self.assertEqual(crop.bands.band_ids, (1, 4, 5))
        self.assertEqual(sparse.bands.band_ids, (4, 8))
        self.assertEqual(sparse.bands.fwhm, (5, 9))
        with self.assertRaises(ValueError):
            sample(obs, bands, {"mode": "range_crop", "lower": 100, "upper": 200, "unit": "nm"})
        for coverage in ("none", "affine"):
            model = self.model(coverage)
            with torch.no_grad():
                actual = forward_partition(model, "fixture", obs, band_metadata=bands)
                sliced = forward_partition(model, "fixture", full.observation, band_metadata=full.bands)
                self.assertEqual(actual[-1]["spectral_metadata"]["metadata_hash"], full_record["selected_metadata_hash"])
                self.assertEqual(actual[-1]["spectral_metadata"]["original_band_ids"], (1, 4, 5, 8))
                for left, right in zip(actual[:3], sliced[:3], strict=True):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
                    self.assertTrue(torch.equal(left.argmax(1), right.argmax(1)))
                if coverage == "affine":
                    self.assertEqual(actual[-1]["coverage"]["metadata_hash"], full.bands.fingerprint)
                    self.assertEqual(actual[-1]["coverage"]["description"]["out_of_reference_count"], 1)
                for view in (crop, sparse):
                    result = forward_partition(model, "fixture", view.observation, band_metadata=view.bands)
                    self.assertTrue(all(torch.isfinite(head).all() for head in result[:3]))

    def test_adapter_permutation_gradients_and_distinct_cache_identity(self):
        bands = self.load()
        model = self.model("none").backbone.adapter
        x = torch.tensor(self.observation().spectral, device=DEVICE, requires_grad=True)
        order = [4, 3, 2, 0, 1]
        with torch.no_grad():
            y, info = model(x, band_metadata=bands, return_attention=True)
            self.assertEqual(info["original_band_ids"], (1, 4, 5, 8))
            cache = next(iter(model._eval_attention_cache))
            permuted = model(x[..., order], band_metadata=bands.subset(order))
            torch.testing.assert_close(y, permuted, rtol=1e-5, atol=1e-6)
            other = replace(bands, source_id="fixture:new-source")
            torch.testing.assert_close(y, model(x, band_metadata=other), rtol=0, atol=0)
            self.assertNotEqual(cache, next(iter(model._eval_attention_cache)))
            self.assertEqual(model(x[..., :1], band_metadata=bands.subset([0])).shape[:2], x.shape[:2])
        model(x, band_metadata=bands).square().mean().backward()
        self.assertGreater(float(model.anchor_offsets.grad.abs().sum()), 0)
        self.assertEqual(float(x.grad[..., 1].abs().sum()), 0)

    def test_registered_checkpoint_reload_and_wrong_release_rejected(self):
        bands = self.load()
        obs = self.observation()
        view, record = sample(obs, bands, {"mode": "uniform_stride", "stride": 2, "offset": 0})
        registration = register("fixture", obs, bands, record)
        path = self.root / "model.pt"
        model = self.model("affine")
        save_model_checkpoint(model, path, spectral_registration=registration)
        with self.assertRaisesRegex(ValueError, "spectral registration"):
            load_model_checkpoint(path, DEVICE)
        different = replace(bands, source_id="fixture:different-source")
        changed = replace(bands, band_ids=(1, 3, 6, 5, 8))
        for invalid in (different, changed):
            _, other_record = sample(obs, invalid, record["config"])
            other = register("fixture", obs, invalid, other_record)
            with self.assertRaisesRegex(ValueError, "spectral registration"):
                load_model_checkpoint(path, DEVICE, spectral_registration=other)
            with self.assertRaisesRegex(ValueError, "sampling record identity"):
                restore(registration, "fixture", obs, invalid)
        restored = restore(registration, "fixture", obs, bands)
        with torch.no_grad():
            original = forward_partition(model, "fixture", view.observation, band_metadata=view.bands)
            loaded = load_model_checkpoint(path, DEVICE, spectral_registration=registration)
            actual = forward_partition(loaded, "fixture", restored.observation, band_metadata=restored.bands)
        for left, right in zip(original[:3], actual[:3], strict=True):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        legacy = self.root / "legacy.pt"
        save_model_checkpoint(model, legacy)
        self.assertNotIn("spectral_registration", load_model_checkpoint(legacy, DEVICE).training_config)
