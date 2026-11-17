"""Calibration verifies observed bytes and the exact ordered development role."""
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from scripts.calibrate_wp05_frozen_front import check_views, view_identity


class CalibrationBindingTests(TestCase):
    def test_changed_role_bytes_order_count_source_rejected(self):
        tiles = [{"id": 3, "box": [0, 2, 0, 2], "partition": "train"},
                 {"id": 7, "box": [2, 4, 2, 4], "partition": "train"}]
        authority = {"allowed": {t["id"]: t for t in tiles}, "shape": [6, 6],
                     "expected_bands": "the-frozen-band-order", "plan_hash": "plan"}

        def materialize(tile):
            y0,y1,x0,x1 = tile["box"]
            y, x = np.mgrid[y0:y1,x0:x1]
            obs = SimpleNamespace(partition="train", plan_hash="plan",
                                  coordinates=np.stack((y/5, x/5), axis=-1).astype(np.float32),
                                  global_ids=y*6+x, spectral=np.ones((2, 2, 3), dtype=np.float32))
            bands = SimpleNamespace(fingerprint="the-frozen-band-order", band_ids=(0, 2, 4))
            return SimpleNamespace(observation=obs, bands=bands)

        original = [(tile, materialize(tile)) for tile in tiles]
        existing = [view_identity(t, v, **authority) for t, v in original]
        self.assertEqual(len(check_views(original, existing, authority)), 2)
        wrong = [{**existing[0], "spectral_digest": "f"*64}, existing[1]]
        for contexts, records in ((original, wrong), (original[::-1], existing), (original[:1], existing)):
            with self.assertRaisesRegex(ValueError, "bytes, order or census"):
                check_views(contexts, records, authority)
        for field, value in (("partition", "val"), ("plan_hash", "different-source")):
            first = original[0][1]
            changed_obs = SimpleNamespace(**{**vars(first.observation), field: value})
            changed = [(tiles[0], SimpleNamespace(observation=changed_obs, bands=first.bands)), original[1]]
            with self.assertRaisesRegex(ValueError, "authority"):
                check_views(changed, existing, authority)
        original[0][1].observation.spectral[0, 0, 0] = 7.
        with self.assertRaisesRegex(ValueError, "bytes"):
            check_views(original, existing, authority)
