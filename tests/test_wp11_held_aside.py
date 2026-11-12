"""Verify only eligible source pixels are exported and raw labels stay unread."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat

from evaluation.wp11_confirmation import unused_tiles
from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan, tile_slice
from research_runtime import code_snapshot, write_json
from scripts.prepare_wp11_held_aside import prepare
from tests.test_wp11_confirmation import file_pin


class HeldAsideTests(unittest.TestCase):
    def test_exact_candidates_and_image_pin(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, value):
                path = root / (name + ".json")
                write_json(path, value)
                return file_pin(path)

            plan = make_tile_plan((80, 80), 20, 2)
            gt = np.resize(np.arange(1, 10), (80, 80))
            record = make_development(gt, plan, list(range(1, 10)))
            image = root / "image.mat"
            cube = np.arange(80*80*4, dtype=np.float32).reshape(80, 80, 4)
            savemat(image, {"xuzhou": cube, "xuzhou_gt": gt})
            spec = dict(scene="xuzhou", plan=write("plan", plan), development=write("development", record),
                        image=file_pin(image), preflight=write("preflight", dict(status="passed",
                        raw_labels_read=False, optimizer_updates=0, rows=[dict(scene="xuzhou", shape=list(cube.shape),
                            plan_sha256=plan["plan_hash"], image_sha256=file_pin(image)["sha256"])])))
            code = code_snapshot()
            with patch("scripts.prepare_wp11_held_aside.loadmat", wraps=loadmat) as read:
                result = prepare(write("registration", spec), root / "out", code)
            self.assertEqual(read.call_args.kwargs["variable_names"], ["xuzhou"])
            self.assertEqual([t["tile_id"] for t in result["tiles"]], unused_tiles(plan, record))
            for row in result["tiles"]:
                np.testing.assert_array_equal(np.load(row["file"]["path"]), cube[tile_slice(plan["tiles"][row["tile_id"]])])
            self.assertIsNone(result["role_assignment"])
            self.assertFalse(result["raw_labels_read"])
            spec["image"]["sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "accepted image-only"):
                prepare(write("bad", spec), root / "rejected", code)
            self.assertFalse((root / "rejected").exists())
