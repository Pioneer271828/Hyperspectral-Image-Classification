"""Input identity, CPU ownership and memory bounds for opt-in observation reuse."""
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from scripts.wp12_decoded_observation_cache import DecodedObservationCache, storage_bytes


class DecodedObservationCacheTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.path = self.root / "observation.pt"
        q = torch.sparse_coo_tensor([[0, 1], [0, 1]], torch.tensor([1., 2.]), (4, 2)).coalesce()
        self.observation = {"net_input": torch.arange(12.).reshape(2, 2, 3), "Q": q,
                            "A": torch.eye(2), "wavelengths": torch.ones(3),
                            "fwhm": None, "superpixel_scale": 2}
        self.pin = self.save(self.observation)

    def save(self, observation):
        torch.save(observation, self.path)
        return {"path": str(self.path), "sha256": sha256(self.path.read_bytes()).hexdigest()}

    def test_once_verified_then_defensive_copies_of_dense_and_coo(self):
        seen = []
        cache = DecodedObservationCache(self.root, self.pin, lambda item: seen.append(tuple(item)))
        first = cache.checkout()
        self.assertEqual(cache.retained_bytes, storage_bytes(self.observation))
        first["net_input"].zero_()
        first["Q"]._values().zero_()
        first["A"].zero_()
        with patch("scripts.wp12_decoded_observation_cache.load_snapshot", side_effect=AssertionError):
            second = cache.checkout()
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(second["net_input"], self.observation["net_input"])
        torch.testing.assert_close(second["Q"], self.observation["Q"])
        torch.testing.assert_close(second["A"], self.observation["A"])

    def test_changed_file_same_length_and_restored_mtime_refused(self):
        cache = DecodedObservationCache(self.root, self.pin, lambda _: None)
        cache.checkout()
        before = self.path.stat()
        content = self.path.read_bytes()
        with self.path.open("r+b") as stream:
            stream.write(bytes([content[0] ^ 1]))
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "changed during"):
            cache.checkout()

    def test_symlink_and_mismatched_hash_fail_closed(self):
        invalid = {**self.pin, "sha256": "0" * 64}
        cache = DecodedObservationCache(self.root, invalid, lambda _: None)
        with self.assertRaisesRegex(ValueError, "Pinned artifact content differs"):
            cache.checkout()
        cache = DecodedObservationCache(self.root, self.pin, lambda _: None)
        cache.checkout()
        renamed = self.path.with_name("replacement.pt")
        self.path.rename(renamed)
        self.path.symlink_to(renamed)
        with self.assertRaisesRegex(ValueError, "approved runs root"):
            cache.checkout()

    def test_label_fields_and_over_cap_archive_fail(self):
        with self.assertRaisesRegex(ValueError, "byte limit"):
            DecodedObservationCache(self.root, self.pin, lambda _: None, max_bytes=1)
        pin = self.save({**self.observation, "val_labels": torch.tensor([0])})
        cache = DecodedObservationCache(self.root, pin, lambda _: self.fail("unexpected external validation"))
        with self.assertRaisesRegex(ValueError, "label-free observation"):
            cache.checkout()
        self.assertEqual(cache.retained_bytes, 0)


if __name__ == "__main__":
    unittest.main()
