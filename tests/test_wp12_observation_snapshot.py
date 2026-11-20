import fcntl
from hashlib import sha256
import mmap
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from evaluation.wp12_observation_snapshot import load_snapshot, verified_snapshot
from joint_training import _tree_fingerprint


class VerifiedObservationSnapshotTests(unittest.TestCase):
    def test_verified_bytes_survive_replacement_and_descriptor_close(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path = root / "observation.pt"
            item = {"dense": torch.arange(5), "sparse": torch.eye(3).to_sparse().coalesce()}
            torch.save(item, path)
            pin = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
            with verified_snapshot(root, pin) as snapshot:
                with snapshot.open("rb") as stream:
                    self.assertTrue(fcntl.fcntl(stream, fcntl.F_GET_SEALS) & fcntl.F_SEAL_WRITE)
                torch.save({"replacement": torch.ones(1)}, path)
                loaded = load_snapshot(snapshot)
            self.assertEqual(_tree_fingerprint(loaded), _tree_fingerprint(item))
            with self.assertRaisesRegex(ValueError, "content differs"), verified_snapshot(root, pin):
                self.fail("Changed contents reached the decoder")

    def test_symbolic_source_and_shared_mapping_are_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root / "observation.pt"
            torch.save(torch.zeros(1), source)
            path = root / "alias.pt"
            path.symlink_to(source)
            pin = {"path": str(path), "sha256": sha256(source.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "approved runs root"), verified_snapshot(root, pin):
                self.fail("Symbolic source was accepted")
            pin["path"] = str(source)
            with verified_snapshot(root, pin) as snapshot, torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
                with self.assertRaisesRegex(ValueError, "private PyTorch mapping"):
                    load_snapshot(snapshot)
