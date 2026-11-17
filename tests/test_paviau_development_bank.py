import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from transfer.paviau_development_bank import FixedImage


class FixedPaviaImageTests(unittest.TestCase):
    def test_only_bound_crops_and_image_variable_are_decoded(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "PaviaU.mat"
            train = {"id": 0, "role": "devtrain", "box": [0, 1, 0, 2]}
            devval = {"id": 2, "role": "devval", "box": [3, 4, 0, 2]}
            release = SimpleNamespace(shape=(4, 2, 2), rows=(train, devval),
                                      statistics={"minimum": [2, 5], "maximum": [4, 5]})
            values = np.full(release.shape, np.nan, dtype=np.float32)
            values[0] = [[2, 5], [4, 6]]
            values[3] = [[0, 9], [6, 3]]
            savemat(source, {"paviaU": values, "forbidden": np.ones(8)}, do_compression=True)
            pin = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            reader = FixedImage(pin, root, {}, release)
            with patch.object(reader, "_image", side_effect=AssertionError("decoded a rejected request")):
                for invalid in ({**train, "box": [0, 2, 0, 2]}, {**devval, "role": "devtrain"},
                                {"id": 1, "role": "test", "box": [1, 3, 0, 2]}):
                    with self.assertRaisesRegex(ValueError, "escapes"):
                        reader(invalid)
            from scipy.io import loadmat
            with patch("transfer.paviau_development_bank.loadmat", wraps=loadmat) as decode:
                np.testing.assert_array_equal(reader(train), np.float32([[[0, 0], [1, 1]]]))
                np.testing.assert_array_equal(reader(devval), np.float32([[[-1, 4], [2, -2]]]))
                self.assertEqual(decode.call_count, 1)
                self.assertEqual(decode.call_args.kwargs["variable_names"], ["paviaU"])
            corrupted = FixedImage({**pin, "sha256": "0"*64}, root, {}, release)
            with patch("transfer.paviau_development_bank.loadmat") as decode:
                with self.assertRaisesRegex(ValueError, "image bytes differ"):
                    corrupted(train)
                decode.assert_not_called()
