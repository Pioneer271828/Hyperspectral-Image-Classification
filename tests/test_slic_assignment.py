import unittest
from unittest.mock import patch

import numpy as np

from scripts.audit_slic_portability import compare
from transfer.slic_assignment import slic_assignment


class SlicAssignmentTests(unittest.TestCase):
    def test_sparse_q_uses_raster_order_even_if_native_ids_are_nonconsecutive(self):
        image = np.zeros((2, 3, 3), dtype=np.float32)
        old = np.array([[99, 99, 4], [8, 4, 8]])
        with patch("skimage.segmentation.slic", return_value=old) as slic:
            labels, q = slic_assignment(image, scale=2)
        np.testing.assert_array_equal(labels, [[0, 0, 1], [2, 1, 2]])
        np.testing.assert_array_equal(q.toarray(), np.eye(3, dtype=np.float32)[labels.ravel()])
        self.assertEqual(slic.call_args.kwargs["n_segments"], 3)
        self.assertFalse(slic.call_args.kwargs["convert2lab"])

    def test_invalid_cube_never_calls_the_backend(self):
        with patch("skimage.segmentation.slic") as slic:
            for image in (np.zeros((2, 3, 3)), np.ones((3, 6), dtype=np.float32),
                          np.full((2, 3, 3), np.nan, dtype=np.float32)):
                with self.assertRaisesRegex(ValueError, "finite FP32"):
                    slic_assignment(image, scale=2)
        slic.assert_not_called()

    def test_audit_detects_split_or_join_with_same_shape_and_q_nnz(self):
        case = dict(name="fixture", scene="synthetic", role="synthetic", input_sha256="test", input={},
                    shape=[2, 2, 3], scale=2, q_columns=2, q_nnz=4, elapsed_ms=[1, 2, 3],
                    labels=[0, 0, 1, 1])
        result = compare(case, dict(case, labels=[0, 1, 0, 1]))
        self.assertFalse(result["equal_partition"])
        self.assertEqual(result["changed_coassignment_pairs"], 4)
        self.assertEqual(result["changed_four_neighbor_borders"], 4)
        with self.assertRaisesRegex(ValueError, "Pair inputs differ"):
            compare(case, dict(case, input_sha256="other"))
