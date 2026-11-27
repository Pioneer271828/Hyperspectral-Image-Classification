import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.learnable_superpixel_graph import LearnableSuperpixelGraph
from model.region_pixel_alignment import broadcast_regions, pool_regions
from research_runtime import tensor_hash
from scripts.audit_m3_partition_freeze import geometry, scores
from scripts.m3_partition_graph import grid_labels, slic_graph
from transfer.slic_assignment import slic_assignment


def sparse(matrix):
    return dict(shape=list(matrix.shape), indices=matrix.indices().tolist(), values=matrix.values().tolist())


class M3PartitionTests(unittest.TestCase):
    def test_irregular_region_count_is_consumed_without_padding(self):
        feature = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        labels = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        builder = LearnableSuperpixelGraph(3, assignment_dim=4)
        state = slic_graph(builder, feature, labels, (2, 2))
        self.assertIsNone(state.grid_shape)
        self.assertEqual(state.region_count, 3)
        self.assertEqual(state.A0.shape, (3, 3))
        flat = feature.reshape(8, 3)
        pooled = pool_regions(state.Q, flat)[:, :2]
        propagated = torch.sparse.mm(state.A0, pooled)
        pixels = broadcast_regions(state.Q, propagated)
        row = dict(graph=dict(q_sha256=tensor_hash(state.Q), a_sha256=tensor_hash(state.A0)),
                   witness=dict(Q=sparse(state.Q), A0=sparse(state.A0), positions=state.region_positions.tolist(),
                                input_columns=flat[:, :2].tolist(), pooled=pooled.tolist(),
                                broadcast=pixels.tolist(), region_output=propagated.tolist()))
        block = dict(shape=[2, 4, 3], grid=[2, 2], native_budget=4, slic_members=labels.ravel().tolist(),
                     points=[dict(offset=0, class_id=1), dict(offset=1, class_id=2)])
        result, actual = geometry(row, block, "slic")
        self.assertEqual(result["actual_regions"], 3)
        self.assertEqual(result["known_semantic_boundaries"], 1)
        self.assertEqual(result["captured_semantic_boundaries"], 0)
        np.testing.assert_array_equal(actual, labels.ravel())
        row["witness"]["pooled"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "backprojection"):
            geometry(row, block, "slic")

    def test_grid_matches_original_and_slic_requests_exact_nominal_budget(self):
        pixels = torch.randn(7, 9, 4)
        state = LearnableSuperpixelGraph(4).build_graph(pixels, 9, assignment_mode="fixed_grid")
        np.testing.assert_array_equal(state.Q.indices()[1].numpy(), grid_labels((7, 9), state.grid_shape).ravel())
        cube = np.zeros((7, 9, 3), dtype=np.float32)
        with patch("skimage.segmentation.slic", return_value=np.zeros((7, 9), dtype=int)) as probe:
            _, q = slic_assignment(cube, regions=state.region_count)
        self.assertEqual(probe.call_args.kwargs["n_segments"], state.region_count)
        self.assertEqual(q.shape, (63, 1))

    def test_reported_score_has_actual_per_class_denominator(self):
        probabilities = [[.9, .1], [.8, .2], [.1, .9]]
        result, cm, _, _ = scores([1, 2, 2], probabilities, 2)
        self.assertEqual(cm.tolist(), [[1, 0], [1, 1]])
        self.assertEqual(result["support"], 3)
        self.assertAlmostEqual(result["oa"], 2/3)
        self.assertAlmostEqual(result["aa"], .75)
