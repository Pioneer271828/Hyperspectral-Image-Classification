"""Fixed operator mathematics and qualification identity for FP64 edge accumulation."""
from copy import deepcopy
import math
import unittest

import numpy as np
import torch

from transfer import p4_coverage_batch as batch, p4_cuda_qualification as qual, p4_runtime_device as device


class StableSparseWeightTests(unittest.TestCase):
    def test_bounded_fp64_dots_against_fsum_and_finite_difference(self):
        rng = np.random.default_rng(14071)
        edges = device.WEIGHT_CHUNK + 11
        for channels in (64, 128):
            features = torch.from_numpy(rng.normal(size=(channels, 19)).astype(np.float32)).T
            sensitivity = torch.from_numpy(rng.normal(size=(channels, 13)).astype(np.float32)).T
            rows = torch.arange(edges) % 13
            cols = torch.arange(edges) % 19
            self.assertFalse(features.is_contiguous())
            self.assertFalse(sensitivity.is_contiguous())
            actual = device.sparse_weight_gradient64(sensitivity, features, rows, cols)
            sample = (0, 97, device.WEIGHT_CHUNK-1, device.WEIGHT_CHUNK, edges-1)
            for index in sample:
                a, b = sensitivity[rows[index]].numpy(), features[cols[index]].numpy()
                expected = math.fsum(float(x)*float(y) for x,y in zip(a,b,strict=True))
                self.assertAlmostEqual(actual[index].item(), expected, places=12)
                # Independent double forward objective of one sparse edge, at +/- epsilon.
                eps = 2**-12
                plus = np.dot((.5+eps)*b.astype(np.float64), a.astype(np.float64))
                minus = np.dot((.5-eps)*b.astype(np.float64), a.astype(np.float64))
                self.assertAlmostEqual(actual[index].item(), (plus-minus)/(2*eps), places=10)

    def test_noncontiguous_coo_duplicates_transpose_zero_and_unused_gradients(self):
        rng = np.random.default_rng(7345)
        base = np.array([[0, 0, 0, 2, 2, 3], [0, 1, 1, 3, 3, 4]])
        for indices, shape in ((base, (5,6)), (base[::-1].copy(), (6,5))):
            for weight_grad, feature_grad in ((True,True),(True,False),(False,True)):
                weights = torch.tensor(rng.uniform(.05,.5,len(indices[0])).astype(np.float32),requires_grad=weight_grad)
                x = torch.from_numpy(rng.normal(size=(64,shape[1])).astype(np.float32)).T.requires_grad_(feature_grad)
                dy = torch.from_numpy(rng.normal(size=(64,shape[0])).astype(np.float32)).T
                dy[0] = 0
                graph = torch.sparse_coo_tensor(torch.from_numpy(indices), weights, shape)
                y = device.sparse_row_product_weight64(graph, x)
                y.backward(dy)
                forward, edge, dense = qual.sparse_reference(indices,weights.detach().numpy(),x.detach().numpy(),dy.numpy())
                self.assertTrue(qual.compare(y.detach().numpy(),forward)['compatible'])
                if weight_grad:
                    np.testing.assert_array_equal(weights.grad.numpy(),edge.astype(np.float32))
                else:
                    self.assertIsNone(weights.grad)
                if feature_grad:
                    self.assertTrue(qual.compare(x.grad.numpy(),dense)['compatible'])
                else:
                    self.assertIsNone(x.grad)
                self.assertEqual(y.dtype,torch.float32)

    def test_unchanged_three_way_gates_and_fixed_geometry(self):
        config = {'tile_size': 28, 'model': {'learned_assignment_radius': 1, 'learned_graph_radius': 1}}
        shapes = ((8,23),(8,28),(28,18),(28,23),(28,28))
        allowed = {'train':[{'box':[0,h,0,w]} for h,w in shapes],'val':[]}
        contract = qual.geometry_contract({'shape':[92,107],'plan_hash':'fixture'}, allowed,config)
        # Pin the old 8x23 fixture and the newly observed 28x23 shape in one fixed stream.
        self.assertIn([8,23],contract['shapes'])
        self.assertIn([28,23],contract['shapes'])
        results, patterns = qual.run_fixtures(contract,'cpu',device.CUDA_FP64_WEIGHT)
        self.assertEqual(patterns,qual.expected_patterns(contract))
        self.assertTrue(all(qual.check_comparisons(name,row) for name,row in results.items()))

    def test_v4_receipt_binds_source_protocol_device_and_dtype(self):
        geometry = {'geometry_sha256':'test','shapes':[[8,23]],'token_stride':10,'assignment_radius':1,
                    'adjacency_radius':1,'pool_channels':64}
        results, patterns = qual.run_fixtures(geometry,'cpu',device.CUDA_FP64_WEIGHT)
        code = {'commit':'candidate','source_hash':'source'}
        gpu = 'GPU-11111111-1111-1111-1111-111111111111'
        row = {'schema':qual.WEIGHT64_SCHEMA,'status':'passed',**code,'parent_sha256':batch.PARENT_SHA,
               'geometry':geometry,'tolerances':{'rtol':qual.RTOL,'atol':qual.ATOL},
               'numeric':{'protocol':device.CUDA_FP64_WEIGHT,'device':'cuda:0','physical_uuid':gpu,
                          'deterministic_algorithms':True,'deterministic_warn_only':False,'tf32':False,
                          'float32_matmul_precision':'highest','graph_kernels':'adaptive_bin_mean_and_sparse_row_product_weight64_v4',
                          'sparse_weight_contract':device.FP64_WEIGHT_CONTRACT},
               'optimizer_updates':0,'query_accessed':False,'raw_gt_loaded':False,'scope_restored':True,
               'results':results,'coo_patterns':patterns,'fixture_seed':qual.FIXTURE_SEED,
               'interceptions':{'adaptive_avg_pool2d':1,'sparse_mm':6}}
        self.assertEqual(qual.check_receipt(row,device.CUDA_FP64_WEIGHT,code,geometry),gpu)
        for change in (lambda r:r.update(schema=qual.SCHEMA,status='failed'),
                       lambda r:r.update(source_hash='different'),
                       lambda r:r['numeric'].update(physical_uuid='unknown'),
                       lambda r:r['numeric']['sparse_weight_contract'].update(weight_gradient_accumulation_dtype='torch.float32')):
            wrong = deepcopy(row)
            change(wrong)
            with self.assertRaises(ValueError):
                qual.check_receipt(wrong,device.CUDA_FP64_WEIGHT,code,geometry)
