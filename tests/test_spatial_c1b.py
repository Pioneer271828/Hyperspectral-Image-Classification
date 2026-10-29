import copy
from dataclasses import fields
import json
import unittest
import numpy as np
import torch
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from loadData.spatial_protocol import (make_spatial_plan, validate_plan, fit_train_statistics,
    observe_partition, training_labels, forward_partition, class_coverage, PartitionObservation)

torch.set_num_threads(1)

class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.cube=np.random.default_rng(7).normal(size=(80,20,6)).astype(np.float32)
        self.plan=make_spatial_plan((80,20),gap=2)
        self.stats=fit_train_statistics(self.cube,self.plan)

    def test_partitions_and_guards_are_disjoint(self):
        groups=[set(observe_partition(self.cube,self.plan,self.stats,p).global_ids.ravel()) for p in ('train','val','test')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertLess(len(set.union(*groups)),80*20)
        altered=copy.deepcopy(self.plan);altered['boxes']['val'][0]=0
        with self.assertRaises(ValueError):validate_plan(altered)
        with self.assertRaises(ValueError):make_spatial_plan((10,10),gap=4)
        with self.assertRaises(ValueError):make_spatial_plan((80,20),fractions=(.5,.4,.4))

    def test_only_train_values_affect_statistics_and_input(self):
        other=self.cube.copy();other[48:]=1e6
        self.assertEqual(self.stats,fit_train_statistics(other,self.plan))
        a=observe_partition(self.cube,self.plan,self.stats,'train')
        b=observe_partition(other,self.plan,self.stats,'train')
        np.testing.assert_array_equal(a.spectral,b.spectral)
        target=observe_partition(other,self.plan,self.stats,'test')
        self.assertGreater(target.spectral.max(),1)
        self.assertNotIn('labels',{f.name for f in fields(PartitionObservation)})
        self.assertFalse(a.spectral.flags.writeable)

    def test_label_access_and_boundary_rejection(self):
        obs=observe_partition(self.cube,self.plan,self.stats,'val')
        labels=np.ones((80,20),dtype=int)
        with self.assertRaises(ValueError):training_labels(labels,self.plan,obs)
        obs=observe_partition(self.cube,self.plan,self.stats,'train')
        self.assertEqual(training_labels(labels,self.plan,obs).shape,obs.spectral.shape[:2])
        with self.assertRaises(ValueError):observe_partition(self.cube,self.plan,self.stats,'train',(40,60,0,10))
        wrong=copy.deepcopy(self.stats);wrong['maximum'][0]+=1
        with self.assertRaises(ValueError):observe_partition(self.cube,self.plan,wrong,'train')

    def test_statistics_roundtrip_constant_band_and_coverage(self):
        cube=np.ones_like(self.cube)
        stats=json.loads(json.dumps(fit_train_statistics(cube,self.plan)))
        obs=observe_partition(cube,self.plan,stats,'test')
        self.assertTrue(np.isfinite(obs.spectral).all())
        self.assertEqual(float(obs.spectral.max()),0)
        labels=np.ones((80,20),dtype=int);labels[-5:]=2
        coverage=class_coverage(labels,self.plan)
        self.assertIn(2,coverage['missing_classes']['train'])
        self.assertFalse(coverage['all_classes_present'])

    def test_independent_graph_forward_backward_and_eval_guard(self):
        torch.manual_seed(7)
        model=MultiDatasetVCASAHGCN({'demo':2},latent_channels=8,adapter_embed_dim=8,
            norm_type='group_norm',graph_mode='learned',locality_window=None)
        obs=observe_partition(self.cube,self.plan,self.stats,'train',(0,12,0,12))
        out=forward_partition(model,'demo',obs,superpixel_scale=16)
        (-out[0][:,0].clamp_min(1e-7).log().mean()).backward()
        gradient=model.backbone.learnable_graph.pixel_query.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()),0)
        target=observe_partition(self.cube,self.plan,self.stats,'test',(70,80,0,12))
        with self.assertRaises(ValueError):forward_partition(model,'demo',target)
        model.eval()
        with torch.no_grad():
            a=forward_partition(model,'demo',obs)[0]
            altered=self.cube.copy();altered[48:]+=10000
            b=forward_partition(model,'demo',observe_partition(altered,self.plan,self.stats,'train',(0,12,0,12)))[0]
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(forward_partition(model,'demo',target)[0].shape,(120,2))
        static=MultiDatasetVCASAHGCN({'demo':2},latent_channels=8,adapter_embed_dim=8,graph_mode='static')
        with self.assertRaises(ValueError):forward_partition(static,'demo',obs)

if __name__=='__main__':unittest.main()
