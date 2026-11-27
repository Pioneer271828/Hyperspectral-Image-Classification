"""Numeric reference, invariance, fixed-budget and input-policy checks on synthetic data."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn._loss.loss import HalfBinomialLoss
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.neighbors import NearestCentroid

from features import transform
from models import objective_gradient,initial,fit,centroid,predict,sigmoid


class NumericalBaselines(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2026);self.X=rng.uniform(-1,1,(10,8));self.y=np.array([0]*5+[1]*5,dtype=np.float64);self.theta=initial(64)
    def test_logistic_objective_and_gradient_match_maintained_reference(self):
        loss,grad=objective_gradient(self.theta,self.X,self.y)
        expected=LinearModelLoss(HalfBinomialLoss(),fit_intercept=True).loss_gradient(self.theta,self.X,self.y,l2_reg_strength=.01,n_threads=1)
        self.assertAlmostEqual(loss,float(expected[0]),places=13);np.testing.assert_allclose(grad,expected[1],rtol=1e-12,atol=1e-12)
    def test_gradient_matches_independent_central_difference(self):
        _,grad=objective_gradient(self.theta,self.X,self.y);ref=[]
        for i in range(9):
            plus=self.theta.copy();minus=self.theta.copy();plus[i]+=1e-6;minus[i]-=1e-6;ref.append((objective_gradient(plus,self.X,self.y)[0]-objective_gradient(minus,self.X,self.y)[0])/2e-6)
        np.testing.assert_allclose(grad,ref,rtol=1e-6,atol=1e-6)
    def test_intercept_is_not_l2_penalized(self):
        theta=np.r_[np.zeros(8),2.0];a=objective_gradient(theta,self.X,self.y,l2=.01);b=objective_gradient(theta,self.X,self.y,l2=0)
        self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    def test_fixed_steps_descend_and_checkpoint_reload_preserves_float64(self):
        with tempfile.TemporaryDirectory() as d:
            a=fit(self.X,self.y,seed=64,steps=64,checkpoint_dir=Path(d));b=fit(self.X,self.y,seed=64,steps=64)
        self.assertEqual(a['optimizer_updates'],64);np.testing.assert_array_equal(a['theta'],b['theta']);self.assertLessEqual(a['training_trace'][-1]['regularized_training_objective'],a['training_trace'][0]['regularized_training_objective'])
    def test_interrupted_32_plus_32_matches_64(self):
        a=fit(self.X,self.y,seed=512,steps=32);b=fit(self.X,self.y,seed=512,steps=32,start=a['theta']);c=fit(self.X,self.y,seed=512,steps=64);np.testing.assert_array_equal(b['theta'],c['theta'])
    def test_initialization_does_not_mutate_global_rng(self):
        state=np.random.get_state();initial(2048);after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
    def test_centroid_means_and_predictions_match_reference(self):
        model=centroid(self.X,self.y);ref=NearestCentroid(metric='euclidean').fit(self.X,self.y);np.testing.assert_allclose(model['class_means'],ref.centroids_,atol=1e-14);np.testing.assert_array_equal(predict(model,self.X)[2],ref.predict(self.X))
    def test_stable_extreme_probabilities_and_ground_tie(self):
        p=sigmoid(np.array([-10000.,0.,10000.]));np.testing.assert_array_equal(p,[0.,.5,1.]);m={'theta':[0.]*9};self.assertTrue(np.all(predict(m,self.X)[2]==0))
    def test_band_permutation_preserves_distribution_features(self):
        x=np.random.default_rng(42).normal(size=(3,31,7,7));a=transform(x);b=transform(x[:,::-1]);np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
    def test_samplewise_positive_affine_change_has_no_fitted_population_step(self):
        x=np.random.default_rng(42).normal(size=(3,31,7,7));np.testing.assert_allclose(transform(x),transform(x*7.3+15),atol=1e-12,rtol=1e-12)
    def test_constant_or_zero_spectra_keep_examples_with_zero_descriptor(self):
        for x in (np.zeros((2,31,7,7)),np.ones((2,31,7,7))):np.testing.assert_array_equal(transform(x),np.zeros((2,8)))
    def test_invalid_inputs_do_not_silently_create_labels_or_features(self):
        with self.assertRaises(ValueError):transform(np.full((1,31,7,7),np.nan))
        with self.assertRaises(ValueError):objective_gradient(self.theta,self.X,np.full(10,2))
        with self.assertRaises(ValueError):fit(self.X,self.y,seed=64,steps=129)


if __name__=='__main__':unittest.main()
