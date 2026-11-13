"""Reproduce the implicit graph-scale switch and verify bank selector isolation."""
from itertools import product
from unittest import TestCase

import numpy as np
import torch

from loadData.observation_sampling import BandMetadata, sample
from loadData.spatial_protocol import PartitionObservation
from model.research_config import digest
from transfer import native_released_capacity as capacity
from transfer.native_scale_observation import Lattice, _materialize, forward
from transfer.native_scale_training import baseline
from transfer.spatial_multiscale import initialize


class GeometryTests(TestCase):
    def observation(self, factor):
        box = (5, 47, 7, 54)
        lattice = Lattice.make(box, (60, 70), factor)
        original = Lattice.make(box, (60, 70), 1)
        y, x = np.mgrid[5:47, 7:54]
        v = np.random.default_rng(64).normal(size=(42, 47, 7)).astype(np.float32)
        bands = BandMetadata("isolated-bank", tuple(range(7)), tuple(float(i) for i in range(450, 800, 50)),
                             "nm", "approximate", (True,) * 7)
        obs = PartitionObservation("val", v, original.coordinates(), y*70+x, digest(box))
        view, _ = sample(obs, bands, {"mode": "none"})
        return _materialize("xuzhou", 9, view, lattice, {"bank": "synthetic-no-known-physical"}, "devval", "test")

    def test_same_bank_old_mixed_grid_new_explicit_grid(self):
        torch.set_num_threads(1)
        base = baseline(9, 64)
        model = initialize(base, "joint", 7397)
        actual = self.observation(2)
        missing, constant = (capacity.condition_on_bank(actual, kind) for kind in ("missing", capacity.CONSTANT))
        physical = actual.receipt["physical"]
        self.assertIsNone(physical["original_gsd_m"])
        self.assertIsNone(constant.gsd_m)
        with torch.no_grad():
            old = [forward(model, actual, "test", "devval", condition=cond)[1] for cond in (missing, constant)]
            fixed = [forward(model, actual, "test", "devval", graph_scale=100, condition=cond)[1]
                     for cond in (missing, constant)]
        self.assertNotEqual(old[0]["q_shape"][1], old[1]["q_shape"][1])
        self.assertEqual(fixed[0]["q"], fixed[1]["q"])
        self.assertEqual(fixed[0]["a0"], fixed[1]["a0"])
        self.assertEqual(actual.verify("test", "devval")["physical"], physical)
        with self.assertRaisesRegex(ValueError, "observation"):
            forward(model, self.observation(4), "test", "devval", graph_scale=100, condition=constant)
        for conditional, fallback in (("metadata", "mean"), ("joint", "content")):
            a, b = (initialize(base, arm, 7397) for arm in (conditional, fallback))
            common = dict(a.named_parameters())
            for name, parameter in b.named_parameters():
                parameter.data.copy_(common[name].data)
            results = [forward(m.eval(), actual, "test", "devval", graph_scale=100)[0] for m in (a, b)]
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(results[0][:3], results[1][:3])))
            for result in results:
                sum(t[0,0] for t in result[:3]).backward()
            for name, value in a.named_parameters():
                if name not in dict(b.named_parameters()):
                    self.assertIsNone(value.grad)
                else:
                    prior = dict(b.named_parameters())[name].grad
                    if prior is not None:
                        self.assertTrue(torch.equal(prior, value.grad))
