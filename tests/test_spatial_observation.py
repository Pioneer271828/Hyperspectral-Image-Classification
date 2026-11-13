"""Independent arithmetic, pixel lattice, and synthetic condition checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import math

import numpy as np

from transfer import development_bank as bank, development_bank_training as fixtures
from transfer import spatial_observation as spatial, spatial_multiscale as multi
from transfer.spatial_multiscale_training import contract, Run


def scalar(values, mode):
    h,w,b = values.shape
    output = np.empty_like(values)
    weights = [math.exp(-d*d/2) for d in range(-3,4)]
    denom = sum(weights)**2
    for y in range(h):
        for x in range(w):
            for k in range(b):
                if mode == "gaussian":
                    output[y,x,k] = sum(wy*wx*float(values[max(0,min(h-1,y+dy)),max(0,min(w-1,x+dx)),k])
                        for dy,wy in zip(range(-3,4),weights) for dx,wx in zip(range(-3,4),weights))/denom
                else:
                    f = int(mode[-1])
                    a,c = y//f*f,x//f*f
                    cell = [float(values[i,j,k]) for i in range(a,min(a+f,h)) for j in range(c,min(c+f,w))]
                    output[y,x,k] = sum(cell)/len(cell)
    return output


class SpatialObservationTests(TestCase):
    def bank(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        release, arrays = fixtures.fixture(9)
        return bank.Bank(bank.publish(root/"bank",root,release,lambda row: arrays[row["id"]]),root,release)

    def test_independent_scalar_tiny_partial_and_narrow(self):
        for shape in ((5,7,3),(1,3,1),(8,2,4)):
            cube = np.random.default_rng(2).normal(size=shape).astype(np.float32)
            for mode in spatial.MODES[1:]:
                np.testing.assert_allclose(spatial.degrade(cube,mode),scalar(cube,mode),atol=1e-7,rtol=1e-6)
            self.assertIs(spatial.degrade(cube,"identity"),cube)

    def test_partial_cell_and_label_unknown(self):
        cube = np.arange(15,dtype=np.float32).reshape(3,5,1)
        self.assertEqual(float(spatial.degrade(cube,"area4")[-1,-1,0]),9.)
        row = dict(box=[1,4,1,6],samples=[[11,1],[12,2],[35,1]])
        result = spatial.observability(row,"area4",10,2)
        self.assertEqual(result["conflicting_cells"],1)
        self.assertEqual(result["known_per_class"],[2,1])
        self.assertEqual(result["unknown_pixels"],12)

    def test_nan_shape_dtype_and_unknown(self):
        for value,mode in ((np.zeros((1,1,1)),"area2"),(np.full((2,1,1),np.nan,dtype=np.float32),"gaussian"),
                           (np.zeros((0,3,1),dtype=np.float32),"area2"),(np.ones((1,1,1),dtype=np.float32),"invalid")):
            with self.assertRaises(ValueError):
                spatial.degrade(value,mode)

    def test_same_lattice_new_q_no_full_band_leak(self):
        opened = self.bank()
        row = opened.role_rows("devtrain")[0]
        subset = fixtures._condition("even_retained_positions")
        identity = spatial.materialize(opened,row,subset,"identity","test")
        mode = spatial.materialize(opened,row,subset,"gaussian","test")
        self.assertEqual(mode[-1]["positions"],[0,2,4,6])
        for key in ("coordinates","ids","statistics","samples"):
            self.assertEqual(mode[-1][key],identity[-1][key])
        self.assertNotEqual(mode[-1]["observation"],identity[-1]["observation"])
        model = fixtures._baseline(opened.release,64)
        _, a = multi.forward(model,"off",identity)
        _, b = multi.forward(model,"off",mode)
        self.assertNotEqual(a["q"],b["q"])
        self.assertEqual(a["grid"],b["grid"])
        self.assertFalse(model.backbone.adapter._eval_attention_cache)
        self.assertFalse(mode[2].observation.spectral.flags.writeable)
        with patch.object(opened,"materialize",side_effect=AssertionError("read invalid request")):
            with self.assertRaises(ValueError):
                spatial.materialize(opened,row,subset,"bad","test")

    def test_boundary_poison_outside_block(self):
        image = np.full((11,14,2),1e8,dtype=np.float32)
        block = np.arange(70,dtype=np.float32).reshape(5,7,2)
        image[2:7,3:10] = block
        for mode in spatial.MODES:
            reference = spatial.degrade(block,mode).copy()
            image[7:] = -1e8
            np.testing.assert_array_equal(spatial.degrade(image[2:7,3:10],mode),reference)

    def test_conditions_only_and_single_candidate(self):
        opened = self.bank()
        source = fixtures._baseline(opened.release,64)
        evidence = multi.condition_audit(opened,source,"test")
        self.assertEqual(evidence["synthetic_forwards"],21)

    def test_contract_and_checkpoint_reject_before_observation(self):
        opened = self.bank()
        spec = contract(opened,64,"off","producer")
        run = Run(opened,spec,"producer")
        altered = spec["exposure"][0].copy()
        altered["spatial"] = "area4"
        spec["transforms"]["area4"]["factor"] = 2
        run.start()
        with patch.object(opened,"materialize",side_effect=AssertionError("read altered observation")):
            with self.assertRaises(ValueError):
                run.update(altered)
            with self.assertRaises(ValueError):
                Run(opened,spec,"producer")
            with self.assertRaises(ValueError):
                run.restore(dict(run.pack(),contract="other"),0)
