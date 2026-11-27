"""Negative role gates and independently decoded synthetic TIFF bytes."""
import io
import unittest

import numpy as np
import tifffile

from audit_intake import manual_tiff,check_roles
from catalog import train_member


class AdmissionGates(unittest.TestCase):
    def test_train_whitelist_rejects_validation_test_and_unknown_before_fetch(self):
        train_member('flair_aerial_train/D006_2020/Z1/img/IMG_1.tif')
        for domain in ('D004_2021','D015_2020','D999_2020'):
            with self.assertRaises(ValueError):train_member('flair_aerial_train/'+domain+'/IMG_1.tif')
        with self.assertRaises(ValueError):train_member('unknown/IMG_1.tif')
    def test_manual_pixels_are_exact_for_contiguous_and_planar_samples(self):
        a=np.arange(6*8*5,dtype=np.uint8).reshape(6,8,5)
        for separate in (False,True):
            f=io.BytesIO();tifffile.imwrite(f,a.transpose(2,0,1) if separate else a,photometric='rgb',planarconfig='separate' if separate else 'contig',metadata=None)
            _,actual=manual_tiff(f.getvalue());self.assertTrue(np.array_equal(actual,a))
    def test_compressed_or_truncated_tiff_is_not_misreported_as_independent_success(self):
        f=io.BytesIO();tifffile.imwrite(f,np.ones((6,8),dtype=np.uint8),compression='deflate',metadata=None)
        with self.assertRaises(ValueError):manual_tiff(f.getvalue())
        f=io.BytesIO();tifffile.imwrite(f,np.ones((6,8),dtype=np.uint8),metadata=None)
        with self.assertRaises(ValueError):manual_tiff(f.getvalue()[:-10])


if __name__=='__main__':unittest.main()
