"""Negative role gates and independently decoded synthetic TIFF bytes."""
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile
from PIL import Image

from audit_intake import manual_tiff,check_roles
from catalog import train_member
from fetch_metadata import fetch
from public_reader import read_aerial
import requests


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
    def test_independent_lzw_decoder_matches_pillow_across_dictionary_width_changes(self):
        a=np.random.default_rng(17).integers(0,256,size=(180,240),dtype=np.uint8)
        f=io.BytesIO();Image.fromarray(a).save(f,format='TIFF',compression='tiff_lzw')
        _,decoded=manual_tiff(f.getvalue());self.assertTrue(np.array_equal(decoded[:,:,0],a))
    def test_quarantined_metadata_binding_rejected_before_payload_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan={'role':'public_train_intake_only','samples':[{'image_id':'IMG_synthetic'}]};raw=json.dumps(plan).encode();(root/'plan.json').write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
            (root/'manifest.json').write_text(json.dumps({'plan_sha256':digest,'members':[]}));receipt=root/'audit.json';receipt.write_text(json.dumps({'passed':True,'plan_sha256':digest,'samples':[{'sample_opaque_id':hashlib.sha256(b'IMG_synthetic').hexdigest()[:16],'metadata_center_binding_admitted':False}]}))
            with self.assertRaisesRegex(ValueError,'quarantined'):read_aerial(root,0,audit_receipt=receipt)
    def test_failed_fetch_never_reports_signed_redirect_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('fetch_metadata.requests.get',side_effect=requests.HTTPError('https://example.invalid/?signed=synthetic-secret')):
                with self.assertRaises(RuntimeError) as caught:fetch('https://example.invalid/path?signature=synthetic-secret',root,'metadata.json',1024)
            self.assertNotIn('synthetic-secret',str(caught.exception));receipt=(root/'metadata.json.failure.json').read_text();self.assertNotIn('synthetic-secret',receipt);self.assertEqual(json.loads(receipt)['partial_file_bytes'],0)


if __name__=='__main__':unittest.main()
