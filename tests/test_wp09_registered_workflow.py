"""Label custody and effective support checks for joint WP09 publication."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from evaluation.wp09_registered_workflow import _counts, _reserve
from evaluation.registered_open_set import _load
from tests.test_registered_open_set import fixture


class RegisteredWP09WorkflowTests(unittest.TestCase):
    def test_distinguishes_repeated_exposure_and_unique_observation_support(self):
        frames=[{"frame_id":n, "observation":k} for n,k in (("f1","a"),("f2","a"),("f3","b"))]
        reg=SimpleNamespace(value={"frames":frames},observation_identity=lambda f:f["observation"])
        labels={"f1":np.array([1,0]),"f2":np.array([1,0]),"f3":np.array([2,0])}
        support=_counts(reg,labels,[1,2,3])
        self.assertEqual(support["excluded_zero_exposures"],3)
        self.assertEqual(support["excluded_zero_unique"],2)
        self.assertEqual(support["per_class"],[
            dict(class_id=1,frame_support=2,distinct_observation_offset_support=1),
            dict(class_id=2,frame_support=1,distinct_observation_offset_support=1),
            dict(class_id=3,frame_support=0,distinct_observation_offset_support=0),
        ])
        labels["f2"]=np.array([2,0])
        with self.assertRaisesRegex(ValueError,"change its offline ground truth"):
            _counts(reg,labels,[1,2,3])

    def test_reserves_deferred_paths_and_detects_hardlinks(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp)
            custody=base/"private-label-directory"
            custody.mkdir()
            label=custody/"later.txt"
            pre=base/"observation.txt"
            pre.write_text("synthetic")
            _reserve({"path":str(pre),"sha256":"a"*64},(label,))
            os.link(pre,label)
            with self.assertRaisesRegex(ValueError,"cannot alias"):
                _reserve({"path":str(pre),"sha256":"a"*64},(label,))
            with self.assertRaisesRegex(ValueError,"deferred label directory"):
                _reserve({"path":str(custody/"not-released"),"sha256":"a"*64},(label,))

    def test_source_validation_freeze_refuses_a_deferred_label_inode_alias(self):
        with TemporaryDirectory() as tmp:
            reg, value, _ = fixture(Path(tmp))
            os.link(value["model"]["path"], value["labels"]["path"])
            with self.assertRaisesRegex(ValueError,"cannot alias source-role evidence"):
                _load(reg,source_validation_only=True)
