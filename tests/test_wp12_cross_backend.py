"""Regression tests for the production multi-backend comparison and WP12 custody."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from evaluation import wp12_cross_backend as comparison, wp12_cross_backend_fixture as fixtures
from evaluation.frozen_training_stages import json_pin
from scripts.run_frozen_band_region import _write
from scripts.run_native_released_points import arithmetic
from transfer import wp_cross_backend as backend
from transfer.g3_training import state_hash


class CrossBackendComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads,torch.get_num_threads())
        torch.set_num_threads(1)
        cls.tmp = TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.code = {"commit": "a"*40}
        cls.request = fixtures.fixture(cls.root/"released","xuzhou","a",[64],views=("full",))
        with patch.object(comparison,"require_frozen_code"), arithmetic(cls.root,None) as execution:
            cls.capture = comparison.run_case(cls.root,cls.request,cls.root/"capture.json",cls.code,execution,test_history=[])

    def test_approved_points_recover_all_backends_and_disabled_head_parity(self):
        fixtures.validate_controls(self.root,self.capture)
        with patch.object(comparison,"require_frozen_code"):
            receipt = comparison.review_case(self.root,self.capture,self.root/"review.json",self.code,test_history=[])
        reviewed = json_pin(self.root,receipt)
        self.assertEqual(set(reviewed["roles"]),{"support","calibration","evaluation"})
        result = reviewed["cohorts"]["s64"]
        self.assertEqual({row["semantics"]["backend"] for row in result["arms"].values()},set(backend.BACKENDS))
        self.assertEqual(result["pairs"]["off/fixed1"]["full"]["accuracy"]["fused"]["delta_oa"],0)
        spatial = result["pairs"]["off/local"]["full"]["spatial"]
        self.assertTrue(spatial["boundary"]["n"]>0 and spatial["narrow"]["n"]>0)
        self.assertFalse(spatial["narrow"]["qualification"]["fused"]["protection_evaluable"])
        self.assertFalse(reviewed["query_accessed"] or reviewed["raw_gt_loaded"])

    def test_wrong_taxonomy_budget_phase_and_calibration_are_rejected(self):
        manifest = json_pin(self.root,self.request)
        group = manifest["groups"]["s64"]
        protocol = deepcopy(group["protocol"])
        protocol["class_axis"][0] = 0
        foreign = deepcopy(manifest)
        foreign["groups"]["s64"]["protocol"] = protocol
        bad = _write(self.root/"zero-class.json",foreign)
        with self.assertRaisesRegex(ValueError,"taxonomy"):
            comparison.open_case(self.root,bad,test_history=[])
        baseline = json_pin(self.root,group["arms"]["off"])
        wrong = deepcopy(baseline)
        wrong["step"] = 1
        with self.assertRaisesRegex(ValueError,"cursor"):
            comparison.validate_endpoint(wrong,baseline["protocol"],64)
        changed = deepcopy(manifest)
        wrong = deepcopy(baseline)
        wrong["phase"] = "selected"
        changed["groups"]["s64"]["arms"]["off"] = _write(self.root/"selected-off.json",wrong)
        with self.assertRaisesRegex(ValueError,"endpoint phases"):
            comparison.open_case(self.root,_write(self.root/"mixed-phase.json",changed),test_history=[])
        with self.assertRaisesRegex(ValueError,"isolated engineering history"):
            comparison.open_case(self.root,self.request)

    def test_invalid_role_order_observation_and_query_cannot_enter_wp12(self):
        edits = {
            "support": ("role","support","role or admitted use"),
            "point-order": ("order","f"*64,"points, observation"),
            "observation": ("observation","a"*64,"label-blind cost"),
            "query": ("query_accessed",True,"scope, provenance"),
        }
        for name,(key,value,reason) in edits.items():
            with self.subTest(name=name), patch.object(comparison,"require_frozen_code"):
                source = json_pin(self.root,self.capture)
                row = source["cohorts"]["s64"]["arms"]["off"]["rows"][0]
                place = source if key=="query_accessed" else row["accuracy"] if key=="order" else row["cost"] if key=="observation" else row
                place[key] = value
                pin = _write(self.root/(name+"-capture.json"),source)
                with self.assertRaisesRegex(ValueError,reason):
                    comparison.review_case(self.root,pin,self.root/("invalid-"+name+".json"),self.code,test_history=[])

    def test_backend_config_is_required_and_checkpoint_precision_cannot_cast(self):
        manifest = json_pin(self.root,self.request)
        spec = json_pin(self.root,manifest["groups"]["s64"]["arms"]["combined"])["checkpoint"]
        with self.assertRaisesRegex(ValueError,"declared backend"):
            backend.load_export(self.root,{**spec,"backend":"native"})
        model = backend.load_export(self.root,spec)
        self.assertEqual(state_hash(model),spec["state"])
        architecture = json_pin(self.root,spec["metadata"])
        weights = dict(model.state_dict())
        first = next(name for name,x in weights.items() if x.is_floating_point())
        weights[first] = weights[first].double()
        with self.assertRaisesRegex(ValueError,"precision"):
            backend.restore(architecture,"region_message",weights,spec["state"])

    def test_truth_class_zero_and_incompatible_source_are_never_paired(self):
        with self.assertRaisesRegex(ValueError,"zero class"):
            comparison.raw_pair([0,1],[1,1],[1,1],9,floor=5)


if __name__ == "__main__":
    unittest.main()
