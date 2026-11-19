"""Delivered-column and fixed-exposure permissions follow the ordinary registration path."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import PropertyMock, patch

import torch

from evaluation.frozen_training_stages import json_pin
from scripts.run_frozen_band_region import _write
from transfer import native_independent_release as independent
from transfer import wp_paired_structure as training
from transfer import wp_paired_structure_fixture as fixture
from transfer import wp_support_match as support


class ScientificAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.out = cls.root/"campaign"
        cls.assets = cls.root/"fixture"
        cls.code = dict(commit="a"*40,source_hash="b"*64)
        cls.files = fixture.generate_scientific(cls.assets,cls.out,cls.code,[])
        cls.request = json_pin(cls.root,cls.files["request"])

    def intake(self, entry):
        return independent.open_independent(self.root,entry["manifest"],entry["authority"],test_history=[])

    def altered(self, entry, name, edit):
        manifest, approval = (json_pin(self.root,entry[k]) for k in ("manifest","authority"))
        edit(manifest,approval)
        pin = _write(self.root/(name+"-intake.json"),manifest)
        approval["approved_manifest_sha256"] = pin["sha256"]
        return {**entry,"manifest":pin,"authority":_write(self.root/(name+"-authority.json"),approval)}

    def test_column_order_and_deletions_are_approved_without_wavelengths(self):
        entry = self.request["releases"]["xz-anchor"]
        bank,intake,roles = training.admit(self.root,self.out,entry,test=True)
        self.assertFalse(intake["physical_wavelengths_verified"] or intake["uses"]["cost"]["eligible"])
        observed = {}
        for view in independent.COLUMN_VIEWS:
            actual,_,targets = independent.materialize_reviewed(bank,intake,roles,"evaluation",
                roles["evaluation"][0],self.code["commit"],view,use="independent_accuracy")
            observed[view] = actual.receipt["source"]
            self.assertEqual(actual.view.bands.wavelength_unit,"normalized")
            self.assertEqual(len(targets.points),63)
        self.assertEqual(observed["full"]["spectral_metadata_mode"],"index")
        self.assertEqual(observed["metadata_missing"]["spectral_metadata_mode"],"missing")
        self.assertEqual(observed["full"]["band_ids"],observed["metadata_missing"]["band_ids"])
        self.assertNotEqual(observed["central_gap"]["band_ids"],observed["matched_dispersed"]["band_ids"])
        self.assertEqual(len(observed["central_gap"]["band_ids"]),len(observed["matched_dispersed"]["band_ids"]))
        with self.assertRaisesRegex(ValueError,"separate registrations"):
            training.admit(self.root,self.out,{k:v for k,v in entry.items() if k!="observation_protocol"},test=True)
        with self.assertRaisesRegex(ValueError,"approved actual column order"):
            self.intake(self.altered(entry,"order",lambda m,a:m["delivered_columns"]["axis"].reverse()))

    def test_cost_only_uses_observation_projection_and_never_enters_training(self):
        from evaluation.frozen_training_stages import file_pin

        entry = json_pin(self.root,file_pin(self.assets/"cost-only-entry.json"))
        bank,intake,roles = self.intake(entry)
        with patch.object(bank,"role_row",side_effect=AssertionError("label row read")), \
                patch.object(independent,"bank_view",side_effect=AssertionError("label binding")):
            actual,labels,targets = independent.materialize_reviewed(bank,intake,roles,"evaluation",
                roles["evaluation"][0],self.code["commit"],"metadata_missing",use="cost")
        self.assertIsNone(labels)
        self.assertIsNone(targets)
        self.assertNotIn("samples",actual.receipt["source"])
        with self.assertRaisesRegex(ValueError,"eligible three-role"):
            training.admit(self.root,self.out,entry,test=True)

    def test_k_seed_and_spatial_inventory_and_foreign_bank_fail(self):
        releases,groups = self.request["releases"],self.request["groups"]
        for suffix,factor in (("k1","support_k"),("k10","support_k"),("draw64","support_seed"),
                              ("draw2048","support_seed"),("layout-b","spatial_layout")):
            keys = ["xz-anchor","xz-"+suffix]
            banks = []
            for key in keys:
                bank,_,roles = self.intake(releases[key])
                banks.append((bank,roles,support.inventory(bank,roles,releases[key]["k"])))
            args = (factor,*banks,*(groups[k] for k in keys),*(releases[k] for k in keys))
            readout = support.validate_contrast(*args)
            self.assertEqual(readout["same_evaluation_points"],factor!="spatial_layout")
            if factor=="support_k":
                bank = banks[0][0]
                foreign = deepcopy(banks[1][0].members)
                foreign[next(iter(foreign))]["content"]["array_sha256"] = "0"*64
                with patch.object(type(bank),"members",new_callable=PropertyMock,
                                  side_effect=[bank.members, foreign]):
                    with self.assertRaisesRegex(ValueError,"actual observations"):
                        support.validate_contrast(*args)
            changed = {**groups[keys[1]],"seed":512}
            with self.assertRaisesRegex(ValueError,"identical model seed"):
                support.validate_contrast(*args[:4],changed,*args[5:])
        self.assertEqual(support.choose([(1,1),(2,2),(3,1),(4,2)],1),[2,3])


if __name__ == "__main__":
    unittest.main()
