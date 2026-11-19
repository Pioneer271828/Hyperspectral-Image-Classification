"""Verify the existing native stage-two route through the released-point binding."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from evaluation.frozen_training_stages import json_pin
from scripts.run_native_released_points import arithmetic
from transfer import wp_cross_backend as backend, wp_region_graph as region
from transfer.controlled_regional_support import invariants, match_support
from transfer.native_support import NativeSupportModel
from transfer.wp_paired_structure import base_model, admit
from transfer.wp_paired_structure_fixture import generate_regions
from transfer.g3_training import state_hash


class RegionalGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads,torch.get_num_threads())
        torch.set_num_threads(1)
        directory = TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        cls.code = dict(commit="a"*40,source_hash="b"*64)
        cls.campaign = cls.root/"campaign"
        fixture = generate_regions(cls.root/"fixture",cls.campaign,cls.code,[])
        cls.request = json_pin(cls.root,fixture["request"])
        cls.bank,cls.intake,cls.roles = admit(cls.root,cls.campaign,cls.request["releases"]["xz-k1"],test=True)

    def test_legacy_weights_restore_with_exact_inventory_and_capacity(self):
        baseline,_ = base_model(self.bank,64,None)
        legacy = NativeSupportModel.initialize(baseline,7385)
        current = backend.restore(legacy.export_config(),region.BACKEND,legacy.state_dict(),state_hash(legacy))
        self.assertEqual(legacy.export_config(),current.export_config())
        self.assertEqual(state_hash(legacy),state_hash(current))
        self.assertEqual(sum(p.numel() for p in current.parameters())-sum(p.numel() for p in baseline.parameters()),4096)

    def test_same_budget_changes_real_edges_and_rejects_tampering(self):
        from transfer.native_independent_release import materialize_reviewed

        tile = self.roles["evaluation"][1]
        actual,_,targets = materialize_reviewed(self.bank,self.intake,self.roles,"evaluation",tile,
                                                self.code["commit"],"full",use="independent_accuracy")
        base,_ = base_model(self.bank,64,None)
        model = region.RegionSupportModel.initialize(base,7385).eval()
        source = deepcopy(model).requires_grad_(False)
        proofs = {}
        with torch.no_grad(), arithmetic(self.root,None) as exec:
            for policy in ("original","random_fixed","spectral_fixed","spectral_bounded"):
                program = region.GraphProgram(source,policy,19064,40)
                _,g,_ = region.forward_region(model,actual,self.code["commit"],"devval",exec,program)
                proofs[policy] = region.verify_trace(g["regional_support"],actual.receipt["observation"])
        old = proofs["original"]
        for key,new in proofs.items():
            self.assertEqual(new["positive_nnz"],old["positive_nnz"])
            self.assertEqual(new["counts"]["receive"],old["counts"]["receive"])
            self.assertEqual(new["first_indices"],old["first_indices"])
            self.assertEqual(new["newly_added"] > 0,key!="original")
        altered = deepcopy(proofs["spectral_fixed"])
        altered["positive_nnz"] -= 1
        with self.assertRaisesRegex(ValueError,"Recounted"):
            region.verify_trace(altered,actual.receipt["observation"])
        signal = self.bank.role_row("devval",tile)["origin"]["regional_signal"]
        target_y = [y for _,y in targets.points]
        visible = region.signal_visibility(signal,old,actual.receipt["source"]["band_ids"],
                                           targets.cells.tolist(),target_y)
        self.assertEqual(visible["case"],"across_region")
        self.assertTrue(visible["cue_present"])
        missing = region.signal_visibility(signal,old,[],targets.cells.tolist(),target_y)
        self.assertFalse(missing["cue_present"])
        self.assertTrue(all(c["reachable"]==0 for c in missing["classes"].values()))

    def test_strict_match_has_exact_directional_degrees_or_is_infeasible(self):
        base = sorted([(i,i) for i in range(4)]+[(0,2),(1,3)])
        pool = sorted(base+[(0,3),(1,2)])
        coords = [(0,0),(2,0),(1,1),(1,-1)]
        matched = match_support(base,pool,[],coords,[0]*4,list(range(4)),seed=64)
        self.assertEqual(matched["status"],"MATCHED")
        self.assertEqual(invariants(base,coords),invariants(matched["pairs"],coords))
        self.assertEqual(match_support(base,base,[],coords,[0]*4,list(range(4)),seed=64)["status"],"INFEASIBLE")
