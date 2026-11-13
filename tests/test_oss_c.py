"""Test the isolated short worker and audit input contracts with no extra libraries."""

import copy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from joint_training import _tree_fingerprint, schedule_for
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot
from scripts.run_oss_c_pair import authorized_launch, run_case
from test_calibrated_selection import items
from test_joint_training import checkpoints, hashes, settings
from transfer.oss_c_oracles import HEADS, confusion, qualification, require_production, validate_indices
from transfer.oss_c_vca import ARMS, GlobalReferenceRuntime, settings as study_settings


DEVICE = os.environ.get("OSS_C_DEVICE", "cpu")


class AuditContracts(TestCase):
    def test_roles_ids_and_missing_class(self):
        labels = torch.tensor([1,3,1])
        scores = torch.tensor([[.8,.1,.1],[.1,.2,.7],[.2,.3,.5]])
        one = dict(fused=scores)
        observed = confusion(labels,one,3,role="val",active_heads=HEADS[:1])
        self.assertEqual(observed["fused"]["empty_classes"],[2])
        self.assertEqual(observed["fused"]["confusion"],[[1,0,1],[0,0,0],[0,0,1]])
        with self.assertRaisesRegex(ValueError,"Only authorized"):
            confusion(labels,one,3,role="query",active_heads=HEADS[:1])
        with self.assertRaisesRegex(ValueError,"out-of-order heads"):
            confusion(labels,dict(cnn=scores,fused=scores,graph=scores),3,role="val",active_heads=HEADS)
        with self.assertRaisesRegex(ValueError,"one-based"):
            confusion(labels-1,one,3,role="train",active_heads=HEADS[:1])
        valid = torch.tensor([1,3,8])
        validate_indices(valid,valid)
        for wrong in ([3,1,8],[1,1,8],[1,3]):
            with self.assertRaisesRegex(ValueError,"row identities"):
                validate_indices(torch.tensor(wrong),valid)

    def test_failed_research_receipt_is_never_production(self):
        witness = qualification("pyg_coo_sum",test_execution_passed=True,
                                failed_tensors={"gradient":{"max_abs":1e-4}},numeric_scope="fixture")
        self.assertEqual(witness["numerical_qualification"],"failed")
        for value in (witness,dict(witness,production_authorized=True)):
            with self.assertRaisesRegex(ValueError,"cannot authorize"):
                require_production(value)
        with self.assertRaisesRegex(ValueError,"independently issued"):
            authorized_launch(None,None,{}, "native-64")


class VCAWorkerTests(TestCase):
    def test_fixed_coordinate_masks_and_existing_fallbacks(self):
        opts = study_settings(settings(),"exclude_out_of_reference")
        config = ResearchConfig.from_dict(opts["config"])
        adapter = GlobalReferenceRuntime("exclude_out_of_reference").build_model(config,{"paviaU":9}).backbone.adapter.eval()
        control = GlobalReferenceRuntime("matched_boundary_dropout").build_model(config,{"paviaU":9}).backbone.adapter.eval()
        control.load_state_dict(adapter.state_dict())
        x = torch.rand(2,3,5)
        wave = torch.tensor([400.,510.,820.,2450.,2508.])
        with torch.no_grad():
            out,t = adapter(x,wave,return_attention=True)
            ref,r = control(x,wave,return_attention=True)
            self.assertEqual(t["band_indices"].tolist(),[0,1,2,3])
            self.assertEqual(r["band_indices"].tolist(),[1,2,3,4])
            self.assertTrue(torch.equal(t["anchors"],r["anchors"]))
            order = torch.tensor([3,4,0,2,1])
            torch.testing.assert_close(out,adapter(x[:,:,order],wave[order]),rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(ref,control(x[:,:,order],wave[order]),rtol=1e-5,atol=1e-6)
            self.assertEqual(adapter(x[:,:,:1],wave[:1]).shape[-1],128)
            self.assertEqual(adapter(x,None).shape[-1],128)
            self.assertEqual(adapter(x,torch.full_like(wave,3000.)).shape[-1],128)
            adapter(x,wave,band_mask=torch.tensor([True,True,False,True,True]))
            self.assertEqual(len(adapter._eval_attention_cache),1)
            before = adapter(x,wave)
            adapter.value_proj[0].bias.add_(.5)
            self.assertFalse(torch.equal(before,adapter(x,wave)))
            with self.assertRaisesRegex(ValueError,"retain at least one"):
                adapter(x,wave,band_mask=torch.zeros(5,dtype=torch.bool))

    def test_three_arms_run_and_resume_exactly(self):
        data = items()
        for row in data.values():
            row["wavelengths"][-1] = 2508.
        source = code_snapshot()
        arms = {}
        init = None
        for arm in ARMS:
            opt = study_settings(settings(),arm)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(64)
                model = GlobalReferenceRuntime(arm).build_model(ResearchConfig.from_dict(opt["config"]),
                                                {name:item["class_num"] for name,item in data.items()})
            actual = _tree_fingerprint(model.state_dict())
            init = actual if init is None else init
            self.assertEqual(init,actual)
            arms[arm] = dict(arm=arm,settings=opt,settings_sha256=digest(opt),
                             schedule_sha256=digest(schedule_for(opt)),validation_steps=[3,6,9,12])
        reg = dict(cases=arms,data_manifest="synthetic-fixture")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for arm in ARMS:
                full,split = root/(arm+"-full"),root/(arm+"-split")
                self.assertEqual(run_case(reg,arm,data,source,full,device=DEVICE)["updates_this_call"],12)
                self.assertEqual(run_case(reg,arm,data,source,split,device=DEVICE,stop_after=5)["updates_this_call"],5)
                with self.assertRaisesRegex(ValueError,"identity mismatch"):
                    run_case(reg,next(key for key in ARMS if key!=arm),data,source,split,device=DEVICE,resume=True)
                self.assertEqual(run_case(reg,arm,data,source,split,device=DEVICE,resume=True)["updates_this_call"],7)
                f,s = checkpoints(full),checkpoints(split)
                for key in ("model","optimizer","scheduler","rng","trace","history","cursor"):
                    self.assertEqual(_tree_fingerprint(f[key]),_tree_fingerprint(s[key]),key)
                self.assertNotEqual(f["model"]["backbone.adapter.anchor_offsets"].count_nonzero().item(),0)
                self.assertEqual(set(f["best_probabilities"]["paviaU"]["val"]),set(HEADS))
                before = hashes(split)
                self.assertEqual(run_case(reg,arm,data,source,split,device=DEVICE,resume=True)["updates_this_call"],0)
                self.assertEqual(hashes(split),before)
