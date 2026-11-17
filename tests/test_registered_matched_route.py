"""Metadata and execution-boundary regressions on actual synthetic 9/22-class route models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import milp
import torch

from loadData.observation_sampling import BandMetadata
from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot
from transfer import matched_route_admission as admission, registered_development as dev
from transfer import registered_matched_route as pipeline
from transfer.curriculum_exposure import ExposurePlan
from transfer.development_training import prepare_scene
from transfer.graph_route_context import RouteContextBank, route_state_hash


def synthetic_candidate(root,classes,*,fixed=False,steps=24):
    code = code_snapshot()
    inputs = root/"inputs"
    inputs.mkdir(parents=True)
    layout = make_tile_plan((60,60),20,1,2026)
    record = make_development(np.resize(np.arange(1,classes+1),(60,60)),layout,list(range(1,classes+1)))
    cube = np.random.default_rng(3).normal(size=(60,60,8)).astype(np.float32)
    bands = BandMetadata("synthetic-route",tuple(range(8)),tuple(np.linspace(400,1000,8)),
                         "nm","approximate",(True,)*8)
    scene = prepare_scene(cube,bands,layout,record)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(512)
        source = MultiDatasetVCASAHGCN({scene.dataset:classes},latent_channels=8,adapter_embed_dim=8,
            locality_window=None,norm_type="group_norm",graph_mode="learned",
            graph_route={"budget":4,"mode":"weight","seed":64}).eval()
    source.backbone.adapter.cache_attention = False
    source_path = inputs/"source.pt"
    save_model_checkpoint(source,source_path)
    development = dev.export_development(scene,record,inputs/"data")
    plan = ExposurePlan.freeze(scene,record,updates=24,seed=64,
        minimum_class_blocks={str(c):1 for c in record["classes"]},minimum_support_visits=1)
    bank = bank_pin = None
    if fixed:
        ids = [row["id"] for row in record["roles"] if row["role"] in ("devtrain","devval")]
        bank = RouteContextBank.freeze(source,scene,record,tile_ids=ids,step=0,expected_source_state=route_state_hash(source))
        bank_pin = dev._json_new(inputs/"bank.json",bank.to_dict())
    reg = dev.register_route(source_path,development,plan,inputs/"registration.json",lr=.0001,torch_seed=512,
        expected_commit=code["commit"],bank_path=None if bank_pin is None else bank_pin["path"],
        expected_bank_hash=None if bank is None else bank.bank_hash)
    dev.execute_route(reg,out=root/"donor",to_step=steps,expected_commit=code["commit"])
    return {"name":f"synthetic-{classes}","registration":reg,"attempt":dev._pin(root/"donor/status.json"),
            "provenance":None,"source_issue":403}


def attest(root,candidate,*,changes=None):
    root.mkdir(parents=True)
    summary,*_ = admission.metadata(candidate)
    proof = {**summary,"policy":admission.ORIGIN,"status":"accepted","synthetic":True,
        "runtime":admission.runtime_identity(),"registration":candidate["registration"],"attempt":candidate["attempt"],
        "raw_gt_accessed":False,"query_accessed":False,**(changes or {})}
    candidate = {**candidate,"provenance":dev._json_new(root/"provenance.json",proof)}
    return candidate,dev._json_new(root/"catalog.json",{"policy":admission.CATALOG,"candidates":[candidate]})


class ProducerCompatibilityTests(unittest.TestCase):
    def test_cross_commit_uses_entire_non_test_source_in_both_directions(self):
        def snapshot(commit, files):
            return {"commit":commit, "dirty":False, "source_hash":digest(files), "files":files}

        runtime = {path:"old" for path in ("research_runtime.py", "train_vca_multidataset.py",
            "inference/light_cnn.py", "compression/int8_model.py", "configs/route.json")}
        producer = snapshot("producer", {**runtime, "tests/test_old.py":"old"})
        consumer = snapshot("consumer", {**runtime, "tests/test_new.py":"new"})
        self.assertEqual(admission.producer_compatibility(producer, consumer), "producer_runtime_files_identical")
        self.assertEqual(admission.producer_compatibility(producer, producer), "exact_source")
        for path in runtime:
            with self.subTest(path=path, change="edit"):
                with self.assertRaisesRegex(ValueError, "not byte-compatible"):
                    admission.producer_compatibility(producer, snapshot("consumer", {**runtime, path:"new"}))
            with self.subTest(path=path, change="delete"):
                with self.assertRaisesRegex(ValueError, "not byte-compatible"):
                    admission.producer_compatibility(producer, snapshot("consumer", {k:v for k,v in runtime.items() if k!=path}))
        for path in ("inference/__init__.py", "configs/new.json", "scripts/new_route_cli.py"):
            with self.subTest(path=path, change="add"):
                with self.assertRaisesRegex(ValueError, "not byte-compatible"):
                    admission.producer_compatibility(producer, snapshot("consumer", {**runtime, path:"new"}))
        with self.assertRaisesRegex(ValueError, "Consumer source manifest"):
            admission.producer_compatibility(producer, {**consumer, "source_hash":"invalid"})


class RegisteredMatchedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.code = code_snapshot()
        cls.candidate,cls.catalog = attest(cls.root/"authority",synthetic_candidate(cls.root/"nine",9))
        cls.scope = pipeline.prepare(cls.catalog,cls.candidate["name"],cls.root/"scope",cls.code,allow_synthetic=True)

    def test_metadata_first_synthetic_gate_and_missing_provenance(self):
        no_proof = {**self.candidate,"provenance":None}
        with patch("torch.load",side_effect=AssertionError("Before checkpoint decoding")):
            with self.assertRaisesRegex(ValueError,"synthetic"):
                admission.restore(self.candidate,self.code)
            with self.assertRaisesRegex(ValueError,"provenance"):
                admission.restore(no_proof,self.code,allow_synthetic=True)
            bad = {**self.candidate,"attempt":{**self.candidate["attempt"],"sha256":"0"*64}}
            with self.assertRaisesRegex(ValueError,"identity changed"):
                admission.restore(bad,self.code,allow_synthetic=True)

    def test_wrong_origin_scene_seed_runtime_and_incomplete_attempt(self):
        for key,change in (("scene","whu-honghu"),("model_seed",99),("support_seed",99),("runtime",{})):
            cand,_ = attest(self.root/f"bad-{key}",self.candidate,changes={key:change})
            with patch("torch.load",side_effect=AssertionError("Must refuse metadata first")):
                with self.assertRaisesRegex(ValueError,"scene, seeds"):
                    admission.restore(cand,self.code,allow_synthetic=True)
        old = admission.read(self.candidate["attempt"])
        altered = self.root/"short"
        altered.mkdir()
        attempt = dev._json_new(altered/"status.json",{**old,"status":"partial","completed_updates":7})
        cand,_ = attest(self.root/"short-proof",{**self.candidate,"attempt":attempt})
        with patch("torch.load",side_effect=AssertionError("Must refuse incomplete donor")):
            with self.assertRaisesRegex(ValueError,"completed original dynamic"):
                admission.restore(cand,self.code,allow_synthetic=True)

    def test_fixed_donor_missing_tile_label_tamper_and_wrong_source_rejected(self):
        cand,_ = attest(self.root/"fixed-proof",synthetic_candidate(self.root/"fixed",9,fixed=True,steps=2))
        with patch("torch.load",side_effect=AssertionError("Must refuse fixed donor")):
            with self.assertRaisesRegex(ValueError,"fixed bank"):
                admission.restore(cand,self.code,allow_synthetic=True)
        reg = admission.read(self.candidate["registration"])
        raw = admission.read(reg["development"])
        for role in ("devtrain","devval"):
            altered = self.root/f"missing-{role}"
            altered.mkdir()
            copied = deepcopy(raw)
            copied["blocks"].remove(next(row for row in copied["blocks"] if row["role"]==role))
            data = dev._json_new(altered/"development.json",copied)
            with self.assertRaisesRegex(ValueError,"Missing entire"):
                admission.file_pins(data,copied)
        source = Path(reg["source"]["path"])
        original = source.read_bytes()
        try:
            source.write_bytes(original+b"x")
            with patch("torch.load",side_effect=AssertionError("Wrong source must fail before decoding")):
                with self.assertRaisesRegex(ValueError,"Original input hash"):
                    admission.restore(self.candidate,self.code,allow_synthetic=True)
        finally:
            source.write_bytes(original)
        data = Path(reg["development"]["path"])
        original = data.read_bytes()
        try:
            raw["record"]["validation_support"]["labels"].reverse()
            data.write_text(json.dumps(raw))
            with patch("torch.load",side_effect=AssertionError("Changed labels must fail before decoding")):
                with self.assertRaisesRegex(ValueError,"identity changed"):
                    admission.restore(self.candidate,self.code,allow_synthetic=True)
        finally:
            data.write_bytes(original)

    def test_frozen_review_all_labels_class_protection_and_no_updates(self):
        with patch("torch.optim.Adam.step",side_effect=AssertionError("Review cannot update")):
            pin = pipeline.frozen_review(self.scope,self.root/"review",self.code)
        result = admission.read(pin)
        self.assertEqual(result["status"],"passed")
        self.assertTrue(result["state_and_rng_unchanged"])
        for arm in result["independent"].values():
            self.assertEqual(arm["status"],"evaluated")
            for head in arm["heads"].values():
                self.assertEqual(sum(row["support"] for row in head["per_class"]),45)
                self.assertEqual(len(head["class_protection"]),9)

    def test_solver_timeout_keeps_incomplete_arms_and_no_recovery_acceptance(self):
        fake = SimpleNamespace(status=1,success=False,message="synthetic timeout")
        with patch("evaluation.matched_edge_controls.milp",return_value=fake):
            scope = pipeline.prepare(self.catalog,self.candidate["name"],self.root/"timeout",self.code,
                                     allow_synthetic=True)
        plan = admission.read(scope)
        self.assertEqual(plan["status"],"registered_incomplete")
        self.assertIn("synthetic timeout",plan["arms"]["random:64"]["reasons"][0]["reason"])
        proof = admission.read(pipeline.frozen_review(scope,self.root/"timeout-review",self.code))
        self.assertEqual(proof["status"],"incomplete_or_failed")
        self.assertEqual(proof["independent"]["low_weight"]["status"],"unavailable")
        audit = pipeline.audit_recovery(scope,self.root/"timeout-audit",self.code)
        self.assertEqual(admission.read(audit)["status"],"incomplete_or_failed")
        self.assertFalse((self.root/"timeout-audit"/"acceptance.json").exists())

    def test_infeasible_training_tile_blocks_the_whole_review_arm(self):
        registration = admission.read(self.candidate["registration"])
        blocks = admission.read(registration["development"])["blocks"]
        training_index = next(i for i,row in enumerate(blocks) if row["role"]=="devtrain")
        failing_call = 4*training_index+1
        calls = 0
        def fail_first_training_control(*args,**kwargs):
            nonlocal calls
            calls += 1
            if calls==failing_call:
                return SimpleNamespace(status=2)
            return milp(*args,**kwargs)
        with patch("evaluation.matched_edge_controls.milp",side_effect=fail_first_training_control):
            scope = pipeline.prepare(self.catalog,self.candidate["name"],self.root/"infeasible",self.code,
                                     allow_synthetic=True)
        value = admission.read(scope)
        preflight = admission.read(value["preflight"])
        row = value["arms"]["low_weight"]
        self.assertEqual(row["status"],"unavailable")
        self.assertEqual(len(row["reasons"]),1)
        self.assertIn(row["reasons"][0]["tile_id"],preflight["donor"]["devtrain_tiles"])
        review_plan = admission.read(value["review_plan"])
        self.assertTrue(all(x["control_plan"]["controls"]["low_weight"]["status"]=="optimal"
                            for x in review_plan["contexts"]))
        result = admission.read(pipeline.frozen_review(scope,self.root/"infeasible-review",self.code))
        self.assertEqual(result["status"],"incomplete_or_failed")
        self.assertIsNone(result["original"]["controls"]["low_weight"]["metrics"])
        self.assertEqual(result["independent"]["low_weight"]["status"],"unavailable")
        self.assertEqual(result["independent"]["none"]["status"],"evaluated")

    def test_three_identical_random_masks_still_have_separate_arm_identity(self):
        objectives = []
        def shared_random_objective(cost,*args,**kwargs):
            objectives.append(cost.copy())
            group_position = (len(objectives)-1)%4
            if group_position>1:
                cost = objectives[-group_position]
            return milp(cost,*args,**kwargs)
        with patch("evaluation.matched_edge_controls.milp",side_effect=shared_random_objective):
            scope = pipeline.prepare(self.catalog,self.candidate["name"],self.root/"duplicate-scope",self.code,
                                     allow_synthetic=True)
        result = admission.read(scope)
        self.assertEqual(result["duplicate_random_banks"],[["random:64","random:512","random:2048"]])
        for name in ("random:64","random:512"):
            proof = pipeline.execute(scope,name,"split-first",self.root/f"duplicate-{name[-2:]}",self.code)
            if name=="random:64":
                wrong = admission.read(proof)["attempt"]
        with self.assertRaisesRegex(ValueError,"Cross-arm"):
            pipeline.execute(scope,"random:512","split-last",self.root/"duplicate-wrong",self.code,resume=wrong)

    def test_allocation_cross_arm_and_frozen_bank_pin(self):
        a = pipeline.execute(self.scope,"none","split-first",self.root/"none-first",self.code)
        b = pipeline.execute(self.scope,"important","split-first",self.root/"important-first",self.code)
        with self.assertRaisesRegex(ValueError,"Cross-arm"):
            pipeline.execute(self.scope,"important","split-last",self.root/"wrong-arm",self.code,
                             resume=admission.read(a)["attempt"])
        with self.assertRaises(FileExistsError):
            pipeline.execute(self.scope,"none","split-first",self.root/"repeat",self.code)
        self.assertNotEqual(admission.read(a)["attempt"],admission.read(b)["attempt"])
        scope = admission.read(self.scope)
        bank = Path(scope["arms"]["none"]["bank"]["path"])
        original = bank.read_bytes()
        try:
            bank.write_bytes(original+b" ")
            with self.assertRaisesRegex(ValueError,"input hash differs"):
                pipeline.load_scope(self.scope,self.code)
        finally:
            bank.write_bytes(original)
        scope["arms"]["none"] = scope["arms"]["important"]
        alias = self.root/"alias"
        alias.mkdir()
        bad_scope = dev._json_new(alias/"scope.json",scope)
        with self.assertRaisesRegex(ValueError,"registered matched mask"):
            pipeline.load_scope(bad_scope,self.code)


if __name__=="__main__":
    unittest.main()
