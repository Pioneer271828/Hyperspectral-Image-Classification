"""Tests for the non-24 schedule and its real execution boundaries."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model.checkpoint_io import save_model_checkpoint
from research_runtime import code_snapshot
from tests.test_graph_route_context import setup
from transfer import registered_development as dev, route_campaign_allocation as allocation
from transfer import xuzhou_route_campaign as campaign
from transfer.curriculum_exposure import ExposurePlan
from transfer.development_training import WP02_SPATIAL_RECORD


class RouteCampaignTests(unittest.TestCase):
    def test_budget_prefixes_only_use_the_original_sampler(self):
        _,scene,record,_,_ = setup()
        original = deepcopy(record)
        with patch("torch.nn.Module.__call__",side_effect=AssertionError("No model forwards for choosing the budget")), \
             patch("torch.optim.Adam.step",side_effect=AssertionError("No updates for choosing the budget")):
            rows,selected = campaign.budget_prefixes(scene,record)
        self.assertEqual(record,original)
        self.assertEqual([row["budget"] for row in rows],[64,128,256])
        self.assertEqual(selected.updates,next(row["budget"] for row in rows if row["eligible"]))
        self.assertEqual([row["prefix_hash"] for row in rows],[
            self.plan(scene,record,256).progress(n)["prefix_hash"] for n in (64,128,256)])

    @staticmethod
    def plan(scene,record,budget):
        return ExposurePlan.freeze(scene,record,updates=budget,seed=64,
             minimum_class_blocks={str(c):1 for c in record["classes"]},minimum_support_visits=1)

    def test_real_claim_required_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = dev._json_new(root/"development.json",{"record":{"kind":WP02_SPATIAL_RECORD}})
            registration = dev._json_new(root/"registration.json",{"development":record})
            with patch("transfer.registered_development._load_registration",side_effect=AssertionError("Too late")):
                with self.assertRaisesRegex(ValueError,"authorized campaign claim"):
                    dev.execute_route(registration,out=root/"forbidden",to_step=256,expected_commit=code_snapshot()["commit"])
            self.assertFalse((root/"forbidden").exists())
        for wrong in (7,257,True,128.0):
            with self.assertRaises(ValueError):
                allocation.segments(wrong)

    def test_non24_actual_disk_restore_and_single_use(self):
        code = code_snapshot()
        source,scene,record,_,_ = setup()
        budget = 32
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root/"inputs"
            inputs.mkdir()
            weight = inputs/"source.pt"
            save_model_checkpoint(source,weight)
            data = dev.export_development(scene,record,inputs/"data")
            plan = self.plan(scene,record,budget)
            registrations = {arm:dev.register_route(weight,data,plan,inputs/(arm+".json"),lr=.0001,
                torch_seed=64,expected_commit=code["commit"]) for arm in ("donor","important")}
            config = campaign._allocation(code,budget,registrations)
            pin = dev._json_new(inputs/"allocation.json",config)
            run = lambda arm,stage,resume=None: allocation.execute(root,pin,config,arm,stage,code,resume=resume)
            full = run("donor","full")
            prefix = run("donor","split-first")
            other = run("important","split-first")
            with self.assertRaisesRegex(ValueError,"Cross-arm"):
                run("donor","split-last",campaign.read(other)["attempt"])
            foreign = dev._json_new(inputs/"foreign.json",config)
            with self.assertRaisesRegex(ValueError,"wrong arm, registration or budget"):
                allocation.execute(root,foreign,config,"donor","split-last",code,resume=campaign.read(prefix)["attempt"])
            split = run("donor","split-last",campaign.read(prefix)["attempt"])
            run("donor","completed",campaign.read(split)["attempt"])
            with self.assertRaises(FileExistsError):
                run("donor","full")
            result = campaign._audit_arm(root,pin,config,"donor")
            self.assertEqual(result["updates"],{"full":32,"split_first":7,"split_last":25,"completed":0})
            self.assertEqual(result["attempts"]["full"],campaign.read(full)["attempt"])
            self.assertEqual(campaign.read(dev._pin(root/"attempts"/"donor-full"/"measurement.json"))["new_updates"],2)
