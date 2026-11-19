"""Production regression for the structural worker and the historical released aliases."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.frozen_training_stages import json_pin
from evaluation.wp12_cross_backend_history import archive_alias_rosters
from model.research_config import digest
from scripts.run_frozen_band_region import _write
from scripts.run_native_released_points import arithmetic
from transfer import wp_paired_structure as structure, wp_paired_structure_fixture as fixture, wp_paired_structure_run as run
from transfer.xuzhou_multiscale_audit import _exact


def historical_roster():
    bank = digest("released Bank")
    rows = []
    for block, labels in enumerate(([6,6],[5,5])):
        for band in ("full","central_gap"):
            rows.append(dict(tile=block,points=[block*100+i for i in range(len(labels))],labels=labels[:],
                event=dict(band=band, factor=1, policy="native"),
                evidence=dict(bank=bank, role="devval", released=dict(
                    point_set=digest(["point_set", block]), source=digest(["source",block])))))
    return dict(preflight_population=bank, labels=[6,6,5,5], rows=rows)


class ArchivedPointAliasTests(unittest.TestCase):
    def test_noncontiguous_original_aliases(self):
        groups = archive_alias_rosters(historical_roster(),(5,6))
        self.assertEqual(groups[(1,"full","native")][1][2],[100,101])

    def test_duplicate_foreign_order_and_interchange_refused(self):
        for operation in ("duplicate","wrong_block","point_order","swapped_blocks","source_interchange","cross_role"):
            doc = historical_roster()
            if operation=="duplicate":
                doc["rows"][3]["points"] = [100,100]
            elif operation=="wrong_block":
                doc["rows"][3]["points"] = [0,1]
            elif operation=="point_order":
                doc["rows"][3]["points"].reverse()
            elif operation=="swapped_blocks":
                doc["rows"][1],doc["rows"][3] = doc["rows"][3],doc["rows"][1]
            elif operation=="source_interchange":
                doc["rows"][3]["evidence"]["released"],doc["rows"][1]["evidence"]["released"] = (
                    doc["rows"][1]["evidence"]["released"],doc["rows"][3]["evidence"]["released"])
            else:
                doc["rows"][1]["evidence"]["role"] = "devtrain"
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError,"alias|block identity"):
                archive_alias_rosters(doc,(5,6))


class PairedStructureTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.campaign = self.root/"campaign"
        self.code = {"commit":"a"*40,"source_hash":"c"*64}
        self.input = fixture.generate(self.root/"inputs",self.campaign,self.code,[],qualification=True)

    def register(self):
        return structure.register(self.root,self.campaign,self.code,self.input["request"],self.input["approval"],test=True)

    def test_durable_budget_and_state_exact_reentry(self):
        group,arm = "xz-recovery","combined"
        with patch.object(structure,"require_frozen_code"), patch.object(run,"require_frozen_code"), arithmetic(self.root,None) as execution:
            link = self.register()
            stop = run.execute(self.root,self.campaign,self.code,group,arm,execution,test=True,until=5)
            self.assertEqual((stop["status"],stop["cursor"],stop["updates"]),("drained",5,5))
            with self.assertRaisesRegex(ValueError,"not finished"):
                run.publication(self.root,self.campaign,self.code,test=True)
            self.assertFalse((self.campaign/"publication.json").exists())
            complete = run.execute(self.root,self.campaign,self.code,group,arm,execution,test=True)
            reference = run.execute(self.root,self.campaign,self.code,group,arm,execution,test=True,reference=True)
            reentry = run.execute(self.root,self.campaign,self.code,group,arm,execution,test=True)
            self.assertEqual((complete["updates"],reference["updates"],reentry["updates"]),(7,12,0))
            profile = json_pin(self.root,link)["groups"][group]
            a = run.completed(self.root,self.campaign,group,arm,link,profile)
            b = run.completed(self.root,self.campaign,group,arm,link,profile,reference=True)
            self.assertTrue(_exact(a[3],b[3]))
            self.assertGreater(a[1]["groups"]["messages"]["updated"],0)

    def test_mismatched_budget_approval_and_missing_roles_refused(self):
        body = json_pin(self.root,self.input["approval"])
        body["optimizer_updates"] -= 1
        invalid = _write(self.root/"wrong-approval.json",body)
        with patch.object(structure,"require_frozen_code"), self.assertRaisesRegex(ValueError,"training approval"):
            structure.register(self.root,self.campaign,self.code,self.input["request"],invalid,test=True)
        self.assertFalse(self.campaign.exists())
        request = json_pin(self.root,self.input["request"])
        invalid_group = deepcopy(request["groups"]["xz-recovery"])
        invalid_group["steps"] += 1
        with self.assertRaisesRegex(ValueError,"calibration clock"):
            structure.validate_group(invalid_group,request["releases"],test=True)
        entry = request["releases"]["xz"]
        manifest = json_pin(self.root,entry["manifest"])
        manifest["roles"]["evaluation"] = []
        new_pin = _write(self.root/"missing-roles.json",manifest)
        owner = json_pin(self.root,entry["authority"])
        owner["approved_manifest_sha256"] = new_pin["sha256"]
        with self.assertRaisesRegex(ValueError,"every published block"):
            structure.admit(self.root,self.campaign,{**entry,"manifest":new_pin,
                "authority":_write(self.root/"owner-new.json",owner)},test=True)
        with self.assertRaisesRegex(ValueError,"paired training request"):
            structure.request_spec(self.root,self.input["request"],test=False)

    def test_native_and_disabled_multiscale_share_parameter_identity(self):
        entry = json_pin(self.root,self.input["request"])["releases"]["xz"]
        bank,_,_ = structure.admit(self.root,self.campaign,entry,test=True)
        source = structure.open_source(self.root,bank,entry["source"])
        group = dict(seed=64,arms={"off":fixture.OFF,"fixed1":fixture.FIXED},pairs=[fixture.pair("off","fixed1",equal=True)])
        result = structure.start_arms(bank,group,source)
        a,b = result["off"],result["fixed1"]
        self.assertEqual(a["state"],b["state"])
        self.assertEqual(len(a["source"]["copied"]),29)
        self.assertEqual(len(a["source"]["graph"]),6)
        self.assertTrue(all(a["inventory"][k]==b["inventory"][k] for k in a["inventory"]))


if __name__=="__main__":
    unittest.main()
