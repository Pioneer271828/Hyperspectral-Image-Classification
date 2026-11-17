"""Full synthetic two-stage roster, budget, recovery and closed gates."""
from copy import deepcopy
import csv
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation import allscene_probe_registry as scenes, frozen_probe_audit as audit, frozen_probe_registry as reg
from evaluation import linear_probe, v2_paired_probe_campaign as campaign, v2_paired_probe_registry as registry
from g4_matched_target_study import METHODS, MODES, case_name
from research_runtime import code_snapshot, hash_file, write_json
from tests.test_stable_probe_numerics import probe_fixture


def synthetic_sources(root):
    """Replace only pinned upstream release authority and historical sensitivity reference."""
    pins,sources,cases,legacy = {},{},{},{}
    paths = {"full":[5],"split":[2,3,5],"completed":[5]}
    columns = {stage:reg.probe_columns(64,64,w) for stage,w in (("stage1",128),("stage2",64))}

    def pin(path):
        sha = hash_file(path)
        pins[str(path)] = sha
        return {"path":str(path),"sha256":sha}

    for key in ("scope","execution","review","diagnosis","draft"):
        path = root/f"source-{key}"/"receipt.json"
        path.parent.mkdir()
        write_json(path,{"synthetic_fixture":key})
        sources[key] = pin(path)
    plans = {}
    for scene,classes in scenes.CLASSES.items():
        pristine,plan = probe_fixture(classes)
        plans[scene] = {**plan,"steps":5}
        overlay = {f"{role}_{name}":pristine[key][role] for role in ("train","val")
                   for key,name in (("labels","labels"),("row_ids","indices"))}
        for seed,method,mode in product(reg.SEEDS,METHODS[:2],MODES):
            name = case_name(scene,seed,method,mode)
            case = {"name":name,"target":scene,"seed":seed,"method":method,"mode":mode,
                "initial_step":0,"terminal_step":500,"stages":{},"regional_effective_dimensions":{},
                "released_rows_sha256":linear_probe.fingerprint(overlay),
                "authorized_roles_fingerprints":{key:linear_probe.fingerprint(pristine[key]) for key in ("labels","row_ids")}}
            legacy[name] = {}
            for stage in registry.ROLES:
                folder = root/"packs"/name/stage
                folder.mkdir(parents=True)
                pack = deepcopy(pristine)
                identity = {"query_labels_accessed":False,"case":name,"stage":stage,"synthetic":True}
                pack["identity"] = identity
                if stage=="terminal":
                    for feature in pack["features"].values():
                        feature["train"] *= 1.5
                path = folder/"pack.pt"
                torch.save(pack,path)
                reference = {layer:{role:torch.zeros((len(labels),classes),dtype=torch.float64)
                                    for role,labels in pack["labels"].items()} for layer in reg.LAYERS}
                previous = folder/"selected-logp.pt"
                torch.save(reference,previous)
                capture = folder/"receipt.json"
                write_json(capture,identity)
                case["stages"][stage] = {"pack":pin(path),"capture":pin(capture),"legacy_probabilities":pin(previous),
                    "model_state_sha256":hash_file(capture),"pack_identity":identity,"fingerprint":linear_probe.fingerprint(pack)}
                old = {layer:{"selected_step":1,"roles":{role:{"selected":{"oa":1/classes,"nll":2.}}
                       for role in ("train","val")}} for layer in reg.LAYERS}
                legacy[name][stage] = {"layers":old}
            cases[name] = case
    return registry.Evidence(sources,list(cases.values()),plans,paths,columns,pins,legacy,synthetic=True)


class V2PairedProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        cls.root = Path(scratch.name)
        cls.evidence = synthetic_sources(cls.root)
        cls.code = code_snapshot()
        auth = patch.object(registry,"authenticate",return_value=cls.evidence)
        auth.start()
        cls.addClassCleanup(auth.stop)
        cls.scope = registry.prepare((),cls.root/"scope",cls.code)
        cls.scope_sha = hash_file(cls.scope)
        cls.execution = campaign.run(cls.scope,cls.scope_sha,cls.root/"run",cls.code)
        cls.execution_sha = hash_file(cls.execution)
        cls.review = campaign.review(cls.scope,cls.scope_sha,cls.execution,cls.execution_sha,cls.root/"review",cls.code)

    def test_new_v2_complete_roster_budget_selected_audits_and_zero_resume(self):
        scope = json.loads(self.scope.read_text())
        execution = json.loads(self.execution.read_text())
        result = json.loads(self.review.read_text())
        self.assertTrue(result["synthetic"])
        self.assertEqual(scope["planned_total_updates"],4320)
        self.assertEqual(execution["updates_by_stage"],{"initial":2160,"terminal":2160})
        self.assertEqual((result["status"],result["accepted_pairs"],result["accepted_heads"]),("passed",54,432))
        self.assertEqual(result["independent_selected_probability_endpoints"],864)
        self.assertEqual(result["completed_resume_updates"],0)
        self.assertEqual(result["v1_heads_reused"],0)
        self.assertTrue(self.review.with_name("acceptance.json").is_file())
        for row in execution["cases"]:
            for stage in registry.ROLES:
                value = row["stages"][stage]
                self.assertEqual([segment["probe_updates"] for segment in value["segments"]],[20,8,4,8,0])
        stage = json.loads(self.review.with_name(result["cases"][0]["case"]+".json").read_text())["initial"]
        self.assertTrue(stage["exact_continuous_split_and_completed"])
        self.assertTrue(all(layer["independent_normalization"]["floor_branches_exact"] for layer in stage["layers"].values()))
        for filename,expected in (("metrics.csv",1728),("class-metrics.csv",23040),("trajectories.csv",432),
                                  ("v1-sensitivity.csv",864),("paired-status-v2/class-contrasts.csv",69120)):
            with (self.review.parent/filename).open() as stream:
                self.assertEqual(sum(1 for _ in csv.DictReader(stream)),expected)

    def test_tampered_scope_hash_roles_order_policy_budget_rejected_before_decode(self):
        with patch("torch.load",side_effect=AssertionError("must not decode")):
            with self.assertRaisesRegex(ValueError,"Pinned JSON"):
                registry.load(self.scope,"0"*64,self.code)
            for label,alter in (
                ("reorder",lambda d:d["cases"].reverse()),
                ("omit",lambda d:d["cases"].pop()),
                ("cross-scene",lambda d:d["cases"][0].update(target="xuzhou")),
                ("stages",lambda d:d["cases"][0]["stages"].update(initial=d["cases"][0]["stages"]["terminal"])),
                ("old-policy",lambda d:d["plans"]["paviaU"].update(policy=linear_probe.POLICY)),
                ("budget",lambda d:d.update(planned_total_updates=4324)),
                ("reuse",lambda d:d.update(historical_head_reuse=216))):
                doc = json.loads(self.scope.read_text())
                alter(doc)
                path = self.root/f"scope-{label}.json"
                write_json(path,doc)
                with self.assertRaisesRegex(ValueError,"roles, input hashes or budget"):
                    registry.load(path,hash_file(path),self.code)

    def test_pack_role_class_order_labels_shape_and_bad_sha(self):
        case = deepcopy(self.evidence.cases[0])
        source = case["stages"]["initial"]
        path = source["pack"]["path"]
        pack = torch.load(path,map_location="cpu",weights_only=True)
        classes = scenes.CLASSES[case["target"]]
        bad = deepcopy(case)
        bad["stages"]["initial"]["pack"]["sha256"] = "0"*64
        with patch("torch.load",side_effect=AssertionError("must not decode")):
            with self.assertRaises(ValueError):
                registry.load_pack(bad,"initial",classes)
        for name,alter in (
            ("cross-role",lambda x:x["identity"].update(stage="terminal")),
            ("class",lambda x:x["labels"]["train"].fill_(0)),
            ("row-order",lambda x:x["row_ids"].update(train=x["row_ids"]["train"].flip(0))),
            ("label-drift",lambda x:x["labels"].update(val=(x["labels"]["val"]+1)%classes)),
            ("layer-shape",lambda x:x["features"]["stage1_graph"].update(train=x["features"]["stage1_graph"]["train"][:,:64]))):
            changed,example = deepcopy(case),deepcopy(pack)
            alter(example)
            path = self.root/f"pack-{name}.pt"
            torch.save(example,path)
            changed["stages"]["initial"].update(pack={"path":str(path),"sha256":hash_file(path)},
                                                fingerprint=linear_probe.fingerprint(example))
            with self.assertRaisesRegex(ValueError,"pack identity|class or row support|ordered labels|feature shape"):
                registry.load_pack(changed,"initial",classes)

    def test_duplicate_allocation_and_over_budget_ledger(self):
        alternate = registry.prepare((),self.root/"another-scope",self.code)
        self.assertEqual(hash_file(alternate),self.scope_sha)
        with patch.object(registry,"preflight",side_effect=AssertionError("must not re-preflight")):
            with self.assertRaisesRegex(ValueError,"already has"):
                campaign.run(self.scope,self.scope_sha,self.root/"second-run",self.code)
            with self.assertRaisesRegex(ValueError,"already has"):
                campaign.run(alternate,self.scope_sha,self.root/"second-run",self.code)
        original = reg.pinned_json
        def read(path,sha):
            result = original(path,sha)
            if path==self.execution:
                result["cases"][0]["stages"]["terminal"]["segments"][0]["probe_updates"] += 4
            return result
        with patch.object(reg,"pinned_json",side_effect=read):
            with self.assertRaisesRegex(ValueError,"Execution interval"):
                campaign.checked_ledger(self.execution,self.execution_sha,self.scope,self.scope_sha,
                                        json.loads(self.scope.read_text()),self.code)

    def test_missing_stage_probability_file_blocks_independent_review(self):
        scope = json.loads(self.scope.read_text())
        recorded = json.loads(self.execution.read_text())["cases"][0]
        case = scope["cases"][0]
        path = self.execution.parent/case["name"]/"terminal"/"selected-logp.pt"
        backup = path.with_suffix(".saved")
        path.rename(backup)
        try:
            with self.assertRaises(FileNotFoundError):
                campaign.audit_stage(self.execution.parent,case,"terminal",recorded["stages"]["terminal"],
                                     scope,self.scope_sha,self.evidence.legacy[case["name"]]["terminal"])
        finally:
            backup.rename(path)

    def test_one_probability_failure_propagates_without_acceptance(self):
        original = audit.audit_probes
        touched = False
        def fail(*args,**kwargs):
            nonlocal touched
            result = original(*args,**kwargs)
            if not touched:
                touched = True
                proof = result["layers"]["stage1_cnn"]["roles"]["val"]["cross_backend"]
                proof["probabilities_close"] = False
                result["numerical_failures"] = [{"layer":"stage1_cnn","role":"val",**proof}]
            return result
        with patch.object(audit,"audit_probes",side_effect=fail):
            result_path = campaign.review(self.scope,self.scope_sha,self.execution,self.execution_sha,self.root/"rejected",self.code)
        result = json.loads(result_path.read_text())
        self.assertEqual((result["status"],result["accepted_pairs"],result["accepted_heads"]),("incomplete_or_failed",53,431))
        self.assertFalse(result_path.with_name("acceptance.json").exists())
        self.assertEqual(result["independent_selected_probability_endpoints"],864)
        summary = json.loads((result_path.parent/"paired-status-v2"/"summary.json").read_text())
        self.assertGreater(summary["contrast_status_counts"]["failed"],0)
        denied = [row for row in summary["class_protection"] if row["audit_status"]=="failed"]
        self.assertTrue(denied)
        self.assertTrue(all(row["formal_passes_2pp"] is None for row in denied))


if __name__ == "__main__":
    unittest.main()
