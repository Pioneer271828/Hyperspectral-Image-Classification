"""A small complete mixed-authority 54-pair test and closed-gate regressions."""
from copy import deepcopy
import csv
from pathlib import Path
from shutil import copyfile
import unittest
from unittest.mock import patch

from evaluation import frozen_probe_audit as audit,frozen_probe_campaign as core
from evaluation import initial_final_probe_campaign as campaign,initial_final_probe_registry as reg
from evaluation import initial_probe_pack as pack,terminal_probe_history as history
from g4_support_context import read
from model.research_config import digest
from research_runtime import hash_file,write_json
from tests import test_allscene_frozen_probes as old


def declare_test_initializations(launch):
    common = {}
    for case in launch["cases"]:
        target,seed = case["target"],case["seed"]
        shared = common.setdefault((target,seed),{"target":target,"seed":seed,
            "fresh_state_sha256":digest([target,seed,"synthetic"])})
        sha = digest([case["name"],"synthetic_step_zero"])
        case.update(initial_state_sha256=sha,fresh_state_sha256=shared["fresh_state_sha256"],
            initialization={"initial_state_sha256":sha,"head_seed":seed+1000},graph_binding={"synthetic":target},
            target_context={"query_labels_accessed":False})
    launch["common_initials"] = list(common.values())


def initial_captures(root,scope,launch,boundary):
    root.mkdir()
    ordered = {case["name"]:case for case in launch["cases"]}
    common = {(row["target"],row["seed"]):row for row in launch["common_initials"]}
    producer = read(scope["cases"][0]["capture_receipt"])["code"]
    fields = {"policy":pack.CAPTURE_POLICY,"freeze_sha256":scope["freeze_sha256"],"analysis_sha256":scope["analysis_sha256"],
        "boundary_sha256":digest(boundary),"options":pack.OPTIONS,"optimizer_updates":0,"bn_updates":0,
        "raw_labels_deserialized":False,"new_query_release":False}
    rows = []
    for selected in scope["cases"]:
        name = selected["name"]
        case = ordered[name]
        folder = root/name
        folder.mkdir()
        capture = Path(selected["capture_receipt"])
        old_receipt = read(capture)
        for filename in old_receipt["outputs"]:
            copyfile(capture.parent/filename,folder/filename)
        path = folder/"receipt.json"
        write_json(path,{**fields,"status":"passed","case":name,"case_sha256":selected["case_sha256"],
            "stage_role":"pre_adaptation","selected_step":0,"model_forwards":1,
            "producer_commit":producer["commit"],"producer_source_hash":producer["source_hash"],
            "initialization":case["initialization"],"source_spec":case["source_spec"],
            "initial_state_sha256":case["initial_state_sha256"],"common_fresh":common[(case["target"],case["seed"])],
            "graph_binding":case["graph_binding"],"evidence":{**old_receipt["evidence"],"state_sha256":case["initial_state_sha256"]},
            "outputs":{file.name:hash_file(file) for file in folder.iterdir()}})
        rows.append({"case":name,"status":"captured","receipt":str(path),"sha256":hash_file(path),"model_forwards":1})
    path = root/"receipt.json"
    write_json(path,{**fields,"status":"passed","producer":producer,"base_commit":producer["commit"],
        "base_source_hash":producer["source_hash"],"initialization_only":False,"model_forwards":54,"cases":rows,
        "protected_files":boundary,"protected_files_unchanged":True})
    return path,producer


class InitialFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = old.AllSceneProbeTests
        cls.addClassCleanup(fixture.doClassCleanups)
        make_grid = old.make_grid
        with patch.object(old,"make_grid",side_effect=lambda root: make_grid(root,configure_launch=declare_test_initializations)):
            fixture.setUpClass()
        cls.root,cls.code = fixture.root,fixture.code
        cls.history_paths = (fixture.scope,fixture.execution,fixture.review)
        hashes = {key:hash_file(path) for key,path in zip(("scope","execution","review"),cls.history_paths,strict=True)}
        hashes["acceptance"] = hash_file(fixture.review.with_name("acceptance.json"))
        selected = read(fixture.scope)
        first = read(selected["cases"][0]["capture_receipt"])
        launch = read(Path(read(fixture.freeze)["training_root"])/"launch.json")
        cls.capture,producer = initial_captures(cls.root/"initial-captures",selected,launch,first["protected_files"])
        for module,values in ((history,{"COMMIT":cls.code["commit"],"SOURCE":cls.code["source_hash"],"SHA":hashes}),
                              (pack,{"PRODUCER":producer["commit"],"SOURCE":producer["source_hash"]})):
            guard = patch.multiple(module,**values)
            guard.start()
            cls.addClassCleanup(guard.stop)
        cls.scope = reg.prepare(cls.capture,hash_file(cls.capture),cls.history_paths,cls.root/"pair-scope",cls.code,synthetic=True)
        cls.sha = hash_file(cls.scope)
        cls.execution = campaign.run(cls.scope,cls.sha,cls.root/"pair-execution",cls.code)
        cls.run_sha = hash_file(cls.execution)
        cls.review = campaign.review(cls.scope,cls.sha,cls.execution,cls.run_sha,cls.root/"pair-review",cls.code)

    def test_exact_complete_pair_roles_and_terminal_reuse(self):
        result = read(self.review)
        self.assertEqual((result["status"],result["accepted_pairs"],result["paired_probe_endpoints"]),("passed",54,432))
        self.assertEqual((result["historical_probe_updates"],result["new_probe_updates"],result["total_probe_updates"]),
                         (2592,2592,5184))
        self.assertEqual(len([row for row in read(self.execution)["cases"] if row["status"]=="executed"]),54)
        metrics = read(self.review.with_name("summary.json"))
        self.assertEqual((metrics["endpoint_rows"],metrics["class_rows"],metrics["contrast_rows"],metrics["class_contrast_rows"]),
                         (1728,23040,5184,69120))
        with self.review.with_name("class-contrasts.csv").open() as stream:
            table = list(csv.DictReader(stream))
        hk = [row for row in table if row["scene"]=="whu-honghu" and row["contrast"]=="terminal_minus_initial"]
        self.assertEqual({int(row["class_id"]) for row in hk},set(range(1,23)))
        self.assertTrue(all(float(row["recall"])==0 for row in hk))
        with self.assertRaises(FileExistsError):
            campaign.run(self.scope,self.sha,self.root/"duplicate-pair-run",self.code)

    def test_historical_missing_original_file_blocks_consumer(self):
        row = next(row for row in read(self.history_paths[1])["cases"] if row["status"]=="executed")
        file = self.history_paths[1].parent/row["case"]/"full-0006"/"state.pt"
        backup = file.with_name("backup.pt")
        file.rename(backup)
        try:
            with self.assertRaises((ValueError,FileNotFoundError)):
                reg.load(self.scope,self.sha,self.code)
        finally:
            backup.rename(file)

    def test_changed_initial_roles_and_budget_fail_closed(self):
        selected = read(self.scope)
        bad = self.root/"changed-budget.json"
        write_json(bad,{**selected,"planned_new_probe_updates":selected["expected_probe_updates"]})
        with self.assertRaisesRegex(ValueError,"identities or budget"):
            reg.load(bad,hash_file(bad),self.code)
        raw = read(self.capture)
        capture = Path(raw["cases"][0]["receipt"])
        original = capture.read_bytes()
        original_manifest = self.capture.read_bytes()
        try:
            altered = read(capture)
            altered["stage_role"] = "selected_model"
            write_json(capture,altered)
            raw["cases"][0]["sha256"] = hash_file(capture)
            write_json(self.capture,raw)
            with self.assertRaisesRegex(ValueError,"role or state proof"):
                pack.authenticate(self.capture,hash_file(self.capture),read(self.history_paths[0]),{},synthetic=True)
        finally:
            capture.write_bytes(original)
            self.capture.write_bytes(original_manifest)

    def test_pack_rows_and_scientific_role_cannot_be_swapped(self):
        scope,_,cases,_,_,_ = reg.load(self.scope,self.sha,self.code)
        case,bound,local = next(row for row in cases if row[0]["target"]=="paviaU")
        folder = self.execution.parent/case["name"]
        plan = reg.case_scope(scope,case)
        loaded,*_ = core.checked_execution(folder,hash_file(folder/"execution.json"),plan,self.sha,case,local,contract=pack.contract)
        for key in ("stage_role","reordered","swapped","missing-layer","wrong-scene"):
            value = deepcopy(loaded)
            if key=="stage_role":
                value["identity"]["stage_role"]="selected_model"
            elif key=="reordered":
                value["row_ids"]["train"] = value["row_ids"]["train"].roll(1)
            elif key=="swapped":
                value["labels"]["train"],value["labels"]["val"] = value["labels"]["val"],value["labels"]["train"]
            elif key=="wrong-scene":
                value["identity"]["case"]="xuzhou-s64-none-target_scratch_full"
            else:
                value["features"].pop("stage2_graph")
            with self.assertRaises(ValueError):
                pack.independent(value,bound,plan,case,local)

    def test_strict_numerical_failure_keeps_all_rows(self):
        original = audit.numpy_logp
        observed = []

        def altered(*args):
            output = original(*args)
            if not observed:
                output[0,output[0].argmax()] = -1e6
            observed.append(True)
            return output

        with patch.object(audit,"numpy_logp",side_effect=altered):
            path = campaign.review(self.scope,self.sha,self.execution,self.run_sha,self.root/"numerical-failure",self.code)
        result = read(path)
        self.assertEqual((result["status"],result["accepted_pairs"],result["paired_probe_endpoints"]),
                         ("incomplete_or_failed",53,432))
        self.assertEqual(result["failed_initial_probes"],1)
        self.assertFalse(path.with_name("acceptance.json").exists())
