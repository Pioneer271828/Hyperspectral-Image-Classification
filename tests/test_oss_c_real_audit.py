"""Exercise the independent audit on complete synthetic producer artifacts and tampering."""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from joint_calibrated_selection import observations
from joint_training import _tree_fingerprint, schedule_for
from model.bn_calibration import observation_id
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, hash_file
from scripts.audit_oss_c_real import (
    ARTIFACTS, CLASSES, STEPS, compare, evaluate_record,
    read, readouts, state, verify_artifacts, verify_checkpoint, verify_history, verify_trace,
)
from scripts.run_oss_c_pair import run_case
from scripts.train_joint_m4 import graph_binding
from test_calibrated_selection import items
from test_joint_training import settings
from transfer.oss_c_oracles import torchmetrics_oracle
from transfer.oss_c_vca import ARMS
from transfer.oss_c_vca import GlobalReferenceRuntime, settings as study_settings


class IndependentAuditTests(TestCase):
    @classmethod
    def setUpClass(cls):
        workspace = TemporaryDirectory()
        cls.addClassCleanup(workspace.cleanup)
        cls.execution = Path(workspace.name)
        cls.key = "exclude_out_of_reference-64"
        cls.directory = cls.execution / cls.key
        data = items()
        for item in data.values():
            item["wavelengths"][-1] = 2508.
        arm = "exclude_out_of_reference"
        opts = study_settings(settings(), arm)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(64)
            initial = GlobalReferenceRuntime(arm).build_model(ResearchConfig.from_dict(opts["config"]), CLASSES)
        cls.spec = dict(arm=arm, seed=64, settings=opts, settings_sha256=digest(opts), updates=12,
                        initial_state_sha256=_tree_fingerprint(initial.state_dict()),
                        schedule_sha256=digest(schedule_for(opts)), validation_steps=list(STEPS))
        cls.registered = dict(cases={cls.key: cls.spec}, data_manifest="fixture")
        cls.source = code_snapshot()
        result = run_case(cls.registered, cls.key, data, cls.source, cls.directory, device="cpu")
        if result["updates_this_call"] != 12:
            raise AssertionError("Expected a complete synthetic producer case")
        metadata = {scene: {"support": {role: [1] * count for role in ("train", "val")}}
                    for scene, count in CLASSES.items()}
        cls.labels = readouts(data, metadata)
        cls.graph = graph_binding(data)
        cls.observations = observation_id(observations(data))
        cls.result = read(cls.directory / "result.json")
        cls.history = read(cls.directory / "history.json")
        cls.probabilities = state(cls.directory / "selected-probabilities.pt")
        cls.write_snapshots(cls.directory)

    @classmethod
    def write_snapshots(cls, directory, key=None):
        rows = "".join(f"{hash_file(path)}  ./{path.name}\n" for path in sorted(directory.iterdir()))
        for phase in ("before", "after"):
            (cls.execution / f"{key or cls.key}.{phase}.sha256").write_text(rows)

    def test_selected_probabilities_and_all_checkpoints_recompute(self):
        actual = evaluate_record(self.result["selected_validation"], self.labels, probabilities=self.probabilities)
        self.assertEqual(sum(len(actual["metrics"][scene]["val"]["fused"]["per_class"]) for scene in CLASSES), 40)
        hashes = verify_artifacts(self.directory, self.execution, self.key)[1]
        last, scores = verify_checkpoint(self.directory, self.result, hashes,
                                         self.result["selected_step"], self.spec["settings"], self.observations)
        self.assertEqual(_tree_fingerprint(scores), _tree_fingerprint(self.probabilities))
        self.assertGreater(last["backbone.adapter.anchor_offsets"].count_nonzero().item(), 0)

    def test_optional_torchmetrics_counts_equal_offline(self):
        source = os.environ.get("OSS_C_AUDIT_TEST_TORCHMETRICS")
        if source is None:
            self.skipTest("Fixed TorchMetrics source is explicit opt-in")
        record = self.result["selected_validation"]
        expected = evaluate_record(record, self.labels, probabilities=self.probabilities)
        self.assertEqual(expected, evaluate_record(record, self.labels, probabilities=self.probabilities,
                                                   crosscheck=torchmetrics_oracle(Path(source))))
        with self.assertRaisesRegex(ValueError, "Fixed TorchMetrics source bytes"):
            torchmetrics_oracle(self.directory / "trace.json")

    def test_rejects_wrong_metric_wrong_macro_and_wrong_probability_order(self):
        recorded = self.result["selected_validation"]
        bad = deepcopy(recorded)
        bad["metrics"]["paviaU"]["val"]["fused"]["aa"] += .1
        with self.assertRaisesRegex(ValueError, "aa differs from confusion"):
            evaluate_record(bad, self.labels, probabilities=self.probabilities)
        bad = deepcopy(recorded)
        bad["macro_oa"] += .1
        with self.assertRaisesRegex(ValueError, "Macro OA differs"):
            evaluate_record(bad, self.labels, probabilities=self.probabilities)
        bad_scores = deepcopy(self.probabilities)
        bad_scores["paviaU"] = dict(reversed(tuple(bad_scores["paviaU"].items())))
        with self.assertRaisesRegex(ValueError, "Probability scene or role order"):
            evaluate_record(recorded, self.labels, probabilities=bad_scores)
        bad_scores = deepcopy(self.probabilities)
        bad_scores["paviaU"]["val"]["fused"] = bad_scores["paviaU"]["val"]["fused"].roll(1, 0)
        with self.assertRaisesRegex(ValueError, "Probability confusion|NLL differs"):
            evaluate_record(recorded, self.labels, probabilities=bad_scores)

    def test_rejects_missing_replaced_and_rehashed_artifacts(self):
        from shutil import copytree

        with TemporaryDirectory() as workspace:
            directory = Path(workspace) / "copy"
            copytree(self.directory, directory)
            path = directory / "selected-probabilities.pt"
            original = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
                verify_artifacts(directory, self.execution, self.key)
            path.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "Artifact hash"):
                verify_artifacts(directory, self.execution, self.key)
            revised = deepcopy(self.result)
            revised["artifacts"][path.name] = hash_file(path)
            (directory / "result.json").write_text(json.dumps(revised))
            with self.assertRaisesRegex(ValueError, "Completed-resume artifact hashes"):
                verify_artifacts(directory, self.execution, self.key)
            self.assertEqual(len(revised["artifacts"]), len(ARTIFACTS))

    def test_rejects_wrong_selection_and_extra_updates(self):
        selected = self.result["selected_step"]
        self.assertEqual(len(verify_history(self.history, self.labels, selected)), 4)
        with self.assertRaisesRegex(ValueError, "first maximum"):
            verify_history(self.history, self.labels, next(step for step in STEPS if step != selected))
        with self.assertRaisesRegex(ValueError, "budget or chronology"):
            verify_history(self.history[::-1], self.labels, selected)
        trace = read(self.directory / "trace.json")
        opts = self.spec["settings"]
        self.assertEqual(len(verify_trace(trace, schedule_for(opts), opts)["graph_changed_elements_by_step"]), 12)
        with self.assertRaisesRegex(ValueError, "budget or schedule"):
            verify_trace(trace + trace[-1:], schedule_for(opts), opts)

    def test_gate_requires_both_controls_and_protects_each_small_class(self):
        values = evaluate_record(self.result["selected_validation"], self.labels, probabilities=self.probabilities)
        cases = {f"{arm}-{seed}": {"selected": deepcopy(values)} for arm in ARMS for seed in (64, 512, 2048)}
        for seed in (64, 512, 2048):
            candidate = cases[f"exclude_out_of_reference-{seed}"]["selected"]
            candidate["macro_oa"] += .05
            candidate["selection_macro_oa"] += .05
            candidate_class = candidate["metrics"]["paviaU"]["val"]["fused"]["per_class"][0]
            native_class = cases[f"native-{seed}"]["selected"]["metrics"]["paviaU"]["val"]["fused"]["per_class"][0]
            native_class["recall"] = candidate_class["recall"] + .03
        paired = compare(cases)
        self.assertTrue(paired["native"]["gates"]["primary"])
        self.assertFalse(paired["native"]["development_gate_passed"])
        self.assertEqual(paired["native"]["small_class_failures"][0]["label"], 1)
        self.assertTrue(paired["matched_boundary_dropout"]["development_gate_passed"])
