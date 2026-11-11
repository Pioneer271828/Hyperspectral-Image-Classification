"""Authorized real-schema fixture, complete synthetic role grid and negative gates."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluation import frozen_probe_audit as audit, frozen_probe_campaign as campaign, frozen_probe_registry as reg
from evaluation.linear_probe import fingerprint
from g4_support_context import read
from research_runtime import code_snapshot, hash_file, write_json
from tests.test_authenticated_probe_pack import fixture


def synthetic_grid(root, *, original_stage=False):
    capture, freeze, analysis, _, inputs = fixture(root)
    template = read(capture)
    frozen = read(freeze)
    family_hash = read(inputs[2])["family_hash"]
    selected = reg.grid_rows(frozen, "xuzhou")
    for row in selected:
        row["family_hash"] = family_hash
    write_json(freeze, frozen)
    accepted = read(analysis)
    accepted["freeze_sha256"] = hash_file(freeze)
    write_json(analysis, accepted)
    pins = {str(freeze): hash_file(freeze), str(analysis): hash_file(analysis), **accepted["protected_files"],
            **{str(analysis.parent/name): sha for name, sha in accepted["outputs"].items()}}
    stages = {name: torch.load(capture.parent/name, map_location="cpu", weights_only=True)
              for name in ("stage1-features.pt", "stage2-features.pt")}
    rng = torch.Generator().manual_seed(187)
    index_dir = root/"index"
    index_dir.mkdir()
    rows = []
    for row in selected:
        name = row["name"]
        folder = root/("archive-"+name)
        folder.mkdir()
        for file, values in stages.items():
            # This fixture's superpixels repeat across classes. Use a constant graph
            # control with identical logits for the passing path and test nonconstant
            # captures and decision mismatches separately.
            features = values if original_stage else {
                **values, "cnn": torch.randn(values["cnn"].shape, generator=rng, dtype=values["cnn"].dtype),
                "graph": torch.zeros_like(values["graph"])}
            torch.save(features, folder/file)
        for stage in (1, 2):
            write_json(folder/f"stage{stage}-summary.json", {"synthetic": True})
        path = folder/"receipt.json"
        write_json(path, {**template, "case": name, "freeze_sha256": hash_file(freeze),
            "analysis_sha256": hash_file(analysis), "protected_files": pins,
            "outputs": {file.name: hash_file(file) for file in folder.iterdir()}})
        rows.append({"case": name, "receipt": str(path), "sha256": hash_file(path), "origin": "captured"})
    index = index_dir/"capture-index.json"
    write_json(index, {"captures": rows})
    return freeze, analysis, index


def register(root, code, *, steps=6, original_stage=False):
    freeze, analysis, index = synthetic_grid(root, original_stage=original_stage)
    scope = reg.prepare(freeze, hash_file(freeze), analysis, hash_file(analysis), index, hash_file(index),
        root/"registration", code, synthetic=True, plan={**reg.PLAN, "steps": steps},
        paths={"full": [steps], "split": [2, 4, steps], "completed": [steps]})
    return scope, index


class FrozenProbeCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.code = code_snapshot()
        cls.scope_path, cls.index = register(cls.root, cls.code)
        cls.sha = hash_file(cls.scope_path)
        cls.scope, _, cls.cases, cls.pins, _ = reg.load(cls.scope_path, cls.sha, cls.code)
        cls.run_path = campaign.run(cls.scope_path, cls.sha, cls.root/"run", cls.code)
        cls.run_sha = hash_file(cls.run_path)

    def first(self):
        case, bound, local = self.cases[0]
        folder = self.run_path.parent/case["name"]
        return case, bound, local, folder, hash_file(folder/"execution.json")

    def test_entire_grid_exact_recovery_and_independent_accepted_evidence(self):
        self.assertEqual(len(self.cases), 18)
        self.assertEqual(self.scope["planned_probe_updates"], 864)
        self.assertEqual(read(self.run_path)["probe_updates"], 864)
        path = campaign.review(self.scope_path, self.sha, self.run_path, self.run_sha, self.root/"review", self.code)
        result = read(path)
        self.assertEqual((result["status"], result["accepted_cases"], result["accepted_probes"],
                          result["probe_updates"], result["audit_probe_updates"]), ("passed", 18, 72, 864, 0))
        self.assertEqual(read(path.parent/"acceptance.json")["probes"], 72)
        report = read(path.parent/"summary.json")
        self.assertEqual((report["metric_rows"], report["paired_rows"]), (288, 240))
        self.assertEqual(report["history_selections_checked"], 72)
        self.assertTrue(all(row["seed_count"] == 3 for row in report["three_seed_paired"]))
        self.assertTrue(result["protected_files_unchanged"])
        with self.assertRaises(FileExistsError):
            campaign.run(self.scope_path, self.sha, self.run_path.parent, self.code)
        with self.assertRaises(ValueError):
            campaign.run(self.scope_path, self.sha, self.scope_path.parent/"nested", self.code)

    def test_wrong_seed_plan_and_incomplete_or_repeated_capture_index(self):
        original = read(self.scope_path)
        for field in ("cases", "probe_plan", "layers"):
            altered = deepcopy(original)
            if field == "cases":
                altered[field][0]["seed"] = 999
            elif field == "probe_plan":
                altered[field]["lr"] = .2
            else:
                altered[field].pop()
            path = self.root/f"bad-{field}.json"
            write_json(path, altered)
            with self.assertRaisesRegex(ValueError, "Registered matrix"):
                reg.load(path, hash_file(path), self.code)
        good = read(self.index)["captures"]
        for name, rows in (("missing", good[:-1]), ("duplicate", [*good[:-1], good[0]]),
                           ("wrong-role", [{**good[0], "receipt": good[1]["receipt"], "sha256": good[1]["sha256"]}, *good[1:]])):
            index = self.index.parent/f"{name}.json"
            write_json(index, {"captures": rows})
            with patch("torch.load", side_effect=AssertionError("Must reject before archive decode")):
                with self.assertRaises(ValueError):
                    reg.authenticate_inputs(original["freeze"], original["freeze_sha256"], original["analysis"],
                        original["analysis_sha256"], index, hash_file(index), synthetic=True)

    def test_tampered_state_and_protected_stage_file_are_rejected(self):
        case, _, local, folder, sha = self.first()
        path = folder/"split-0004"/"state.pt"
        prior = path.read_bytes()
        try:
            path.write_bytes(prior+b"tamper")
            with self.assertRaisesRegex(ValueError, "segment files"):
                campaign.checked_execution(folder, sha, self.scope, self.sha, case, local)
        finally:
            path.write_bytes(prior)
        original = read(self.scope_path)
        stage = Path(case["capture_receipt"]).parent/"stage2-features.pt"
        prior = stage.read_bytes()
        try:
            stage.write_bytes(prior+b"tamper")
            with patch("torch.load", side_effect=AssertionError("Must reject before archive decode")):
                with self.assertRaisesRegex(ValueError, "Pinned support/observation"):
                    reg.load(self.scope_path, self.sha, self.code)
        finally:
            stage.write_bytes(prior)
        self.assertEqual(hash_file(Path(original["capture_index"])), original["capture_index_sha256"])

    def test_independent_auditor_rejects_reordered_swapped_labels_and_missing_layer(self):
        case, bound, local, folder, sha = self.first()
        pack, state, report, recorded, _, _ = campaign.checked_execution(folder, sha, self.scope, self.sha, case, local)
        for kind in ("reorder", "swap", "missing-layer"):
            bad = deepcopy(pack)
            if kind == "reorder":
                bad["row_ids"]["train"] = bad["row_ids"]["train"].roll(1)
            elif kind == "swap":
                bad["labels"]["train"], bad["labels"]["val"] = bad["labels"]["val"], bad["labels"]["train"]
            else:
                del bad["features"]["stage2_graph"]
            with self.assertRaises(ValueError):
                audit.audit_pack(bad, bound, self.scope, case, local)
        bad = deepcopy(state)
        bad["layers"]["stage2_graph"]["best"]["step"] = 100
        bad.pop("checksum")
        bad["checksum"] = fingerprint(bad)
        with self.assertRaisesRegex(ValueError, "first-minimum"):
            audit.audit_probes(pack, self.scope, bad, report, recorded)

    def test_exact_decision_gate_and_failed_batch_keeps_evidence(self):
        a = np.array([[-1., -1.-1e-15]])
        self.assertFalse(audit.comparison(a, a[:, ::-1])["strict_decisions"])
        self.assertTrue(audit.comparison(a, a[:, ::-1])["probabilities_close"])
        oracle = audit.numpy_logp
        observed = []

        def altered(*args):
            result = oracle(*args)
            if not observed:
                winner = result[0].argmax()
                result[0, winner] = -1e6
            observed.append(1)
            return result

        with patch.object(audit, "numpy_logp", side_effect=altered):
            path = campaign.review(self.scope_path, self.sha, self.run_path, self.run_sha,
                                   self.root/"review-numerical-failure", self.code)
        self.assertEqual(read(path)["status"], "incomplete_or_failed")
        self.assertEqual(read(path)["accepted_cases"], 17)
        self.assertFalse((path.parent/"acceptance.json").exists())
        result = read(path.parent/"summary.json")
        self.assertEqual(result["reviewed_probes"], 72)
        self.assertEqual(len(result["numerical_gate_failures"]), 1)

    def test_actual_initialized_network_archive_enters_named_probe_paths(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            scope_path, _ = register(root, self.code, original_stage=True)
            sha = hash_file(scope_path)
            scope, _, cases, _, _ = reg.load(scope_path, sha, self.code)
            case, bound, local = cases[0]
            out = root/"case"
            execution = campaign.run_case(out, bound, case, local, scope, sha, lambda receipt: None)
            pack, state, report, recorded, counts, _ = campaign.checked_execution(out, execution, scope, sha, case, local)
            self.assertEqual(counts, {"full": 24, "split": 24, "completed": 0})
            self.assertEqual(audit.audit_pack(pack, bound, scope, case, local)["feature_cells_checked"], 51840)
            self.assertEqual(audit.audit_probes(pack, scope, state, report, recorded)["history_selection_roles"], 4)
