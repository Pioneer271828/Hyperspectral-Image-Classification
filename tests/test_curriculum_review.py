"""Exercise authenticated bands, cross-process observation recovery and role-isolated review."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compression.registered_precision_recovery import stages
from research_runtime import ROOT, code_snapshot
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_observation_admission import (ObservationRun, audited_forward, bind_sample, execute_synthetic,
                                                      observation_policies, register_observation)
from transfer.curriculum_support_review import _stratify, review
from transfer.observation_curriculum import CurriculumPolicy, ORDERS
from transfer.xuzhou_multiscale import read_stage
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact, _packet


class CurriculumReviewTests(unittest.TestCase):
    def test_fixed_small_class_bins_and_tie_break(self):
        teacher = {i: dict(label=1, tile=3, nll=(i//2)/10, prediction=2, probability=.5) for i in (4, 3, 2, 1, 0)}
        exposure = {str(i): dict(raw=1, effective=1.) for i in teacher}
        row = _stratify(teacher, teacher, exposure)["1"]
        self.assertEqual([r["index"] for r in row["samples"]], list(range(5)))
        self.assertEqual([row["bins"][str(k)]["n"] for k in range(3)], [2, 2, 1])
        self.assertIsNone(row["association"]["teacher_nll_vs_effective"])

    @unittest.skipUnless(os.environ.get("WP01_CUR_S_BASE_FIXTURES"), "completed sample curriculum fixture archives required")
    def test_actual_9_and_22_class_observation_and_mechanism(self):
        parent = os.environ.get("WP06_CUR_FIXTURES")
        if parent:
            root = Path(parent)
            root.mkdir()
            self.matrix(root)
        else:
            with TemporaryDirectory() as directory:
                self.matrix(Path(directory))

    def matrix(self, parent):
        code = code_snapshot()
        source = Path(os.environ["WP01_CUR_S_BASE_FIXTURES"])
        for classes in (9, 22):
            with self.subTest(classes=classes):
                base = source/str(classes)/"run"
                target = parent/str(classes)
                target.mkdir()
                sample = dev._pin(base/"registration.json")
                reg, model, scene, record, plans = bind_sample(sample, code)
                proof = review(sample, dev._pin(base/"audit.json"), target/"mechanism", code)
                result = admission.read(proof)
                self.assertEqual(result["train_support_per_endpoint"], classes*10)
                self.assertEqual(result["devval_support_per_endpoint"], classes*5)
                self.assertEqual(len(result["cases"]), 6)
                pin = register_observation(sample, target/"observation", code)
                value = admission.read(pin)
                self.assertTrue(value["marginal_match"])
                self.assertEqual(value["preflight_forwards"], 16)
                self.assertEqual(value["optimizer_updates"], 0)
                policies = observation_policies(scene, record, reg["budget"])
                self.negatives(model, scene, record, plans["64"], policies["fixed"])
                roots = Path(value["root"])
                for order in ORDERS:
                    name = "s64-"+order
                    execute_synthetic(pin, code, name, "full", None)
                    pre = execute_synthetic(pin, code, name, "prefix", None)
                    argv = [sys.executable, str(ROOT/"scripts/run_curriculum_review.py"), "--expected-commit", code["commit"],
                            "synthetic-stage", "--registration", pin["path"], pin["sha256"], "--case", name, "--stage", "resume",
                            "--prior", pre["path"], pre["sha256"]]
                    child = subprocess.run(argv, capture_output=True, text=True)
                    self.assertEqual(child.returncode, 0, child.stderr)
                    resume = dev._pin(roots/"attempts"/name/"resume/status.json")
                    with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("extra update")):
                        execute_synthetic(pin, code, name, "completed", resume)
                    intervals = stages(reg["budget"], reg["prefix"])
                    full, _ = read_stage(roots, pin, name, "full", None, arms={name}, stages=intervals)
                    split, _ = read_stage(roots, pin, name, "resume", pre, arms={name}, stages=intervals)
                    end, _ = read_stage(roots, pin, name, "completed", resume, arms={name}, stages=intervals)
                    a, b, c = [_packet(v, value["contracts"][name], reg["budget"], packet_format=ObservationRun.FORMAT)
                               for v in (full, split, end)]
                    self.assertTrue(all(_exact(a[k], v[k]) for v in (b, c) for k in SCIENCE))
                    self.assertEqual(end["new_updates"], 0)
                    self.assertEqual(Counter(r["forward_evidence"]["band_count"] for r in a["history"]), {8:1, 6:1, 4:1, 2:1})
                    scheduled = value["coverage"][name]
                    self.assertEqual([e["record_hash"] for e in scheduled["events"]],
                                     [r["observation"]["record_hash"] for r in a["history"]])
                denied = dev._json_new(target/"real-no-stage.json", dict(value, scope="wp08_xuzhou_off"))
                with self.assertRaisesRegex(ValueError, "Only registered synthetic"):
                    execute_synthetic(denied, code, "s64-fixed", "full", None)

    def negatives(self, model, scene, record, plan, policy):
        step = 2
        tile = scene.plan["tiles"][plan.event(step)["tile_id"]]
        val = next(t for t in record["roles"] if t["role"] == "devval")
        with self.assertRaisesRegex(ValueError, "development role"):
            policy.view(scene, record, scene.plan["tiles"][val["id"]], phase="training", step=step)
        view = policy.view(scene, record, tile, phase="training", step=step)
        standard = CurriculumPolicy(policy.to_dict()).view(scene, record, tile, phase="training", step=step)
        self.assertTrue(np.array_equal(view.observation.spectral, standard.observation.spectral))
        obs = policy.to_dict()["training"][step-1]
        changes = [
            (replace(view, bands=replace(view.bands, band_ids=(9999, *view.bands.band_ids[1:]))), "band IDs"),
            (replace(view, bands=replace(view.bands, wavelengths=(9999., *view.bands.wavelengths[1:]))), "metadata"),
            (replace(view, observation=replace(view.observation, spectral=view.observation.spectral+.01)), "statistics"),
        ]
        for changed, reason in changes:
            with self.assertRaisesRegex(ValueError, reason):
                audited_forward(model, scene, record, tile, changed, obs)
        omitted = next(i for i in range(len(scene.bands.band_ids)) if i not in obs["selected_positions"])
        changed = scene.cube.copy()
        changed[..., omitted] += 999.
        poisoned = replace(scene, cube=changed)
        alternate = policy.view(poisoned, record, tile, phase="training", step=step)
        _, clean = audited_forward(model, scene, record, tile, view, obs)
        _, comparison = audited_forward(model, poisoned, record, tile, alternate, obs)
        self.assertEqual(clean, comparison)
        graph = model.backbone.learnable_graph
        original = graph.build_graph
        y0, y1, x0, x1 = tile["box"]
        unregistered = torch.tensor(scene.cube[y0:y1, x0:x1, omitted]).unsqueeze(-1)
        with patch.object(graph, "build_graph", lambda features, *a, **k: original(features+unregistered, *a, **k)):
            with self.assertRaisesRegex(ValueError, "outside the selected-band"):
                audited_forward(model, scene, record, tile, view, obs)
