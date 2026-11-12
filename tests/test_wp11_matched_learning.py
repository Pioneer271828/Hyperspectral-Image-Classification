"""Matched 9/22-class learning and actual deferred custody-to-confirmation chain."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluation.wp11_confirmation import HISTORY_POLICY, build_requests
from evaluation.wp11_model_pair import LIGHT_PINS, MODEL_FILES, load_pair
from inference.tile_cascade import gate_score
from loadData.observation_sampling import BandMetadata
from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan, tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from research_runtime import ROOT, code_snapshot
from scripts import prepare_wp11_confirmation as preparation
from scripts import run_wp11_confirmation as confirmation
from scripts.prepare_wp11_gate import run as register_gate
from scripts.run_wp11_calibration import run as calibrate
from tests.test_wp11_calibration import read, save
from tests.test_wp11_confirmation import file_pin
from tests.test_wp11_gate_registration import synthetic_registration
from transfer.curriculum_exposure import ExposurePlan
from transfer.development_training import prepare_scene
from transfer.g3_training import state_hash
from transfer.observation_policy import scene_identity
from transfer.registered_development import export_development
from transfer.registered_wp11_learning import ARMS, execute, register


def fixture(root, classes):
    plan = make_tile_plan((100, 100), 20, 1, 2026)
    y = np.zeros((100, 100), np.int64)
    for tile in [tile for tile in plan["tiles"] if tile["partition"] == "train"][:2]:
        local = y[tile_slice(tile)]
        local[...] = np.resize(np.arange(1, classes + 1), local.shape)
    record = make_development(y, plan, list(range(1, classes + 1)))
    bands = BandMetadata("synthetic-wp11", tuple(range(8)), tuple(np.linspace(460, 850, 8)),
                         "nm", "approximate", (True,) * 8)
    cube = np.random.default_rng(42).normal(size=(100, 100, 8)).astype(np.float32)
    scene = prepare_scene(cube, bands, plan, record)
    torch.manual_seed(64)
    model = MultiDatasetVCASAHGCN({scene.dataset: classes}, latent_channels=8, adapter_embed_dim=8,
                                 locality_window=None, graph_mode="learned", norm_type="group_norm").eval()
    model.backbone.adapter.cache_attention = False
    weights = root / "source.pt"
    save_model_checkpoint(model, weights)
    dev = export_development(scene, record, root / "development")
    audited = save(root, "old-audit", dict(fixture=True, original_gt_read=False))
    usage = {key: [r["id"] for r in record["roles"] if r["role"] == role] for key, role in
             (("fit", "devtrain"), ("training", "devtrain"), ("selection", "devval"))}
    history = save(root, "old-history", dict(policy=HISTORY_POLICY, scope="all_same_scene_ancestry",
                   tile_usage=usage, full_state=state_hash(model), scene_identity=scene_identity(scene, record),
                   history_evidence_sha256=audited["sha256"]))
    source = dict(development=dev, full_weights=file_pin(weights), full_metadata=file_pin(root / "source.pt.json"),
                  full_history=history, history_evidence=audited,
                  training=dict(updates=4, seed=64, lr=1e-4, class_balance=True))
    exposure = ExposurePlan.freeze(scene, record, updates=4, seed=64,
                                   minimum_class_blocks={str(c): 1 for c in record["classes"]},
                                   minimum_support_visits=1)
    request = dict(scope="synthetic", origin=None, exposure=save(root, "exposure", exposure.to_dict()), inputs=source)
    return request, scene, record


def cli(code, operation, pin, note, root, tag, *flags):
    argv = [sys.executable, str(ROOT / "scripts/run_wp11_matched.py"), "--expected-commit", code["commit"],
            operation, "--registration", pin["path"], "--registration-sha256", pin["sha256"],
            "--notice", note["path"], "--notice-sha256", note["sha256"], *flags]
    run = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    (root / f"{tag}.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise AssertionError(run.stderr)


class MatchedWP11Tests(unittest.TestCase):
    def test_matched_learning_frozen_pair_and_independent_confirmation(self):
        affinity = os.sched_getaffinity(0)
        torch.set_num_threads(1)
        try:
            os.sched_setaffinity(0, {min(affinity)})
            evidence = os.environ.get("WP11_SYNTHETIC_EVIDENCE")
            if evidence:
                root = Path(evidence)
                root.mkdir()
                self.matrix(root)
            else:
                with TemporaryDirectory() as name:
                    self.matrix(Path(name))
        finally:
            os.sched_setaffinity(0, affinity)

    def matrix(self, root):
        code = code_snapshot()
        for classes in (9, 22):
            with self.subTest(classes=classes):
                self.one(root / str(classes), code, classes)

    def one(self, root, code, classes):
        root.mkdir()
        request, scene, record = fixture(root, classes)
        attempt = root / "attempt"
        bad = dict(request, inputs=dict(request["inputs"], training=dict(request["inputs"]["training"], seed=65)))
        with self.assertRaisesRegex(ValueError, "class-first seed"):
            register(save(root, "wrong-seed", bad), root / "bad", code)
        missing = dict(read(request["exposure"]), minimum_support_visits=5)
        with self.assertRaisesRegex(ValueError, "original support|exposure"):
            register(save(root, "altered-exposure", dict(request, exposure=save(root, "wrong-plan", missing))),
                     root / "bad", code)
        register(save(root, "request", request), attempt, code)
        registration = file_pin(attempt / "registration.json")
        notice = save(root, "notice", dict(registration=registration["sha256"], commit=code["commit"],
                      total_updates=16, device="single_queue_cpu", estimate="synthetic", basis="fixed four-step fixture"))
        for arm in ARMS:
            for stage in ("continuous", "prefix", "resume"):
                cli(code, "run", registration, notice, root, f"{arm}-{stage}", "--arm", arm, "--stage", stage)
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("completed optimizer call")):
                ended = execute(registration, notice, code, arm, "completed")
            self.assertEqual(ended["updates"], 0)
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("duplicate optimizer call")):
                with self.assertRaises(FileExistsError):
                    execute(registration, notice, code, arm, "continuous")
        release_dir = root / "frozen-models"
        cli(code, "audit", registration, notice, root, "audit", "--out", str(release_dir))
        release_pin = file_pin(release_dir / "release.json")
        release = read(release_pin)
        audit = read(release["history_evidence"])
        self.assertEqual(audit["actual_updates"], 16)
        self.assertTrue(audit["exact_continuous_resume_completed"])
        self.assertEqual(audit["endpoints"]["full"]["selected"]["fused"]["count"], 5 * classes)
        self.assertEqual(set(audit["endpoints"]["light"]["selected"]), {"light"})
        self.assertTrue(all(p["changed_steps"] for p in audit["parameters"]["light"].values()))
        self.assertEqual(audit["stages"]["light"]["continuous"]["graph_calls"], 0)
        self.assertGreater(audit["stages"]["full"]["continuous"]["graph_calls"], 0)
        self.assertNotEqual(release["source_state"], release["full_state"])

        spec = {key: release[key] for key in (*MODEL_FILES, *LIGHT_PINS)}
        spec.update(matched_release=release_pin, tiles=[])
        full, light, _, _ = load_pair(spec, scene, record)
        gates = synthetic_registration(root, code, with_design=True, prepared=(spec, full, light, scene, record))
        labels = file_pin(root / "confirmation/labels.json")
        raw = Path(labels["path"]).read_bytes()
        Path(labels["path"]).unlink()
        (root / "confirmation/receipt.json").unlink()
        (root / "confirmation-source.json").unlink()
        requests = build_requests(scene, read(gates["confirmation"]["blocks"]), gates["scale"])
        if len(requests) >= 2:
            def score(row):
                return gate_score(light.model(row["dataset"], row["x"], wavelengths=row["wavelengths"]),
                                  "max_probability")
            scores = sorted(map(score, requests))
            cutoff = (scores[0] + scores[-1]) / 2
            calibration = build_requests(scene, read(gates["calibration"]["blocks"]), gates["scale"])
            if max(map(score, calibration)) >= cutoff and scores[0] < cutoff < scores[-1]:
                gates["settings"]["thresholds"] = [cutoff]

        early_pin = save(root, "early-gate", gates)
        early = register_gate(early_pin, root / "registered-gate", code)
        sealed = root / "calibrated"
        cal = calibrate(early["calibration"], sealed, code)
        self.assertEqual(cal["status"], "adaptive_frozen")
        self.assertFalse(cal["confirmation_labels_read"])
        handoff = dict(handoff=early["confirmation"], freeze=file_pin(sealed / "gate-freeze.json"),
                       freeze_receipt=file_pin(sealed / "receipt.json"), confirmation_labels=labels)
        with self.assertRaisesRegex(ValueError, "differs from custody"):
            preparation.run(save(root, "bad-role", dict(handoff, confirmation_labels=release["development"])),
                            root / "bad", code)
        prepared = preparation.run(save(root, "prepare-confirmation", handoff), root / "ready", code)
        self.assertFalse(Path(labels["path"]).exists())
        final = read(prepared["confirmation"])
        changed = (
            dict(final, matched_release=dict(final["matched_release"], sha256="0"*64)),
            dict(final, full_weights=request["inputs"]["full_weights"]),
            dict(final, fixed=save(root, "wrong-fixed", dict(read(final["fixed"]), min_coverage=.5))),
            dict(final, gate_freeze=save(root, "wrong-freeze", dict(read(final["gate_freeze"]), selected=0))),
            dict(final, comparison=save(root, "wrong-cost", dict(read(final["comparison"]),
                 relative_tolerances=dict(p50_ms=1., p95_ms=1.)))),
        )
        for index, candidate in enumerate(changed):
            with patch.object(confirmation, "load_development", side_effect=AssertionError("premature model data")):
                with self.assertRaises(ValueError):
                    confirmation.run(save(root, f"rejection-{index}", candidate), root / "bad", code)
        Path(labels["path"]).write_bytes(raw)
        result = confirmation.run(prepared["confirmation"], root / "confirmed", code)
        timings = result["timing"]["summary"]
        self.assertEqual(timings["adaptive"]["calls"], timings["random"]["calls"])
        self.assertTrue(all(n > 0 for n in result["support"]))
        self.assertEqual(len(result["support"]), classes)
        self.assertEqual(result["query"], 0)
