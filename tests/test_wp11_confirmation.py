"""Exercise the isolated tile transport and actual fixed path with synthetic models."""

import copy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.dynamic_cost_pairs import POLICY as PAIRS
from evaluation.wp11_confirmation import POLICY, HISTORY_POLICY, admit, build_requests, confirmation_environment, layout, unused_tiles
from inference.gate_calibration import GateProtocol, calibrate
from inference.route_controls import fixed_cascade, request_hash
from inference.trained_light import export_trained_light, load_trained_light, trained_cascade
from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan, tile_slice
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot, hash_file, write_json
from scripts.run_wp11_confirmation import run
from tests.test_gate_calibration import data_pin
from tests.test_int8_model import fixture
from transfer.development_training import prepare_scene
from transfer.g3_training import state_hash
from transfer.light_head_training import LightHeadRun
from transfer.observation_policy import scene_identity
from transfer.registered_development import export_development


def file_pin(path):
    return dict(path=str(path), sha256=hash_file(path))


def synthetic_package(root):
    def pin(name, value):
        path = root / f"{name}.json"
        write_json(path, value)
        return file_pin(path)

    full, example, _ = fixture()
    plan = make_tile_plan((80, 80), 20, 1)
    truth = np.resize(np.arange(1, 10), (80, 80))
    record = make_development(truth, plan, list(range(1, 10)))
    cube = np.random.default_rng(42).normal(size=(80, 80, 8)).astype(np.float32)
    scene = prepare_scene(cube, example.bands, plan, record)
    light_run = LightHeadRun(full, scene, record, updates=1, seed=64, lr=.0001, class_balance=True)
    light_run.advance(1)
    checkpoint, light_path = root / "training.pt", root / "light.pt"
    training_sha = light_run.save(checkpoint)
    light_sha = export_trained_light(full, scene, record, checkpoint, light_path,
                                    expected_training_sha256=training_sha, expected_contract_hash=light_run.contract_hash)
    published = load_trained_light(full, light_path, expected_sha256=light_sha,
                                   expected_training_sha256=training_sha, expected_contract_hash=light_run.contract_hash)
    blocks = [dict(id=role, role=role, tile_id=i, label_indices=list(range(9)))
              for i, role in zip(unused_tiles(plan, record), ("calibration", "confirmation"))]
    requests = build_requests(scene, blocks, 12)
    gate_blocks = [dict(id=role, role=role, scene=scene.dataset, condition="full",
                        context=next(row["box"] for row in record["roles"] if row["role"] == kind),
                        label_indices=[0])
                   for role, kind in (("training", "devtrain"), ("selection", "devval"))]
    for block in blocks:
        gate_blocks.append(dict(id=block["id"], role=block["role"], scene=scene.dataset, condition="full",
                                context=plan["tiles"][block["tile_id"]]["box"], label_indices=block["label_indices"]))
    records = {}
    for r in requests:
        a = trained_cascade(full, published, [r], mode="fast")["outputs"][0]["probabilities"].numpy()
        b = trained_cascade(full, published, [r])["outputs"][0]["probabilities"].numpy()
        y = np.zeros(len(a), dtype=int)
        y[:9] = np.arange(1, 10)
        records[r["id"]] = dict(id=r["id"], labels=y, fast=a, full=b, coverage_known=True, coverage_fraction=1.)
    gate = dict(schema="dy-gate-protocol-v1", name="synthetic independent tile control", full_state=state_hash(full),
                light_artifact=light_sha, criterion="max_probability", thresholds=[0.], min_coverage=1., classes=9,
                min_class_support=1, max_drop=dict(oa=1., aa=1., recall=1.), blocks=gate_blocks,
                data_hashes={k: data_pin([v], gate_blocks) for k, v in records.items()})
    frozen = calibrate(GateProtocol(gate, digest(gate)), [records["calibration"]])
    fixed = dict(format="dy-fixed-path-v1", input_hash=request_hash(requests[1:]), full_state=state_hash(full),
                 light_state=state_hash(published.model), min_coverage=1.,
                 routes=[dict(id="confirmation", route="full")])
    labels = {"confirmation": list(range(1, 10))}
    evidence = pin("synthetic-history-evidence", {"fixture": True})
    train = [t["id"] for t in record["roles"] if t["role"] == "devtrain"]
    validation = [t["id"] for t in record["roles"] if t["role"] == "devval"]
    history = dict(policy=HISTORY_POLICY, scope="all_same_scene_ancestry",
                   tile_usage=dict(fit=train, training=train, selection=validation), full_state=state_hash(full),
                   scene_identity=scene_identity(scene, record), history_evidence_sha256=evidence["sha256"])
    admission = dict(policy=POLICY, history_sha256=digest(history), light_spec_sha256=digest(published.specification),
                     light_artifact_sha256=light_sha, blocks=blocks, request_sha256=request_hash(requests),
                     labels_sha256=digest(labels), scale=12, benchmark=dict(seed=64, warmups=1, pairs=20))
    policy_hashes = dict(full=digest({"mode": "full"}), fast=digest({"mode": "fast", "min_coverage": 1.}),
                         adaptive=digest(frozen), fixed=digest(fixed))
    env = confirmation_environment()
    curve = dict(schema="dy-development-curve-v1", gate_protocol=digest(gate), role="confirmation",
                 points=[dict(id=k, method=k, policy_hash=v) for k, v in policy_hashes.items()],
                 environment_hash=digest(env), timing_scope=env["scope"], warmups=1, repeats=20,
                 cost_interval_ms=[.001, 10000.], area_definition="all_points_cost_sorted_piecewise_linear_trapezoid_no_envelope",
                 ece_bins=2, nll_epsilon=1e-8)
    pairs = dict(policy=PAIRS, curve_protocol_hash=digest(curve),
                 relative_tolerances=dict(p50_ms=.1, p95_ms=.1), pairs=[dict(adaptive="adaptive", fixed="fixed")])
    tiles = []
    for block in blocks:
        tile = plan["tiles"][block["tile_id"]]
        path = root / f"held-{tile['id']}.npy"
        np.save(path, cube[tile_slice(tile)], allow_pickle=False)
        tiles.append(dict(tile_id=tile["id"], file=file_pin(path)))
    full_path = root / "full.pt"
    save_model_checkpoint(full, full_path)
    inputs = dict(development=export_development(scene, record, root / "development"),
                  full_weights=file_pin(full_path), full_metadata=file_pin(root / "full.pt.json"),
                  full_history=pin("history", history), history_evidence=evidence, light=file_pin(light_path),
                  light_training_sha256=training_sha, light_contract_hash=light_run.contract_hash,
                  admission=pin("admission", admission), tiles=tiles, gate_protocol=pin("gate", gate),
                  gate_freeze=pin("freeze", frozen), fixed=pin("fixed", fixed), confirmation_labels=pin("labels", labels),
                  curve_protocol=pin("curve", curve), comparison=pin("pairs", pairs))
    return pin("registration", inputs), (full, published, scene, record, history, admission, fixed, requests)


class ConfirmationTests(unittest.TestCase):
    def test_pinned_cli_pipeline_and_actual_fixed_calls(self):
        affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, {min(affinity)})
            with TemporaryDirectory() as directory:
                root = Path(directory)
                registration, (full, published, scene, record, history, admission, fixed, requests) = synthetic_package(root)
                before = state_hash(full), state_hash(published.model), request_hash(requests)
                with patch.object(published.model, "forward", wraps=published.model.forward) as light:
                    output = fixed_cascade(full, published, requests[1:], fixed, expected_plan_hash=digest(fixed))
                self.assertEqual(light.call_count, 0)
                self.assertEqual(output["counts"]["full_calls"], 1)
                with patch.object(full, "forward", wraps=full.forward) as full_path:
                    fast = dict(fixed, routes=[dict(id="confirmation", route="fast")])
                    fixed_cascade(full, published, requests[1:], fast, expected_plan_hash=digest(fast))
                self.assertEqual(full_path.call_count, 0)
                self.assertEqual(before, (state_hash(full), state_hash(published.model), request_hash(requests)))
                out = root / "result"
                receipt = run(registration, out, code_snapshot())
                self.assertEqual(receipt["support"], [1]*9)
                self.assertEqual(receipt["timing"]["summary"]["fixed"]["calls"]["light_calls"], 0)
                self.assertTrue((out / "matched-cost-report.json").is_file())
        finally:
            os.sched_setaffinity(0, affinity)

    def test_selection_query_and_unproven_history_rejected(self):
        with TemporaryDirectory() as directory:
            _, (full, published, scene, record, history, admission, _, _) = synthetic_package(Path(directory))
            for forbidden in ("devval", "devtrain", "test"):
                blocks = copy.deepcopy(admission["blocks"])
                blocks[-1]["tile_id"] = next(r["id"] for r in record["roles"] if r["role"] == forbidden)
                with self.assertRaisesRegex(ValueError, "outside training, selection and test"):
                    layout(scene.plan, record, blocks)
            bad = dict(history, policy="whole_scene_transductive")
            altered = dict(admission, history_sha256=digest(bad))
            with self.assertRaisesRegex(ValueError, "Audited complete"):
                admit(full, published, scene, record, bad, altered, expected_registration_hash=digest(altered))
            bad = copy.deepcopy(history)
            bad["tile_usage"]["fit"].append(admission["blocks"][-1]["tile_id"])
            altered = dict(admission, history_sha256=digest(bad))
            with self.assertRaisesRegex(ValueError, "footprint escapes"):
                admit(full, published, scene, record, bad, altered, expected_registration_hash=digest(altered))
            changed = dict(admission, request_sha256="0"*64)
            with self.assertRaisesRegex(ValueError, "input bytes differ"):
                admit(full, published, scene, record, history, changed, expected_registration_hash=digest(changed))
