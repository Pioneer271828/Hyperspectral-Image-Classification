"""Admit held-aside development tiles and run frozen confirmation controls."""

import numpy as np
import torch

from evaluation.dynamic_curve import pin, require
from evaluation.wp12_resident_cost import environment
from inference.gate_calibration import _validate_freeze, confirm
from inference.route_controls import _fixed_routes, _output_hash, benchmark_controls, fixed_cascade, request_hash
from inference.trained_light import trained_cascade
from loadData.observation_sampling import sample
from loadData.spatial_tiles import observe_tile
from model.research_config import digest
from transfer.development_training import validate
from transfer.g3_training import state_hash
from transfer.observation_policy import scene_identity


POLICY = "dy-independent-development-v1"
HISTORY_POLICY = "dy-audited-devtrain-devval-footprint-v1"
TIMING_SCOPE = "resident_cpu_batch_public_api_validation_and_compute"


def confirmation_environment():
    value = environment("cpu", "native")
    for key in ("memory", "package", "excluded"):
        del value[key]
    value["scope"] = TIMING_SCOPE
    return value


def unused_tiles(plan, record):
    """Whole rectangles outside all train, selection and test roles."""
    validate(record, plan)
    return [tile["id"] for tile, role in zip(plan["tiles"], record["roles"])
            if tile["partition"] == "val" and role["role"] == "unused"]


def layout(plan, record, blocks):
    available = set(unused_tiles(plan, record))
    seen, roles, ids = set(), set(), set()
    require(isinstance(blocks, list) and blocks and plan["gap"] >= 1,
            "Registered held-aside tiles and positive spatial buffer required")
    for block in blocks:
        require(set(block) == {"id", "tile_id", "role", "label_indices"}
                and isinstance(block["id"], str) and bool(block["id"]) and block["id"] not in ids
                and type(block["tile_id"]) is int and block["tile_id"] in available - seen
                and block["role"] in ("calibration", "confirmation"),
                "Each held-aside tile has one role, outside training, selection and test")
        y0, y1, x0, x1 = plan["tiles"][block["tile_id"]]["box"]
        indices = block["label_indices"]
        require(isinstance(indices, list) and bool(indices)
                and all(type(i) is int and 0 <= i < (y1-y0)*(x1-x0) for i in indices)
                and indices == sorted(set(indices)), "Authorized local label indices required")
        ids.add(block["id"])
        seen.add(block["tile_id"])
        roles.add(block["role"])
    require(roles == {"calibration", "confirmation"}, "Separate calibration and confirmation tiles required")


def build_requests(scene, blocks, scale):
    require(type(scale) is int and scale >= 4, "Explicit graph scale required")
    require(all(scene.bands.valid), "The full-band confirmation protocol requires all declared bands")
    requests = []
    for block in blocks:
        tile = scene.plan["tiles"][block["tile_id"]]
        observation = observe_tile(scene.cube, scene.plan, scene.stats, tile)
        view, _ = sample(observation, scene.bands, {"mode": "none"})
        x = np.concatenate((view.observation.spectral, view.observation.coordinates), axis=-1)
        requests.append(dict(id=block["id"], dataset=scene.dataset, x=torch.tensor(x, dtype=torch.float32),
                             wavelengths=torch.tensor(view.bands.wavelengths, dtype=torch.float32), scale=scale,
                             coverage_known=True, coverage_fraction=1.0))
    return requests


def admit(full, published, scene, record, history, registration, *, expected_registration_hash):
    """Verify pinned provenance and crop only held-aside observations. No label reads."""
    require(pin(expected_registration_hash) and digest(registration) == expected_registration_hash,
            "Pinned confirmation registration required")
    require(set(registration) == {"policy", "history_sha256", "light_spec_sha256", "light_artifact_sha256",
                                  "blocks", "request_sha256", "labels_sha256", "scale", "benchmark"}
            and registration["policy"] == POLICY
            and pin(registration["labels_sha256"]) and pin(registration["light_artifact_sha256"]),
            "Exact confirmation registration required")
    identity = scene_identity(scene, record)
    require(set(history) == {"policy", "full_state", "scene_identity", "history_evidence_sha256"}
            and history["policy"] == HISTORY_POLICY and digest(history) == registration["history_sha256"]
            and pin(history["history_evidence_sha256"]) and history["full_state"] == state_hash(full)
            and history["scene_identity"] == identity, "Audited complete full-model footprint required")
    light = published.specification
    require(published.validate(full) == scene.dataset and digest(light) == registration["light_spec_sha256"]
            and light["training_spec"]["identity"] == identity, "Light training and selection footprint differs")
    layout(scene.plan, record, registration["blocks"])
    requests = build_requests(scene, registration["blocks"], registration["scale"])
    require(request_hash(requests) == registration["request_sha256"], "Held-aside input bytes differ")
    return requests


def run_confirmation(full, published, scene, record, history, registration, *, expected_registration_hash,
                     protocol, frozen, expected_freeze_hash, read_labels, fixed, expected_fixed_hash):
    """Evaluate one frozen candidate and its preregistered fixed path on confirmation."""
    _validate_freeze(protocol, frozen, expected_freeze_hash)
    selected = frozen["candidates"][frozen["selected"]]
    require(selected["kind"] == "adaptive", "Calibration retained the full path")
    requests = admit(full, published, scene, record, history, registration,
                     expected_registration_hash=expected_registration_hash)
    spec = protocol.specification
    require(spec["full_state"] == history["full_state"] and spec["classes"] == scene.classes
            and spec["light_artifact"] == registration["light_artifact_sha256"], "Gate model identity differs")
    blocks = {b["id"]: b for b in registration["blocks"]}
    expected = [dict(id=b["id"], scene=scene.dataset, condition="full", role=b["role"],
                     context=scene.plan["tiles"][b["tile_id"]]["box"], label_indices=b["label_indices"])
                for b in blocks.values()]
    declared = [b for b in spec["blocks"] if b["role"] in ("calibration", "confirmation")]
    require(sorted(declared, key=lambda b: b["id"]) == sorted(expected, key=lambda b: b["id"]),
            "Gate calibration and confirmation geometry differs")
    requests = [r for r in requests if blocks[r["id"]]["role"] == "confirmation"]
    require(fixed["min_coverage"] == spec["min_coverage"], "Fixed and adaptive coverage rules differ")
    routes = _fixed_routes(full, published, requests, fixed, expected_fixed_hash)
    budget = registration["benchmark"]
    require(set(budget) == {"seed", "warmups", "pairs"}
            and all(type(budget[k]) is int for k in budget)
            and 0 <= budget["seed"] < 2**32 and budget["warmups"] >= 1
            and 20 <= budget["pairs"] <= 100 and budget["pairs"] % 20 == 0,
            "Balanced five-arm and curve-compatible repeat budget required")
    observed_environment = confirmation_environment()
    require(len(observed_environment["affinity"]) == 1, "One CPU core required")
    # Only this deferred loader sees confirmation values. No raw GT reader is used.
    labels = read_labels()
    require(digest(labels) == registration["labels_sha256"] and isinstance(labels, dict)
            and labels.keys() == routes.keys(), "Pinned confirmation labels required")
    for key, values in labels.items():
        require(isinstance(values, list) and len(values) == len(blocks[key]["label_indices"])
                and all(type(y) is int and 1 <= y <= scene.classes for y in values),
                "Confirmation labels differ from the authorized subset")
    counts = np.bincount([y for values in labels.values() for y in values], minlength=scene.classes+1)[1:]
    require((counts >= spec["min_class_support"]).all(), "Independent confirmation lacks class support")
    adaptive_kwargs = dict(mode="adaptive", criterion=spec["criterion"], threshold=selected["threshold"],
                           min_coverage=spec["min_coverage"])
    outputs = dict(full=trained_cascade(full, published, requests),
                   fast=trained_cascade(full, published, requests, mode="fast", min_coverage=spec["min_coverage"]),
                   adaptive=trained_cascade(full, published, requests, **adaptive_kwargs),
                   fixed=fixed_cascade(full, published, requests, fixed, expected_plan_hash=expected_fixed_hash))
    records = []
    for a, b, req in zip(outputs["fast"]["outputs"], outputs["full"]["outputs"], requests):
        key = req["id"]
        y = np.zeros(b["probabilities"].shape[0], dtype=np.int64)
        y[blocks[key]["label_indices"]] = labels[key]
        records.append(dict(id=key, labels=y, fast=a["probabilities"].numpy(), full=b["probabilities"].numpy(),
                            coverage_known=req["coverage_known"], coverage_fraction=req["coverage_fraction"]))
    confirmation = confirm(protocol, frozen, records, expected_freeze_hash=expected_freeze_hash)
    timing = benchmark_controls(full, published, requests, adaptive_kwargs=adaptive_kwargs,
                                fixed=(fixed, expected_fixed_hash), **budget)
    require(all(_output_hash(value) == timing["output_hashes"][arm] for arm, value in outputs.items()),
            "Scored and timed outputs differ")
    require(observed_environment["scope"] == timing["scope"], "Benchmark scope changed")
    receipt = dict(status="passed", registration_sha256=expected_registration_hash, confirmation=confirmation,
                   environment=observed_environment, timing=timing, support=counts.tolist(),
                   full_history_sha256=digest(history), optimizer_updates=0, query=0)
    policy_hashes = dict(full=digest({"mode": "full"}), fast=digest({"mode": "fast", "min_coverage": spec["min_coverage"]}),
                         adaptive=expected_freeze_hash, fixed=expected_fixed_hash)
    points = [dict(id=arm, policy_hash=policy_hashes[arm], full_state=spec["full_state"],
                   light_artifact=spec["light_artifact"], data_hash=confirmation["confirmation_hash"],
                   environment_hash=digest(observed_environment), timing_scope=timing["scope"],
                   warmups=timing["warmups"], routes=[dict(id=r["id"], fast=r["route"] == "fast") for r in value["outputs"]],
                   wall_ms=[r["timings"][arm]["wall_ms"] for r in timing["samples"]],
                   execution_receipt_sha256=digest(receipt))
              for arm, value in outputs.items()]
    return receipt, records, points
