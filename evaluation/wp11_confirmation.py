"""Admit held-aside development tiles and run frozen confirmation controls."""

import io
import os
from pathlib import Path
import platform

import numpy as np
import torch

from evaluation.dynamic_curve import pin, require

from inference.gate_calibration import SPARSE_PROTOCOL, _validate_freeze, confirm
from inference.route_controls import _fixed_routes, _output_hash, benchmark_controls, fixed_cascade, request_hash
from inference.trained_light import trained_cascade
from loadData.observation_sampling import sample
from loadData.spatial_tiles import observe_tile, tile_slice
from model.research_config import digest
from transfer.development_training import validate
from transfer.g3_training import state_hash
from transfer.observation_policy import scene_identity
from transfer.registered_development import _read


POLICY = "dy-independent-development-v1"
HISTORY_POLICY = "dy-audited-devtrain-devval-footprint-v1"
TIMING_SCOPE = "resident_cpu_batch_public_api_validation_and_compute"
DESIGN_PINS = {"confirmation_design_sha256", "confirmation_labels_file_sha256"}
MATCHED_PIN = {"matched_release_sha256"}


def confirmation_environment():
    """Retain the reviewed WP11 timing environment schema without the WP12 worker."""
    if torch.get_num_threads() != 1:
        raise ValueError("Registered confirmation requires one CPU thread")
    cpu = {line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
           if line.startswith("model name")}
    return dict(host=platform.node(), kernel=platform.release(), cpu=sorted(cpu),
                affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
                torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                device="cpu", arithmetic="native", scope=TIMING_SCOPE,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32)


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


def load_held_aside_tiles(scene, tiles):
    """Copy authenticated FP32 observations into an already admitted layout."""
    for tile_id, pin in tiles.items():
        sl = tile_slice(scene.plan["tiles"][tile_id])
        data = np.load(io.BytesIO(_read(pin)), allow_pickle=False)
        require(isinstance(data, np.ndarray) and data.shape == scene.cube[sl].shape
                and data.dtype == np.float32 and np.isfinite(data).all(), "Finite registered FP32 tile required")
        scene.cube[sl] = data


def validate_full_history(full, scene, record, history):
    """Check the declared footprint against the isolated development scene."""
    identity = scene_identity(scene, record)
    require(set(history) == {"policy", "scope", "tile_usage", "full_state", "scene_identity", "history_evidence_sha256"}
            and history["policy"] == HISTORY_POLICY
            and history["scope"] == "all_same_scene_ancestry"
            and pin(history["history_evidence_sha256"]) and history["full_state"] == state_hash(full)
            and history["scene_identity"] == identity, "Audited complete full-model footprint required")
    usage = history["tile_usage"]
    require(isinstance(usage, dict) and set(usage) == {"fit", "training", "selection"},
            "Full history must enumerate fitting, training and selection tiles")
    for phase, role in (("fit", "devtrain"), ("training", "devtrain"), ("selection", "devval")):
        allowed = {row["id"] for row in record["roles"] if row["role"] == role}
        ids = usage[phase]
        require(isinstance(ids, list) and bool(ids)
                and all(type(i) is int and i in allowed for i in ids) and ids == sorted(set(ids)),
                f"Audited full-model {phase} footprint escapes its development role")
    return identity


def validate_admission(full, published, scene, record, history, registration, *, expected_registration_hash):
    """Verify provenance and layout without reading held-aside observations."""
    require(pin(expected_registration_hash) and digest(registration) == expected_registration_hash,
            "Pinned confirmation registration required")
    fields = {"policy", "history_sha256", "light_spec_sha256", "light_artifact_sha256",
              "blocks", "request_sha256", "labels_sha256", "scale", "benchmark"}
    require(set(registration) in (fields, fields | DESIGN_PINS, fields | DESIGN_PINS | MATCHED_PIN)
            and registration["policy"] == POLICY
            and all(pin(registration[key]) for key in ({"labels_sha256", "light_artifact_sha256"} |
                                                       ((DESIGN_PINS | MATCHED_PIN) & registration.keys()))),
            "Exact confirmation registration required")
    require(digest(history) == registration["history_sha256"], "Full history registration differs")
    identity = validate_full_history(full, scene, record, history)
    light = published.specification
    require(published.validate(full) == scene.dataset and digest(light) == registration["light_spec_sha256"]
            and light["training_spec"]["identity"] == identity, "Light training and selection footprint differs")
    layout(scene.plan, record, registration["blocks"])


def admit(full, published, scene, record, history, registration, *, expected_registration_hash):
    """Verify pinned provenance and crop only held-aside observations. No label reads."""
    validate_admission(full, published, scene, record, history, registration,
                       expected_registration_hash=expected_registration_hash)
    requests = build_requests(scene, registration["blocks"], registration["scale"])
    require(request_hash(requests) == registration["request_sha256"], "Held-aside input bytes differ")
    return requests


def validate_gate_inputs(spec, scene, history, registration):
    require(spec["full_state"] == history["full_state"] and spec["classes"] == scene.classes
            and spec["light_artifact"] == registration["light_artifact_sha256"], "Gate model identity differs")
    blocks = {b["id"]: b for b in registration["blocks"]}
    expected = [dict(id=b["id"], scene=scene.dataset, condition="full", role=b["role"],
                     context=scene.plan["tiles"][b["tile_id"]]["box"], label_indices=b["label_indices"])
                for b in blocks.values()]
    declared = [b for b in spec["blocks"] if b["role"] in ("calibration", "confirmation")]
    require(sorted(declared, key=lambda b: b["id"]) == sorted(expected, key=lambda b: b["id"]),
            "Gate calibration and confirmation geometry differs")
    if spec["schema"] == SPARSE_PROTOCOL:
        require(spec["admission_hash"] == digest(registration)
                and spec["data_hashes"]["confirmation"] == registration["labels_sha256"],
                "Gate admission or deferred confirmation commitment differs")
    return blocks


def endpoint_records(fast, full, requests, blocks, labels):
    records = []
    for a, b, req in zip(fast["outputs"], full["outputs"], requests, strict=True):
        key = req["id"]
        require(a["id"] == b["id"] == key, "Scored endpoint order differs")
        y = np.zeros(b["probabilities"].shape[0], dtype=np.int64)
        y[blocks[key]["label_indices"]] = labels[key]
        records.append(dict(id=key, labels=y, fast=a["probabilities"].numpy(), full=b["probabilities"].numpy(),
                            coverage_known=req["coverage_known"], coverage_fraction=req["coverage_fraction"]))
    return records


def validate_benchmark(budget):
    require(set(budget) == {"seed", "warmups", "pairs"}
            and all(type(budget[k]) is int for k in budget)
            and 0 <= budget["seed"] < 2**32 and budget["warmups"] >= 1
            and 20 <= budget["pairs"] <= 100 and budget["pairs"] % 20 == 0,
            "Balanced five-arm and curve-compatible repeat budget required")


def run_confirmation(full, published, scene, record, history, registration, *, expected_registration_hash,
                     protocol, frozen, expected_freeze_hash, read_labels, fixed, expected_fixed_hash):
    """Evaluate one frozen candidate and its preregistered fixed path on confirmation."""
    _validate_freeze(protocol, frozen, expected_freeze_hash)
    selected = frozen["candidates"][frozen["selected"]]
    require(selected["kind"] == "adaptive", "Calibration retained the full path")
    requests = admit(full, published, scene, record, history, registration,
                     expected_registration_hash=expected_registration_hash)
    spec = protocol.specification
    blocks = validate_gate_inputs(spec, scene, history, registration)
    if spec["schema"] == SPARSE_PROTOCOL:
        calibration = [r for r in requests if blocks[r["id"]]["role"] == "calibration"]
        require(request_hash(calibration) == spec["calibration_request_hash"], "Calibrated input bytes differ")
    requests = [r for r in requests if blocks[r["id"]]["role"] == "confirmation"]
    require(fixed["min_coverage"] == spec["min_coverage"], "Fixed and adaptive coverage rules differ")
    routes = _fixed_routes(full, published, requests, fixed, expected_fixed_hash)
    budget = registration["benchmark"]
    validate_benchmark(budget)
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
    records = endpoint_records(outputs["fast"], outputs["full"], requests, blocks, labels)
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
                   light_artifact=spec["light_artifact"], data_hash=confirmation.get("observed_records_hash", confirmation["confirmation_hash"]),
                   environment_hash=digest(observed_environment), timing_scope=timing["scope"],
                   warmups=timing["warmups"], routes=[dict(id=r["id"], fast=r["route"] == "fast") for r in value["outputs"]],
                   wall_ms=[r["timings"][arm]["wall_ms"] for r in timing["samples"]],
                   execution_receipt_sha256=digest(receipt))
              for arm, value in outputs.items()]
    return receipt, records, points
