"""Zero-update probes using the accepted Xuzhou development release and two blocks."""

from dataclasses import asdict
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from loadData.observation_sampling import BandMetadata
from loadData.spatial_tiles import tile_slice
from research_runtime import external_directory, tensor_hash
from transfer import registered_development as dev, wp02_route_development as wp02
from transfer.development_training import WP02_SPATIAL_RECORD, validate
from transfer.g3_training import Scene, _forward, state_hash
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_support import NativeSupportModel, capture_context, make_selector, native_forward
from transfer.regional_support_evolution import Schedule


WP07_REGISTRATION = "711e90e6b08d9c4554a051bd2e5ae3d133e62a2fc6a897c362b7545ff1984604"
WP08_SOURCE_CAMPAIGN = "2dd3eff9c32300e1df6c330daaa7c79539301a699030c37dc2a4ae6ab8cfb6a4"
CURATOR_RECEIPT = "e012bce1197154f6b6e81978758ae4d8ee3f11fdc374813dd7d317b74b0e789e"


def _read(pin):
    return json.loads(dev._read(pin))


def _two_blocks(pin):
    raw = _read(pin)
    record, plan = raw["record"], raw["plan"]
    if raw["dataset"] != "xuzhou" or record["kind"] != WP02_SPATIAL_RECORD:
        raise ValueError("Expected accepted Xuzhou WP02 development custody")
    dev._released_record(record)
    validate(record, plan)
    original = wp02.verify_record(record, plan, "xuzhou")
    meta = BandMetadata(**raw["bands"])
    bands = len(meta.band_ids)
    cube = np.zeros((*plan["shape"], bands), dtype=np.float32)
    stats = raw["statistics"]
    chosen = [next(r for r in record["roles"] if r["role"] == role) for role in ("devtrain", "devval")]
    members = {row["id"]: row for row in raw["blocks"]}
    used = []
    folder = Path(pin["path"]).parent
    for role in chosen:
        key = role["id"]
        row = members[key]
        path = (folder / f"tile-{key}.npy").resolve(strict=True)
        if path.parent != folder or row["name"] != path.name or row["role"] != role["role"]:
            raise ValueError("Selected block differs from authorized role or manifest")
        member = dict(path=str(path), sha256=row["sha256"])
        block = np.load(io.BytesIO(dev._read(member)), allow_pickle=False)
        if block.dtype != np.float32 or block.shape != cube[tile_slice(role)].shape or not np.isfinite(block).all():
            raise ValueError("Malformed published development tile")
        cube[tile_slice(role)] = block
        used.append(member)
    scene = Scene("xuzhou", cube, plan, meta, len(record["classes"]), stats)
    wp02.verify_scene(scene, original)
    return scene, chosen, used


@torch.no_grad()
def probe(out, wp07, wp08, curator, code):
    started = perf_counter()
    if (wp07["sha256"], wp08["sha256"], curator["sha256"]) != (WP07_REGISTRATION, WP08_SOURCE_CAMPAIGN, CURATOR_RECEIPT):
        raise ValueError("Only accepted release and source registrations can authorize real arrays")
    prior, campaign, custody = map(_read, (wp07, wp08, curator))
    rows = {row["scene"]: row for row in custody["rows"]}
    if (prior["issue"] != 411 or campaign["issue"] != 407 or custody["status"] != "completed"
            or rows["xuzhou"]["status"] != "prepared" or rows["whu-honghu"]["status"] != "blocked"):
        raise ValueError("Prior WP07 and WP02 role evidence differs")
    baseline = dev._load_source(prior["source"], prior["sidecar"])
    original_state = state_hash(baseline)
    if original_state != prior["source_state"]:
        raise ValueError("Accepted WP07 source state differs")
    scene, roles, files = _two_blocks(campaign["development"])
    support_model = NativeSupportModel.initialize(baseline, 7321)
    variants = {
        "graph_radius2": MultiscaleTrial(graph_multiscale="fixed", graph_radius=2),
        "mean": MultiscaleTrial(cnn_multiscale="mean", graph_multiscale="mean"),
        "metadata": MultiscaleTrial(cnn_multiscale="metadata", graph_multiscale="metadata"),
        "content": MultiscaleTrial(cnn_multiscale="content", graph_multiscale="content"),
        "joint": MultiscaleTrial(cnn_multiscale="joint", graph_multiscale="joint"),
    }
    models = {name: trial.initialize(baseline, 7321).eval() for name, trial in variants.items()}
    initial = {name: state_hash(model) for name, model in {**models, "native": support_model, "off": baseline}.items()}
    result = []
    schedule = asdict(Schedule(10, 6, 3, 7, 6, 48, pool=5))
    for role in roles:
        tile = scene.plan["tiles"][role["id"]]
        outputs = {"off": _forward(baseline, scene, tile)}
        for name, model in models.items():
            outputs[name] = _forward(model, scene, tile, multiscale_condition_only=True)
        for a, b in (("metadata", "mean"), ("joint", "content")):
            if not all(torch.equal(x, y) for x, y in zip(outputs[a][:3], outputs[b][:3], strict=True)):
                raise ValueError("Unavailable Xuzhou metadata must fall back to matching three-head predictions")
        entry = capture_context(support_model, scene, role, campaign["development"]["sha256"], original_state)
        selector = make_selector(entry, dict(mode="joint", schedule=schedule), 19064)
        native, info = native_forward(support_model, scene, tile, selector=selector)
        outputs["native_step_zero"] = native
        if info["trace"]["snapshot"]["step"] != 0:
            raise ValueError("A zero-update probe advanced support")
        result.append(dict(tile=tile["id"], role=role["role"], box=tile["box"],
            probability_hashes={name: [tensor_hash(head) for head in out[:3]] for name, out in outputs.items()},
            class_count=scene.classes, bands=len(scene.bands.band_ids), stage_two=info["trace"],
            radius_two_nnz=outputs["graph_radius2"][-1]["graph_nnz"]))
    if any(state_hash(model) != initial[name] for name, model in {**models, "native": support_model, "off": baseline}.items()):
        raise ValueError("Zero-update forwarding changed checkpoint state")
    for pin in [wp07, wp08, curator, prior["source"], prior["sidecar"], campaign["development"], *files]:
        dev._read(pin)
    return dev._json_new(external_directory(out), dict(version=2, code=code["commit"], updates=0, raw_gt=0,
        query=0, labels_added=0, scope="authorized_development_zero_update", scenes={
            "xuzhou": dict(status="passed", role_arrays=files, released_role_manifest=campaign["development"],
                source=prior["source"], sidecar=prior["sidecar"], source_state=original_state,
                actual_forwards=16, rows=result, wavelength_quality=scene.bands.wavelength_quality,
                metadata_fallback="mean", joint_fallback="content", state_unchanged=True),
            "paviaU": dict(status="BLOCKED", reason="Accepted fixed spatial roles exist; no independently pinned development-only tile package and compatible source pair located",
                           role_arrays=[], actual_forwards=0),
            "whu-honghu": dict(status="BLOCKED", reason="Initial WP02 release blocked; the later guarded role release has no independently pinned two-block array package for this entry",
                               role_arrays=[], actual_forwards=0)},
        provenance=[wp07, wp08, curator], wall_seconds=perf_counter() - started))
