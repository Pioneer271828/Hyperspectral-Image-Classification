"""Engineering-only external-format releases for the registered model comparison."""

from dataclasses import asdict

import numpy as np
import torch

from evaluation import wp12_cross_backend as comparison
from evaluation.frozen_training_stages import file_pin, require
from evaluation.wp12_spatial_release import SCHEMA as GEOMETRY
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from scripts.run_frozen_band_region import _write
from transfer import native_independent_release as independent, native_paired_head_fixture as layout
from transfer import wp_cross_backend as backends
from transfer.development_bank_training import _baseline
from transfer.g3_training import state_hash
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_external_paired_fixture import release_bank
from transfer.region_message_attention import ATTENTION_MODES

SCHEMA = "wp05_wp08_synthetic_fixture_v1"


def models(release, seed):
    base = _baseline(release, seed)
    yield "off", "native", base
    yield "fixed1", "multiscale", MultiscaleTrial().initialize(base, seed)
    yield "scale", "multiscale", MultiscaleTrial(cnn_multiscale="content", graph_multiscale="average").initialize(base, seed)
    kwargs = dict(alignment_mode="region", interaction_mode="gated", interaction_direction="both",
                  interaction_strength=.1, interaction_hidden_dim=32, interaction_chunk_pixels=16384)
    attention = dict(width=32, landmark_grid=[2, 2], top_k=3, query_chunk=16)
    variants = [("aligned_off", "aligned", {**kwargs, "interaction_strength": 0.}),
                ("aligned", "aligned", kwargs),
                ("message_off", "region_message", {**kwargs, "interaction_strength": 0.,
                    "region_attention": {**attention, "mode": "local"}})]
    variants.extend((mode, "region_message", {**kwargs, "region_attention": {**attention, "mode": mode}})
                    for mode in ATTENTION_MODES)
    variants.append(("one_direction", "region_message", {**kwargs,
        "interaction_direction": "graph_to_cnn", "region_attention": {**attention, "mode": "combined"}}))
    common = base.state_dict()
    for name, backend, switches in variants:
        cfg = base.export_config()
        cfg["backbone_kwargs"].update(switches)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = backends.make_model(cfg, backend)
        model.load_state_dict({**model.state_dict(), **common}, strict=True)
        yield name, backend, model


def geometry(out, entry, roles, kind, repeat):
    master = layout.SPECS[kind]
    _, classes, channels, _, seed, shape, _ = master
    seed += 29*int(repeat == "b")
    bands = layout.master(kind, channels)
    tiles = {}
    for tile in roles["evaluation"]:
        slot = tile % 100
        _, truth = layout.field(kind, bands, classes, *shape, "erased" if slot == 7 else "retained",
                                tile, seed, repeat == "b" or slot == 8)
        tiles[str(tile)] = dict(labels=truth.tolist(), geometry=(truth > 0).tolist(),
                                 narrow=np.isin(truth, [5, 6] if kind=="xuzhou" else [15, 21]).tolist())
    return _write(out/"geometry.json", dict(schema=GEOMETRY, bank=entry["pin"]["sha256"], roles=digest(roles),
                   scope="synthetic", source=digest([SCHEMA, kind, repeat, "placed_geometry"]), tiles=tiles))


def fixture(out, kind, repeat, seeds, *, views=layout.VIEWS, cost_only=False):
    require(kind in ("xuzhou", "honghu") and repeat in ("a", "b") and seeds, "Invalid engineering scope")
    out.mkdir(parents=True)
    history, _, _ = release_bank(out/"history", kind, 1, 64, "h")
    entry, roles, release = release_bank(out/"candidate", kind, 5, 512, repeat)
    spatial = geometry(out, entry, roles, kind, repeat)
    manifest = _write(out/"bank.json", dict(schema=independent.MANIFEST, candidate=entry, roles=roles,
                      minimum_buffer=8, history=[history], observation_views=list(views), spatial_annotations=spatial))
    approval = _write(out/"owner.json", dict(schema=independent.AUTHORITY,
                      approved_manifest_sha256=manifest["sha256"], historical_bank_sha256=[history["pin"]["sha256"]],
                      uses=dict(independent_accuracy=not cost_only, cost=True, frozen_diagnostics=False),
                      source_mode="synthetic", spatial_annotations_sha256=spatial["sha256"]))
    groups = {}
    for seed in seeds:
        work = out/f"s{seed}"
        work.mkdir()
        bulk = list(models(release, seed))
        protocol = dict(budget=0, data=digest([entry["pin"], roles["support"], roles["calibration"]]),
                        exposure=digest([]), initialization=state_hash(bulk[0][2]),
                        class_axis=list(range(1,release.classes + 1)), band_axis=release.bands.fingerprint,
                        arithmetic=digest("zero_update_fixture_cpu_initialization"), selection=comparison.SELECTION)
        arms = {}
        for name, backend, model in bulk:
            path = work/(name+".pt")
            save_model_checkpoint(model, path)
            exported = dict(backend=backend, weights=file_pin(path), metadata=file_pin(path.with_suffix(".pt.json")),
                            state=state_hash(model))
            arms[name] = _write(work/(name+".json"), dict(schema=comparison.ENDPOINT, protocol=protocol,
                        seed=seed, phase="terminal", step=0, calibration=[dict(step=0,nll=0.)], checkpoint=exported))
        pairs = [["off", name] for name in arms if name!="off"] + [["local", "combined"]]
        groups[f"s{seed}"] = dict(seed=seed, protocol=protocol, arms=arms, pairs=pairs)
    return _write(out/"comparison.json", dict(schema=comparison.DESIGN, intake=manifest, authority=approval, groups=groups))


def validate_controls(root, raw, *, cost_only=False):
    from evaluation.frozen_training_stages import json_pin

    record = json_pin(root, raw)
    for group in record["cohorts"].values():
        arms = group["arms"]
        for name in ("fixed1", "aligned_off", "message_off"):
            for a,b in zip(arms["off"]["rows"], arms[name]["rows"], strict=True):
                if cost_only:
                    require(a["accuracy"] is b["accuracy"] is None, "Cost-only run exposed a score or geometry")
                else:
                    require(a["accuracy"]["heads"] == b["accuracy"]["heads"]
                            and a["accuracy"]["graph"]["q_nnz"] == b["accuracy"]["graph"]["q_nnz"]
                            and a["accuracy"]["graph"]["a0_nnz"] == b["accuracy"]["graph"]["a0_nnz"],
                            "Disabled branches or fixed dilation one changed probabilities or graph support")
