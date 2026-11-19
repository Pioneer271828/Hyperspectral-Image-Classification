"""Bind buffer-safe parent folds, released points and physical band views."""

from dataclasses import replace
from collections import defaultdict

import numpy as np
import torch

from loadData.observation_sampling import BandMetadata, _apply
from loadData.spatial_protocol import validate_statistics
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.development_bank import _gap
from transfer.development_blocks import array_hash
from transfer.frozen_band_region import _spatial_view
from transfer.g3_training import state_hash
from transfer.native_released_points import bind
from transfer.native_scale_observation import Lattice, _materialize
from transfer.p4_spatial import SHARED

SEED = 114031
VIEWS = {
    "full": ("full", "declared"),
    "continuous": ("continuous", "declared"),
    "dispersed": ("dispersed", "declared"),
    "nominal_index": ("dispersed", "index"),
    "missing": ("dispersed", "missing"),
    "shuffled": ("dispersed", "shuffled"),
    "held_range": ("held_range", "declared"),
}
TRAIN_VIEWS = ("full", "continuous", "dispersed", "shuffled", "full", "continuous")


def counts(rows, classes):
    return np.bincount([label for row in rows for _, label in row["samples"]],
                       minlength=classes+1)[1:].tolist()


def parent_folds(rows, classes, gap):
    index = defaultdict(list)
    for row in rows:
        if row["role"] != "devtrain" or row["blocked"] is not None:
            raise ValueError("Spatial diagnostic requires only published training blocks")
        index[row["parent_tile"]].append(row)
    order = sorted(index, key=lambda key: digest([SEED, key]))
    sets = [set(order[::2]), set(order[1::2])]
    result = []
    for held in sets:
        val = [row for key in held for row in index[key]]
        possible = [row for key in set(index)-held for row in index[key]]
        train = [row for row in possible if all(_gap(row["box"], other["box"]) >= gap for other in val)]
        actual_gap = min((_gap(a["box"], b["box"]) for a in train for b in val), default=None)
        train_ids, val_ids = ({i for row in rows for i, _ in row["samples"]} for rows in (train, val))
        if train_ids & val_ids or held & {r["parent_tile"] for r in train}:
            raise ValueError("Fit and held roles have common sample IDs or parents")
        result.append(dict(fit=sorted(r["id"] for r in train), held=sorted(r["id"] for r in val),
                           fit_parents=len({row["parent_tile"] for row in train}), held_parents=len(held),
                           min_gap=actual_gap, buffer_excluded=len(possible)-len(train),
                           fit_counts=counts(train, classes), held_counts=counts(val, classes)))
    return result


def release_statistics(arrays, rows, plan):
    training = [arrays[row["id"]] for row in rows if row["role"] == "devtrain"]
    if not training or any(not np.isfinite(value).all() for value in training):
        raise ValueError("Train-only finite raw observations required")
    low = np.minimum.reduce([value.min(axis=(0, 1)) for value in training])
    high = np.maximum.reduce([value.max(axis=(0, 1)) for value in training])
    stats = dict(schema_version=1, plan_hash=plan, fit_partition="train",
                 minimum=low.tolist(), maximum=high.tolist())
    stats["statistics_hash"] = digest(stats)
    validate_statistics(stats, {"plan_hash": plan})
    return stats


def retained_positions(bands, name):
    if name not in VIEWS and name not in ("central_gap", "matched_dispersed"):
        raise ValueError("Unregistered physical observation")
    valid = np.flatnonzero(bands.valid)
    if len(valid) < 4:
        raise ValueError("Need four original retained-valid positions")
    size = len(valid) // 2
    if name == "central_gap":
        start = (len(valid)-size)//2
        return tuple(map(int,np.r_[valid[:start],valid[start+size:]]))
    if name == "matched_dispersed":
        return tuple(map(int, valid[np.linspace(0, len(valid)-1, len(valid)-size, dtype=int)]))
    shape, _ = VIEWS[name]
    return tuple(int(x) for x in {
        "full": valid,
        "continuous": valid[:size],
        "held_range": valid[-size:],
        "dispersed": valid[np.linspace(0, len(valid)-1, size, dtype=int)],
    }[shape])


def apply_view(full, master, name, box=None):
    valid = tuple(int(i) for i in np.flatnonzero(master.valid))
    if full.bands.fingerprint != master.subset(valid).fingerprint:
        raise ValueError("Band view must start from the authorized original axis")
    positions = retained_positions(master, name)
    local = {source: column for column,source in enumerate(valid)}
    view = _apply(full.observation, full.bands, tuple(local[i] for i in positions))
    _, mode = VIEWS.get(name,("central_gap","declared"))
    effective = view.bands
    if mode in ("index", "missing"):
        coordinates = (tuple(i/(len(master.band_ids)-1) for i in positions) if mode == "index"
                       else tuple(np.linspace(0., 1., len(positions))))
        effective = BandMetadata(master.source_id, view.bands.band_ids, coordinates, "normalized",
                                 "index_only", view.bands.valid)
    if mode == "shuffled":
        widths = effective.fwhm
        effective = replace(effective, wavelengths=effective.wavelengths[::-1],
                            fwhm=widths[::-1] if widths is not None else None)
    return replace(view, observation=_spatial_view(view.observation, box), bands=effective), positions, mode


def metadata_view(view, master, positions, original_mode, policy):
    """Use registered delivery order for index-only fallback, without optical claims."""
    if original_mode != "declared" or policy not in ("declared","missing","ordinal","permuted_ordinal","misaligned"):
        raise ValueError("Independent metadata control must preserve a declared retained-band view")
    if policy == "declared":
        return view, original_mode
    bands=view.bands
    if policy=="misaligned":
        return replace(view,bands=replace(bands,wavelengths=bands.wavelengths[::-1],
                       fwhm=None if bands.fwhm is None else bands.fwhm[::-1])), "shuffled"
    if (len(master.band_ids)<2 or bands.band_ids!=tuple(master.band_ids[i] for i in positions)
            or tuple(sorted(set(positions)))!=positions):
        raise ValueError("Original retained IDs and trusted delivered order differ")
    if policy=="missing":
        centers=tuple(np.linspace(0.,1.,len(positions)))
    else:
        centers=tuple(i/(len(master.band_ids)-1) for i in positions)
        if policy=="permuted_ordinal":
            centers=centers[::-1]
    made=BandMetadata(bands.source_id,bands.band_ids,centers,"normalized","index_only",bands.valid)
    return replace(view,bands=made),"missing" if policy=="missing" else "index"


def bank_observation(bank, row, producer, band, box=None, *, factor=1, profile=None, synthetic_labels=None,
                     metadata=None):
    _, _, original, context = bank.materialize(row, {"mode": "none"})
    release = bank.release
    stats = release.statistics
    if (stats["statistics_hash"] != context["statistics"]["parent_statistics_hash"]
            or original.observation.plan_hash != release.plan_hash):
        raise ValueError("Statistics or space differs from the bank release")
    view, positions, mode = apply_view(original, release.bands, band, box)
    if metadata is not None:
        view, mode = metadata_view(view, release.bands, positions, mode, metadata)
    source = {**context, "positions": list(positions), "band_ids": list(view.bands.band_ids),
              "retained_statistics": digest(dict(plan=release.plan_hash, parent=stats["statistics_hash"],
                                                minimum=[stats["minimum"][i] for i in positions],
                                                maximum=[stats["maximum"][i] for i in positions],
                                                band_ids=list(view.bands.band_ids))),
              "normalized": array_hash(view.observation.spectral), "band_order": view.bands.fingerprint,
              "band_view": band, "spectral_metadata_mode": mode, "registered_box": box}
    if metadata in ("ordinal","permuted_ordinal","misaligned"):
        source["independent_metadata_policy"] = metadata
    if (profile is None)!=(synthetic_labels is None) or (profile is not None and release.profile!="synthetic"):
        raise ValueError("Physical profile and dense truth are restricted to an authenticated synthetic view")
    if synthetic_labels is not None:
        source.update(labels=array_hash(synthetic_labels), master_band_count=len(release.bands.band_ids))
    lattice = Lattice.make(row["box"], release.shape[:2], factor)
    return _materialize(release.scene, release.classes, view, lattice, source, row["role"], producer, profile=profile)


def bank_view(bank, row, producer, band, box=None, *, factor=1, profile=None, synthetic_labels=None, metadata=None):
    actual = bank_observation(bank, row, producer, band, box, factor=factor, profile=profile,
                              synthetic_labels=synthetic_labels,metadata=metadata)
    targets = bind(actual, row, producer, row["role"], synthetic_labels=synthetic_labels)
    return actual, np.asarray(targets.points, dtype=np.int64).reshape((-1, 2)), targets


def model_from_source(source, fresh, target, *, native_source=False):
    """Copy the established P4 shared tensors; leave the target graph and heads fresh."""
    expected = "learned" if native_source else "static"
    if ((not native_source and target in source.heads) or source.backbone.graph_mode != expected
            or fresh.backbone.graph_mode != "learned"):
        raise ValueError("Source tasks or graph role differ from the transfer contract")
    before = fresh.backbone.state_dict()
    origin = source.backbone.state_dict()
    def shared(keys):
        return {key for key in keys if key.startswith(SHARED)
                and not {"norm", "norms", "batch_normalzation2"}.intersection(key.split("."))}

    allowed, actual = shared(before), shared(origin)
    if allowed != actual or len(allowed) != 29:
        raise ValueError("P4 shared tensor inventory differs")
    for key in allowed:
        if origin[key].shape != before[key].shape or origin[key].dtype != before[key].dtype or not torch.isfinite(origin[key]).all():
            raise ValueError("Shared source shape, dtype or finiteness differs")
    graph = {key: tensor_hash(value) for key,value in before.items() if key.startswith("learnable_graph.")}
    fresh.backbone.load_state_dict({**before, **{key: origin[key] for key in allowed}}, strict=True)
    initialized = fresh.backbone.state_dict()
    if len(graph) != 6 or graph != {k: tensor_hash(initialized[k]) for k in graph}:
        raise ValueError("New target graph reset was not preserved")
    return dict(source=state_hash(source), copied=sorted(allowed), graph=graph, target=state_hash(fresh))
