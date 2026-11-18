"""Frozen inference over authorized development tiles and paired band/region views."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace

import numpy as np
import torch

from evaluation.block_input_diagnostics import (
    DEFAULT_SPECTRAL_CAP, XUZHOU_SPECTRAL_CAP, intervene_spectrum,
)
from evaluation.frozen_band_region import HEADS, hard
from loadData.observation_sampling import sample, subset_statistics
from loadData.spatial_protocol import forward_partition
from model.coverage_config import coverage_forward_kwargs
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.development_blocks import array_hash


def band_specs(bands):
    count = len(bands.band_ids)
    if count < 4 or bands.band_ids != tuple(range(count)) or not all(bands.valid):
        raise ValueError("Expected the registered complete ordered band axis")
    deleted = count // 2
    first = (count - deleted) // 2
    last = first + deleted - 1
    waves = bands.wavelengths
    if any(left >= right for left, right in zip(waves, waves[1:])):
        raise ValueError("Nominal positions must be strictly increasing")
    return {
        "full": {"mode": "none"},
        "contiguous": {"mode": "contiguous_gap", "lower": waves[first], "upper": waves[last],
                       "unit": bands.wavelength_unit},
        "scattered": {"mode": "uniform_stride", "stride": 2, "offset": 0},
    }


def region_boxes(shape, offsets, labels, focus):
    """Fix a target and an equal-area nonoverlapping context box before inference."""
    height, width = shape
    side = min(5, height // 2, width // 2)
    if not offsets or side < 1:
        raise ValueError("Region intervention requires released points and a nontrivial tile")
    priority = [offset for offset, label in zip(offsets, labels, strict=True) if label in focus]
    y, x = divmod((priority or offsets)[0], width)
    ty, tx = min(max(y - side // 2, 0), height - side), min(max(x - side // 2, 0), width - side)
    target = (ty, ty + side, tx, tx + side)
    points = [divmod(offset, width) for offset in offsets]
    choices = []
    for a in range(height - side + 1):
        for b in range(width - side + 1):
            if a + side <= ty or ty + side <= a or b + side <= tx or tx + side <= b:
                overlap = sum(a <= c < a + side and b <= d < b + side for c, d in points)
                choices.append((overlap, (ty-a)**2 + (tx-b)**2, a, b))
    _, _, a, b = min(choices)
    return {"target": target, "matched": (a, a + side, b, b + side)}


def _spatial_view(observation, box):
    if box is None:
        return observation
    values = observation.spectral
    if values.size <= DEFAULT_SPECTRAL_CAP or (values.shape[-1] == 436 and max(values.shape[:2]) <= 28):
        cap = DEFAULT_SPECTRAL_CAP if values.size <= DEFAULT_SPECTRAL_CAP else XUZHOU_SPECTRAL_CAP
        altered = intervene_spectrum(torch.tensor(values), {"kind": "occlusion_min", "box": box},
                                     spectral_cap=cap).numpy()
    else:
        # The released Xuzhou/HongHu banks can contain larger tiles or reduced
        # depths. Apply the same operator in independent <=64K scalar slabs.
        if values.size > 48 * 48 * 436 or values.shape[0] * values.shape[1] > DEFAULT_SPECTRAL_CAP:
            raise ValueError("Tile exceeds the registered spatial diagnostic bound")
        slab = DEFAULT_SPECTRAL_CAP // (values.shape[0] * values.shape[1])
        altered = np.concatenate([intervene_spectrum(
            torch.tensor(values[..., start:start + slab]), {"kind": "occlusion_min", "box": box},
        ).numpy() for start in range(0, values.shape[-1], slab)], axis=-1)
    return replace(observation, spectral=altered)


@contextmanager
def _capture_graph(model, traces):
    graph = model.backbone.learnable_graph
    if graph is None or "build_graph" in graph.__dict__:
        raise ValueError("Expected an unwrapped private learned graph builder")
    build = graph.build_graph

    def capture(*args, **kwargs):
        state = build(*args, **kwargs)
        q = state.Q.coalesce()
        traces.append((dict(q=tensor_hash(q), a=tensor_hash(state.A0),
                            q_nnz=q._nnz(), a_nnz=state.A0._nnz()),
                       q.indices().cpu().numpy(), q.values().cpu().numpy()))
        return state

    graph.build_graph = capture
    try:
        yield
    finally:
        del graph.build_graph


@torch.no_grad()
def infer(model, dataset, view, offsets, classes, scale, *, role, wavelength_unit=True):
    obs, bands = view.observation, view.bands
    if model.training or role != "devval" or obs.partition not in ("val", "train"):
        raise ValueError("Frozen development validation only")
    traces, adapter_inputs = [], []
    adapter = model.backbone.adapter
    hook = adapter.register_forward_pre_hook(
        lambda _module, args: adapter_inputs.append(tensor_hash(args[0])))
    kwargs = coverage_forward_kwargs(model, bands)
    if bands.fwhm is not None:
        kwargs.update(fwhm=np.asarray(bands.fwhm, dtype=np.float32), fwhm_unit=bands.fwhm_unit)
    if wavelength_unit:
        kwargs["wavelength_unit"] = bands.wavelength_unit
    try:
        binding = model.partition_bands(bands) if hasattr(model, "partition_bands") else nullcontext()
        with _capture_graph(model, traces), binding:
            output = forward_partition(model, dataset, obs, np.asarray(bands.wavelengths, dtype=np.float32),
                                       scale, **kwargs)
    finally:
        hook.remove()
    if len(traces) != 1 or len(adapter_inputs) != 1:
        raise ValueError("Expected one fresh graph and one VCA input")
    graph, membership, memberships = traces[0]
    scores = {name: values.cpu().numpy()[offsets].copy() for name, values in zip(HEADS, output[:3], strict=True)}
    for values in scores.values():
        hard(values, len(offsets), classes)
    trace = dict(**graph, adapter_input=adapter_inputs[0], metadata=bands.fingerprint,
                 spectral=array_hash(obs.spectral), probabilities={k: array_hash(v) for k, v in scores.items()})
    return scores, trace, membership[:, memberships > 0]


def same_region_groups(offsets, box, shape, membership):
    y0, y1, x0, x1 = box
    _, width = shape
    target = np.array([y0 <= i // width < y1 and x0 <= i % width < x1 for i in offsets])
    pixels, tokens = membership
    covered = (y0 <= pixels // width) & (pixels // width < y1) & (x0 <= pixels % width) & (pixels % width < x1)
    linked = np.isin(offsets, pixels[np.isin(tokens, np.unique(tokens[covered]))])
    return {"in_target_box": target, "outside_sharing_native_q": ~target & linked,
            "outside_other_native_q": ~target & ~linked}


def validate_observation_path(model, dataset, original, bands, stats, spec, offsets, classes, scale,
                              *, role, wavelength_unit=True):
    """Poison only deleted channels and exercise a registered leaking negative."""
    view, record = sample(original, bands, spec)
    kept = record["selected_positions"]
    dropped = sorted(set(range(len(bands.band_ids))) - set(kept))
    if not dropped or list(view.bands.band_ids) != kept:
        raise ValueError("Negative control needs a true subset in source column order")
    dirty = original.spectral.copy()
    dirty[..., dropped] += np.float32(137.)
    poisoned = replace(original, spectral=dirty)
    new, new_record = sample(poisoned, bands, spec)
    modified_stats = {**stats}
    for name in ("minimum", "maximum"):
        values = list(stats[name])
        for idx in dropped:
            values[idx] += 137.
        modified_stats[name] = values
    modified_stats["statistics_hash"] = digest({k: v for k, v in modified_stats.items() if k != "statistics_hash"})
    base_stats, new_stats = (subset_statistics(values, bands, record) for values in (stats, modified_stats))
    if new_record != record or any(base_stats[key] != new_stats[key] for key in ("minimum", "maximum", "band_ids")):
        raise ValueError("Deleted-column extrema leaked into retained metadata/statistics")
    kwargs = dict(role=role, wavelength_unit=wavelength_unit)
    _, base, _ = infer(model, dataset, view, offsets, classes, scale, **kwargs)
    _, test, _ = infer(model, dataset, new, offsets, classes, scale, **kwargs)
    if base != test:
        raise ValueError("Deleted-column poison changed the graph/VCA/three heads")
    leak = dirty[..., kept].copy()
    leak[..., 0] += dirty[..., dropped[0]]
    leaky_view = replace(view, observation=replace(view.observation, spectral=leak))
    _, negative, _ = infer(model, dataset, leaky_view, offsets, classes, scale, **kwargs)
    if negative["adapter_input"] == base["adapter_input"]:
        raise ValueError("Deliberately leaking implementation escaped the VCA input detector")
    return dict(retained_stats_equal=True, deleted_poison_equal=True,
                leaky_vca_input_detected=True, leaky_q_changed=negative["q"] != base["q"],
                leaky_a_changed=negative["a"] != base["a"],
                leaky_heads_changed={k: negative["probabilities"][k] != base["probabilities"][k] for k in HEADS},
                metadata_order=kept, metadata_hash=view.bands.fingerprint)
