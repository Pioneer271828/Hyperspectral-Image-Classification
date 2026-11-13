"""Spatial observation and condition interventions on the existing WP07 consumers."""

from dataclasses import replace
from time import perf_counter

import numpy as np
import torch

from loadData.spatial_protocol import forward_partition
from transfer import development_bank_consumers as old
from transfer.development_block_probe import VIEWS, _graph_trace, _inventory
from transfer.g3_training import state_hash
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_support import _measure
from transfer.spatial_observation import MODES, materialize, observability


ARMS = ("off", "fixed", "mean", "content", "metadata", "joint", "cnn_content", "graph_content")
EXTRA = {
    "fixed": MultiscaleTrial(cnn_dilation=2, graph_multiscale="fixed", graph_radius=2),
    "cnn_content": MultiscaleTrial(cnn_multiscale="content"),
    "graph_content": MultiscaleTrial(graph_multiscale="content"),
    "single_original": MultiscaleTrial(),
}
CONDITIONS = ("matched", "missing", "permuted", "perturbed")


def initialize(baseline, arm, seed):
    if arm in EXTRA:
        model = EXTRA[arm].initialize(baseline, seed).eval()
        model.backbone.adapter.cache_attention = False
        return model
    return old.initialize(baseline, arm, seed)


def fixture_conditions(view, condition, tile_id):
    """Intervene only on the synthetic selector input; the VCA wavelengths stay fixed."""
    if condition not in CONDITIONS:
        raise ValueError("Unregistered physical condition intervention")
    if condition == "missing":
        return None, None, None
    a = tile_id % 2
    profile = 1-a if condition == "permuted" else a
    scale = (0.7, 2.4)[profile]
    offset = (0., 115.)[profile]
    coverage = (0.8, 0.54)[profile]
    if condition == "perturbed":
        scale *= 1.25
        offset += 30.
    bands = replace(view.bands, wavelength_quality="verified", fwhm=(8.,)*len(view.bands.band_ids),
                    fwhm_unit="nm", wavelengths=tuple(wave+offset for wave in view.bands.wavelengths))
    return bands, coverage, scale


def forward(model, arm, actual, *, condition=None):
    scene, tile, view, receipt = actual
    trace = []
    if condition is not None and (receipt["profile"] != "synthetic" or condition not in CONDITIONS):
        raise ValueError("Invalid selector condition")
    with _measure(model.backbone.learnable_graph, "build_graph", {}, "build",
                  lambda graph: trace.append(_graph_trace(graph))):
        if condition is None or arm == "off":
            from transfer.g3_training import _forward
            output = _forward(model, scene, tile, observation_view=view, multiscale_condition_only=True)
        else:
            bands, coverage, gsd = fixture_conditions(view, condition, tile["id"])
            with model.partition_bands(bands), model.physical_conditions(coverage, gsd):
                output = forward_partition(model, scene.dataset, view.observation,
                    np.asarray(view.bands.wavelengths, dtype=np.float32),
                    50 if scene.dataset == "paviaU" else 100, wavelength_unit=view.bands.wavelength_unit)
    if len(trace) != 1:
        raise ValueError("A WP07 observation must build one pixel-to-region assignment")
    graph = output[-1].get("multiscale_graph")
    cnn = output[-1].get("multiscale_cnn")
    weights = {}
    for name, value, label in (("cnn", cnn, "dilations"), ("graph", graph, "radii")):
        if value and "weights" in value:
            weights[name] = dict(scales=list(value[label]), weights=value["weights"].detach().tolist())
    return output, dict(q=trace[0]["q"], a0=trace[0]["a0"], grid=trace[0]["grid"],
                        q_nnz=trace[0]["q_nnz"], a0_nnz=trace[0]["a0_nnz"],
                        range_nnz=None if graph is None else list(graph["nnz"]), weights=weights)


def inventory(baseline, models):
    shared = {name: tensor.detach().clone() for name,tensor in baseline.state_dict().items()}
    return {arm: _inventory(model, shared) for arm,model in models.items()}


@torch.no_grad()
def condition_audit(bank, baseline, producer):
    row = bank.role_rows("devtrain")[0]
    actual = materialize(bank, row, VIEWS["full"], "area2", producer)
    models = {arm: initialize(baseline, arm, 7321) for arm in ARMS}
    result = {}
    for arm in ("off", "mean", "content", "metadata", "joint"):
        result[arm] = {}
        for condition in CONDITIONS:
            output, trace = forward(models[arm], arm, actual, condition=condition)
            result[arm][condition] = dict(heads=old.heads(output, actual), trace=trace)
    for arm in ("off", "mean", "content"):
        if any(result[arm][mode] != result[arm]["matched"] for mode in CONDITIONS):
            raise ValueError("Physical intervention affected a condition-independent arm")
    for arm, fallback in (("metadata", "mean"), ("joint", "content")):
        # These seed-matched variants have identical content weights where applicable.
        if result[arm]["missing"]["heads"] != result[fallback]["matched"]["heads"]:
            raise ValueError("Missing metadata did not reproduce the fallback")
        for change in ("permuted", "perturbed"):
            a,b = result[arm]["matched"], result[arm][change]
            if (a["trace"]["weights"] == b["trace"]["weights"] or a["heads"] == b["heads"]
                    or a["trace"]["q"] != b["trace"]["q"]):
                raise ValueError("Condition intervention did not isolate scale weights on the same Q")
    single = initialize(baseline, "single_original", 7321)
    output, trace = forward(single, "single_original", actual)
    if (old.heads(output, actual) != result["off"]["matched"]["heads"]
            or trace["weights"]["cnn"] != {"scales":[1], "weights":[[1.0]]}):
        raise ValueError("Single original scale differs from off")
    return dict(rows=result, synthetic_forwards=21, updates=0, single_candidate=trace)


@torch.no_grad()
def scan(bank, baseline, previous, producer):
    release = bank.release
    expected = release.source.get("baseline_state", release.source["state"])
    if state_hash(baseline) != expected or previous["bank"] != bank.pin:
        raise ValueError("Frozen bank or model differs")
    arms = ("off",) if release.profile == "fixed" else (*ARMS, "graph_radius2", "native")
    models = {arm: initialize(baseline, arm, 7321) for arm in arms}
    before = inventory(baseline, models)
    selected = set(previous["representatives"])
    old_rows = {(row["context"]["tile"], row["view"]): row for row in previous["rows"]}
    rows, counts, reads = [], {}, 0
    transform_seconds, forward_seconds = {}, {}
    for row in bank.rows.values():
        for band, spec in VIEWS.items():
            if band != "full" and row["id"] not in selected:
                continue
            for spatial in MODES:
                started = perf_counter()
                actual = materialize(bank, row, spec, spatial, producer)
                transform_seconds.setdefault(spatial, []).append(perf_counter()-started)
                outputs, traces = {}, {}
                for arm in (arms if row["id"] in selected else ("off",)):
                    started = perf_counter()
                    if arm == "native":
                        selector = old.native_selector(models[arm], actual, row, expected, 48, 19064)
                        result, trace = old.forward(models[arm], arm, actual, selector)
                        reads += 1
                    else:
                        result, trace = forward(models[arm], arm, actual)
                    forward_seconds.setdefault(arm, []).append(perf_counter()-started)
                    outputs[arm], traces[arm] = old.heads(result, actual), trace
                    counts[arm] = counts.get(arm, 0) + 1
                if spatial == "identity":
                    prior = old_rows[(row["id"], band)]
                    for arm, result in prior["probabilities"].items():
                        if outputs[arm] != result:
                            raise ValueError("Identity changed a frozen #453 probability")
                    prior_graph = prior["graph"]["off"]["built"][0]
                    if any(traces["off"][key] != prior_graph[key] for key in ("q", "a0", "grid", "q_nnz", "a0_nnz")):
                        raise ValueError("Identity changed frozen off graph")
                if "metadata" in outputs and (outputs["metadata"] != outputs["mean"]
                                               or outputs["joint"] != outputs["content"]):
                    raise ValueError("Unknown real metadata did not use established fallback")
                rows.append(dict(band=band, spatial=spatial, observation=actual[-1], outputs=outputs, graphs=traces,
                                 sparse=observability(row, spatial, release.shape[1], release.classes)))
    if inventory(baseline, models) != before or state_hash(baseline) != expected:
        raise ValueError("Real scan mutated source or consumer state")
    def timings(groups):
        return {name: dict(calls=len(values), total_seconds=sum(values),
                           median_seconds=float(np.median(values))) for name,values in groups.items()}
    return dict(rows=rows, counts=counts, native_captures=reads, blocks=len(bank.rows),
                representatives=sorted(selected), models=before, identity_rows=len(old_rows),
                timings=dict(materialization=timings(transform_seconds), forward=timings(forward_seconds),
                             warmup=0, note="whole scan incl first calls; native includes capture; CPU one thread"),
                real_updates=0, raw_gt=0, query=0, new_labels=0)
