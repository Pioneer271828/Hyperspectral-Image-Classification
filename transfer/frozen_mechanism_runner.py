"""Bounded real band-window, graph/feature and spatial mechanism cases."""

from collections import defaultdict
from dataclasses import replace
import resource
from time import perf_counter

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from evaluation.frozen_band_region import HEADS
from evaluation.frozen_mechanism import fit_algebraic_temperature
from loadData.observation_sampling import _apply
from model.research_config import digest
from scripts.run_frozen_band_region import _freeze_bank, _pavia, _shapes, _xuzhou, check
from transfer import frozen_band_region as region
from transfer import frozen_mechanism_graph as graphs, frozen_mechanism_plan as plan
from transfer.g3_training import state_hash


def _configuration(root, scope, parent, scene, name, device):
    bank, intake = _freeze_bank(root, scene, scope["previous"])
    model, seed, identity, step, _, checkpoint = (
        _xuzhou(parent, name) if scene == "xuzhou" else _pavia(parent, name, intake, bank))
    model.to(device).eval()
    model.backbone.adapter.cache_attention = False
    model.backbone.adapter.clear_attention_cache()
    selected = dict(seed=seed, model=identity, step=step, checkpoint=checkpoint)
    check(selected == scope["selected"][scene][name], "Model differs from the frozen window registration")
    return bank, model, selected


def _reader(bank, row, salt, scene):
    _, _, view, context = bank.materialize(row, {"mode": "none"})
    shape, offsets = _shapes(row, bank.release.shape[1])
    labels = [label for _, label in row["samples"]]
    aliases = [plan.point_alias(salt, scene, row["role"], idx) for idx, _ in row["samples"]]
    token = plan.point_alias(salt, scene, row["role"] + "/tile", row["id"])
    check(list(context["positions"]) == list(view.bands.band_ids) and
          tuple(view.observation.spectral.shape[:2]) == shape, "Registered input band order or spatial shape differs")
    return view, shape, offsets, labels, aliases, token


def _forward(model, scene, view, offsets, classes, selected, role, **kwargs):
    return graphs.execute(model, scene, view, offsets, classes, 100 if scene == "xuzhou" else 50,
                          role=role, model_state=selected["model"],
                          registered_units=scene != "xuzhou", **kwargs)


def _training_calibrator(bank, model, scene, selected, salt):
    scores = defaultdict(list)
    population = dict(aliases=[], labels=[], blocks=[])
    trains = plan.population_rows(bank, "devtrain")
    for row in trains:
        view, _, offsets, labels, aliases, tile = _reader(bank, row, salt, scene)
        values = _forward(model, scene, view, offsets, bank.release.classes, selected, "devtrain").probabilities
        for head in HEADS:
            scores[head].append(values[head])
        population["aliases"].extend(aliases)
        population["labels"].extend(labels)
        population["blocks"].append(dict(tile=tile, aliases=aliases))
    scores = {head: np.concatenate(data) for head, data in scores.items()}
    return ({head: fit_algebraic_temperature(values, population["labels"]) for head, values in scores.items()},
            population, {f"calibration_train__{head}": values for head, values in scores.items()}, len(trains))


def _replaced(view, box):
    return replace(view, observation=region._spatial_view(view.observation, box))


def _window_design(window, original, native):
    keep = tuple(window["retained"])
    selected = _apply(original.observation, original.bands, keep)
    check(digest(list(selected.bands.band_ids)) == window["retained_hash"] and
          selected.bands.fingerprint != original.bands.fingerprint and
          np.array_equal(selected.observation.spectral, np.take(original.observation.spectral, keep, axis=-1)) and
          graphs.geometry_hash(selected, "devval") == native.graph.geometry,
          "Actual selected VCA input, metadata order or pixel geometry differs")
    return selected


def _window_outputs(model, scene, view, boxes, offsets, classes, selected, full):
    count = 0

    def run(current, frozen=None):
        nonlocal count
        count += 1
        return _forward(model, scene, current, offsets, classes, selected, "devval", frozen=frozen)

    target = _replaced(view, boxes["target"])
    clean_result, target_result = run(view), run(target)
    clean_cache, target_cache = clean_result.graph, target_result.graph
    graphs.assert_same_view(clean_result, run(view, clean_cache))
    graphs.assert_same_view(target_result, run(target, target_cache))
    output = {
        "clean": clean_result,
        "target": target_result,
        "matched": run(_replaced(view, boxes["matched"])),
        "clean_target_graph": run(view, target_cache),
        "target_clean_graph": run(target, clean_cache),
        "clean_full_graph": run(view, full.graph),
        "target_full_graph": run(target, full.graph),
    }
    if boxes["random"] is not None:
        output["random"] = run(_replaced(view, boxes["random"]))
    for source, candidates in (("clean", ("clean_target_graph", "clean_full_graph")),
                               ("target", ("target_clean_graph", "target_full_graph"))):
        for name in candidates:
            left, right = output[source].trace, output[name].trace
            check(left["feature_input"] == right["feature_input"] and left["observation"] == right["observation"],
                  "Graph intervention affected the retained feature path")
    for left, right in (("clean", "target_clean_graph"), ("target", "clean_target_graph"),
                        ("clean_full_graph", "target_full_graph")):
        check(output[left].graph is output[right].graph and output[left].trace["graph"] == output[right].trace["graph"],
              "Feature intervention changed the assigned fixed graph")
    return output, count


def _append(arrays, name, output):
    for head in HEADS:
        arrays[f"{name}__{head}"].append(output.probabilities[head])


def _spatial_partition(group_masks, offsets, box, shape, full):
    q = full.graph.state.Q.coalesce()
    positive = q.indices()[:, q.values() > 0].detach().cpu().numpy()
    for key, value in region.same_region_groups(offsets, box, shape, positive).items():
        group_masks[key].extend(value.tolist())


def run_windows(root, scope, parent, scene, case, previous, *, device="cpu", limited=False):
    start = perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bank, model, selected = _configuration(root, scope, parent, scene, case, device)
    salt = scope["alias_salt"]
    if limited:
        temperatures, train, calibration, count = {}, None, {}, 0
    else:
        temperatures, train, calibration, count = _training_calibrator(bank, model, scene, selected, salt)
    windows = scope["windows"][scene][:2] if limited else scope["windows"][scene]
    classes = bank.release.classes
    arrays, masks, traces, negative = defaultdict(list), defaultdict(list), [], None
    labels, aliases, points, random_defined = [], [], [], []
    rows = plan.population_rows(bank, "devval")
    for row in rows[:1] if limited else rows:
        full, shape, offsets, current, tagged, tile = _reader(bank, row, salt, scene)
        base = _forward(model, scene, full, offsets, classes, selected, "devval")
        count += 1
        _append(arrays, "full", base)
        boxes = region.region_boxes(shape, offsets, current, plan.FOCUS[scene])
        boxes["random"] = plan.random_control(shape, boxes, row["id"])
        random_defined.extend([boxes["random"] is not None] * len(current))
        _spatial_partition(masks, offsets, boxes["target"], shape, base)
        trace = dict(tile=tile, aliases=tagged, full=base.trace, windows={},
                     area=(boxes["target"][1]-boxes["target"][0])**2,
                     random_available=boxes["random"] is not None)
        for window in windows:
            retained = _window_design(window, full, base)
            group, calls = _window_outputs(model, scene, retained, boxes, offsets, classes, selected, base)
            count += calls
            if "random" not in group:
                # Preserve the row roster; downstream masks exclude unavailable random controls.
                group["random"] = group["matched"]
            trace["windows"][window["id"]] = {cell: result.trace for cell, result in group.items()}
            for cell, result in group.items():
                _append(arrays, f"{window['id']}__{cell}", result)
            if negative is None:
                deleted = full.observation.spectral.copy()
                deleted[..., window["deleted"]] += np.float32(137.)
                dirty = replace(full, observation=replace(full.observation, spectral=deleted))
                poison = _apply(dirty.observation, dirty.bands, tuple(window["retained"]))
                reject = False
                foreign = replace(base.graph, geometry="foreign")
                try:
                    _forward(model, scene, retained, offsets, classes, selected, "devval", frozen=foreign)
                except ValueError:
                    reject = True
                check(reject and graphs.view_hash(poison) == graphs.view_hash(retained),
                      "Foreign graph bypass or dropped data leak")
                different = replace(retained, bands=replace(retained.bands,
                                    wavelengths=tuple(reversed(retained.bands.wavelengths))))
                check(graphs.view_hash(different) != graphs.view_hash(retained),
                      "The retained view cache key ignored changed metadata")
                negative = dict(foreign_graph_rejected=True, deleted_poison_equal=True,
                                reordered_metadata_distinct=True, full_and_retained_cache_separate=True,
                                feature_input_changes_with_target=group["clean"].trace["feature_input"] !=
                                                                  group["target"].trace["feature_input"])
        traces.append(trace)
        labels.extend(current)
        aliases.extend(tagged)
        points.extend(row["samples"])
    arrays = {**{key: np.concatenate(value) for key, value in arrays.items()}, **calibration}
    if device == "cuda":
        torch.cuda.synchronize()
    check(state_hash(model) == selected["model"], "Frozen analysis changed parameters or buffers")
    if not limited:
        check(len(points) == 5 * classes and digest(points) == previous["population"],
              "Window roster differs from the prior result")
        for head in HEADS:
            old = previous["paired"]["contiguous_vs_full"][head]["source_confusion"]
            current = classification_metrics(labels, (arrays[f"full__{head}"].argmax(1)+1).tolist(),
                                             list(range(1, classes+1)))
            check(current["confusion_matrix"] == old, "Current full anchor differs from archived confusion")
    return dict(scene=scene, name=case, observation_scope="devval", selected=selected, model_seed=selected["seed"],
                truth=labels, aliases=aliases, group_masks=dict(masks), train=train, temperatures=temperatures,
                random_available=random_defined, trace=traces, physical_forwards=count, negative=negative,
                group_count={key: sum(mask) for key, mask in masks.items()},
                population=previous["population"], support=previous["bank"], device=device,
                runtime_seconds=perf_counter()-start, hashes=dict(windows=digest(windows), roles=digest(aliases)),
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_peak_bytes=dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
                if device == "cuda" else None), arrays
