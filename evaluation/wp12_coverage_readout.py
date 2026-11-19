"""Verify and publish development-only spacing and branch readout evidence."""

from itertools import product
import os
from tempfile import TemporaryDirectory
import re

import numpy as np

from evaluation.frozen_band_region import HEADS, hard, paired
from evaluation.frozen_mechanism import point_scores
from scripts.run_coverage_readout import ABS, BATCH, WINDOWS, assert_probabilities, parent_case, registration
from scripts.run_frozen_band_region import _read, _write, check, pin
from scripts.run_frozen_mechanisms import read_bundle
from transfer.frozen_branch_readout import BRANCH_TREATMENTS
from transfer.spectral_spacing_recovery import ARMS, DENSITIES, SEEDS, VIEWS


def predictions(scores):
    return scores.argmax(1) + 1


def softmax(logits):
    check(logits.ndim == 2 and np.isfinite(logits).all(), "Nonfinite readout logits")
    exp = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


def _cohort(result, before, size):
    names = result["aliases"]
    check(names == before["aliases"] and result["truth"] == before.get("truth", before.get("labels"))
          and result["sample_n"] == len(names) == len(set(names)) == size
          and all(re.fullmatch(r"[0-9a-f]{32}", alias) for alias in names),
          "Publication accepts only the exact de-identified cohort")


def xuzhou(result, arrays, previous, reg):
    before, original = previous
    _cohort(result, before, 45)
    y = np.asarray(result["truth"])
    focus = reg["focus"]["xuzhou"]
    rivals = {k: reg["foils"]["xuzhou"][str(k)][0] for k in focus}
    windows = {}
    for win in WINDOWS:
        key = win if win == "full" else f"{win}__clean"
        reports = {}
        for head in HEADS:
            archived = original[f"{key}__{head}"]
            assert_probabilities(arrays[f"{win}__original__{head}"], archived)
            reports[head] = paired(hard(arrays[f"{win}__spacing__{head}"], 45, 9),
                                   hard(archived, 45, 9), y, 9)
            reports[head]["protection"] = (
                min(reports[head][f"delta_{metric}"] for metric in ("oa", "aa")) >=
                reg["guards"]["aggregate_min_delta"] and
                all(reports[head]["classes"][c-1]["target_recall"] -
                    reports[head]["classes"][c-1]["source_recall"] >=
                    reg["guards"]["focus_min_delta_recall"] for c in focus))
        logits = {arm: arrays[f"{win}__logit__{arm}"] for arm in BRANCH_TREATMENTS}
        assert_probabilities(softmax(logits["native"]), original[f"{key}__fused"])
        check(np.array_equal(logits["native"], logits["restored"]), "Restore changed native scores")
        contributions = {side: arrays[f"{win}__contribution__{side}"] for side in ("cnn", "graph")}
        bias = arrays[f"{win}__bias"]
        check(np.allclose(logits["native"], sum(contributions.values()) + bias, atol=ABS),
              "Archived branch decomposition does not restore the fused classifier")
        fused, graph, cnn = (predictions(original[f"{key}__{head}"]) for head in HEADS)
        classes, points = {}, []
        for cls in focus:
            foil = rivals[cls]
            matched = y == cls
            missed = matched & (graph == y) & (fused != y)
            counts = {}
            for arm, scores in logits.items():
                pred = predictions(scores)
                counts[arm] = dict(correct=int((matched & (pred == y)).sum()),
                                   recovered=int((missed & (pred == y)).sum()))
            classes[str(cls)] = dict(n=int(matched.sum()), foil=foil,
                                     graph_correct=int((matched & (graph == y)).sum()),
                                     cnn_correct=int((matched & (cnn == y)).sum()),
                                     graph_correct_fused_wrong=int(missed.sum()), arms=counts)
            for index in np.flatnonzero(matched):
                gap = lambda mat: float(mat[index, cls-1] - mat[index, foil-1])
                points.append(dict(alias=result["aliases"][index], truth=cls, foil=foil,
                                   graph_correct_fused_wrong=bool(missed[index]),
                                   original=dict(fused=int(fused[index]), graph=int(graph[index]), cnn=int(cnn[index])),
                                   margin={arm: gap(scores) for arm,scores in logits.items()},
                                   predicted={arm: int(predictions(scores)[index]) for arm,scores in logits.items()},
                                   contribution={side: gap(values) for side,values in {**contributions, "bias": bias}.items()}))
        windows[win] = dict(weights=result["weights"][win], spacing=reports, branch_classes=classes, points=points)
    return windows


def honghu(result, arrays, previous, reg, entry):
    before, original = previous
    _cohort(result, before["populations"]["val"], 110)
    check(result["support_n"] == before["populations"]["train"]["n"] == 22 * entry["k"]
          and result["support_hash"] == entry["train_population"], "Unexpected frozen support population")
    y = np.asarray(result["truth"])
    focus = reg["focus"]["whu-honghu"]
    foes = {cls: reg["foils"]["whu-honghu"][str(cls)] for cls in focus}
    indexes = [i for i, label in enumerate(y) if label in focus]
    outcomes = {}
    groups = {alias: name for name, members in result["spatial"].items() for alias in members}
    blocks = {alias: index for index, members in enumerate(result["blocks"]) for alias in members}
    check(set(blocks) == set(result["aliases"]) and set(groups) <= set(blocks), "Spatial view escaped the cohort")
    for head in HEADS:
        forecast = arrays[f"{head}__native"]
        assert_probabilities(forecast, original[f"nominal__selected__val__{head}"])
        distance, between, counts = (arrays[f"{head}__{kind}"] for kind in ("distance", "between", "counts"))
        check(distance.shape == (110, 22) and between.shape == (22, 22) and counts.shape == (22,)
              and np.all(counts == entry["k"]) and np.isfinite(distance).all() and np.isfinite(between).all()
              and np.min(distance) >= 0 and np.min(between) >= 0
              and np.allclose(between, between.T) and np.all(between.diagonal() == 0),
              "Frozen reference support or distance inventory differs")
        projection, bias = (arrays[f"{head}__{key}"] for key in ("projection", "bias"))
        assert_probabilities(softmax(projection + bias), forecast)
        native = point_scores(forecast, y, foes)
        reference = point_scores(softmax(-distance), y, foes)
        native_labels, reference_labels = predictions(forecast), (-distance).argmax(1) + 1
        outcomes[head] = dict(
            paired=paired(reference_labels, native_labels, y, 22),
            focus_added_errors=int(((native_labels == y) & (reference_labels != y) & np.isin(y, focus)).sum()),
            reference_kind="same_embedding_width_one_frozen_centroid_per_class",
            points=[dict(alias=result["aliases"][i], truth=int(y[i]), block=blocks[result["aliases"][i]],
                         spatial_group=groups.get(result["aliases"][i], "no_registered_ablation_group"),
                         native=native[i], centroid=reference[i],
                         readout={str(foil): dict(
                             bias_gap=float(bias[y[i]-1] - bias[foil-1]),
                             projection_gap=float(projection[i,y[i]-1] - projection[i,foil-1]),
                             within_distance=float(distance[i,y[i]-1]),
                             rival_distance=float(distance[i,foil-1]),
                             between_centroids=float(between[y[i]-1,foil-1]))
                             for foil in foes[int(y[i])]}) for i in indexes])
    return outcomes


def synthetic(directory, registration_sha, code):
    output = directory / "synthetic"
    link = pin(output / "receipt.json")
    receipt = _read(link)
    expected = {f"{seed}-{density}-{arm}" for seed,density,arm in product(SEEDS, DENSITIES, ARMS)}
    check(receipt["status"] == "passed" and receipt["producer"] == code["commit"]
          and receipt["registration"] == registration_sha and set(receipt["cases"]) == expected
          and receipt["synthetic_updates"] == 12 * len(expected)
          and receipt["real_updates"] == 0 and receipt["query_accessed"] is receipt["raw_gt_loaded"] is False,
          "Synthetic matrix is incomplete or mixed")
    output_rows = {}
    for seed, density in product(SEEDS, DENSITIES):
        base = receipt["cases"][f"{seed}-{density}-none"]
        for arm in ARMS:
            name = f"{seed}-{density}-{arm}"
            case = receipt["cases"][name]
            place = output / name
            check(_read(pin(place / "receipt.json")) == case and
                  all(case[k] == base[k] for k in ("common_initial", "labels_root", "inputs"))
                  and [case["phases"][phase]["updates"] for phase in ("full", "prefix", "resume", "completed")] ==
                  [6, 2, 4, 0] and case["parameters"] - base["parameters"] ==
                  (0 if arm in ("none", "spacing") else 194), "Synthetic recovery or matching evidence differs")
            for phase in ("full", "prefix", "resume", "completed"):
                check(pin(place / f"{phase}.pt")["sha256"] == case["checkpoint_sha256"][phase],
                      "Synthetic continuation artifact changed")
            grouped = {band: [row for row in case["validation"] if row["band"] == band] for band in VIEWS}
            summaries = {}
            for band, rows in grouped.items():
                summaries[band] = {}
                for head in HEADS:
                    counts = sum(row["scores"][head]["count"] for row in rows)
                    confusion = sum(np.asarray(row["scores"][head]["point_confusion"]) for row in rows)
                    check(len(rows) == 2 and int(confusion.sum()) == counts and counts > 0,
                          "Synthetic validation population differs")
                    summaries[band][head] = dict(
                        count=int(counts), nll=sum(row["scores"][head]["nll_sum"] for row in rows) / counts,
                        oa=float(confusion.trace() / counts), per_class_n=confusion.sum(1).tolist(),
                        per_class_correct=confusion.diagonal().tolist())
            output_rows[name] = {**{k: case[k] for k in ("arm", "density", "seed", "selected_step",
                                                         "selected_nll", "parameters", "updated", "seconds", "phases")},
                                 "condition_scores": summaries}
    return dict(case_count=len(expected), synthetic_updates=receipt["synthetic_updates"],
                real_updates=0, cases=output_rows), link["sha256"]


def ingest(root, directory, code):
    reg, _, _, batch = registration(root, code)
    manifest = pin(directory / "registration.json")
    check(_read(manifest) == reg and not (directory / "wp12").exists(), "Expected new publication and fixed inputs")
    evidence, cost, sources = {}, {}, {}
    for name, entry in reg["cases"].items():
        link = pin(directory / f"{name}.npz")
        meta, arrays = read_bundle(link)
        check(meta["case"] == name and meta["registration"] == manifest["sha256"]
              and meta["keys"] == sorted(arrays) and meta["real_updates"] == 0
              and meta["raw_gt_loaded"] is meta["query_accessed"] is False,
              "WP12 refuses altered, mixed-role or training evidence")
        previous = parent_case(root, batch, name)
        result = meta["result"]
        evidence[name] = (xuzhou(result, arrays, previous, reg) if name.startswith("xuzhou__")
                          else honghu(result, arrays, previous, reg, entry))
        cost[name] = {k: result[k] for k in ("forwards", "seconds", "stages_seconds", "rss_peak_kib", "sample_n")}
        sources[name] = link["sha256"]
    trained, synthetic_sha = synthetic(directory, manifest["sha256"], code)
    with TemporaryDirectory(prefix=".wp12-", dir=directory) as temp:
        stage = directory / temp / "publication"
        stage.mkdir()
        artifacts = {name: _write(stage / name, value)["sha256"] for name, value in (
            ("frozen-readouts.json", evidence), ("raw-cost.json", cost), ("synthetic-recovery.json", trained))}
        _write(stage / "receipt.json", dict(status="accepted_development_and_synthetic_only",
               producer=code["commit"], parent_batch=BATCH, registration=manifest["sha256"],
               case_count=len(evidence), input_cases=sources, synthetic_source=synthetic_sha,
               artifacts=artifacts, real_updates=0, synthetic_updates=trained["synthetic_updates"],
               query_accessed=False, raw_gt_loaded=False))
        os.rename(stage, directory / "wp12")
    return pin(directory / "wp12/receipt.json")
