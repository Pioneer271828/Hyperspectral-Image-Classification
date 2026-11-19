"""Independently recompute the fixed candidate and publish all-class protection."""

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS, hard, paired
from evaluation.wp12_coverage_readout import softmax
from scripts.run_frozen_band_region import _read, _write, check, pin
from scripts.run_frozen_mechanisms import read_bundle
from scripts.run_transfer_readout import archive, parent_case, registered, stratified
from transfer.frozen_branch_readout import centroid_reference
from transfer.support_readout import SupportReadout, output_scores, support_scale, support_weights, tensor

CONTROL = ("mask_cnn", "mask_graph", "exchange")
TOL = 2e-12


def guard(pairs, focus, safeguards, is_honghu):
    bad = []
    for key in ("oa", "aa"):
        if pairs[f"delta_{key}"] < safeguards["aggregate_min_delta"]:
            bad.append(key)
    for row in pairs["classes"]:
        cls = row["class_id"]
        if cls in focus:
            if row["target_recall"] - row["source_recall"] < safeguards["focus_min_delta_recall"]:
                bad.append(f"recall_{cls}")
            if is_honghu and row["correct_to_error"]:
                bad.append(f"new_error_{cls}")
    return bad


def assert_array(got, expected):
    check(np.array_equal(got, expected) if got.dtype.kind in "iu" else
          (got.shape == expected.shape and np.allclose(got, expected, rtol=TOL, atol=TOL)),
          "Recomputed readout arrays differ")


def packet_predictions(path, item, arrays, win, group, forecasts, train, classes):
    k = f"{win}__{group}"
    binding = item["binding"]
    values = {h: arrays[f"{win}__validation_features__{h}"] for h in HEADS}
    features = {h: arrays[f"{k}__fit__features__{h}"] for h in HEADS}
    scores = {h: arrays[f"{k}__fit__probabilities__{h}"] for h in HEADS}
    labels = arrays[f"{k}__fit__labels"]
    assert_array(labels, np.asarray(train))
    restored = SupportReadout.from_state(torch.load(pin(path, item["sha256"])["path"], map_location="cpu",
                                                      weights_only=True), binding)
    fit = SupportReadout.fit(binding, features, scores, labels, classes)
    for h in HEADS:
        assert_array(fit.centers[h].numpy(), restored.centers[h].numpy())
    assert_array(fit.counts.numpy(), restored.counts.numpy())
    assert_array(fit.weights.numpy(), restored.weights.numpy())
    return (restored.predict(forecasts, values, binding),
            centroid_reference(features, labels, values, classes))


def paired_archive(focus, geometry, probs, centroid, candidate, writer, case, aliases):
    rows = 0
    for head in HEADS:
        proj, bias, dist, between = (geometry[f"nominal__{name}__{head}"]
                                     for name in ("projection", "bias", "distance", "between"))
        for idx, y in enumerate(focus["truth"]):
            foils = focus["foils"].get(y, ())
            for rival in foils:
                writer.writerow(dict(case=case, head=head, alias=aliases[idx], class_id=y, foil=rival,
                                     native_prediction=int(probs[head][idx].argmax())+1,
                                     centroid_prediction=int(centroid[head][idx].argmax())+1,
                                     candidate_prediction=int(candidate[head][idx].argmax())+1,
                                     bias_gap=bias[y-1]-bias[rival-1],
                                     projection_gap=proj[idx,y-1]-proj[idx,rival-1],
                                     distance_advantage=dist[idx,rival-1]-dist[idx,y-1],
                                     between_distance=between[y-1,rival-1]))
                rows += 1
    return rows


def details(case, folder, result, arrays, parent, reg, previous, root, prediction, diagnostics, comparisons):
    before, old = parent
    scene = result["scene"]
    honghu = scene == "whu-honghu"
    val = before["populations"]["val"] if honghu else before
    train = before["populations"]["train"] if honghu else before["train"]
    truth, aliases = (val["labels"] if honghu else val["truth"]), val["aliases"]
    actual = result["population"]
    check((actual["aliases"], actual["labels"]) == (aliases, truth)
          and len(set(aliases)) == len(truth) == (110 if honghu else 45)
          and all(re.fullmatch(r"[0-9a-f]{32}", key) for key in aliases),
          "Point identities or class denominators differ")
    focus = reg["focus"][scene]
    classes = 22 if honghu else 9
    geometry_rows = 0
    for win in result["windows"]:
        native_key = "nominal__selected__val" if honghu else ("full" if win == "full" else f"{win}__clean")
        forecast = {h: arrays[f"{win}__native__{h}"] for h in HEADS}
        for h in HEADS:
            assert_array(forecast[h], old[f"{native_key}__{h}"])
        controls = {arm: {"fused": arrays[f"{win}__{arm}__fused"]} for arm in CONTROL} if not honghu else {}
        if scene == "xuzhou":
            _, stored = archive(root,previous,case)
            for arm in CONTROL:
                assert_array(controls[arm]["fused"], softmax(stored[f"{win}__logit__{arm}"]))
        for group in result["groups"]:
            key = f"{win}__{group}"
            item = result["witnesses"][key]
            if case.startswith("honghu60__"):
                _, saved = archive(root, previous, case)
                distances = {h: saved[f"{h}__distance"] for h in HEADS}
                counts = saved["fused__counts"]
                scales = {h: support_scale(tensor(saved[f"{h}__between"]), tensor(counts)) for h in HEADS}
                weights = support_weights({h: old[f"nominal__selected__train__{h}"] for h in HEADS}, train["labels"], 22)
                derived = output_scores(forecast, distances, counts, scales, weights)
                control = {f"{h}__distance": distances[h] for h in HEADS}
                for h in HEADS:
                    for name in ("distance", "between", "projection", "bias"):
                        assert_array(arrays[f"nominal__{name}__{h}"], saved[f"{h}__{name}"])
            else:
                selection = (np.arange(len(train["labels"])) if honghu else stratified(train["aliases"], train["labels"])[group])
                labels = np.asarray(train["labels"])[selection]
                derived, control = packet_predictions(folder / case / f"{key}.pt", item, arrays, win, group,
                                                      forecast, labels, classes)
            estimates = {}
            for name in ("candidate", "centroid"):
                estimates[name] = {}
                for h in HEADS:
                    array = arrays[f"{key}__{name}__{h}"]
                    expect = derived[h].numpy() if name == "candidate" else softmax(-control[f"{h}__distance"])
                    assert_array(array, expect)
                    estimates[name][h] = array
            if case.startswith("honghu60__"):
                geometry_rows += paired_archive(
                    {"truth": truth, "foils": {15: (18, 10, 12, 1), 21: (18, 10, 16, 12)}},
                    arrays, forecast, estimates["centroid"], estimates["candidate"], diagnostics, case, aliases)
            estimates.update(controls if group == "all" else {})
            estimates["native"] = forecast
            for mode, heads in estimates.items():
                for head, values in heads.items():
                    source, dest = hard(forecast[head], len(truth), classes), hard(values, len(truth), classes)
                    info = paired(dest, source, truth, classes)
                    bad = guard(info, focus, reg["guards"], honghu)
                    worst = min(row["target_recall"] - row["source_recall"] for row in info["classes"])
                    comparisons.append(dict(case=case, scene=scene, win=win, group=group, method=mode,
                                            head=head, failures=bad, worst_delta_recall=worst,
                                            worst_classes=[r["class_id"] for r in info["classes"]
                                                           if abs(r["target_recall"]-r["source_recall"]-worst)<1e-12],
                                            **info))
                    for alias, cls, row, p0, p1 in zip(aliases, truth, values, source, dest, strict=True):
                        prediction.writerow(dict(case=case, scene=scene, window=win, support=group, method=mode,
                                                 head=head, point=alias, truth=cls, original=p0, predicted=p1,
                                                 probabilities=json.dumps(row.tolist(), separators=(",", ":"))))
    return geometry_rows


def summarize(rows):
    result = {}
    keys = sorted({(row["scene"], row["case"].split("__")[0], row["method"], row["head"], row["group"])
                   for row in rows})
    for scene, lane, method, head, group in keys:
        current = [x for x in rows if (x["scene"], x["case"].split("__")[0], x["method"], x["head"], x["group"])
                   == (scene, lane, method, head, group)]
        values = {k: fmean(x[k] for x in current) for k in ("source_oa", "target_oa", "delta_oa",
                                                          "source_aa", "target_aa", "delta_aa")}
        result["/".join((scene,lane,group,method,head))] = dict(
            contrasts=len(current), misses=sum(bool(x["failures"]) for x in current),
            added_correct=sum(x["error_to_correct"] for x in current),
            new_errors=sum(x["correct_to_error"] for x in current), **values)
    primary = [row for row in rows if row["method"] == "candidate" and row["group"] == "all"]
    fused = {key: value for key,value in result.items() if key.endswith("/all/candidate/fused")}
    positive = all(min(row["delta_oa"], row["delta_aa"]) >= -1e-12 for row in fused.values())
    positive = positive and any(max(row["delta_oa"], row["delta_aa"]) > 1e-12 for row in fused.values())
    failures = [dict(case=x["case"], window=x["win"], support=x["group"], head=x["head"], guards=x["failures"])
                for x in rows if x["method"] == "candidate" and x["failures"]]
    return dict(decision="eligible_for_later_confirmation" if positive and not failures else "closed_by_development_guard",
                primary_failures=sum(bool(x["failures"]) for x in primary), all_candidate_failures=failures,
                retrospective_groups=result, validation_dependence=dict(xuzhou="one_seed_four_repeated_observations",
                    paviaU="three_model_seeds_one_validation_space", honghu60="k_initialization_confound",
                    honghu24="two_support_seeds_orthogonal_k_and_initialization_one_validation_space"))


def publish(root, folder, code):
    link, (reg, _, _, batch, previous) = registered(root, folder, code)
    tests = _read(pin(folder / "synthetic.json"))
    check(tests["registration"] == link["sha256"] and tests["status"] == "passed"
          and tests["learned_parameters"] == tests["optimizer_updates"] == tests["real_updates"] == 0
          and tests["query_accessed"] is tests["raw_gt_loaded"] is False and len(tests["cases"]) == 6,
          "Synthetic packet or protection controls missing")
    started = 0
    case_links, cost, results = {}, {}, []
    with TemporaryDirectory(prefix=".wp12-", dir=folder) as temp:
        stage = Path(temp) / "export"
        stage.mkdir()
        with ((stage / "raw-predictions.csv").open("w", encoding="utf-8", newline="") as p,
              (stage / "support-geometry.csv").open("w", encoding="utf-8", newline="") as g):
            prediction = csv.DictWriter(p, ("case","scene","window","support","method","head","point","truth",
                                            "original","predicted","probabilities"))
            diagnostics = csv.DictWriter(g, ("case","head","alias","class_id","foil","native_prediction",
                                              "centroid_prediction","candidate_prediction","bias_gap",
                                              "projection_gap","distance_advantage","between_distance"))
            prediction.writeheader()
            diagnostics.writeheader()
            for case in reg["cases"]:
                saved = pin(folder / case / "receipt.json")
                entry = _read(saved)
                case_links[case] = saved["sha256"]
                check(entry["registration"] == link["sha256"] and case == entry["case"]
                      and entry["real_updates"] == 0 and entry["query_accessed"] is entry["raw_gt_loaded"] is False,
                      "Unexpected case identity or access receipt")
                meta, arrays = read_bundle(pin(folder / case / "arrays.npz", entry["arrays"]))
                check(meta == dict(case=case, scope=link["sha256"]), "Mixed experiment arrays")
                result = entry["result"]
                prev = parent_case(root, batch, case)
                started += details(case, folder, result, arrays, prev, reg, previous, root,
                                   prediction, diagnostics, results)
                cost[case] = dict(forwards=result["forwards"], seconds=result["seconds"],
                                  peak_rss_kib=entry["rss_peak_kib"], witnesses=result["witnesses"])
        _write(stage / "all-class-pairs.json", results)
        _write(stage / "summary.json", summarize(results))
        _write(stage / "cost.json", cost)
        _write(stage / "synthetic.json", tests)
        inventory = {file.name: pin(file)["sha256"] for file in stage.iterdir()}
        _write(stage / "receipt.json", dict(status="accepted_development_only", consumer=code["commit"],
            registration=link["sha256"], case_count=len(case_links), pair_count=len(results),
            total_block_forwards=sum(value["forwards"] for value in cost.values()), geometry_rows=started,
            case_receipts=case_links, artifacts=inventory, raw_gt_loaded=False, query_accessed=False, real_updates=0))
        check(not (folder / "wp12").exists(), "WP12 already published")
        os.rename(stage, folder / "wp12")
    return pin(folder / "wp12/receipt.json")
