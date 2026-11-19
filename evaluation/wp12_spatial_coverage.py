"""Verify the guarded-space coverage and support-only prediction artifacts."""

from collections import Counter
import csv
from itertools import product
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.frozen_band_region import HEADS, hard, paired
from scripts.run_frozen_band_region import _read, _write, check, pin
from scripts.run_frozen_mechanisms import read_bundle
from scripts.run_spatial_coverage_release import BASES, SPATIAL, registered
from transfer.frozen_branch_readout import centroid_distances
from transfer.native_scale_training import _exact
from transfer.p4_coverage import ARMS
from transfer.spatial_coverage_protocol import VIEWS
from transfer.spatial_coverage_training import SCHEMA, SOURCES, SPLITS, load

PAIRS = [(b,"clean","full","clean") for b in BASES if b != "full"] + [
    (b,to,b,source) for b in BASES for to,source in (("target","clean"),("matched","clean"),("target","matched"))]


def recall_flags(report, safeguards, focus, rare):
    unavailable, failures = [], []
    for row in report["classes"]:
        key = row["class_id"]
        if key not in focus:
            continue
        if not row["denominator"]:
            unavailable.append(key)
        else:
            if row["target_recall"]-row["source_recall"] < safeguards["focus_min_delta_recall"]:
                failures.append(f"recall_{key}")
            if rare and row["correct_to_error"]:
                failures.append(f"new_error_{key}")
    for name in ("oa","aa"):
        if report[f"delta_{name}"] is not None and report[f"delta_{name}"] < safeguards["aggregate_min_delta"]:
            failures.append(name)
    return dict(unavailable=unavailable, failures=failures, full_class_guard=not failures and not unavailable
                and all(item["denominator"] for item in report["classes"]))


def centroid_checks(arrays, truth, held, fit, aliases):
    fit_count = np.bincount(truth[fit],minlength=23)[1:]
    seen = np.flatnonzero(fit_count) + 1
    records = []
    for band,head in product(BASES,HEADS):
        values = arrays[f"{band}__embedding__{head}"]
        native = arrays[f"{band}__clean__{head}"][held]
        centers = np.stack([values[fit & (truth == lab)].mean(0) for lab in seen]) if len(seen) else None
        if len(seen) >= 2:
            d = centroid_distances(centers,values[held])
            proposed = seen[np.argsort(d, axis=1,kind="stable")]
        else:
            proposed, d = None, None
        for label in (15,21):
            scores, points, excluded = Counter(), [], 0
            for i,j in enumerate(np.flatnonzero(held)):
                if truth[j] != label:
                    continue
                base = native[i]
                base_order = np.argsort(-base,kind="stable") + 1
                base_rank = int(np.flatnonzero(base_order == label)[0])+1
                has = proposed is not None and label in seen
                rank = int(np.flatnonzero(proposed[i] == label)[0])+1 if has else None
                target = int(proposed[i,0]) if has else None
                excluded += not has
                scores["native_correct"] += int(base_order[0] == label)
                scores["centroid_correct_with_fit"] += int(has and target == label)
                points.append(dict(alias=aliases[j], native_rank=base_rank, centroid_rank=rank,
                                   centroid_prediction=target, classes_in_reference=len(seen),
                                   own_distance=float(d[i, list(seen).index(label)]) if has else None))
            records.append(dict(band=band,head=head,class_id=label,held=len(points),fit=int(fit_count[label-1]),
                                eligible_excluding_missing_fit=len(points)-excluded,
                                unavailable=excluded,points=points,**{key:scores[key] for key in
                                ("native_correct","centroid_correct_with_fit")},fit_counts=fit_count.tolist()))
    return records


def run_comparisons(value, arrays, space, reg, writer, name):
    n = len(value["aliases"])
    classes = space["classes"]
    truth = np.asarray(value["labels"])
    check(len(set(value["aliases"])) == len(truth) == len(value["tiles"]) == n
          and all(re.fullmatch("[0-9a-f]{32}", a) for a in value["aliases"])
          and value["role"] == "historical_devtrain_support" and value["independent_accuracy"] is False,
          "Expected fixed de-identified support roles")
    scene = value["scene"]
    check(value["band_ids"] == {name:space["bands"][name] for name in BASES}, "Original band IDs changed")
    scope_tiles = set().union(*(set(g["held"]) for g in space["folds"]))
    check(set(value["tiles"]) <= scope_tiles, "Prediction role escaped released training geometry")
    released = {f"{b}__{mode}__{h}" for b,mode,h in product(BASES,SPATIAL,HEADS)}
    embeds = {f"{b}__embedding__{h}" for b,h in product(BASES,HEADS)} if scene == "whu-honghu" else set()
    check(set(arrays) == released | embeds, "Unexpected readout or trace inventory")
    for key in released:
        band,mode,h = key.split("__")
        scores = arrays[key]
        for uid,y,p,label in zip(value["aliases"],truth,scores,hard(scores,n,classes),strict=True):
            writer.writerow(dict(case=name, band=band, spatial=mode,head=h,point=uid,truth=int(y), predicted=label,
                                 probabilities=json.dumps(p.tolist(),separators=(",",":"))))
    folds, effects = [], []
    for fold_number,fold in enumerate(space["folds"]):
        held = np.isin(value["tiles"],fold["held"])
        fit = np.isin(value["tiles"],fold["fit"])
        check(not np.any(held & fit), "Support folds overlap")
        row = dict(fold=fold_number, held=int(held.sum()), fit=int(fit.sum()),
                   held_counts=np.bincount(truth[held],minlength=classes+1)[1:].tolist(),
                   fit_counts=np.bincount(truth[fit],minlength=classes+1)[1:].tolist())
        if scene == "whu-honghu":
            row["centroid"] = centroid_checks(arrays,truth,held,fit,value["aliases"])
        folds.append(row)
        for band,kind,reference,reference_kind in PAIRS:
            for head in HEADS:
                old = arrays[f"{reference}__{reference_kind}__{head}"][held]
                new = arrays[f"{band}__{kind}__{head}"][held]
                report = paired(hard(new,len(new),classes),hard(old,len(old),classes),truth[held],classes)
                foils = {5:8,6:3} if scene == "xuzhou" else {}
                changed = {str(lab):{"from":entry["source_predictions"].get(str(rival),0),
                                     "to":entry["target_predictions"].get(str(rival),0)}
                           for lab,rival in foils.items() for entry in (report["classes"][lab-1],)}
                effects.append(dict(case=name,scene=scene,fold=fold_number,band=band,view=kind,
                                    reference_band=reference,reference_view=reference_kind,head=head,
                                    guard=recall_flags(report,reg["guards"],reg["focus"][scene],scene=="whu-honghu"),
                                    predeclared_confusions=changed,**report))
    check(sum(x["held"] for x in folds) == n, "Split does not cover the released cohort exactly once")
    return folds,effects


def synthetic(root, code):
    report = _read(pin(root/"synthetic/receipt.json"))
    expected = {f"{c}-{s}-{split}-{arm}" for c,s,split,arm in product((9,22),SOURCES,SPLITS,ARMS)}
    check(set(report["cases"]) == expected and report["producer"] == code["commit"]
          and report["schema"] == SCHEMA and report["status"] == "passed"
          and report["synthetic_updates"] == 12*len(expected) and report["real_updates"] == 0
          and report["raw_gt_loaded"] is report["query_accessed"] is False, "Synthetic family is incomplete")
    cases = {}
    for name in sorted(expected):
        folder = root/"synthetic"/name
        value = _read(pin(folder/"receipt.json",report["cases"][name]["sha256"]))
        base = _read(pin(root/"synthetic"/(name.rsplit("-",1)[0]+"-none")/"receipt.json"))
        check(value["joint_inputs"] == base["joint_inputs"] and value["source"] == base["source"]
              and value["parameters"]-base["parameters"] == (0 if value["arm"] == "none" else 194)
              and value["reentry"] == 0 and value["optimizer"] == 12 and set(value["validation"])==set(VIEWS),
              "Source, data, arm capacity or continuation budget differs")
        checkpoints = {}
        for phase,sha in value["packet_sha"].items():
            path = folder/f"{phase}.pt"
            check(pin(path)["sha256"] == sha, "Synthetic continuation file changed")
            checkpoints[phase] = load(path)
        check(_exact(checkpoints["full"],checkpoints["resume"]) and
              _exact(checkpoints["resume"],checkpoints["completed"]), "Independent recovery digest differs")
        cases[name] = value
    check(all(all(checks.values()) for checks in report["checks"].values()), "Synthetic negative checks failed")
    return dict(case_count=len(cases),synthetic_updates=report["synthetic_updates"],checks=report["checks"],cases=cases)


def publish(root, folder, code):
    link,(reg,_,_) = registered(root,folder,code)
    engineering = synthetic(folder,code)
    manifests, folds, effects, cost = {}, {}, [], {}
    with TemporaryDirectory(dir=folder,prefix=".wp12-") as temp:
        stage = Path(temp)/"export"
        stage.mkdir()
        with (stage/"support-predictions.csv").open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,("case","band","spatial","head","point","truth","predicted","probabilities"))
            writer.writeheader()
            for label in reg["cases"]:
                artifact=pin(folder/f"{label}.npz")
                record, arrays = read_bundle(artifact)
                check(record["registration"] == link["sha256"] and record["case"] == label
                      and record["query_accessed"] is record["raw_gt_loaded"] is False and record["real_updates"]==0,
                      "Frozen scope or access changed")
                result=record["result"]
                scene = result["scene"]
                folds[label], rows = run_comparisons(result,arrays,reg["spaces"][scene],reg,writer,label)
                manifests[label]=artifact["sha256"]
                effects.extend(rows)
                cost[label] = {key:result[key] for key in ("seconds","rss_peak_kib","forwards")}
        public_spaces = {scene: {**value, "folds": [
            {**{k:v for k,v in f.items() if k not in ("fit","held")},
             "fit_tiles":len(f["fit"]), "held_tiles":len(f["held"])} for f in value["folds"]]}
                         for scene,value in reg["spaces"].items()}
        _write(stage/"space-eligibility.json",dict(spaces=public_spaces,honghu=reg["honghu_feasibility"]))
        _write(stage/"support-spatial-pairs.json",effects)
        _write(stage/"folds.json",folds)
        _write(stage/"cost.json",cost)
        _write(stage/"wp05-budget-draft.json",reg["next_budget"])
        _write(stage/"synthetic.json",engineering)
        contents={file.name:pin(file)["sha256"] for file in stage.iterdir()}
        _write(stage/"receipt.json",dict(status="accepted_support_and_synthetic_only",producer=code["commit"],
                registration=link["sha256"],cases=manifests,pair_rows=len(effects),
                total_forwards=sum(x["forwards"] for x in cost.values()),synthetic_cases=engineering["case_count"],
                synthetic_updates=engineering["synthetic_updates"], independent_cases=0, real_updates=0,
                query_accessed=False,raw_gt_loaded=False,artifacts=contents))
        check(not (folder/"wp12").exists(), "Already published WP12")
        os.rename(stage,folder/"wp12")
    return pin(folder/"wp12/receipt.json")
