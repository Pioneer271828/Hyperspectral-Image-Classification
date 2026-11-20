"""Independently compare frozen full-support, view-refitted and archived native decisions."""

from collections import Counter, defaultdict
import csv

from evaluation.frozen_training_stages import HEADS, VIEWS, file_pin, json_pin, pinned, require
from evaluation.retained_spectral_evidence import LAYERS, SHARED_LAYERS
from scripts.run_wp12_retained_evidence import CASES, LOCKED

SCHEMA = "wp12_locked_support_comparison_v1"
CSV = ("wp05-wp06-real-evidence-20261119-qa/acceptance/same-points.csv",
       "980089e47c89bfc4444310cfbcd27c45e06ac97428f519c5ddd7a7bca722a758")
READOUTS = tuple(f"native_{i}" for i in HEADS) + tuple(f"{i}_{j}" for i in LAYERS for j in ("centroid","ridge"))


def historical_points(root):
    archive = pinned(root,dict(path=str(root/CSV[0]),sha256=CSV[1]))
    groups = defaultdict(list)
    with archive.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            require(row["case"] in CASES and row["view"] in VIEWS and row["readout"] in READOUTS,
                    "Unknown historical true class or endpoint")
            groups[row["case"],row["view"],row["readout"]].append(
                tuple(int(row[key]) for key in ("point_ordinal","truth","predicted")))
    require(len(groups)==len(CASES)*len(VIEWS)*len(READOUTS)
            and sum(map(len,groups.values()))==17085, "Historical same-point inventory differs")
    for roster in groups.values():
        require([row[0] for row in roster] == list(range(1,len(roster)+1)), "Historical point ordinals differ")
    return groups


def transitions(truth, before, after, classes):
    require(len(truth) == len(before) == len(after) and all(1 <= c <= classes for c in (*truth,*before,*after)),
            "Expected aligned all-class decisions")
    counts = Counter(zip(truth,before,after,strict=True))
    result = {}
    for label in range(1,classes+1):
        moves = [[a,b,n] for (y,a,b),n in sorted(counts.items()) if y==label]
        n = sum(x[2] for x in moves)
        before_correct = sum(k for a,b,k in moves if a==label)
        after_correct = sum(k for a,b,k in moves if b==label)
        result[str(label)] = dict(n=n,before=before_correct,after=after_correct,
                                  gains=sum(k for a,b,k in moves if a!=label==b),
                                  losses=sum(k for a,b,k in moves if a==label!=b),moves=moves)
    return result


def validate_fit_modes(views):
    expected = None
    for view in VIEWS:
        item = views[view]
        locked, refit = (item[key] for key in ("fixed_full_support","readout"))
        require(set(locked["layers"]) == set(refit["layers"]) == set(SHARED_LAYERS)
                and locked["fit_class_ids"] == refit["fit_class_ids"], "Readout fit roster or layer scope differs")
        if view == "full":
            require(locked == refit, "The full-support control and full-view fit should coincide")
            expected = locked["fit_sha256"]
        require(locked["fit_sha256"] == expected, "A view-refitted model was mislabeled as a locked full-band head")


def assemble(root, case_dir, producer):
    old = historical_points(root)
    cases = {}
    for name in CASES:
        source = file_pin(case_dir/f"{name}.json")
        receipt = json_pin(root,source)
        require(receipt["schema"] == LOCKED and receipt["producer"] == producer and receipt["result"]["case"] == name
                and receipt["optimizer_updates"] == 0 and receipt["query_accessed"] is receipt["raw_gt_loaded"] is False,
                "Independent precision claims require an actual external evaluation receipt")
        case = receipt["result"]
        views = case["views"]
        validate_fit_modes(views)
        full = views["full"]
        classes = case["classes"]
        by_view = {}
        for band in VIEWS:
            data = views[band]
            truth = data["truth"]
            require(truth == full["truth"], "Historical published labels changed across views")
            legacy = {}
            historical = {}
            for method in READOUTS:
                history = old[name,band,method]
                base = old[name,"full",method]
                require(truth == [y for _,y,_ in history] == [y for _,y,_ in base], "Archived denominator differs")
                legacy[method] = [p for _,_,p in history]
                historical[method] = transitions(truth,[p for _,_,p in base],legacy[method],classes)
            for head in HEADS:
                require([row["predicted"] for row in data["native"][head]] == legacy[f"native_{head}"],
                        "Archived original three-head predictions differ")
            by_layer = {}
            for layer in SHARED_LAYERS:
                by_layer[layer] = {}
                for method in ("centroid","ridge"):
                    refit = [row["predicted"] for row in data["readout"]["layers"][layer][method]]
                    fixed = [row["predicted"] for row in data["fixed_full_support"]["layers"][layer][method]]
                    original = [row["predicted"] for row in full["fixed_full_support"]["layers"][layer][method]]
                    require(refit == legacy[f"{layer}_{method}"], "Archived view-refitted predictions differ")
                    by_layer[layer][method] = dict(
                        representation_same_fit=transitions(truth,original,fixed,classes),
                        refit_increment=transitions(truth,fixed,refit,classes),
                        frozen_native_vs_fixed_readout=transitions(truth,legacy["native_fused"],fixed,classes))
            by_view[band] = dict(historical_view_transitions=historical,shared=by_layer,
                support_counts=data["roles"]["devtrain"]["label_counts"],
                devval_counts=data["roles"]["devval"]["label_counts"],
                fixed_fit_sha256=data["fixed_full_support"]["fit_sha256"],view_fit_sha256=data["readout"]["fit_sha256"])
        cases[name] = dict(scene=case["scene"],step=case["stage"],focus=case["focus"],source_sha256=source["sha256"],
                           layers=list(SHARED_LAYERS),views=by_view)
    return dict(schema=SCHEMA, producer=producer, scope="released_development_diagnostics_only",
                independent_real_accuracy=None, historical_sha256=CSV[1],historical_rows=17085,
                raw_sensor_alignment=None, raw_sensor_omission="retained dimensions differ; no certified physical mapping",
                fit_role="devtrain", evaluation_role="devval", cases=cases,
                real_updates=0, raw_gt_loaded=False,query_accessed=False)
