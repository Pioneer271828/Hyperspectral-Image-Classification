"""Paired support effects on fixed points and descriptive effects across layouts."""

import numpy as np

from evaluation.frozen_training_stages import HEADS, require
from evaluation.wp12_cross_backend import FORMAL_FLOOR, raw_pair


def same_points(first, second, classes):
    require([(r["view"],r["tile"]) for r in first] == [(r["view"],r["tile"]) for r in second],
            "Fixed-point contrast has different ordered blocks or views")
    views = {}
    for left, right in zip(first,second,strict=True):
        a,b = left["accuracy"],right["accuracy"]
        require((a is None) == (b is None), "Fixed-point contrast lost a released accuracy block")
        if a is None:
            continue
        require(all(a[k] == b[k] for k in ("truth","order","bands","spatial")),
                "Support comparison may only pair the same actual points")
        view = views.setdefault(left["view"], dict(truth=[], before={h:[] for h in HEADS},
                               after={h:[] for h in HEADS}, blocks={}))
        truth = a["truth"]
        view["truth"].extend(truth)
        block = {}
        for head in HEADS:
            before,after = ((np.asarray(p["heads"][head]).argmax(axis=1)+1).tolist() for p in (a,b))
            view["before"][head].extend(before)
            view["after"][head].extend(after)
            block[head] = raw_pair(truth,before,after,classes,floor=FORMAL_FLOOR)
        view["blocks"][str(left["tile"])] = dict(points=len(truth),heads=block)
    return {name: dict(points=len(v["truth"]), blocks=v["blocks"], heads={
        h:raw_pair(v["truth"],v["before"][h],v["after"][h],classes,floor=FORMAL_FLOOR) for h in HEADS})
        for name,v in views.items()}


def method_effects(first,second):
    require(set(first["pairs"]) == set(second["pairs"]), "Cross-release method pairs were changed")
    effects = {}
    for pair, views in first["pairs"].items():
        compare = second["pairs"][pair]
        require(set(views) == set(compare), "Cross-release views were changed")
        effects[pair] = {}
        for view,a in views.items():
            b = compare[view]
            if a["accuracy"] is None or b["accuracy"] is None:
                effects[pair][view] = None
                continue
            arms = ({h:side["accuracy"][h] for h in HEADS} for side in (a,b))
            left,right = arms
            effects[pair][view] = {
                head:dict(
                    baseline_support=[r["n"] for r in left[head]["transitions"].values()],
                    comparison_support=[r["n"] for r in right[head]["transitions"].values()],
                    baseline_blocks=len(a["blocks"]), comparison_blocks=len(b["blocks"]),
                    baseline_delta_aa=left[head]["delta_aa"], comparison_delta_aa=right[head]["delta_aa"],
                    descriptive_range_aa=sorted([left[head]["delta_aa"],right[head]["delta_aa"]]),
                    difference_of_deltas_aa=right[head]["delta_aa"]-left[head]["delta_aa"],
                    baseline_delta_oa=left[head]["delta_oa"], comparison_delta_oa=right[head]["delta_oa"],
                    descriptive_range_oa=sorted([left[head]["delta_oa"],right[head]["delta_oa"]]),
                    difference_of_deltas_oa=right[head]["delta_oa"]-left[head]["delta_oa"])
                for head in HEADS}
    return effects


def summarize(plan, raw, reviewed, cases):
    output = {}
    for name, proof in plan.get("support_contrasts",{}).items():
        left,right = (proof[k] for k in ("baseline","comparison"))
        a,b = ({**plan["groups"][k],"name":k} for k in (left,right))
        arms = sorted(a["arms"])
        exposure = {arm:cases[left][arm]["class_exposure"] for arm in arms}
        require(all(exposure[arm] == cases[right][arm]["class_exposure"] == exposure[arms[0]]
                    and len(set(exposure[arm])) == 1 for arm in arms),
                "Measured supervised class exposure differs across registered contrast")
        a_raw,b_raw = (raw[g["release"]]["cohorts"][g["name"]]["arms"] for g in (a,b))
        a_score,b_score = (reviewed[g["release"]]["cohorts"][g["name"]] for g in (a,b))
        scene = plan["inputs"][a["release"]]["scene"]
        classes = {"xuzhou":9,"paviaU":9,"whu-honghu":22}[scene]
        same = {arm:same_points(a_raw[arm]["rows"],b_raw[arm]["rows"],classes) for arm in arms
                } if proof["same_evaluation_points"] else None
        output[name] = dict(
            registration=proof, class_exposure=exposure,
            unit="paired_same_released_points" if same is not None else "separate_spatial_layouts",
            same_point=same, within_release_method_effects=method_effects(a_score,b_score),
            interpretation="descriptive_fixed_design_no_independent_seed_or_pixel_interval")
    return output
