"""Independent review of the released two-scene attribution receipts."""

import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np

from evaluation import wp12_attribution_exposure as gradients
from evaluation.frozen_band_region import paired as transitions
from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, require
from evaluation.wp12_linear_attribution import FOCUS, RIVALS
from evaluation.wp12_paired_history import SCHEMA as HISTORICAL
from evaluation.wp12_real_attribution import SCHEMA, XZ_CURSORS
from model.research_config import digest
from research_runtime import ROOT

OLD = "wp05-wp06-external-paired-20261119-qa"
PREVIOUS = {
    "historical_index": (OLD+"/historical5/index.json",
        "c4ed2f38ae9a0c88f4dd1a6672935ea42f38783b955de7eb8ed1a32b34af1b3d",
        "fe2ec05185d12811ab3c83f0a8fef57ddc309c1f"),
    "final_delivery_index": (OLD+"/delivery/wp12-external-paired-index.json",
        "642c38f8262dc7bce187f717f5aab9ee218af5034b9250a95caae2f711013979",
        "732c5da9c27b6bcebdb274bb9b9f8e68c27311dd"),
}
SOURCES = ROOT/"docs/references/wp05-two-scene-attribution-sources-20261119.json"


def historical(root):
    out = {}
    history = {}
    for key,(path,sha,commit) in PREVIOUS.items():
        link = dict(path=str(root/path),sha256=sha)
        data = json_pin(root,link)
        require(data["schema"]==HISTORICAL and data["producer"]==commit
                and data["real_optimizer_updates"]==0 and data["query_accessed"] is data["raw_gt_loaded"] is False,
                "Prior WP12 frozen index identity or scope differs")
        out[key] = dict(pin=link,producer=commit,external_review="external_paired_review" in data)
        history[key] = {k:v for k,v in data.items() if k not in ("producer","external_paired_review")}
    require(out["historical_index"]["external_review"] is False and out["final_delivery_index"]["external_review"] is True
            and history["historical_index"]==history["final_delivery_index"],"WP12 historical and final objects were confused")
    return out,history["historical_index"]


def check_sources(xz, hh):
    source = json.loads(SOURCES.read_text(encoding="utf-8"))
    scenes = source["scenes"]
    require(set(scenes)=={"xuzhou","whu-honghu"},"Source inspection omitted a registered scene")
    for scene,count,bands in (("xuzhou",9,436),("whu-honghu",22,270)):
        entry = scenes[scene]
        require(entry["local_numeric_class_ids"]==list(range(1,count+1))
                and entry["local_delivered_columns"]==bands
                and entry["per_band_measured_or_mapping_verified"] is False
                and entry["manufacturer_detector_mapping"] is None,"Foreign scene, axis or class claim")
    require(len(xz["bands"]["full"])==436 and xz["identities"]["classes"]==9
            and all(item["populations"]["val"]["per_class"]==5 for item in hh),
            "Observed diagnostic populations differ from the source review")
    alias_file = ROOT/"scripts/honghu_replacement/honghu_common.py"
    names = {"XUZHOU_CLASS_NAMES","HONGHU_CLASS_NAMES"}
    assignments = {target.id:ast.literal_eval(item.value) for item in ast.parse(alias_file.read_text(encoding="utf-8")).body
                   if isinstance(item,ast.Assign) for target in item.targets
                   if isinstance(target,ast.Name) and target.id in names}
    honghu = assignments["HONGHU_CLASS_NAMES"]
    expected = scenes["whu-honghu"]["publisher_table_spot_checks"]
    require(list(honghu)==list(range(1,23)) and all(honghu[int(key)]==value for key,value in expected.items())
            and list(assignments["XUZHOU_CLASS_NAMES"])==list(range(1,10)), "Local named compatibility inventory differs")
    return dict(reference_sha256=file_pin(SOURCES)["sha256"],inventory=source,
                local_reporting_aliases=assignments,
                local_class_status="verified_numeric_order_within_each_approved_model_and_role",
                cross_release_compatible=False,physical_claim="original_delivered_column_ids_only")


def scan_rows(body, scene):
    classes = len(body["classes"])
    rows = body["rows"]
    keys = [(r["block"],r["ordinal"]) for r in rows]
    require(len(set(keys))==len(rows),"Duplicate spatial point ordinal")
    scores = {}
    for y in range(1,classes+1):
        points = [r for r in rows if r["truth"]==y]
        declared = body["classes"][str(y)]
        require(declared["n"]==len(points) and points,"Readout cannot promote an absent class")
        numeric = {}
        for head in HEADS:
            parts = np.array([r["terms"][head] for r in points],dtype=float)
            rivals = {p["rival"] for p in points}
            require(np.isfinite(parts).all() and parts.shape==(len(points),6)
                    and rivals=={RIVALS[scene].get(y,(2 if y==1 else 1,))[0]}
                    and np.allclose(parts[:,0],parts[:,1]+parts[:,2],rtol=2e-6,atol=2e-5),
                    "Fixed class pair or original linear reconstruction differs")
            m = declared["heads"][head]
            count = sum(r["predicted"][head]==y for r in points)
            mean = parts[:,:3].mean(0)
            errors = abs(parts[:,0]-parts[:,1]-parts[:,2])
            require(count==m["correct"] and np.allclose(mean,[m["mean_fixed_margin"],m["mean_bias"],
                    m["mean_inner_product"]],rtol=0,atol=1e-12)
                    and np.allclose(errors,parts[:,5],rtol=0,atol=1e-12),"Independent class tally differs")
            step_terms = [r["from_initial"][head] for r in points if "from_initial" in r]
            numeric[head] = dict(correct=count,wrong=len(points)-count,mean_margin=mean[0],mean_bias=mean[1],
                mean_dot=mean[2],max_error=float(errors.max()),
                predicted_counts=dict(Counter(r["predicted"][head] for r in points)),
                before_after_bias_removal=[int((parts[:,0]>0).sum()),int((parts[:,2]>0).sum())],
                trajectory=None if not step_terms else np.array(step_terms).mean(0).tolist())
        scores[str(y)] = dict(n=len(points),heads=numeric,space=declared["blocks"])
    return dict(n=len(rows),classes=scores,ordered_point_identity=digest([(r["block"],r["ordinal"],r["truth"]) for r in rows]))


def across_views(a,b,classes):
    require([(r["block"],r["ordinal"],r["truth"],r["rival"]) for r in a]==
            [(r["block"],r["ordinal"],r["truth"],r["rival"]) for r in b],
            "Different released point order, class or fixed competing row")
    truth = [r["truth"] for r in a]
    output = {}
    for head in HEADS:
        before = [r["predicted"][head] for r in a]
        after = [r["predicted"][head] for r in b]
        paired = transitions(after,before,truth,classes)
        terms = {}
        for y in linear_classes(classes):
            left,right = (np.array([r["terms"][head] for r in role if r["truth"]==y]) for role in (a,b))
            require(np.array_equal(left[:,1],right[:,1]),"Observation-only comparison changed the frozen bias")
            secondary = None
            if classes==9:
                aa,bb = (np.array([r["secondary"]["terms"][head] for r in role if r["truth"]==y]) for role in (a,b))
                require(np.array_equal(aa[:,1],bb[:,1]),"Alternative fixed opponent was changed")
                secondary = float((bb[:,0]-aa[:,0]).mean())
            primary = paired["classes"][y-1]
            terms[str(y)] = dict(n=len(left),before=primary["source_correct"],after=primary["target_correct"],
                gained=primary["error_to_correct"],lost=primary["correct_to_error"],losers=primary["lost_to"],
                margin_delta=float((right[:,0]-left[:,0]).mean()),dot_delta=float((right[:,2]-left[:,2]).mean()),
                bias_delta=0.,secondary_fixed_margin_delta=secondary)
        output[head] = dict(total=len(truth),oa_delta=paired["delta_oa"],aa_delta=paired["delta_aa"],focus=terms,
                            confusion_before=paired["source_confusion"],confusion_after=paired["target_confusion"])
    return output


def linear_classes(classes):
    return (5,6) if classes==9 else (15,21)


def read_case(root,directory,key,producer,group):
    case = file_pin(directory/(key+".json"))
    row = json_pin(root,case)
    require(row["schema"]==SCHEMA and row["producer"]==producer and row["evidence"]["group"]==group
            and row["evidence"]["case"]==key and row["raw_gt_loaded"] is row["query_accessed"] is False,
            "Foreign attribution shard")
    return case,row["evidence"]


def xz_summary(xz, previous):
    output = {}
    for name,cursor in XZ_CURSORS.items():
        detail = xz["fit"][name]
        require(detail["cursor"]==cursor==previous[name]["step"]
                and detail["checkpoint"]["sha256"]==previous[name]["checkpoint"],"Wrong physical Xuzhou checkpoint")
        roles = {role:{band:scan_rows(body,"xuzhou") for band,body in entries.items()}
                 for role,entries in detail["roles"].items()}
        values = detail["roles"]["val"]
        pairs = {key:across_views(values[a]["rows"],values[b]["rows"],9) for key,a,b in (
            ("central_gap_vs_full","full","central_gap"),("dispersed_vs_full","full","matched_dispersed"),
            ("central_gap_vs_dispersed","matched_dispersed","central_gap"))}
        for band in ("central_gap","matched_dispersed"):
            key = "central_gap_vs_full" if band=="central_gap" else "dispersed_vs_full"
            for head in HEADS:
                for cls,value in pairs[key][head]["focus"].items():
                    old = previous[name]["views"][band][head]["focus"][cls]
                    require(tuple(value[f] for f in ("n","before","after","gained","lost"))==
                            tuple(old[f] for f in ("n","before","after","gains","losses")),"Archived Xuzhou point counts differ")
        output[name] = dict(cursor=cursor,roles=roles,comparisons=pairs,weights=detail["parameter_delta"])
    columns = xz["bands"]
    require(len(columns["central_gap"])==len(columns["matched_dispersed"]) and
            set(columns["central_gap"])!=set(columns["matched_dispersed"]),"Placement controls need actual matched retention counts")
    return dict(endpoints=output,role_identity=xz["identities"],exposure=xz["exposure"],retained_original_columns=columns,
                support_spectra=xz["normalized_before_after"],layer_changes=xz["layer_delta"],
                graph_policy="same_frozen_model_and_native_graph_builder_rebuild_per_view",
                views_added=0,missing_comparison="no_authority_for_additional_shifted_bands")


def honghu_summary(records):
    groups = defaultdict(dict)
    for r in records:
        subgroup = f's{r["support_seed"]}-k{r["k"]}-{r["method"]}-m{r["model_seed"]}'
        if r.get("history")=="confusion_only":
            groups[subgroup][r["case"]] = dict(margins_only=True,audit=r["fit"])
            continue
        stage = {key:dict(cursor=fit["cursor"],roles={role:scan_rows(body,"whu-honghu")
                  for role,body in fit["roles"].items()},params=fit["parameter_delta"])
                 if "roles" in fit else fit for key,fit in r["fit"].items()}
        initial,selected = stage["initial"],stage["selected"]
        require(all(initial["roles"][key]["ordered_point_identity"]==selected["roles"][key]["ordered_point_identity"]
                    for key in ("train","val")),"Historical support or development labels changed across endpoints")
        different = r["fit"]["selected"]["selected_full_development"]
        if different:
            stage["selected"]["full_development"] = scan_rows(different,"whu-honghu")
        contrast = {}
        for cls in ("15","21"):
            train = selected["roles"]["train"]["classes"][cls]
            val = selected["roles"]["val"]["classes"][cls]
            contrast[cls] = {head:dict(train_n=train["n"],dev_n=val["n"],
                train_correct=train["heads"][head]["correct"],dev_correct=val["heads"][head]["correct"],
                interpretation=("all_released_support_also_wrong" if train["heads"][head]["correct"]==0 else
                   "support_all_correct_but_development_incomplete" if train["heads"][head]["correct"]==train["n"]
                   and val["heads"][head]["correct"]<val["n"] else "mixed_or_no_development_loss"))
                for head in HEADS}
        groups[subgroup][r["case"]] = dict(stage=stage,populations=r["populations"],exposure=r["exposure"],
             same_protocol_support_and_development=contrast,holdout_independent=False,
             archived_path_updates=r.get("archived_path_updates",stage["last"]["cursor"]),
             reviewed_combined_path_updates=r.get("reviewed_combined_path_updates"),
             tensor_gradient_record=r["tensor_gradient_record"],historic_event_margins=r["historic_event_margins"])
    require(all(len(values)==5 for values in groups.values()),"Historical subgroup omitted a preregistered coverage arm")
    return dict(groups=dict(groups),spatial_stratum="tile-relative periphery and native-head disagreement")


def review(root, out, code, directories, producers, floor_pin):
    from scripts.run_frozen_band_region import _write
    from scripts.run_wp12_real_attribution import roster

    archive,previous = historical(root)
    keys = roster(root)
    cases, receipts = {}, {}
    for group, labels in keys.items():
        subprocess.run(["git","merge-base","--is-ancestor",producers[group],code["commit"]],check=True)
        pairs = [read_case(root,directories[group],case,producers[group],group) for case in labels]
        cases[group] = [row for _,row in pairs]
        receipts[group] = {row["case"]:link for link,row in pairs}
    xz, = cases["xuzhou"]
    require(len(cases["honghu_short"])==40 and len(cases["honghu_120"])==10,"Incomplete prior HongHu roster")
    floor = json_pin(root,floor_pin)
    require(floor["schema"]=="wp12_formal_class_floor_regression_v1" and len(floor["cases"])==10
            and floor["new_real_labels"] is floor["actual_real_accuracy_qualified"] is False
            and floor["optimizer_updates"]==0,"A threshold engineering fixture was promoted")
    out.parent.mkdir(parents=True,exist_ok=True)
    return _write(out,dict(schema="wp12_real_error_attribution_review_v1",producer=code["commit"],
        attribution_producers=producers,receipts=receipts,historical_objects=archive,
        xuzhou=xz_summary(xz,previous["frozen_actual_retained_views"]),
        honghu_short=honghu_summary(cases["honghu_short"]),honghu_120=honghu_summary(cases["honghu_120"]),
        recorded_gradient_exposure=dict(xuzhou=gradients.xuzhou(root),
             honghu_short=gradients.honghu(root,cases["honghu_short"],True),
             honghu_120=gradients.honghu(root,cases["honghu_120"],False)),
        provenance=check_sources(xz,cases["honghu_short"]+cases["honghu_120"]),formal_class_floor=floor_pin,
        independent_real_accuracy=None,raw_gt_loaded=False,query_accessed=False,optimizer_updates=0,wp04=False))
