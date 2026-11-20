"""Join published supervision to recorded gradient-bearing optimizer steps."""

from collections import Counter

from evaluation.frozen_training_stages import json_pin, require, stage_plan
from evaluation.wp12_release_results import proof
from scripts import run_frozen_band_region as archive
from scripts.run_wp12_retained_evidence import RELEASE, pin
from transfer import p4_frozen_diagnostics as hh, p4_update_evidence as update
from evaluation.wp12_real_attribution import XZ_CURSORS, checkpoint
from model.research_config import digest

HEADS = {"fused":"heads","graph":"hgnn_heads","cnn":"cnn_heads"}


def axes(scene):
    return {f"{head}/{leaf}":f"{prefix}.{scene}.{leaf}"
            for head,prefix in HEADS.items() for leaf in ("weight","bias")}


def join(steps,cursors,classes,channels):
    """Count points on an observed active step. Tensor-level flags cannot isolate a class derivative."""
    counts = {channel:Counter() for channel in channels}
    steps_used = {channel:Counter() for channel in channels}
    events = {"0":{channel:dict(labeled_visits=[0]*classes,steps=[0]*classes) for channel in channels}}
    for i,(labels,active) in enumerate(steps,1):
        require(set(active)<=set(channels),"Unknown recorded gradient channel")
        classes_here = set(labels)
        require(classes_here<=set(range(1,classes+1)),"Unreleased training label")
        for channel in active:
            counts[channel].update(labels)
            steps_used[channel].update(classes_here)
        if i in cursors:
            events[str(i)] = {channel:dict(labeled_visits=[counts[channel][c] for c in range(1,classes+1)],
                steps=[steps_used[channel][c] for c in range(1,classes+1)]) for channel in channels}
    require(set(events)=={"0",*(str(x) for x in cursors)},"Recorded optimization exposure misses a cursor")
    return events


def xuzhou(root):
    plan = stage_plan(root,pin(root,RELEASE))
    data = json_pin(root,plan["development"])
    training = data["record"]["train_support"]
    require(digest(training)==plan["identities"]["support"],"Foreign training role")
    support = dict(zip(training["indices"],training["labels"],strict=True))
    state = checkpoint(root,plan["stages"]["last"]["state"])
    names = axes("xuzhou")
    keys = [f"{name}/{flag}" for name in names for flag in ("nonzero","updated")]
    rows = []
    for event in state["history"]:
        require(event["step"]==len(rows)+1,"Unordered recorded gradient evidence")
        labels = [support[p] for p in event["support_ids"]]
        active = []
        for name,param in names.items():
            entry = event["parameter_evidence"][param]
            if entry["gradient_l1"]:
                active.append(name+"/nonzero")
            if entry["changed"]:
                active.append(name+"/updated")
        rows.append((labels,active))
    return dict(available="per_tensor_checked_on_each_step",per_class_derivative_isolated=False,
                updates_observed=len(rows),support=plan["identities"]["support"],
                by_cursor=join(rows,set(XZ_CURSORS.values())-{0},9,keys))


def p4_context(root, short):
    if short:
        path,sha = archive.PINS["hh_scope"]
        scope = json_pin(root,dict(path=str(root/path),sha256=sha))
        _,reg,plan,geometry,_,allowed = archive._honghu_context(scope)
        cases = scope["cases"]
    else:
        review = json_pin(root,proof(root,"p4"))
        scope = json_pin(root,review["scientific_scope"])
        path = scope["registration"]
        reg = json_pin(root,dict(path=path,sha256=scope["protected_files"][path]))
        plan,geometry = (json_pin(root,dict(path=reg[key],sha256=reg["pins"][reg[key]]))
                         for key in ("plan","development"))
        _,allowed = hh.guarded.bind_inputs(reg,plan,geometry)
        cases = review["cases"]
    return reg,plan,geometry,allowed,{r["case"]["id"]:r["case"] for r in cases}


def honghu(root, cases, short):
    reg,plan,geometry,allowed,cases_by_id = p4_context(root,short)
    output = {}
    for r in cases:
        if r.get("history")=="confusion_only":
            output[r["case"]] = dict(available="checkpoint_unavailable")
            continue
        case = cases_by_id[r["case"]]
        require(all(case[k]==r[k] for k in ("support_seed","k","method","arm")),"Foreign archived support case")
        release = hh.prior.released_roles(reg,plan,geometry,allowed,case)
        support = {tile["id"]:local[local>0].tolist() for tile,local in release["train"]}
        state = checkpoint(root,r["checkpoint"])
        expected = axes("whu-honghu")
        named = {}
        if not short:
            audit = state["parameter_audit"]
            require(audit["schema"]==update.SCHEMA,"Unknown logged Adam flags")
            named = {entry["name"]:i for i,entry in enumerate(audit["layout"])}
            require(set(expected.values())<=set(named),"The saved layout omitted a recognition head")
        channels = ["learned_graph/nonzero","learned_graph/updated"]
        if not short:
            channels.extend(f"{word}/{tag}" for word in expected for tag in ("nonzero","updated"))
        rows = []
        for event in state["trace"]:
            require(event["step"]==len(rows)+1,"Unordered HongHu step evidence")
            labels = support[event["tile"]]
            records = []
            if event["graph_gradient_l1"]:
                records.append("learned_graph/nonzero")
            if event["graph_changed_elements"]:
                records.append("learned_graph/updated")
            if not short:
                flags = bytes.fromhex(event["parameter_flags"])
                require(len(flags)==len(named) and set(flags)<=update.VALID_FLAGS,"Recorded named gradient flags differ")
                for word,param in expected.items():
                    flag = flags[named[param]]
                    if flag&update.NONZERO:
                        records.append(word+"/nonzero")
                    if flag&update.CHANGED:
                        records.append(word+"/updated")
            rows.append((labels,records))
        cursors = set(map(int,r["exposure"]))-{0}
        output[r["case"]] = dict(available="learned_graph_only" if short else "graph_and_named_head_tensors",
            per_class_derivative_isolated=False,updates_observed=len(rows),by_cursor=join(rows,cursors,22,channels))
    return output
