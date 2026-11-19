"""Recompute the native, matched-deletion and retraining evidence for WP12."""

from collections import Counter
import csv
import json
import os
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.classification_metrics import classification_metrics
from evaluation.frozen_band_region import HEADS, hard
from scripts.run_frozen_band_region import _read, _write, check, pin
from transfer.band_graph_campaign import PHASES, scope
from transfer.band_graph_control import ARMS, FIXED_ARMS
from transfer.band_graph_fixture import SEEDS, LAYOUTS, TASKS, VIEWS, STEPS


def endpoint(value, matches, arm, mode, key, writer):
    y=value["labels"]
    check(len(y)==8 and Counter(y)=={1:4,2:4},"Validation point exposure differs")
    for row in value["traces"]:
        plan=matches[f'devval/{row["tile"]}']
        check(row["band_ids"]==plan["band_ids"],"Original band IDs differ")
        if arm!="native":
            removed=(0 if arm=="fixed_none" else len(plan["baseline"]["removed"]) if arm=="important"
                     else len(plan["controls"][arm]["removed"]))
            check(row["nnz"]==plan["original_nnz"]-removed and row["q_nnz"]==plan["assignment"],
                  "Actual Q/A support does not match the physical deletion")
    summaries={}
    for head in HEADS:
        scores=np.asarray(value["scores"][head],dtype=np.float64)
        pred=hard(scores,8,9)
        metrics=classification_metrics(np.asarray(y),np.asarray(pred),range(1,10))
        p=scores[np.arange(8),np.asarray(y)-1]
        nll=float(-np.log(np.maximum(p,1e-8)).mean())
        for i,(truth,label,row) in enumerate(zip(y,pred,scores,strict=True)):
            writer.writerow(dict(case=key,mode=mode,arm=arm,head=head,point=i,truth=truth,predicted=label,
                                 scores=json.dumps(row.tolist(),separators=(",",":"))))
        summaries[head]=dict(nll=nll,**metrics)
    return summaries


def match_checks(matches, band, layout):
    check(set(matches)=={*(f"devtrain/{i}" for i in range(4)),"devval/4","devval/5"},
          "Matching plan does not cover all train/devval views")
    for row in matches.values():
        n=9
        removed=set(row["baseline"]["removed"])
        check(row["points"]==4 and len(row["band_ids"])==(8 if band=="full" else 4)
              and row["assignment"]==12*32 and row["symmetric"]==(layout=="vertical")
              and row["original_nnz"]==(25 if layout=="vertical" else 17)
              and len(removed)==(8 if layout=="vertical" else 4)
              and all(e//n!=e%n and row["topology"]["isolated_self"] not in divmod(e,n) for e in removed),
              "Unexpected original edge support or mandatory isolated self-loop")
        for item in row["controls"].values():
            if item["status"]=="optimal":
                ids=set(item["removed"])
                check(len(ids)==len(removed) and ids!=removed
                      and item["counts"]==row["baseline"]["counts"],"Unmatched edge group")
                if row["symmetric"]:
                    check(all((e%n)*n+e//n in ids for e in ids),"Unpaired reciprocal mask")
            else:
                check(item["removed"] is None,"Unavailable mask may not supply an intervention")


def fit_checks(path, item, manifest, arm):
    check(item["status"]=="passed" and item["scope"]==manifest and item["arm"]==arm
          and item["budget"]==STEPS and item["equal"] is True and item["selected"] in (0,6,12),
          "Incorrect native recoverable fit")
    phases=item["stage"]
    arrays=item["checkpoint_sha"]
    check(set(phases)==set(arrays)==set(PHASES)
          and [phases[tag]["updates"] for tag in PHASES]==[12,4,8,0]
          and phases["full"]["state"]==phases["resume"]["state"]==phases["completed"]["state"],
          "Scheduled native steps or final recovery identity differs")
    for name,sha in arrays.items():
        check(pin(path/f"{name}.pt")["sha256"]==sha,"Training checkpoint bytes changed")


def science(cases):
    out={}
    for task in TASKS:
        details=[]
        for layout,seed in product(LAYOUTS,SEEDS):
            key=f"{task}-{layout}-{seed}"
            case=cases[key]
            values={}
            for band in VIEWS:
                catalog=case[band]
                values[band]=dict(
                    native=catalog["fit"]["native"]["heads"]["fused"]["oa"],
                    fixed=catalog["fit"]["fixed_none"]["heads"]["fused"]["oa"],
                    important=catalog["frozen"]["important"]["graph"]["oa"],
                    frozen_full=catalog["frozen"]["fixed_none"]["graph"]["oa"],
                    weak=(catalog["frozen"]["low_weight"]["graph"]["oa"] if "low_weight" in catalog["frozen"] else None))
            spectral=min(values["full"]["fixed"],values["continuous"]["fixed"])-values["dispersed"]["fixed"]
            native=min(values["full"]["native"],values["continuous"]["native"])-values["dispersed"]["native"]
            full=values["full"]
            graph_ok=full["weak"] is not None and full["frozen_full"]-full["important"]>=.10 and abs(
                     full["frozen_full"]-full["weak"])<=.10
            details.append(dict(case=key,values=values,spectral_fixed_delta=spectral,
                                spectral_native_delta=native,graph_signal=graph_ok))
        out[task]=dict(cases=details,spectral_both=sum(d["spectral_fixed_delta"]>=.10 and d["spectral_native_delta"]>=.10
                                                      for d in details),graph_signature=sum(d["graph_signal"] for d in details))
    out["claims"]=dict(central_frequency=out["spectral"]["spectral_both"]>=4,
         graph_remote=out["remote"]["graph_signature"]>=4 and out["negative"]["graph_signature"]<4,
         note="trials use controlled independent synthetic tiles only")
    return out


def publish(folder,out,code,expected_scope):
    reg=scope(folder,code)
    check(reg["sha256"]==expected_scope and not out.exists(),"Wrong immutable campaign or publication")
    cases, case_hashes, witnesses = {}, {}, {}
    updates=0
    unavailable_count=0
    fit_count=0
    with TemporaryDirectory(prefix=".wp12-band-graph-",dir=out.parent) as temp:
        stage=Path(temp)/"result"
        stage.mkdir()
        with (stage/"probabilities.csv").open("w",encoding="utf-8",newline="") as file:
            writer=csv.DictWriter(file,("case","mode","arm","head","point","truth","predicted","scores"))
            writer.writeheader()
            for task,layout,seed in product(TASKS,LAYOUTS,SEEDS):
                key=f"{task}-{layout}-{seed}"
                path=folder/key
                link=pin(path/"receipt.json")
                record=_read(link)
                check(record["scope"]==reg["sha256"] and (record["task"],record["layout"],record["seed"])==(task,layout,seed)
                      and record["real_updates"]==0 and record["raw_gt_loaded"] is record["query_accessed"] is False
                      and set(record["views"])==set(VIEWS),"Foreign case or label access")
                cases[key]={}
                case_hashes[key]=link["sha256"]
                witnesses[key]=record["truth"]
                origins,exposure=set(),set()
                for band in VIEWS:
                    base=path/band
                    case=_read(pin(base/"receipt.json",record["views"][band]))
                    check(case["band"]==band and case["scope"]==reg["sha256"] and case["foreign_band_rejected"],
                          "Band/graph scope differs")
                    match_checks(case["matches"],band,layout)
                    bad=case["unavailable"]
                    unavailable_count+=len(bad)
                    check(set(case["fits"])==set(ARMS)-set(bad) and set(case["frozen"])==set(FIXED_ARMS)-set(bad),
                          "Incomplete or partly unmatched control")
                    row=dict(fit={},frozen={},unavailable=bad,matches=case["matches"])
                    for arm,sha in case["fits"].items():
                        f=base/arm.replace(":","-")
                        fit=_read(pin(f/"receipt.json",sha))
                        fit_checks(f,fit,reg["sha256"],arm)
                        report=endpoint(fit["endpoint"],case["matches"],arm,"retrained",key+"/"+band,writer)
                        row["fit"][arm]=dict(heads=report,selected=fit["selected"],stage_cost=fit["stage"],
                                             inferences=fit["endpoint"]["traces"],seconds=fit["seconds"],peak_kib=fit["peak_kib"])
                        origins.add(fit["source"])
                        exposure.add(fit["exposure"])
                        updates+=sum(value["updates"] for value in fit["stage"].values())
                        fit_count+=1
                    for arm,item in case["frozen"].items():
                        check(item["scores"]["cnn"]==case["frozen"]["fixed_none"]["scores"]["cnn"],
                              "Frozen CNN invalidated graph-only causality")
                        row["frozen"][arm]=endpoint(item,case["matches"],arm,"frozen",key+"/"+band,writer)
                        row.setdefault("frozen_cost",{})[arm]=item["traces"]
                    cases[key][band]=row
                check(len(origins)==len(exposure)==1,"Initialization or labeled tile exposure differs by band/arm")
        check(fit_count+unavailable_count==270 and updates==24*fit_count,"Actual budget is not the declared matched matrix")
        _write(stage/"case-metrics-cost.json",cases)
        _write(stage/"hypotheses.json",science(cases))
        _write(stage/"physical-witness.json",witnesses)
        files={f.name:pin(f)["sha256"] for f in stage.iterdir()}
        _write(stage/"receipt.json",dict(status="accepted_controlled_synthetic_only",producer=code["commit"],
               scope=expected_scope,cases=case_hashes,fit_cases=fit_count,unavailable=unavailable_count,
               synthetic_updates=updates,real_updates=0,query_accessed=False,raw_gt_loaded=False,artifacts=files))
        os.rename(stage,out)
    return pin(out/"receipt.json")
