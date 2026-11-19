"""Publish frozen localization and controlled repair in the WP12 evidence namespace."""

from collections import Counter
import csv
import json
import os
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from evaluation.classification_metrics import classification_metrics
from evaluation.representation_localization import HEADS, STAGES
from research_runtime import hash_file
from transfer.registered_development import _json_new, _pin
from transfer.representation_campaign import CASES, PHASES, load, read, scope, state_fingerprint
from transfer.representation_bypass import ARMS, BANDS, PREFIX, STEPS, TASKS


def need(condition,message):
    if not condition:
        raise ValueError(message)


def numeric(row, truth, competitors):
    probs = row["probabilities"]
    for i,(p,label,other) in enumerate(zip(probs,truth,competitors,strict=True)):
        pred = max(range(len(p)),key=p.__getitem__)+1
        rank = 1+sum(v>p[label-1] for v in p)
        need(len(p)>1 and abs(sum(p)-1.)<2e-6 and all(z>=0 for z in p)
             and pred==row["predicted"][i] and rank==row["rank"][i]
             and abs(p[label-1]-p[other-1]-row["margin"][i])<1.5e-7,
             "Probability margin, rank or prediction changed")


def slopes(row,truth,competitors):
    for key in ("base","plus","minus","donor"):
        need(set(row[key])==set(HEADS),"Incomplete frozen heads")
        for head in HEADS:
            numeric(row[key][head],truth,competitors)
    for head in HEADS:
        record = row["slope"][head]
        approx=[(a-b)/(2*row["epsilon"]) for a,b in zip(row["plus"][head]["margin"],row["minus"][head]["margin"],strict=True)]
        need(all(abs(a-b)<3e-6 for a,b in zip(approx,record["finite_difference"],strict=True))
             and all(err is (None) if tangent is None else abs(err-abs(tangent-fd))<1e-8
                     for err,tangent,fd in zip(record["absolute_error"],record["directional"],record["finite_difference"],strict=True)),
             "Finite-difference arithmetic changed")


def real_table(archive,writer,code_path):
    need(archive["status"]=="passed" and archive["query"] is archive["raw_gt"] is False and archive["real_updates"]==0
         and archive["common_diagnostic_sha256"]==hash_file(code_path)
         and set(archive["scenes"])=={"xuzhou","whu-honghu"}
         and archive["historical"]["old_probe_models"]==54,
         "Historical bridge or released proof differs")
    result = {}
    for scene,item in archive["scenes"].items():
        collection = {}
        focus = {5:8,6:3} if scene=="xuzhou" else {15:1,21:18}
        expected = Counter({i:5 for i in focus})
        observed = Counter()
        for window in item["windows"]:
            truth,other = window["truth"],window["competitors"]
            observed.update(truth)
            need(other==[focus[y] for y in truth] and window["retained_projection_equal"]
                 and window["self_replacement_exact"] and window["unhooked_repeat_exact"]
                 and set(window["interventions"])=={"sensor",*STAGES},"Opened a different released point or donor view")
            for head in HEADS:
                for case in ("natural_full","natural_gap","frozen_gap","frozen_full"):
                    numeric(window[case][head],truth,other)
            for stage,entry in window["interventions"].items():
                slopes(entry,truth,other)
                for head in HEADS:
                    if stage!="sensor":
                        need(entry["base"][head]==window["frozen_gap"][head],"Replacement changed the fixed target baseline")
                    for j,(label,competitor) in enumerate(zip(truth,other,strict=True)):
                        old=entry["base"][head]["predicted"][j]
                        new=entry["donor"][head]["predicted"][j]
                        base=entry["base"][head]["margin"][j]
                        target=entry["donor"][head]["margin"][j]
                        slope=entry["slope"][head]["directional"][j]
                        error=entry["slope"][head]["absolute_error"][j]
                        writer.writerow((scene,window["window"],stage,head,j,label,competitor,base,target,
                                         old,new,slope,entry["slope"][head]["finite_difference"][j],error))
                        key=f"{stage}/{head}/{label}"
                        collection.setdefault(key,[]).append((label,base,target,old,new,error))
        need(observed==expected,"Lost released monitoring denominator")
        rows = {}
        for k,values in collection.items():
            n=len(values)
            errs=[x[5] for x in values if x[5] is not None]
            rows[k]=dict(count=n,base_margin=fmean(x[1] for x in values),donor_margin=fmean(x[2] for x in values),
                         base_correct=sum(x[3]==x[0] for x in values),donor_correct=sum(x[4]==x[0] for x in values),
                         became_correct=sum(x[3]!=x[0] and x[4]==x[0] for x in values),
                         flipped=sum(x[3]!=x[4] for x in values),max_absolute_error=max(errs,default=None))
        result[scene]=dict(model=item["model"],denominators=dict(observed),windows=len(item["windows"]),stages=rows,
                           topology_changes=sum(w["natural_topology_changed"] for w in item["windows"]))
    return result


def synthetic_table(item,case,arm,writer):
    metrics = {}
    for band in BANDS:
        values = item["endpoint"][band]
        y=values["labels"]
        metrics[band]={}
        for head in HEADS:
            p=values["probabilities"][head]
            predictions=[max(range(3),key=q.__getitem__)+1 for q in p]
            built=classification_metrics(y,predictions,[1,2,3])
            original=values["metrics"][head]
            need(all(original[key]==v for key,v in built.items()),"Independent confirmation classification differs")
            need(all(len(q)==3 and abs(sum(q)-1.)<2e-6 for q in p),"Invalid synthetic probability simplex")
            for i,(truth,boundary,q) in enumerate(zip(y,values["truth_boundary"],p,strict=True)):
                writer.writerow((case,arm,band,head,i,truth,int(boundary),*q))
            metrics[band][head]=original
    return metrics


def outcomes(values):
    result={}
    for task in TASKS:
        rows=[]
        for case,item in values.items():
            if not case.startswith(task+"-"):
                continue
            arms={arm:a["gap"]["fused"]["oa"] for arm,a in item.items()}
            delta=arms["aligned"]-max(arms["closed"],arms["matched_permutation"])
            passed = ((arms["aligned"]>=.8 and delta>=.2) if task=="recoverable"
                      else min(arms.values())>=.8 if task=="context" else delta<=.1)
            rows.append(dict(case=case,gap_fusion_oa=arms,aligned_over_strongest=delta,passed=passed))
        result[task]=dict(rows=rows,met=sum(r["passed"] for r in rows))
    return dict(tasks=result,repair_passed=all(x["met"]>=4 for x in result.values()))


def publish(folder,out,code,registered_hash,real_pin):
    identity=scope(folder,code,registered_hash)
    archived=read(real_pin)
    if out.exists():
        raise ValueError("Fresh independent WP12 result required")
    with TemporaryDirectory(dir=out.parent,prefix=".representation-wp12-") as temp:
        stage=Path(temp)/"publication"
        stage.mkdir()
        cases,timings,metric={}, {}, {}
        updated=0
        with (stage/"real-effects.csv").open("w",encoding="utf-8",newline="") as file:
            w=csv.writer(file)
            w.writerow(("scene","window","layer","head","ordinal","class","competitor","base_margin","donor_margin",
                        "base_prediction","donor_prediction","directional","finite_difference","error"))
            real=real_table(archived,w,Path(__file__).with_name("representation_localization.py"))
        with (stage/"synthetic-probabilities.csv").open("w",encoding="utf-8",newline="") as file:
            w=csv.writer(file)
            w.writerow(("case","arm","band","head","ordinal","class","boundary","p1","p2","p3"))
            for case in CASES:
                path=folder/case
                link=_pin(path/"receipt.json")
                item=read(link)
                need(item["case"]==case and item["scope"]==identity and item["real_updates"]==0
                     and item["query"] is item["raw_gt"] is False,"Foreign synthetic case")
                cases[case]=link["sha256"]
                timings[case],metric[case]={},{}
                origin,capacities=set(),set()
                for arm in ARMS:
                    base=path/arm
                    report=read({"path":str(base/"receipt.json"),"sha256":item["arms"][arm]})
                    same={}
                    for phase,n in zip(PHASES,(STEPS,PREFIX,STEPS-PREFIX,0),strict=True):
                        note=report["phases"][phase]
                        packet=load(base/f"{phase}.pt",note["checkpoint"])
                        need(note["steps"]==n and packet["state"]==note["state"]
                             and state_fingerprint(packet)==note["state"] and packet["cursor"]==note["cursor"],
                             "Restored residual clock or state differs")
                        same[phase]=packet["state"]
                        updated+=n
                    need(same["full"]==same["resume"]==same["completed"],"Fixed-budget complete restore differs")
                    if arm=="aligned":
                        for band,r in item["results"][arm]["diagnostics"].items():
                            need(band!="gap" or not case.startswith("removed-") or r["math"]["direction_l2"]==0,
                                 "Physically missing information was synthesized")
                    origin.add(report["initial"])
                    capacities.add(report["parameters"])
                    metric[case][arm]=synthetic_table(item["results"][arm],case,arm,w)
                    timings[case][arm]=dict(phases=report["phases"],selected=report["chosen"],
                                             chosen_cost=item["results"][arm]["times"],parameters=report["parameters"])
                need(len(origin)==len(capacities)==1,"Controlled arms changed initialization or capacity")
        need(updated==len(CASES)*len(ARMS)*STEPS*2,"Incomplete synthetic budget")
        _json_new(stage/"summary.json",dict(real=real,synthetic=outcomes(metric),historical=archived["historical"]))
        _json_new(stage/"synthetic-metrics.json",metric)
        _json_new(stage/"costs.json",timings)
        files={item.name:_pin(item)["sha256"] for item in stage.iterdir()}
        pin=_json_new(stage/"receipt.json",dict(status="accepted_frozen_development_and_synthetic",commit=code["commit"],
                      real_source=real_pin["sha256"],scope=identity,cases=cases,fits=len(CASES)*len(ARMS),
                      synthetic_updates=updated,complete_reentry_updates=0,real_updates=0,raw_gt=False,query=False,files=files))
        os.rename(stage,out)
    return {"path":str(out/"receipt.json"),"sha256":pin["sha256"]}
