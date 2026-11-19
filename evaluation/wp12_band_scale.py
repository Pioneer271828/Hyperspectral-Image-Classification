"""Consume the registered band-scale evidence and independently recount released endpoints."""

from collections import Counter
import csv
from itertools import product
import json
import os
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from evaluation.classification_metrics import classification_metrics
from scripts.run_frozen_band_region import _read, _write, check, pin
from transfer import band_scale_real as real
from transfer.band_scale import BANDS, FACTORS, HEADS
from transfer.band_scale_campaign import PHASES, SYNTHETIC, registered
from transfer.band_scale_fixture import ARMS, PREFIX, STEPS, TASKS
from transfer.spatial_coverage_training import load
from transfer.native_scale_training import _exact


def choose(probs):
    return max(range(1,len(probs)+1),key=lambda i:probs[i-1])


def pack(case, policy, writer):
    rows = sorted((x for x in case["rows"] if all(x["event"][k]==v for k,v in policy.items())),key=lambda x:x["tile"])
    values = {head:[] for head in HEADS}
    for row in rows:
        for j,(point,label,groups) in enumerate(zip(row["points"],row["labels"],row["groups"],strict=True)):
            for head in HEADS:
                probs = row["probabilities"][head][j]
                competing = case["competitor"].get(str(label))
                if competing is None:
                    competing = max((i for i in range(1,len(probs)+1) if i!=label),key=lambda i:probs[i-1])
                margin = probs[label-1]-probs[competing-1]
                predicted = choose(probs)
                value = dict(point=point,label=label,predicted=predicted,other=competing,margin=margin,**groups)
                values[head].append(value)
                writer.writerow((case["case"],policy["band"],policy["factor"],policy["policy"],head,point,
                                 label,predicted,competing,margin,json.dumps(probs,separators=(",",":"))))
    check(all([row["label"] for row in seq]==case["labels"] for seq in values.values()),"Real population or order changed")
    return values


def compare(base, treated):
    groups = {
        "released_all":lambda _a,_b:True,
        "observed_neighbor_disagreement":lambda a,b:a["released_boundary"]==b["released_boundary"]=="known_neighbor_disagrees",
        "unknown_boundary":lambda a,b:a["released_boundary"]==b["released_boundary"]=="unknown_from_sparse_labels",
        "mixed_released_region":lambda _a,b:b["region_known_classes"]>1,
        "at_most_one_released_class_per_region":lambda _a,b:b["region_known_classes"]<=1,
        "complete_boundary":lambda _a,_b:False,
        "thin_structure":lambda _a,_b:False,
    }
    result = {}
    for head in HEADS:
        rows = list(zip(base[head],treated[head],strict=True))
        check(all((a["point"],a["label"])==(b["point"],b["label"]) for a,b in rows),"Pairing changed original point")
        rows_by_class = {}
        for label in sorted({a["label"] for a,_b in rows}):
            arm = {}
            for group,include in groups.items():
                points = [(a,b) for a,b in rows if a["label"]==label and include(a,b)]
                n = len(points)
                source_correct = sum(a["predicted"]==label for a,_ in points)
                target_correct = sum(b["predicted"]==label for _,b in points)
                afternll = fmean(b["margin"] for _,b in points) if n else None
                beforenll = fmean(a["margin"] for a,_ in points) if n else None
                arm[group] = dict(denominator=n, correct_before=source_correct,correct_after=target_correct,
                                  flipped=sum(a["predicted"]!=b["predicted"] for a,b in points),
                                  lost=sum(a["predicted"]==label and b["predicted"]!=label for a,b in points),
                                  gained=sum(a["predicted"]!=label and b["predicted"]==label for a,b in points),
                                  before_oa=source_correct/n if n else None,after_oa=target_correct/n if n else None,
                                  before_mean_margin=beforenll,after_mean_margin=afternll,
                                  delta_margin=None if not n else afternll-beforenll,
                                  passes_protection=None if not n else (target_correct-source_correct)/n>=-.02,
                                  true_to_competitor_before=sum(a["predicted"]==a["other"] for a,_ in points),
                                  true_to_competitor_after=sum(b["predicted"]==b["other"] for _,b in points))
            rows_by_class[str(label)] = arm
        result[head]=rows_by_class
    return result


def real_table(item, writer):
    data = {}
    for band,factor,policy in product(BANDS,FACTORS,item["supported_policies"]):
        event = dict(band=band,factor=factor,policy=policy)
        data[(band,factor,policy)] = pack(item,event,writer)
    contrasts = {}
    for factor in FACTORS:
        contrasts[f"gap_f{factor}"] = compare(data[("full",factor,"native")],data[("central_gap",factor,"native")])
    for band in BANDS:
        contrasts[f"area2_{band}"] = compare(data[(band,1,"native")],data[(band,2,"native")])
    for band,factor,kind in product(BANDS,FACTORS,item["supported_policies"][1:]):
        contrasts[f"{kind}_{band}_f{factor}"] = compare(data[(band,factor,"native")],data[(band,factor,kind)])
    interactions = {}
    if "both1" in item["supported_policies"]:
        for band,factor in product(BANDS,FACTORS):
            key = f"{band}_f{factor}"
            controls = [contrasts[f"{name}_{key}"] for name in ("cnn1","graph1","both1")]
            interactions[key] = {head:{label:controls[2][head][label]["released_all"]["delta_margin"]
                                            -controls[0][head][label]["released_all"]["delta_margin"]
                                            -controls[1][head][label]["released_all"]["delta_margin"]
                                       for label in controls[0][head]} for head in HEADS}
    return dict(scene=item["scene"],seed=item["model_seed"],architecture=item["architecture"],model=item["model"],
                selected_step=item["selected_step"],competitor=item["competitor"],labels=dict(Counter(item["labels"])),
                forwards=len(item["rows"]),pairs=contrasts,margin_interaction=interactions,
                no_cross_reason=item["no_cross_reason"],dependency=item["checkpoint"])


def synthetic_table(row, case, arm, writer):
    reports = {}
    for band,factor in product(BANDS,FACTORS):
        found = [x for x in row["endpoints"] if x["event"]["band"]==band and x["event"]["factor"]==factor]
        truth = [label for x in found for label in x["labels"]]
        check(Counter(truth)=={1:4,2:4},"Synthetic verification denominator changed")
        reports[f"{band}/{factor}"] = {}
        for head in HEADS:
            predictions = [p for x in found for p in x["scores"][head]]
            for i,(label,probs) in enumerate(zip(truth,predictions,strict=True)):
                check(len(probs)==9 and abs(sum(probs)-1.)<3e-6 and all(p>=0 for p in probs),"Invalid scale endpoint probabilities")
                writer.writerow((case,arm,band,factor,head,i,label,choose(probs),json.dumps(probs,separators=(",",":"))))
            reports[f"{band}/{factor}"][head] = classification_metrics(truth,[choose(p) for p in predictions],list(range(1,10)))
    return reports


def tests(scored):
    signals = {}
    for task in TASKS:
        data = []
        for name,matrix in scored.items():
            if not name.startswith(task+"-"):
                continue
            signals_delta = (matrix["off"]["full/1"]["fused"]["oa"]-matrix["off"]["central_gap/1"]["fused"]["oa"]
                             if task=="spectral" else matrix["graph_content"]["full/1"]["fused"]["oa"]
                             -matrix["cnn_content"]["full/1"]["fused"]["oa"])
            data.append(dict(case=name,delta_oa=signals_delta,passed=signals_delta>=.10))
        signals[task] = dict(rows=data,met=sum(x["passed"] for x in data))
    return dict(**signals,spectral_supported=signals["spectral"]["met"]>=4,
                graph_context_supported=signals["context"]["met"]>=4 and signals["negative"]["met"]<4)


def publish(root,folder,out,code,expected):
    sha = registered(root,folder,code,expected)["sha256"]
    check(not out.exists(),"Existing scale publication")
    historical_ref = pin(folder/"history.json")
    history = _read(historical_ref)
    check(history["scope"]==sha and history["pavia"]["phase_receipts"]==117
          and len(history["xuzhou"])==8 and len(history["honghu"])==40,"Historical evidence roster incomplete")
    with TemporaryDirectory(prefix=".wp12-band-scale-",dir=out.parent) as temp:
        stage = Path(temp)/"publication"
        stage.mkdir()
        synthetic, cases, timing, new_real = {}, {}, {}, {}
        updates = 0
        with (stage/"synthetic-probabilities.csv").open("w",encoding="utf-8",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("case","arm","band","factor","head","point","truth","predicted","probabilities"))
            for name in SYNTHETIC:
                path = folder/name
                seal = pin(path/"receipt.json")
                item = _read(seal)
                check(item["scope"]==sha and item["case"]==name and item["real_updates"]==0
                      and item["raw_gt_loaded"] is item["query_accessed"] is False,"Foreign synthetic scale receipt")
                synthetic[name], cases[name], timing[name] = {}, seal["sha256"], {}
                source = set()
                for arm in ARMS:
                    base = path/arm
                    fit = _read(pin(base/"receipt.json",item["arms"][arm]))
                    check(fit["scope"]==sha and fit["arm"]==arm,"Foreign frozen arm")
                    source.add(fit["initial"])
                    by_phase = [fit["phases"][phase]["updates"] for phase in PHASES]
                    check(by_phase==[STEPS,PREFIX,STEPS-PREFIX,0],"Synthetic recovery schedule differs")
                    states = {}
                    for phase in PHASES:
                        states[phase] = load(Path(pin(base/f"{phase}.pt",fit["snapshots"][phase])["path"]))
                        check(pin(base/f"{phase}.pt")["sha256"]==fit["snapshots"][phase],"Altered checkpoint bytes")
                    check(_exact(states["full"],states["resume"]) and _exact(states["full"],states["completed"])
                          and min(states["full"]["validation"],key=lambda row:row["fused_nll"])["step"]==fit["chosen"],
                          "Closed synthetic recovery state or first-min rule differs")
                    if arm=="joint":
                        check(fit["physical_controls"]["fallback_matches_content"],"Lost physical fallback result")
                    updates+=sum(by_phase)
                    synthetic[name][arm] = synthetic_table(fit,name,arm,writer)
                    timing[name][arm] = dict(phases=fit["phases"],branch_updates=fit["updates_by_branch"],
                                              parameters=fit["parameters"],chosen=fit["chosen"])
                check(len(source)==1,"Synthetic arms differ in baseline")
        with (stage/"real-probabilities.csv").open("w",encoding="utf-8",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("case","band","factor","selector","head","point","truth","predicted","competitor","margin","probabilities"))
            for name in real.CASES:
                ref = pin(folder/f"{name}.json")
                current = _read(ref)
                check(current["scope"]==sha and current["case"]==name and current["real_updates"]==0
                      and current["raw_gt_loaded"] is current["query_accessed"] is False,"Foreign actual scale receipt")
                cases[name] = ref["sha256"]
                new_real[name] = real_table(current,writer)
        for name,ref in history["pavia"]["tables"].items():
            source = Path(ref["path"])
            check(pin(source)["sha256"]==ref["sha256"],"Changed historical Pavia table")
            with source.open(encoding="utf-8",newline="") as left, (stage/f"historical-{name}.csv").open("w",encoding="utf-8",newline="") as right:
                rows = csv.DictReader(left)
                fields = [key for key in rows.fieldnames if key!="physical_gpu"]
                dest = csv.DictWriter(right,fields,extrasaction="ignore")
                dest.writeheader()
                dest.writerows(rows)
        _write(stage/"scope-summary.json",dict(synthetic=tests(synthetic),new_real_forwards=sum(v["forwards"] for v in new_real.values()),
               historical_pavia_phases=117,historical_pavia_endpoints=468,historical_xuzhou_endpoints=16,
               historical_honghu_model_cases=40))
        _write(stage/"paired-real.json",new_real)
        _write(stage/"synthetic-metrics.json",synthetic)
        _write(stage/"synthetic-cost.json",timing)
        _write(stage/"historical-sources.json",history)
        files = {x.name:pin(x)["sha256"] for x in stage.iterdir()}
        _write(stage/"receipt.json",dict(status="accepted_released_only_and_controlled_synthetic",
               scope=sha,producer=code["commit"],cases=cases,artifacts=files,historical=historical_ref["sha256"],
               synthetic_updates=updates,completed_reentry_updates=0,real_updates=0,query_accessed=False,raw_gt_loaded=False))
        os.rename(stage,out)
    return pin(out/"receipt.json")
