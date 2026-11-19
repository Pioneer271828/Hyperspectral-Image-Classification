"""Publish separately audited native deployment cost and synthetic engineering quality."""

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.wp11_deployment_cost import SCOPES, document, need, read_plan
from evaluation.wp11_deployment_synthetic import BUDGETS, SEEDS, SUPPORT
from model.research_config import digest
from research_runtime import hash_file, require_frozen_code
from scripts.run_frozen_band_region import _write, pin


def audited_cost(root,out,code,plan_sha):
    plan_pin,plan=read_plan(root,out,code,plan_sha)
    batch=out/"stages"
    rows,sources=[],{}
    for folder in sorted(batch.iterdir()):
        manifest=folder/"device.json"
        if not manifest.exists():
            continue
        identity=pin(manifest)
        value=document(root,identity)
        device=folder.name
        need(value["plan"]==plan_pin["sha256"] and value["device"]==device
             and len(value["stages"])==len(SCOPES)*len(plan["cases"]),"Incomplete physical execution roster")
        expected={(case,scope) for case in plan["cases"] for scope in SCOPES}
        seen,predictions=set(),{}
        for link in value["stages"]:
            record=document(root,link)
            l=record["lineage"]
            case,scope=l["case"],l["scope"]
            need((case,scope) in expected-seen and l["device"]==device and l["plan"]==plan_pin["sha256"]
                 and l["environment"]==value["environment"] and record["optimizer_updates"]==0
                 and record["query_accessed"] is False,"Foreign benchmark scope receipt")
            seen.add((case,scope))
            samples=record["samples"]
            registered=plan["cases"][case]
            count=len(registered["requests"])
            graph_count=count if registered["mode"]=="full" else 0
            need(len(samples)==plan["repeats"][scope] and
                 all(s["wall_ms"]>0 and s["graph_calls"]==graph_count and s["light_calls"]+s["graph_calls"]==count
                     and s["io_reads"]==(0 if scope=="resident_forward" else registered["io_per_warm"])
                     and (s["gpu_event_ms"] is not None)==(device!="cpu") for s in samples),
                 "Timing samples do not represent their declared operations")
            reps={s["prediction"] for s in samples}
            need(len(reps)==1 and predictions.setdefault(case,next(iter(reps)))==next(iter(reps)),
                 "Deployment prediction changed across timing scopes")
            timing=record["summary"]
            elapsed=[r["wall_ms"] for r in samples]
            p50,p95=np.quantile(elapsed,[.5,.95],method="linear")
            need(math.isclose(p50,timing["p50_ms"],rel_tol=1e-12) and
                 math.isclose(p95,timing["p95_ms"],rel_tol=1e-12)
                 and timing["rss_peak_kib"]==max(r["rss_kib"] for r in samples),
                 "Independent timing percentile or host RSS differs")
            if scope=="concurrent_throughput":
                need(len({r["pid"] for r in samples})==plan["concurrent_workers"] and
                     math.isclose(len(samples)*1000/timing["wall_concurrent_ms"],
                                  timing["concurrent_requests_per_second"],rel_tol=1e-12),
                     "Concurrent measured duration or worker roster differs")
            elif scope=="cold_process_first_prediction":
                need(len({r["pid"] for r in samples})==len(samples) and all(
                        r["process_exit_ms"]>=r["wall_ms"]>=r["inside_request_ms"] for r in samples),
                     "Cold process ordering does not enclose first available prediction")
            rows.append(dict(case=case,scope=scope,device=device,mode=registered["mode"],workload=registered["workload"],
                             digest=link["sha256"],sample_count=len(samples),
                             warmups=record["warmups"],summary=timing,batch=count,
                             pixels=sum(math.prod(s["shape"]) for s in plan["cases"][case]["requests"])))
        sources[device]=identity["sha256"]
    need("cpu" in sources,"CPU deployment costs are required before optional physical qualification")
    return rows,sources,plan_pin,plan


def audited_synthetic(root,folders,code):
    out={}
    for folder in folders:
        link=pin(folder/"routing.json")
        result=document(root,link)
        need(result["producer"]==code["commit"] and result["scope"]=="isolated_synthetic_engineering"
             and result["query_accessed"] is False and result["optimizer_updates"]==0
             and len(result["rows"])==len(SUPPORT)*len(SEEDS)*len(BUDGETS),"Foreign synthetic routing result")
        labels=Counter((row["support"],row["seed"],row["budget"]) for row in result["rows"])
        need(len(labels)==len(result["rows"]),"Duplicate budget case")
        for row in result["rows"]:
            adaptive=row["output"]["adaptive"]
            random=row["output"]["matched_random"]
            full=row["output"]["full"]
            for name in ("light_calls","full_calls","graph_builds","coverage_fallbacks"):
                need(adaptive["counts"][name]==random["counts"][name],"Random control no longer matches executed work")
            ref=full["metrics"]
            for sample in row["output"].values():
                for measured in sample["timing"].values():
                    need(len(measured["wall_ms"])==4 and measured["warmups"]==2
                         and math.isclose(np.quantile(measured["wall_ms"],.95),measured["p95_ms"],rel_tol=1e-12),
                         "Synthetic route timing samples changed")
                y=sample["metrics"]
                matrix=np.asarray(y["confusion_matrix"])
                need(int(matrix.sum())==y["count"]==ref["count"] and
                     math.isclose(float(matrix.trace()/matrix.sum()),y["oa"],rel_tol=1e-12),
                     "Synthetic result denominators or OA differ")
            need(row["overrun"]==max(0,adaptive["counts"]["full_calls"]-row["limit"]),"Fallback overrun not retained")
        key="cpu" if result["physical_admission"] is None else result["physical_admission"]["physical_uuid"]
        need(key not in out,"Duplicate synthetic execution device")
        out[key]=dict(artifact=link["sha256"],summaries=result["summaries"],failures=result["failures"])
    return out


def publish(root,run,code,plan_sha,synthetic):
    costs,devices,plan_pin,plan=audited_cost(root,run,code,plan_sha)
    routes=audited_synthetic(root,synthetic,code)
    out=run/"report"
    need(not out.exists(),"Fresh external deployment publication required")
    with TemporaryDirectory(dir=run,prefix=".deployment-") as scratch:
        stage=Path(scratch)/"report"
        stage.mkdir()
        _write(stage/"costs.json",costs)
        _write(stage/"legacy-scopes.json",plan["archive"])
        _write(stage/"synthetic-routing.json",routes)
        with (stage/"costs.csv").open("x",encoding="utf-8",newline="") as file:
            writer=csv.writer(file)
            a=("case","scope","device","mode","workload","batch","pixels","sample_count","warmups")
            b=("p50_ms","p95_ms","rss_peak_kib","cuda_process_peak_bytes","graph_ms","concurrent_requests_per_second")
            writer.writerow((*a,*b))
            for row in costs:
                writer.writerow((*(row[k] for k in a),*(row["summary"][k] for k in b)))
        require_frozen_code(code)
        output=_write(stage/"receipt.json",dict(status="deployment_engineering_accepted",producer=code["commit"],
                 plan=plan_pin["sha256"],devices=devices,synthetic_devices=list(routes),new_cost_scopes=len(costs),
                 measured_request_samples=sum(x["sample_count"] for x in costs),os_cache=plan["os_cache"],
                 real_independent_confirmation_available=False,query_accessed=False,new_optimizer_updates=0,
                 files={p.name:hash_file(p) for p in stage.iterdir()}))
        os.rename(stage,out)
    return dict(path=str(out/"receipt.json"),sha256=output["sha256"])
