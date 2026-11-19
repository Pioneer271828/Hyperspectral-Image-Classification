"""Authorized real classification and hardware evidence for the native release."""

from collections import defaultdict
import csv
from itertools import product
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from tempfile import TemporaryDirectory

from evaluation.classification_metrics import metrics_from_confusion
from evaluation.wp12_cost_primitives import timing_summary
from evaluation.wp12_evidence import accuracy_records, join_hardware
from evaluation.wp12_native_coverage import readout
from evaluation.wp12_p4_released_evidence import paviau_prepared, paviau_trained
from evaluation.wp12_resident_cost import assemble
from model.research_config import digest
from research_runtime import hash_file, require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer.development_blocks import read_pin
from transfer.native_coverage_queue import SCENES
from transfer.native_coverage_replay import EXPECTED_ARCHIVES

SCHEMA = "wp12_native_real_release_results_v1"
HEADS = ("fused", "graph", "cnn")
FROZEN = "wp12-cost-frozen54-campaign02"
NATIVE = "wp07-paviau-real-results-20261118.TC3eXtC5/aggregate/results"
BAND = "wp05-wp06-wp07-band-scale-20261119"
P4_REVIEW = "wp05-independent-queue-real-authority01/partial-audit02/review.json"
PROOFS = {
    "registration": (FROZEN+"/registration.json", "8bcd2c0b29f96b45d1883eab654881b271804bede86ce7e5b8d15e903c6ff0c8"),
    "amendment": (FROZEN+"/registration-amendment.json", "15b2d36a449248af00b8b3b2819f74e43572c73b3d635fc2a06551cd1e80b3ee"),
    "measurements": (FROZEN+"/measurements.json", "732e321a6943e38537e4ff87dd470c7128e65b82c6c89b2db7190360e3d3b102"),
    "plan": (FROZEN+"/cost-plan.json", "148b9c00c6138eb1e8b1122bfbe1e3c5c9c70f713dc7e33ca7d73acea08c5afd"),
    "old_combined": (FROZEN+"/combined-report.json", "342af1b6f9e83c1022f563156d1f1541d1862ccce0da1c7050a8d5ff1adda1cf"),
    "hardware": (FROZEN+"/hardware-receipt.json", "364ebda4801f9864a8177abc47950b8cbde77721fa68a5cc7ff4d97cecaea170"),
    "independent": ("wp12-cost-frozen54-independent01/receipt.json", "efa741072cfeed09e69345181cac9a8ec39b3e042bc12aac8ad8e9a3bd216e35"),
    "pavia": (NATIVE+"/comparison-audit.json", "728076ffe4751662cea2580e66c3b915d3fc5b30cecc1308ff1d761a3b8cc1bb"),
    "p4_supplement": ("wp12-p4-real-intake02/published/supplement.json", "c73fcb8dce2d9880b20bafd4c7a5f6cd22e7b148a497c6ff934352ab31bae678"),
    "p4": (P4_REVIEW, "a60134e6447550ce1d0455199a9b70f1f42648b860c61e87c72fe9ec0b8fa0fb"),
    "band_registration": (BAND+"/registration.json", "70b1c645051e55d6df7e0b6e44d99640238b167b3b5cdf8cb708d07f416a10b7"),
}
BAND_PINS = {**EXPECTED_ARCHIVES,
             "paviaU-512": "5f660c26de4fe5554dffa35dac23aa863fa2ff02123a665eea480d09d8fa0662",
             "paviaU-2048": "796572a49b68a70978cd9f0e7edde6c94bc9aa75f27ec7ef77b8b05c6662ab28"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def proof(root, key):
    file, sha = PROOFS[key]
    return dict(path=str(root/file), sha256=sha)


def document(root, pin):
    return json.loads(read_pin(pin, root))


def same_metrics(built, previous):
    for field in ("oa","aa","kappa","macro_f1"):
        a, b = built[field], previous[field]
        require(a is b is None or a is not None and b is not None and math.isclose(a,b,abs_tol=1e-12),
                "Recomputed released metric differs from the frozen export")


def metrics(matrix, classes, nll=None):
    out = metrics_from_confusion(matrix, range(1, classes+1))
    out["nll"] = nll
    return out


def cost_missing(reason):
    return dict(status="unmeasured", reason=reason, scope=None, p50_ms=None, p95_ms=None,
                peak_allocated_bytes=None, graph_seconds=None, end_to_end_ms=None)


def case_key(row):
    return tuple(row[field] for field in ("source","protocol","scene","seed","support","method","arm","role","band",
                                         "factor","endpoint","device","workload","measurement_scope"))


def frozen54(root):
    pins = {key: proof(root,key) for key in ("registration","amendment","measurements","plan","old_combined",
                                            "hardware","independent")}
    data = {key: document(root, pin) for key,pin in pins.items()}
    reg, audit = data["registration"], data["independent"]
    require(audit["status"] == "passed" and audit["case_count"] == len(reg["cases"]) == 54
            and audit["timed_samples"] == 648 and audit["warmups"] == 162
            and audit["optimizer_updates"] == audit["new_query_reads"] == 0
            and audit["preregistered_benefit_conclusion"] == "primary_benefit_not_supported"
            and data["amendment"]["revised_registration_sha256"] == pins["registration"]["sha256"]
            and data["amendment"]["changed_models_or_cost_budget"] is False,
            "Frozen54 campaign or independent audit differs")
    approval = reg["inputs"]["accuracy"]
    accuracy = accuracy_records(*(document(root,approval[key]) for key in ("analysis","receipt","freeze")),
                                analysis_sha=approval["analysis"]["sha256"], freeze_sha=approval["freeze"]["sha256"])
    plan = data["plan"]
    rows, cost_by_key, originals = [], {}, []
    require(len(data["measurements"])==54, "Incomplete accepted campaign manifest")
    for index, (link, item) in enumerate(zip(data["measurements"],reg["cases"],strict=True),1):
        accepted_path = root/FROZEN/f"accepted-{index:02}.json"
        require(document(root,dict(path=str(accepted_path),sha256=hash_file(accepted_path))) == link,
                "Accepted campaign pointer differs from the sealed roster")
        result = document(root,link)
        original_job = root/FROZEN/f"job-{index:02}.json"
        job_sha = hash_file(original_job)
        job = document(root,dict(path=str(original_job),sha256=job_sha))
        key = tuple(item[k] for k in ("scene","seed","arm"))
        require(key not in cost_by_key and result["job"] == job and result["job_sha256"] == job_sha
                and result["fresh_process"] is True and result["producer_commit"] == audit["worker_commit"]
                and result["environment"]["gpu"] == audit["hardware"] and job["workload"] == reg["workloads"][key[0]]
                and list(job["case"][k] for k in ("scene","seed","arm")) == list(key)
                and job["checkpoint"] == item["checkpoint"] and job["sidecar"] == item["sidecar"],
                "Raw cost record has a foreign case, workload, device or checkpoint")
        admission = result["resource_admission"]
        require(admission["free_mib"] >= audit["min_admitted_free_mib"]
                and admission["utilization_percent"] <= audit["max_admitted_utilization_percent"],
                "Admitted hardware resources differ from independent audit")
        measured = result["row"]["cost"]
        timing = timing_summary(measured["wall_ms"],repeats=reg["repeats"],blocks=job["workload"]["work_items"])
        resource = dict(status="measured", scope=reg["scope"], warmups=reg["warmups"], repeats=reg["repeats"],
                        **timing, peak_allocated_bytes=measured["peak_bytes"], parameters=measured["parameters"],
                        package_bytes=measured["package_bytes"], graph_seconds=None, end_to_end_ms=None,
                        reason="Graph is included in resident latency. Build-only, loading and I/O time were not separated.",
                        receipt=link["sha256"])
        cost_by_key[key] = (resource,job["workload"]["tensor_sha256"])
        originals.append(result)
    rebuilt = assemble(plan,originals,plan_sha256=pins["plan"]["sha256"])
    combined = join_hardware(accuracy,plan,rebuilt,cost_plan_sha=pins["plan"]["sha256"],baseline=reg["baseline"])
    require(rebuilt==data["hardware"] and combined==data["old_combined"]["combined"],
            "Reassembled raw hardware or legacy paired report differs")
    for item in accuracy["cases"]:
        scene, seed, arm = (item[k] for k in ("scene","seed","arm"))
        method, mode = arm.split("/")
        cost, workload = cost_by_key[scene,seed,arm]
        rows.append(dict(source="frozen54",protocol=item["evaluation_sha256"],scene=scene,seed=seed,support=None,
                         method=method,arm=mode,role="historical_frozen_authorized_aggregate",
                         band="ordered_registered_full_observation",factor=None,endpoint="frozen_primary",
                         device=audit["hardware"]["uuid"],workload=workload,measurement_scope=cost["scope"],
                         model=item["checkpoint_sha256"],cost=cost,
                         heads={h:metrics(item["heads"][h],len(item["class_ids"]),item["nll"][h]) for h in HEADS}))
    lifecycle = dict(cases=54, existing_accepted=54, pending=0, new_measurements=0, planned_warmups=162,
                     timed_samples=648, validation_forwards=audit["validation_forwards"],
                     planned_case_timeout_seconds=reg["case_timeout_seconds"], planned_machine_hours=None,
                     recorded_timed_sample_seconds=math.fsum(x["cost"]["mean_ms"]*reg["repeats"] for x in rows)/1000,
                     independent_audit_seconds=audit["elapsed_seconds"], gpu=audit["hardware"],
                     benefit=accuracy["preregistered_benefit_conclusion"], receipt=pins["independent"]["sha256"],
                     old_combined=pins["old_combined"]["sha256"])
    return rows,lifecycle


def csv_rows(root, pin):
    return list(csv.DictReader(read_pin(pin,root).decode("utf-8").splitlines()))


def native_pavia(root):
    link = proof(root,"pavia")
    audit = document(root,link)
    admitted = paviau_trained(link, root, paviau_prepared(audit["preflight"],root))
    tables = admitted["tables"]
    endpoints, classes = {}, {}
    inference, stages = defaultdict(list), defaultdict(list)
    for row in csv_rows(root,tables["endpoints"]):
        key = tuple(row[k] for k in ("case","endpoint","head","band"))
        require(key not in endpoints,"Repeated historical endpoint")
        endpoints[key]=row
    for row in csv_rows(root,tables["per-class"]):
        key = tuple(row[k] for k in ("case","endpoint","head","band","class_id"))
        require(key not in classes,"Repeated historical class row")
        classes[key]=row
    for row in csv_rows(root,tables["resident-inference"]):
        infer_key = row["case"],row["band"]
        wall = json.loads(row["wall_ms"])
        measured = timing_summary(wall,repeats=int(row["repeats"]),blocks=1)
        require(math.isclose(measured["p50_ms"],float(row["p50_ms"]),abs_tol=1e-12)
                and math.isclose(measured["p95_ms"],float(row["p95_ms"]),abs_tol=1e-12),
                "Historical sparse resident timing differs from samples")
        inference[infer_key].append(dict(scope=row["scope"],warmups=int(row["warmups"]),
                                         step=int(row["step"]),physical_gpu=row["physical_gpu"],**measured,
                                         peak_allocated_bytes=int(row["peak_allocated_bytes"]),
                                         peak_reserved_bytes=int(row["peak_reserved_bytes"])))
    for row in csv_rows(root,tables["stage-costs"]):
        stages[row["case"]].append({k:float(row[k]) for k in ("graph_seconds","forward_seconds","transform_seconds",
                                                               "train_update_seconds","validation_seconds")})
    rows = []
    for pin in audit["case_replays"]:
        item = document(root,pin)
        name = item["case"]
        _, seed, factor, arm = name.split("/")
        for endpoint, band in product(audit["endpoints"],audit["bands"]):
            packed = item["head_metrics" if endpoint=="selected" else "last_head_metrics"]
            step = item["selected_step" if endpoint=="selected" else "last_step"]
            heads = {}
            for h in HEADS:
                raw = packed[h][band]
                built = metrics(raw["confusion_matrix"],9,raw["nll"])
                key = name,endpoint,h,band
                exported = endpoints[key]
                same_metrics(built,{k:float(exported[k]) for k in ("oa","aa","kappa","macro_f1")})
                require(built["count"] == int(exported["denominator"]) and step == int(exported["step"])
                        and exported["physical_gpu"] == item["actual_cuda_uuid"],"Historical endpoint identity differs")
                require(all(v["support"]==int(classes[(*key,str(v["class_id"]))]["denominator"])
                            and math.isclose(v["recall"],float(classes[(*key,str(v["class_id"]))]["recall"]),abs_tol=1e-12)
                            for v in built["per_class"]), "Historical class denominators differ from raw confusion")
                heads[h] = built
            samples = inference[name,band] if endpoint=="selected" else []
            scope = samples[0]["scope"] if samples else None
            cost = cost_missing("Archive measured synchronized representative blocks for selected models only.")
            if samples:
                require(len(samples)==1 and samples[0]["step"]==step
                        and samples[0]["physical_gpu"]==item["actual_cuda_uuid"],"Cost belongs to a foreign Pavia endpoint")
                cost.update(samples[0], status="measured_representative_block",
                            end_to_end_ms=None, graph_seconds=None, reason="Fixed host block differs from the complete validation roster.")
            cost["training_phase_totals"]={key:math.fsum(r[key] for r in stages[name]) for key in stages[name][0]}
            role="devval_selected_development_protocol"
            rows.append(dict(source="native_pavia",protocol=digest([audit["bank"],audit["point_set"]["devval"]]),
                             scene="paviaU",seed=int(seed),support=512,method="native_released",arm=arm,
                             role=role,band=band,factor=int(factor),endpoint=endpoint,device=item["actual_cuda_uuid"],
                             workload=digest([role,int(factor),band,"representative_host_block"]),
                             measurement_scope=scope, step=step, model=item["selected_model_sha256"] if endpoint=="selected" else None,
                             receipt=pin["sha256"],cost=cost,heads=heads))
    return rows


def band_diagnostics(root):
    plan = document(root,proof(root,"band_registration"))
    require(set(plan["real"])==BAND_PINS.keys() and plan["query_accessed"] is plan["raw_gt_loaded"] is False
            and plan["real_updates"] == 0, "Unexpected frozen band diagnostic plan")
    rows = []
    for case in plan["real"]:
        link = dict(path=str(root/BAND/(case+".json")),sha256=BAND_PINS[case])
        result = document(root,link)
        scene = result["scene"]
        require(result["case"]==case and result["role"]=="devval" and result["real_updates"]==0
                and result["raw_gt_loaded"] is result["query_accessed"] is False,
                "Frozen band result must retain published validation provenance")
        for band,factor in product(plan["bands"],plan["factors"]):
            slots = [x for x in result["rows"] if x["event"] == dict(band=band,factor=factor,policy="native")]
            require([c for r in slots for c in r["labels"]]==result["labels"] and bool(slots),
                    "Incomplete or reordered released diagnostic population")
            heads={}
            for h in HEADS:
                matrix,nll=readout(((r["labels"],r["probabilities"][h]) for r in slots),SCENES[scene])
                heads[h]=metrics(matrix,SCENES[scene],nll/len(result["labels"]))
            cost = dict(status="measured_observation_trace",scope="native_single_forward_per_distinct_released_block",
                        timings_per_distinct_block=[{k:r["computation"][k] for k in
                            ("build_seconds","elapsed_seconds","forward_seconds")} for r in slots],
                        graph_seconds=math.fsum(r["computation"]["build_seconds"] for r in slots),
                        materialize_seconds=math.fsum(r["timings"]["materialize_seconds"] for r in slots),
                        p50_ms=None,p95_ms=None,peak_allocated_bytes=None,end_to_end_ms=None,
                        warmups=None,repeats=None,reason="Single observations on distinct blocks do not form timing repetitions.")
            rows.append(dict(source="wp06_frozen_band_diagnostics",protocol=digest([result["preflight_population"],plan["focus"][scene]]),
                             scene=scene,seed=result["model_seed"],support=512,method="frozen_native",
                             arm=result["architecture"], role="devval_released_focus_only",band=band,
                             factor=factor,endpoint="frozen_selected",device="cpu_archived_diagnostics",
                             workload=digest([result["preflight_population"],band,factor,[r["tile"] for r in slots]]),
                             measurement_scope=cost["scope"],model=result["model"],receipt=link["sha256"],cost=cost,heads=heads))
    return rows


def p4_partial(root):
    link=proof(root,"p4")
    reviewed=document(root,link)
    supplement=document(root,proof(root,"p4_supplement"))
    require(supplement["p4"]["review"]==link and reviewed["status"]=="partial"
            and reviewed["raw_gt_loaded"] is reviewed["query_accessed"] is False
            and reviewed["audit_optimizer_updates"]==0 and len(reviewed["cases"])==4,
            "Foreign five-arm partial development evidence")
    rows=[]
    for item in reviewed["cases"]:
        cfg=item["case"]
        paired=item["paired"]
        heads={}
        for h, raw in paired["heads"].items():
            built=metrics(raw["confusion"],22,raw["nll"])
            same_metrics(built,raw)
            require([entry["recall"] for entry in built["per_class"]]==raw["recall"],
                    "Partial WP05 released recalls differ from confusion")
            heads[h]=built
        cost=cost_missing("Only case-audit time was recorded here. No comparable inference benchmark for this arm and cohort.")
        cost["audit_seconds"]=item["audit_wall_seconds"]
        rows.append(dict(source="wp05_five_arm_partial",protocol=digest([reviewed["release"],cfg["pair_sha256"]]),
                         scene="whu-honghu",seed=cfg["model_seed"],support=cfg["support_seed"],
                         method=cfg["method"],arm=cfg["arm"],role="devval_selected_development_protocol",
                         band="archived_registered_observation",factor=None,endpoint="selected",device=cfg["physical_uuid"],
                         workload=cfg["observation_sha256"],measurement_scope=None,model=paired["state"],
                         receipt=link["sha256"],cost=cost,heads=heads))
    return rows, {"registered_arms":["none","anchor_only","binary","continuous","shuffled"],
                  "observed_arms":sorted({r["arm"] for r in rows}),"missing_arms":{"shuffled":"No accepted same-contract historical case"},
                  "historical_training_spans":reviewed["timing"],"complete_true_comparison":False}


def select(rows, **criteria):
    """Find rows by exact scene, seed, protocol, model, device or other published identity."""
    if rows and criteria.keys()-rows[0].keys():
        raise ValueError("Unknown result-index filter")
    return [row for row in rows if all(row[field]==value for field,value in criteria.items())]


def cohorts(rows):
    grouped=defaultdict(list)
    for row in rows:
        key=case_key(row)
        cohort=key[:3]+key[4:]
        grouped[cohort].append(row)
    output=[]
    for key, items in sorted(grouped.items(),key=lambda pair:str(pair[0])):
        seeds=sorted(row["seed"] for row in items)
        require(len(seeds)==len(set(seeds)),"Cohort duplicates the same model seed")
        first=items[0]
        fields=("source","protocol","scene","role","band","factor","method","arm","support","device","endpoint","measurement_scope")
        for h in HEADS:
            values={field:[item["heads"][h][field] for item in items] for field in ("oa","aa","kappa","macro_f1","count")}
            matrices=[item["heads"][h]["per_class"] for item in items]
            require(all([x["support"] for x in matrix]==[x["support"] for x in matrices[0]] for matrix in matrices),
                    "Paired seed cohort changed the class denominators")
            output.append(dict(**{field:first[field] for field in fields},head=h,model_seeds=seeds,
                               **{name:dict(mean=fmean(v) if all(x is not None for x in v) else None,
                                            sample_sd=stdev(v) if len(v)>1 and all(x is not None for x in v) else None)
                                  for name,v in values.items()}))
    return output


def publish(root, out, code):
    frozen, lifecycle = frozen54(root)
    sparse = native_pavia(root)
    diagnostics = band_diagnostics(root)
    partial, gaps = p4_partial(root)
    cases = [*frozen,*sparse,*diagnostics,*partial]
    require(len({case_key(row) for row in cases})==len(cases),"Conflicting registered release case identity")
    summary = cohorts(cases)
    require(not out.exists(),"Use a fresh output directory")
    with TemporaryDirectory(dir=out.parent,prefix=".unified-real-") as tmp:
        stage = Path(tmp)/"publication"
        stage.mkdir()
        _write(stage/"cases.json",cases)
        _write(stage/"cohorts.json",summary)
        with (stage/"real-per-class.csv").open("x",encoding="utf-8",newline="") as file:
            w=csv.writer(file)
            w.writerow(("source","protocol","scene","seed","support","method","arm","role","band","factor",
                        "endpoint","device","class","head","support_n","recall","f1"))
            for row in cases:
                for head,values in row["heads"].items():
                    for c in values["per_class"]:
                        w.writerow((*[row[k] for k in ("source","protocol","scene","seed","support","method","arm","role","band",
                                                      "factor","endpoint","device")],
                                    c["class_id"],head,c["support"],c["recall"],c["f1"]))
        for name,items in (("real-results.csv",cases),("real-seed-summary.csv",summary)):
            with (stage/name).open("x",encoding="utf-8",newline="") as file:
                write_csv(file,name,items)
        files={path.name:hash_file(path) for path in stage.iterdir()}
        require_frozen_code(code)
        receipt=_write(stage/"receipt.json",dict(schema=SCHEMA,status="published",producer=code["commit"],
            files=files,qualified_historical_joint_cases=len(frozen),native_development_rows=len(sparse),
            bounded_diagnostic_rows=len(diagnostics),partial_five_arm_rows=len(partial),coverage=gaps,
            cost_lifecycle=lifecycle,formal_independent_accuracy_rows=0,new_query_reads=0,new_optimizer_updates=0))
        os.rename(stage,out)
    return dict(path=str(out/"receipt.json"),sha256=receipt["sha256"])


def write_csv(file, name, items):
    identity=("source","protocol","scene","method","arm","role","band","factor","endpoint","device","measurement_scope")
    metrics_fields=("oa","aa","kappa","macro_f1","count")
    w=csv.writer(file)
    if name=="real-results.csv":
        w.writerow((*identity,"seed","support_seed","model","head",*metrics_fields,"cost_status",
                    "p50_ms","p95_ms","peak_allocated_bytes","graph_seconds","end_to_end_ms","cost_reason"))
        for row in items:
            cost=row["cost"]
            for head,values in row["heads"].items():
                w.writerow((*[row[k] for k in identity],row["seed"],row["support"],row["model"],head,
                            *[values[k] for k in metrics_fields],
                            *[cost.get(k) for k in ("status","p50_ms","p95_ms","peak_allocated_bytes","graph_seconds","end_to_end_ms","reason")]))
    else:
        w.writerow((*identity,"head","seeds",*(f"{m}_{stat}" for m in metrics_fields for stat in ("mean","sample_sd"))))
        for row in items:
            w.writerow((*[row[k] for k in identity],row["head"]," ".join(map(str,row["model_seeds"])),
                        *(row[m][stat] for m in metrics_fields for stat in ("mean","sample_sd"))))
