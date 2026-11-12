"""Join an authenticated real factorial to the existing real production rows."""

from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from evaluation.classification_metrics import classification_metrics
from evaluation.wp12_low_rank_evidence import verify_guard
from evaluation.wp12_production_evidence import FULL_HEADS, metrics, missing, provenance
from model.research_config import digest
from research_runtime import hash_file
from transfer.curriculum_factorial import ARMS
from transfer.curriculum_factorial_audit import effects
from transfer.curriculum_factorial_real import BUDGET, CASES, FORMAT, NEW_UPDATES, REFERENCED_UPDATES
from transfer.curriculum_factorial_real_audit import BASELINES, METRICS, POINTS
from transfer.curriculum_observation_admission import CONDITIONS
from transfer.curriculum_real import SEEDS
from transfer.registered_wp11_learning import read


def _receipts(spec):
    if set(spec) != {"registration","audit","diagnostic"}:
        raise ValueError("Real factorial registration, audit and joint diagnostics required")
    reg, audit, diagnosis = (read(spec[k]) for k in ("registration","audit","diagnostic"))
    complete, run, frozen, controls = (read(p) for p in (audit["completion"],audit["campaign"],audit["frozen"],reg["controls"]))
    teacher = read(diagnosis["teacher"])
    cases = {c["case"]:c for c in CASES}
    fresh = {c["case"] for c in reg["cases"]}
    if (reg["format"] != FORMAT or reg["all_cases"] != list(CASES) or reg["budget"] != BUDGET
            or reg["comparison_updates"] != NEW_UPDATES+REFERENCED_UPDATES
            or reg["maximum_updates"] != NEW_UPDATES or reg["historical_control_updates"] != REFERENCED_UPDATES
            or reg["historical_batches"] != {"sample":6408,"observation":4272}
            or reg["historical_full_rerun_proposal"]["status"] != "NOT_RUN"
            or audit["status"] != "passed_real_mixed_factorial" or audit["producer"] != reg["commit"]
            or audit["registration"] != spec["registration"] or audit["new_updates"] != NEW_UPDATES
            or audit["cases"].keys() != cases.keys() or audit["guards"].keys() != fresh
            or complete["registration"] != spec["registration"] or complete["updates"] != NEW_UPDATES
            or complete["attempts"].keys() != fresh or run["new_updates"] != NEW_UPDATES
            or run["producer"] != reg["commit"] or run["registration"] != spec["registration"]["sha256"]
            or run["completion"] != audit["completion"]["sha256"] or run["notice"] != complete["notice"]["sha256"]
            or frozen["registration"] != spec["registration"] or frozen["completion"] != audit["completion"]
            or frozen["cases"].keys() != cases.keys() or frozen["optimizer_updates"] != 0
            or controls["status"] != "passed" or controls["request"] != reg["request"]
            or controls["commit"] != reg["commit"] or controls["historical_updates"] != REFERENCED_UPDATES
            or controls["cases"].keys() != cases.keys()-fresh
            or diagnosis["status"] != "passed_joint_supervision" or diagnosis["audit"] != spec["audit"]
            or diagnosis["cases"].keys() != cases.keys() or teacher["teacher"] != reg["teacher"]
            or teacher["state"] != reg["state"] or teacher["status"] != "passed_frozen_train_teacher"
            or any(v[k] for v in (reg,audit,diagnosis,complete,controls,teacher) for k in ("optimizer_updates","gt","query"))
            or any(run[k] for k in ("gt","query"))):
        raise ValueError("Factorial producer, arm completeness, budget, freeze or role identity differs")
    tags = {f"{name}-{stage}" for name in fresh for stage in ("full","prefix","resume","completed")} | {"complete"}
    if run["status"] != "passed" or set(run["commands"]) != tags:
        raise ValueError("Missing real factorial campaign commands")
    logs = Path(audit["campaign"]["path"]).parent
    for tag, value in run["commands"].items():
        path = logs/f"{tag}.exit.json"
        if (value["exit"] != 0 or not math.isfinite(value["seconds"]) or value["seconds"] <= 0
                or json.loads(path.read_text()) != value):
            raise ValueError("Recorded execution time or subprocess exit differs")
    if not math.isfinite(run["wall_seconds"]) or run["wall_seconds"] < math.fsum(v["seconds"] for v in run["commands"].values()):
        raise ValueError("Invalid sequential campaign wall time")
    return reg, audit, diagnosis, complete, run, frozen, controls, teacher


def _snapshot(value, targets, classes, record_hash):
    if (value["targets"] != targets or value["condition_record"] != record_hash or value["labels"] != len(targets)
            or value["forwards"] != 5 or len(value["blocks"]) != 5 or value["seconds"] <= 0
            or not math.isfinite(value["seconds"])):
        raise ValueError("Wrong fixed factorial validation condition or label order")
    quality = metrics(value["metrics"], classes, FULL_HEADS, matrix_key="confusion_matrix")
    for head, q in quality.items():
        if (classification_metrics(targets, value["predictions"][head], classes)["confusion_matrix"] != q["confusion_matrix"]
                or any(q[k] != value["metrics"][head][k] for k in ("kappa","macro_f1"))
                or not math.isfinite(q["nll"]) or q["nll"] < 0 or len(value["probabilities"][head]) != len(targets)):
            raise ValueError("Factorial predictions or quality differ from the frozen replay")
    return quality


def _stage_cost(case, complete, campaign, labels, blocks):
    amounts = (178,7,171,0)
    result = Counter()
    for stage, budget in zip(("full","prefix","resume","completed"),amounts,strict=True):
        status = read(complete["attempts"][case["case"]["case"]][stage])
        actual = case["stage_costs"][stage]
        if (status["status"] != "passed" or status["new_updates"] != budget
                or any(status[k] != v for k,v in actual.items())
                or actual["timing"].get("training_forward",{}).get("calls",0) != budget
                or actual["timing"]["full_v5"]["calls"] != budget+(2 if stage in ("resume","completed") else 0)):
            raise ValueError("Factorial stage timing does not come from the original registered run")
        for k in ("restore_or_prepare_seconds","update_loop_seconds","checkpoint_finish_seconds"):
            if not math.isfinite(actual[k]) or actual[k] < 0:
                raise ValueError("Factorial stage time must be finite and nonnegative")
            result[k] += actual[k]
        result["subprocess_seconds"] += campaign["commands"][f"{case['case']['case']}-{stage}"]["seconds"]
    visits = 2*sum(len(row["support_ids"]) for row in case["history"])
    return dict(scope="registered_cpu_new_factorial_stages",updates=sum(amounts),label_accesses=visits,
                training_v5_events=sum(amounts),training_v5_forwards=blocks*sum(amounts),training_v5_labels=labels*sum(amounts),
                restore_v5_events=4,restore_v5_forwards=4*blocks,restore_v5_labels=4*labels,**result)


def adapt(spec, existing):
    reg,audit,diagnosis,complete,run,frozen,controls,teacher = _receipts(spec)
    existing_keys = {(r["module"],r["case_id"]):r for r in existing if "case_id" in r}
    original = existing_keys["WP01-CUR-O","original_fp32"]
    truth = audit["source"]["full"]["targets"]
    classes = original["identity"]["class_ids"]
    released = reg["validation"]
    if (Counter(truth) != Counter(released["labels"]) or len(truth) != 45
            or digest(released) != original["identity"]["evaluation_sha256"]
            or original["state"] != reg["state"] or original["package"] != reg["source"]["weights"]["sha256"]
            or original["identity"]["split"] != reg["preprocessing"] or original["original_training_steps"] != 256):
        raise ValueError("Real factorial source or released support differs from prior production evidence")
    for c in CONDITIONS:
        quality = _snapshot(audit["source"][c],truth,classes,reg["validation_identity"][c])
        if quality != original["condition_quality"]["initial"]["conditions"][c]["quality"]:
            raise ValueError("Real factorial source replay differs from the accepted observation table")
    source_proof = dict(source_commit=original["source_producer_commit"],witness_commit=original["source_producer_commit"],
                        source_evidence=original["source_producer_evidence"],source_steps=original["original_training_steps"])
    new, cells, matches = [], {}, {}
    no_cost = missing("NOT_RUN","No registered real factorial deployment cost")
    for c in CASES:
        name, seed, arm = c["case"], c["seed"], c["arm"]
        item = audit["cases"][name]
        history = item["history"]
        diag = diagnosis["cases"][name]
        if (item["case"] != c or len(history) != BUDGET or digest(history) != item["history_sha256"]
                or diag["case"] != c or diag["producer"] != item["producer"] or item["new"] != (arm=="s_plus_o")):
            raise ValueError("Factorial arm, producer or actual history differs")
        if not item["new"]:
            ref = controls["cases"][name]
            previous = existing_keys["WP01-CUR-O" if arm=="o_only" else "WP01-CUR-S",ref["original_case"]]
            if (ref["producer"] != item["producer"] or previous["recovery_producer_commit"] != item["producer"]
                    or ref["history_sha256"] != item["history_sha256"] or ref["historical_updates"] != 2*BUDGET):
                raise ValueError("Referenced historical control was not in the previous table")
            matches[name] = dict(module=previous["module"],case_id=previous["case_id"],producer=ref["producer"],
                                 registration=ref["registration"],audit=ref["audit"],historical_updates=2*BUDGET)
        chosen = min(history,key=lambda row:row["validation"]["nll"])["step"]
        endpoints = {}
        for point, step in (("selected",chosen),("last",BUDGET)):
            entry = item["endpoints"][point]
            witness = frozen["cases"][name][point]
            package = entry["package"]
            if (entry["step"] != step or any(entry[k] != v for k,v in witness.items()) or package["format"] != "fp32"
                    or package["state"] != history[step-1]["state_hash"] or set(entry["conditions"]) != set(CONDITIONS)
                    or any(hash_file(p["path"]) != p["sha256"] for p in (package["weights"],package["sidecar"]))):
                raise ValueError("Wrong factorial selected/last package or freeze")
            conditions = {}
            for condition in CONDITIONS:
                snapshot = entry["conditions"][condition]
                q = _snapshot(snapshot,truth,classes,reg["validation_identity"][condition])
                condition_row = dict(quality=q,condition_record=snapshot["condition_record"],seconds=snapshot["seconds"],
                                     forwards=snapshot["forwards"],labels=snapshot["labels"])
                if item["new"]:
                    guards = {}
                    for base in BASELINES:
                        reference = audit["cases"][f"s{seed}-{base}"]["endpoints"][point]["conditions"][condition]
                        guards[base] = verify_guard(reference,snapshot,classes,audit["small_classes"],reg["protection"]["limits"])
                    if guards != audit["guards"][name][point][condition]:
                        raise ValueError("Real factorial candidate guard differs from a historical reference")
                    condition_row["protection"] = guards
                conditions[condition] = condition_row
            if not item["new"] and point=="selected" and (conditions["full"]["quality"] != previous["quality"]
                    or package["state"] != previous["state"] or package["weights"]["sha256"] != previous["package"]
                    or step != previous["selected_step"]):
                raise ValueError("Historical selected state differs from the previous evidence table")
            endpoints[point] = dict(step=step,package=package,conditions=conditions)
        for p,step in (("initial",0),("selected",chosen),("last",BUDGET),("full_run",BUDGET)):
            value = diag["supervision"][p]
            if value["step"] != step or value["history_sha256"] != digest(history[:step]):
                raise ValueError("Joint supervision cutoff differs from the frozen endpoint")
        cells[name] = dict(case=c,producer=item["producer"],new=item["new"],endpoints=endpoints,
                           supervision=diag["supervision"],errors_from_source=diag["errors_from_source"])
        if not item["new"]:
            continue
        if item["producer"] != reg["commit"]:
            raise ValueError("New candidate does not carry the production commit")
        chosen_row = endpoints["selected"]
        selected_package = chosen_row["package"]
        cost = _stage_cost(item,complete,run,len(truth),5)
        cells[name]["training_cost"] = cost
        lineage = provenance(**source_proof,recovery_commit=reg["commit"],recovery_steps=BUDGET,
                             release_commit=audit["adapter"],release_evidence=spec["audit"])
        new.append(dict(module="WP01-CUR-SxO",arm=arm,case_id=name,
            identity=dict(original["identity"],sampler_seed=seed),**lineage,
            state=selected_package["state"],package=selected_package["weights"]["sha256"],package_bytes=selected_package["bytes"],
            precision="fp32",budget=BUDGET,actual_optimizer_calls=2*BUDGET,selected_step=chosen_row["step"],selection=reg["selection"],
            quality=chosen_row["conditions"]["full"]["quality"],protection=chosen_row["conditions"]["full"]["protection"]["none"],
            cost=no_cost,training_cost=cost,real_factorial_evidence=spec["audit"]))
    contrasts = {}
    for point in POINTS:
        contrasts[point] = {}
        for condition in CONDITIONS:
            contrasts[point][condition] = {}
            for head in FULL_HEADS:
                contrasts[point][condition][head] = {}
                for metric in METRICS:
                    values = {}
                    for seed in SEEDS:
                        values[str(seed)] = effects([
                            dict(arm=arm,source=reg["state"],labels=digest(released),exposure=reg["plan_hashes"][str(seed)],
                                 budget=BUDGET,metric=metric,seed=seed,condition=condition,
                                 value=cells[f"s{seed}-{arm}"]["endpoints"][point]["conditions"][condition]["quality"][head][metric])
                            for arm in ARMS])
                    contrast = {k:dict(mean=mean(d[k] for d in values.values()),sample_sd=stdev(d[k] for d in values.values()))
                                for k in next(iter(values.values()))}
                    contrasts[point][condition][head][metric] = dict(per_seed=values,summary=contrast)
    cost = {k:sum(r["training_cost"][k] for r in new) for k in ("updates","label_accesses","training_v5_events","training_v5_forwards",
        "training_v5_labels","restore_v5_events","restore_v5_forwards","restore_v5_labels","subprocess_seconds",
        "restore_or_prepare_seconds","update_loop_seconds","checkpoint_finish_seconds")}
    if len(new) != 3 or cost["updates"] != NEW_UPDATES:
        raise ValueError("Real factorial table must add only the three new cases")
    return new,dict(**spec,producer=reg["commit"],audit_commit=audit["adapter"],references=matches,
                    source=original["condition_quality"]["initial"],cells=cells,contrasts=contrasts,
                    historical_batches=reg["historical_batches"],historical_control_updates=REFERENCED_UPDATES,
                    new_training_cost=cost,registration_seconds=reg["preparation_seconds"],campaign_seconds=run["wall_seconds"],
                    zero_update_control_authentication=dict(seconds=controls["seconds"],v5_events=sum(
                        a["v5_events"] for c in controls["cases"].values() for a in c["restoration"]),
                        endpoint_sweeps=sum(c["full_endpoint_replays"] for c in controls["cases"].values())),
                    condition_replay={k:audit[k] for k in ("source_forwards","source_label_accesses","endpoint_forwards","endpoint_label_accesses",
                                                        "replay_seconds","seconds")},
                    teacher=dict(original=reg["teacher"],scope="one_shared_frozen_original_teacher",new_policy_scores=0,
                                 diagnostic_forwards=teacher["forwards"],diagnostic_labels=teacher["labels"],diagnostic_seconds=teacher["seconds"]),
                    diagnosis_seconds=diagnosis["seconds"],deployment=no_cost,independent_spatial=missing("NOT_RUN","Fixed repeated development V5"),
                    interpretation="Single source with three recovery RNG repeats; descriptive paired effects",
                    primary_condition="full",optimizer_updates=0,gt=0,query=0)


def write_factorial(out, result):
    with (out/"real-factorial-effects.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("endpoint","condition","head","metric","effect","seed_64","seed_512","seed_2048","mean","sample_sd"))
        for point,conditions in result["contrasts"].items():
            for c,heads in conditions.items():
                for h,by_metric in heads.items():
                    for metric, values in by_metric.items():
                        for effect,summary in values["summary"].items():
                            writer.writerow((point,c,h,metric,effect,*[values["per_seed"][str(s)][effect] for s in SEEDS],
                                             summary["mean"],summary["sample_sd"]))
    with (out/"real-factorial-conditions.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("case","recovery_seed","arm","producer","new","endpoint","step","condition","condition_sha256",
                         "head",*METRICS,"per_class","reference","protection"))
        for name,cell in result["cells"].items():
            case=cell["case"]
            for p,endpoint in cell["endpoints"].items():
                for c,v in endpoint["conditions"].items():
                    for head,q in v["quality"].items():
                        for reference,guards in v.get("protection",{"":{}}).items():
                            writer.writerow((name,case["seed"],case["arm"],cell["producer"],cell["new"],p,endpoint["step"],c,
                                v["condition_record"],head,*[q[k] for k in METRICS],json.dumps(q["per_class"]),
                                reference,json.dumps(guards.get(head))))
