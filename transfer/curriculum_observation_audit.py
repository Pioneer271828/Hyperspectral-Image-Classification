"""Reconcile executed CUR-O schedules and replay all frozen endpoints by condition."""
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compression.precision_recovery_audit import HEADS, _guard, save_package
from compression.registered_precision_recovery import stages
from evaluation.classification_metrics import metrics_from_confusion
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_observation_admission import audited_forward
from transfer.curriculum_observation_real import MeasuredObservationRun, load_registration
from transfer.curriculum_support_review import _validation_members
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import read_stage
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact, _packet, _verify_history


def observed_exposure(history, plan, policy, scene, record, cache):
    labels = dict(zip(record["train_support"]["indices"], record["train_support"]["labels"], strict=True))
    expected = policy.to_dict()["training"]
    configurations = {}
    bands_total = pixels_total = 0
    records = []
    for step, (row, frozen) in enumerate(zip(history, expected, strict=True), 1):
        event = plan.event(step)
        evidence = row["forward_evidence"]
        tile = scene.plan["tiles"][event["tile_id"]]
        bands = len(frozen["selected_positions"])
        contract = row["observation"]
        if (contract != policy.training_record(step) or evidence["band_count"] != bands
                or evidence["band_ids"] != frozen["selected_band_ids"] or evidence["metadata_sha256"] != frozen["selected_metadata_hash"]
                or evidence["calls"] != {"adapter":1,"denoise":1,"graph":1} or len(evidence["three_heads"]) != 3
                or any(value != 1. for value in row["weights"])):
            raise ValueError("Observed graph/adapter path, band record or unweighted CUR-O loss differs")
        identity = (event["tile_id"], frozen["record_hash"])
        if identity not in cache:
            view = policy.view(scene, record, tile, phase="training", step=step)
            cache[identity] = tensor_hash(view.observation.spectral.copy())
        if evidence["spectral_sha256"] != cache[identity]:
            raise ValueError("Executed selected-band input differs from the admitted original bytes")
        key = digest(frozen["config"])
        if key not in configurations:
            configurations[key] = dict(config=frozen["config"], forwards=0, band_exposures=0, pixels_times_bands=0,
                                       tiles=Counter(), support=Counter(), class_blocks=Counter(), class_label_visits=Counter())
        y0,y1,x0,x1 = tile["box"]
        pixel_bands = (y1-y0)*(x1-x0)*bands
        cfg = configurations[key]
        cfg["forwards"] += 1
        cfg["band_exposures"] += bands
        cfg["pixels_times_bands"] += pixel_bands
        cfg["tiles"][str(tile["id"])] += 1
        ids = event["support_ids"]
        cfg["support"].update(map(str, ids))
        cfg["class_blocks"].update({str(labels[i]) for i in ids})
        cfg["class_label_visits"].update(str(labels[i]) for i in ids)
        records.append(frozen["record_hash"])
        bands_total += bands
        pixels_total += pixel_bands
    return dict(by_config=configurations, train_band_exposures=bands_total, train_pixel_band_exposures=pixels_total,
                record_multiset=digest(sorted(records)), train_forwards=len(history),
                train_label_visits=sum(sum(c["support"].values()) for c in configurations.values()))


@torch.no_grad()
def endpoint(model, scene, record, policy, condition):
    started = perf_counter()
    width = scene.plan["shape"][1]
    members = _validation_members(scene, record)
    per_tile = {}
    for index,label,tile in members:
        per_tile.setdefault(tile,[]).append((index,label))
    matrices = {h: np.zeros((scene.classes,scene.classes), dtype=np.int64) for h in HEADS}
    nlls = Counter()
    targets, predictions, scores = [], {h:[] for h in HEADS}, []
    frozen = policy.to_dict()["validation"][condition]
    for tile_id, rows in per_tile.items():
        tile = scene.plan["tiles"][tile_id]
        y0,_,x0,x1 = tile["box"]
        local = [(i//width-y0)*(x1-x0)+i%width-x0 for i,_ in rows]
        target = np.asarray([label-1 for _,label in rows])
        targets.extend((target+1).tolist())
        view = policy.view(scene, record, tile, phase="validation", condition=condition)
        outputs, evidence = audited_forward(model, scene, record, tile, view, frozen, phase="validation")
        scores.append(evidence["three_heads"])
        for h, tensor in zip(HEADS, outputs[:3], strict=True):
            p = tensor.detach().cpu().numpy()[local]
            choices = p.argmax(axis=1)
            for a,b in zip(target, choices, strict=True):
                matrices[h][a,b] += 1
            predictions[h].extend((choices+1).tolist())
            nlls[h] -= float(np.log(np.maximum(p[np.arange(len(target)),target],1e-8)).sum())
    metrics = {}
    for h in HEADS:
        quality = metrics_from_confusion(matrices[h], record["classes"])
        quality.update(nll=nlls[h]/len(targets),zero_recall=[r["class_id"] for r in quality["per_class"] if r["recall"] == 0.])
        metrics[h] = quality
    return dict(metrics=metrics, predictions=predictions, targets=targets, scores=scores, forwards=len(per_tile),
                labels=len(targets), condition_record=frozen["record_hash"], seconds=perf_counter()-started)


def _pack(source, weights, path):
    model = deepcopy(source).eval()
    model.load_state_dict(weights, strict=True)
    result = save_package(model, "fp32", path)
    result["state"] = state_hash(model)
    return result


def audit(pin, completion, code):
    started = perf_counter()
    root, reg, source, scene, record, plans, trials = load_registration(pin, code)
    batch = admission.read(completion)
    if (completion["path"] != str(root/"batch-complete.json") or batch["registration"] != pin
            or batch["updates"] != reg["maximum_updates"] or any(batch[k] for k in ("optimizer_updates","query","gt"))):
        raise ValueError("Exact observation completion required")
    old = admission.read(reg["prior"])
    output = root/"endpoints"
    output.mkdir()
    intervals = stages(reg["budget"], reg["prefix"])
    cases, seen_inputs = {}, {}
    updates, stage_events = 0, 0
    for name, (case, policy) in trials.items():
        receipts, packets, statuses = {}, {}, {}
        for stage, (_, stop, previous) in intervals.items():
            s,r = read_stage(root,pin,name,stage,receipts.get(previous),arms=trials,stages=intervals)
            if batch["attempts"][name][stage] != r:
                raise ValueError("Execution interval differs from the batch receipt")
            packets[stage] = _packet(s,reg["contracts"][name],stop,packet_format=MeasuredObservationRun.FORMAT)
            receipts[stage],statuses[stage] = r,s
        full = packets["full"]
        if (not all(_exact(full[k], packets[other][k]) for k in SCIENCE for other in ("resume","completed"))
                or not _exact(full["history"][:reg["prefix"]],packets["prefix"]["history"])):
            raise ValueError("Exact CUR-O continuation failed")
        history = full["history"]
        plan = plans[str(case["seed"])]
        chosen = _verify_history(full, plan)
        exposure = observed_exposure(history,plan,policy,scene,record,seen_inputs)
        if any(exposure[k] != old["coverage"][name][k] for k in exposure if k != "train_label_visits"):
            raise ValueError("Actual joint exposure differs from the frozen plan")
        home = output/name
        home.mkdir()
        points = {}
        for point,key,step in (("selected","selected_state",chosen),("last","state",reg["budget"])):
            package = _pack(source,full[key],home/f"{point}.pt")
            if package["state"] != history[step-1]["state_hash"]:
                raise ValueError("Frozen checkpoint state differs from its executed event")
            points[point] = dict(step=step,package=package)
        count = sum(v["new_updates"] for v in statuses.values())
        events = sum(v["timing"]["full_v5_event"]["calls"] for v in statuses.values())
        updates += count
        stage_events += events
        cases[name] = dict(case=case,updates=count,history=history,exposure=exposure,v5_stage_events=events,**points,
            stage_costs={s:{k:v[k] for k in ("timing","new_updates","restore_or_prepare_seconds","update_loop_seconds",
                "checkpoint_finish_seconds")} for s,v in statuses.items()})
    if updates != reg["maximum_updates"]:
        raise ValueError("Actual budget differs from registered observation campaign")
    frozen_selected = dev._json_new(root/"frozen-selected.json",dict(registration=pin,completion=completion,
        packages={name:row["selected"] for name,row in cases.items()}, optimizer_updates=0))
    # No condition sweep runs until all 12 selected states have been committed.
    conditions = reg["validation_conditions"]
    exemplar = next(iter(trials.values()))[1]
    validation_started = perf_counter()
    initial = {name:endpoint(source,scene,record,exemplar,name) for name in conditions}
    original = reg["source_replay"]
    if initial["full"]["scores"] != original["scores"] or initial["full"]["metrics"] != original["metrics"]:
        raise ValueError("Full condition does not replay the original frozen source")
    for name,row in cases.items():
        policy = trials[name][1]
        for point in ("selected","last"):
            entry = row[point]
            package = entry["package"]
            model = dev._load_source(package["weights"],package["sidecar"])
            if state_hash(model) != package["state"]:
                raise ValueError("Frozen endpoint package changed")
            entry["conditions"] = {cond:endpoint(model,scene,record,policy,cond) for cond in conditions}
            fused = entry["conditions"]["full"]["metrics"]["fused"]
            selected = row["history"][entry["step"]-1]["validation"]
            if (fused["confusion_matrix"] != selected["confusion"] or fused["oa"] != selected["oa"]
                    or not math.isclose(fused["nll"],selected["nll"],abs_tol=1e-7)):
                raise ValueError("Frozen full-condition replay differs from the selection event")
        print(f"{name}: audited actual bands, exact resume and all frozen conditions",flush=True)
    support = Counter(r["label"] for r in reg["train_support"])
    small = sorted(c for c,n in support.items() if n <= reg["protection"]["small_support_max"])
    guards = {}
    for name,row in cases.items():
        baseline = cases[f"s{row['case']['seed']}-fixed"]
        guards[name] = {point:{cond:_guard(baseline[point]["conditions"][cond],row[point]["conditions"][cond],record["classes"],small)
                        for cond in conditions} for point in ("selected","last")}
    admission.check_pins([pin,completion,frozen_selected,*reg["protected"]])
    require_frozen_code(code)
    summaries = [v for row in cases.values() for p in ("selected","last") for v in row[p]["conditions"].values()]
    val_blocks = len({t for _,_,t in _validation_members(scene,record)})
    return dev._json_new(root/"audit.json",dict(status="passed_independent_real_observation",registration=pin,
        completion=completion,frozen_selected=frozen_selected,source=initial,cases=cases,guards=guards,small_classes=small,
        actual_updates=updates,actual_training_forwards=updates,
        actual_train_labels=2*sum(r["exposure"]["train_label_visits"] for r in cases.values()),
        training_v5_events=updates,restore_v5_events=stage_events-updates,v5_blocks_per_event=val_blocks,
        labels_per_v5_event=len(reg["validation"]["indices"]),
        source_condition_forwards=sum(v["forwards"] for v in initial.values()),source_condition_labels=sum(v["labels"] for v in initial.values()),
        endpoint_condition_forwards=sum(v["forwards"] for v in summaries),endpoint_condition_labels=sum(v["labels"] for v in summaries),
        condition_review_seconds=perf_counter()-validation_started,independent_input_replays=len(seen_inputs),
        audit_seconds=perf_counter()-started,optimizer_updates=0,query=0,gt=0))
