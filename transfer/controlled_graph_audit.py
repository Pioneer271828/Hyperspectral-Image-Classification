"""Verify trajectory replay, freeze all selected checkpoints and open confirmation once."""

from collections import Counter
import math
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import torch

from evaluation.controlled_signal_learning import (SEEDS, VALIDATION_STEPS, Trajectory, adjacency, confirmation,
    geometry, predict, score_readout, streams, topology)
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_training import CurriculumRun
from transfer.registered_controlled_graph import MAXIMUM, STAGES, _notice, _validate_state, intervals, packet, receipt, registration
from transfer.xuzhou_multiscale_audit import _exact


def _phase_proofs(root,pin,note,reg):
    phase={}
    for group in ("main","fixed_final"):
        expected={c["key"] for c in reg["cases"] if (c["mode"]=="fixed_final")==(group=="fixed_final")}
        for stage in STAGES:
            key=f"{group}-{stage}"
            witness=dev._pin(root/"phase-receipts"/f"{key}.json")
            row=admission.read(witness)
            if (row["registration"]!=pin or row["notice"]!=note or row["stage"]!=stage
                    or row["group"]!=group or set(row["cases"])!=expected
                    or any(row[k] for k in ("real_updates","query","gt"))):
                raise ValueError("Wrong exact phase roster or identity")
            phase[key]=(witness,row)
    return phase


def _history(reg,case,value):
    rows=value["history"]
    chosen=min((r for r in rows if r["validation"] is not None),key=lambda r:r["validation"]["mse"])
    if (chosen["step"]!=value["selected"]["step"] or chosen["validation"]!=value["selected"]["quality"]
            or tuple(r["step"] for r in rows if r["validation"] is not None)!=VALIDATION_STEPS[case["kind"]]):
        raise ValueError("Frozen first-min validation selection differs")
    if not all(any(r["gradient"][name]>0 and r["changes"][name]>0 for r in rows) for name in
               ("hgc1.weight","hgc2.weight")):
        raise ValueError("Both graph layers must show actual updates")
    events={v["step"]:v for v in value["graph_events"]}
    if 1 not in events or len(events)!=len(value["graph_events"]):
        raise ValueError("Support events must begin at step one and remain unique")
    data=streams(case["seed"])["training"]
    active=None
    for i,row in enumerate(rows,1):
        active=events.get(i,active)
        pairs=active["support"]
        if (row["step"]!=i or row["sample_sha256"]!=tensor_hash(data[i-1]) or row["nnz"]!=len(pairs)
                or row["support_sha256"]!=digest(pairs) or row["refresh"]!=active["refresh"]
                or row["budget"]!=active["budget"]
                or row["budget"] is not None and max(Counter(r for r,_ in pairs).values())>row["budget"]):
            raise ValueError("Actual support, sample flow or row budget differs")
    if case["kind"]=="signal" and any(e["support"]!=reg["static_arms"][case["key"]]["pairs"] for e in events.values()):
        raise ValueError("Fixed causal comparison changed support")
    if case["kind"]=="evolution" and not any(r["gradient"]["score.weight"]>0 and r["changes"]["score.weight"]>0 for r in rows):
        raise ValueError("Feature scoring path had no actual gradient and update")
    return events[max(step for step in events if step<=chosen["step"])]["support"]


def freeze(pin,note,code):
    root,reg=registration(pin,code)
    _notice(root,pin,note)
    phases=_phase_proofs(root,pin,note,reg)
    out=root/"frozen-models"
    out.mkdir()
    frozen,counts={},Counter()
    for case in reg["cases"]:
        key=case["key"]
        group="fixed_final" if case["mode"]=="fixed_final" else "main"
        proofs,statuses,packets={},{},{}
        for stage,(_,stop,previous) in intervals(case).items():
            witness,row=receipt(root,pin,case,stage,proofs.get(previous))
            if phases[f"{group}-{stage}"][1]["cases"][key]!=witness:
                raise ValueError("Case status and phase receipt differ")
            proofs[stage],statuses[stage]=witness,row
            packets[stage]=packet(row)
            _validate_state(packets[stage],case,stop)
            counts[case["kind"]]+=row["updates"]
        full=packets["full"]
        prefix=packets["prefix"]
        if (not all(_exact(full,packets[s]) for s in ("resume","completed"))
                or not _exact(full["history"][:7],prefix["history"])):
            raise ValueError("Continuous and independent resumed trajectory differ")
        pairs=_history(reg,case,full)
        if pairs!=full["selected"]["support"]:
            raise ValueError("Selected support is not the actual support at its step")
        if group=="fixed_final":
            donor=key.removesuffix("fixed_final")+"regenerate"
            if donor not in frozen or frozen[donor]["last_support"]!=full["fixed"]:
                raise ValueError("Fixed-final retraining did not use its completed same-seed donor")
        cp=out/f"{key}.pt"
        cp_hash=CurriculumRun._write(full["selected_state"],cp)
        frozen[key]=dict(case=case,selected=full["selected"],last_support=statuses["full"]["final_support"],
                         selected_weights=dict(path=str(cp),sha256=cp_hash),attempts=proofs,
                         initial=full["initial"],donor=statuses["full"]["donor"])
    if dict(counts)!=MAXIMUM:
        raise ValueError("Actual synthetic optimizer totals differ")
    for seed in SEEDS:
        rng=[v["initial"] for v in frozen.values() if v["case"]["seed"]==seed]
        if len(set(rng))!=1:
            raise ValueError("Paired controlled model initialization differs")
        for mode in ("regenerate","random_regenerate"):
            name=f"evolution-bottleneck-{seed}-{mode}"
            if name not in frozen:
                raise ValueError("Both regeneration controls required")
        adaptive=packet(admission.read(frozen[f"evolution-bottleneck-{seed}-regenerate"]["attempts"]["full"]))
        random=packet(admission.read(frozen[f"evolution-bottleneck-{seed}-random_regenerate"]["attempts"]["full"]))
        if any(a["nnz"]!=b["nnz"] or a["budget"]!=b["budget"] for a,b in zip(adaptive["history"],random["history"],strict=True)):
            raise ValueError("Random regeneration must match the actual edge budget at each update")
    require_frozen_code(code)
    return dev._json_new(root/"frozen.json",dict(registration=pin,notice=note,commit=code["commit"],
        phases={k:v[0] for k,v in phases.items()},models=frozen,actual=dict(counts),completed_reentry_updates=0,
        confirmation_accesses=0,optimizer_updates=0,real_updates=0,gt=0,query=0))


def _before_confirmation(root,pin,note,frozen,reg,code):
    f=admission.read(frozen)
    expected={c["key"]:c for c in reg["cases"]}
    if (frozen["path"]!=str(root/"frozen.json") or f["registration"]!=pin or f["notice"]!=note
            or f["commit"]!=code["commit"] or f["actual"]!=MAXIMUM or f["confirmation_accesses"]!=0
            or set(f["models"])!=set(expected)
            or any(v["case"]!=expected[k] for k,v in f["models"].items())):
        raise ValueError("Confirmation requires all exact frozen selected models")
    for item in f["models"].values():
        dev._read(item["selected_weights"])
    return f


@torch.no_grad()
def _check_dependency(run,a,top,pairs):
    g=geometry(top,pairs)
    t=torch.tensor([-.875,.875],dtype=torch.float32)
    p=predict(run.graph,a,top,t)
    perm=list(reversed(range(top["nodes"])))
    moved=predict(run.graph,a,top,t,permutation=perm)
    if not torch.allclose(p,moved,atol=1e-6,rtol=1e-6):
        raise ValueError("Permutation changed the two-stage computation")
    reachable=g["distance"] is not None and g["distance"]<=2
    if not reachable and p[0]!=p[1]:
        raise ValueError("Unreachable source leaked to the target")
    return dict(geometry=g,reachable=reachable,remote_delta=float(p[1]-p[0]),
                permutation_max_delta=float((p-moved).abs().max()))


def audit(pin,note,frozen,code):
    started=perf_counter()
    root,reg=registration(pin,code)
    _notice(root,pin,note)
    bound=_before_confirmation(root,pin,note,frozen,reg,code)
    claim=dev._json_new(root/"confirmation-claim.json",dict(registration=pin,frozen=frozen,once=True))
    results={}
    for key,item in bound["models"].items():
        case=item["case"]
        stages={s:admission.read(p) for s,p in item["attempts"].items()}
        full=packet(stages["full"])
        run=Trajectory(case,fixed=full["fixed"])
        state=torch.load(Path(item["selected_weights"]["path"]),weights_only=True,map_location="cpu")
        if not _exact(state,full["selected_state"]):
            raise ValueError("Frozen selected weight differs")
        run.restore_weights(state)
        selected=item["selected"]
        val=run.streams["validation"]
        pairs=selected["support"]
        with torch.no_grad():
            a=run.matrix_at(selected["step"],pairs=pairs)
            prediction=predict(run.graph,a,run.top,val).tolist()
            if prediction!=selected["predictions"] or score_readout(val,prediction)!=selected["quality"]:
                raise ValueError("Frozen validation replay differs")
            evidence=_check_dependency(run,a,run.top,pairs)
            actual=confirmation(case["seed"])
            estimated=predict(run.graph,a,run.top,actual).tolist()
        dense=a.to_dense()
        results[key]=dict(case=case,selected=selected,validation_signal=val.tolist(),confirmation_signal=actual.tolist(),
            confirmation_predictions=estimated,confirmation_quality=score_readout(actual,estimated),
            selected_support=pairs,selected_values=a.values().tolist(),last_support=item["last_support"],
            graph_events=full["graph_events"],history=full["history"],dependency=evidence,
            phase_costs={s:dict(cost=v["cost"],wall_seconds=v["wall_seconds"],updates=v["updates"]) for s,v in stages.items()},
            confirmed_once=True, dense_row_max_error=float((dense.sum(1)-1).abs().max()))
        print(f"{key} frozen confirmation complete",flush=True)
    summaries={}
    for row in results.values():
        c=row["case"]
        summaries.setdefault("/".join((c["kind"],c["topology"],c["mode"])),{})[str(c["seed"])]=row["confirmation_quality"]
    aggregate={key:{m:dict(mean=mean(v[m] for v in group.values()),sample_sd=stdev(v[m] for v in group.values()))
                    for m in ("mse","mae","sign_accuracy")} for key,group in summaries.items()}
    require_frozen_code(code)
    return dev._json_new(root/"audit.json",dict(status="passed_controlled_synthetic_graph",registration=pin,frozen=frozen,
        claim=claim,commit=code["commit"],cases=results,aggregate=aggregate,actual=bound["actual"],
        confirmation_cases=len(results),optimizer_updates=0,real_updates=0,query=0,gt=0,seconds=perf_counter()-started))
