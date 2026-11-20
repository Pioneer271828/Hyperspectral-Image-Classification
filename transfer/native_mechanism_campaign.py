"""Exact matched support through the shared native training/selection state machine."""

from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, require
from model.research_config import digest
from monitoring.released_region import RegionAction
from research_runtime import require_frozen_code, write_json
from scripts.run_frozen_band_region import _write
from transfer import native_mechanism_edges as edge, native_scale_training as native
from transfer.g3_training import state_hash
from transfer.native_coverage_queue import case_lock
from transfer.native_long_run import packet
from transfer.xuzhou_multiscale_audit import _exact

SCHEMA = edge.SCHEMA
GROUPS = tuple(f"{task}-{seed}" for task,seed in product(edge.TASKS,edge.SEEDS))
REFERENCE = "remote-64"
PHASES = ("prefix","resume","completed","full")
HISTORY = ("wp05-wp06-band-matched-graph-20261119/wp12/receipt.json",
           "7de4ce5e1cf5e73156b87ca5f47f2c085c793d1fee3bf968074dc1cfb7b378fd")
GATES = dict(positive_min_aa=.70,erased_gap_max_aa=.53,rare_added_errors=0,aggregate_delta=-.005,
             ordinary_recall_delta=-.02)


def design():
    return {f"{task}-{seed}":dict(task=task,seed=seed) for task,seed in product(edge.TASKS,edge.SEEDS)}


def archive(root):
    prior=json_pin(root,dict(path=str(root/HISTORY[0]),sha256=HISTORY[1]))
    require(len(prior["cases"])==18,"Historical matched graph cohort differs")
    return dict(sha256=HISTORY[1],cases=18,use="reuse_archived_band_graph_cases_without_training_replay",
                evidence=prior["artifacts"])


def register(root,out,code):
    out.mkdir(parents=True,exist_ok=True)
    path=out/"registration.json"
    if path.exists(): return load(root,out,code["commit"])[1]
    inputs={}
    for name,case in design().items():
        bank,roles,boundary=edge.publish(out/"banks"/name,**case)
        inputs[name]=dict(bank=bank.pin,roles=roles,geometry=digest(boundary))
    value=dict(schema=SCHEMA,producer=code["commit"],cases=design(),arms=list(edge.ARMS),budget=edge.STEPS,
               views=list(edge.VIEWS),support_fit=inputs,gates=GATES,region_action="replace_or_zero_only_during_confirmation",
               optimizer_updates_at_registration=0,historical=archive(root),
               scopes="synthetic_only",real_updates=0,query_accessed=False,raw_gt_loaded=False)
    require_frozen_code(code)
    return _write(path,value)


def load(root,out,producer):
    link=file_pin(out/"registration.json")
    value=json_pin(root,link)
    require(value["schema"]==SCHEMA and value["producer"]==producer and value["cases"]==design()
            and value["arms"]==list(edge.ARMS) and value["budget"]==edge.STEPS and value["gates"]==GATES
            and value["views"]==list(edge.VIEWS) and value["scopes"]=="synthetic_only"
            and value["real_updates"]==value["optimizer_updates_at_registration"]==0
            and value["query_accessed"] is value["raw_gt_loaded"] is False,"Synthetic edge registration differs")
    return value,link


def provider(root,out,plan,name,arm,producer):
    case=plan["cases"][name]
    bank,roles,witness=edge.publish(out/"banks"/name,**case)
    require(plan["support_fit"][name]==dict(bank=bank.pin,roles=roles,geometry=digest(witness)),
            "Synthetic narrow target release or spatial role differs")
    return edge.EdgeProvider(root,bank,roles,case["seed"],arm,producer),witness


def fit(root,out,code,name,execution):
    require(name in GROUPS,"Unknown exact registered edge group")
    plan,link=load(root,out,code["commit"])
    group=out/"cases"/name
    group.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for arm in edge.ARMS:
        p,_=provider(root,out,plan,name,arm,code["commit"])
        place=group/arm.replace(":","-")
        with case_lock(place):
            receipt=place/"receipt.json"
            if receipt.exists():
                previous=json_pin(root,file_pin(receipt))
                require(previous["registration"]==link["sha256"] and previous["edge_arm"]==arm,
                        "Complete edge case belongs to a different arm")
                outputs.append(dict(arm=arm,updates_this_invocation=0,receipt=file_pin(receipt)))
                continue
            spec=p.contract(9,p.seed,p.config.arm,1,p.producer,numeric=execution.numeric)
            registered=place/"contract.json"
            if registered.exists():
                require(json_pin(root,file_pin(registered))==spec,"Edge recovery cannot change device or support")
            else:
                _write(registered,spec)
            phases=PHASES if name==REFERENCE and arm=="fixed_none" else ("full",)
            previous=None
            trained=0
            artifacts={}
            for phase in phases:
                state=place/(phase+".pt")
                if not state.exists():
                    result=native.execute(spec,phase,state,p.producer,
                               prior=previous if phase in ("resume","completed") else None,execution=execution,released=p)
                    _write(place/(phase+".json"),result)
                    trained+=result["updates"]
                checkpoint=file_pin(state)
                end=packet(root,checkpoint)
                require(end["registration"]==digest(spec),"Native graph state registration differs")
                artifacts[phase]=checkpoint
                if phase in ("prefix","resume"): previous=end
            if phases==PHASES:
                require(_exact(end,packet(root,artifacts["resume"])) and _exact(end,packet(root,artifacts["completed"])),
                        "Matched graph optimizer/RNG restoration differed")
            p.validate_history(spec,end)
            require(end["cursor"]==len(end["history"])==edge.STEPS
                    and all(row["supervised_area"]>0 for row in end["history"]),
                    "Exact graph arm has incomplete effective updates")
            best=min(end["validation"],key=lambda row:row["fused_nll"])
            require(best["step"]==end["selected"]["step"] and best["fused_nll"]==end["selected"]["fused_nll"],
                    "Native first minimum calibration selection differs")
            require_frozen_code(code)
            data=dict(schema=SCHEMA,registration=link["sha256"],case=name,edge_arm=arm,producer=p.producer,
                contract=file_pin(registered),states=artifacts,selected=best["step"],device=execution.numeric,
                updates=edge.STEPS,recovery_exact=phases==PHASES,query_accessed=False,raw_gt_loaded=False,real_updates=0)
            outputs.append(dict(arm=arm,updates_this_invocation=trained,receipt=_write(receipt,data)))
    return outputs


def restored(root,out,plan,link,name,arm,producer,device):
    p,witness=provider(root,out,plan,name,arm,producer)
    receipt=json_pin(root,file_pin(out/"cases"/name/arm.replace(":","-")/"receipt.json"))
    require(receipt["schema"]==SCHEMA and receipt["registration"]==link["sha256"]
            and receipt["case"]==name and receipt["edge_arm"]==arm and receipt["producer"]==producer
            and receipt["updates"]==edge.STEPS and receipt["real_updates"]==0
            and receipt["raw_gt_loaded"] is receipt["query_accessed"] is False,"Exact graph checkpoint scope changed")
    spec=json_pin(root,receipt["contract"])
    require(spec==p.contract(9,p.seed,p.config.arm,1,p.producer,numeric=receipt["device"]),
            "Independent edge reader reconstructed different support or schedule")
    state=packet(root,receipt["states"]["full"])
    p.validate_history(spec,state)
    best=min(state["validation"],key=lambda row:row["fused_nll"])
    require(state["registration"]==digest(spec) and state["cursor"]==edge.STEPS and
            state["selected"]["step"]==best["step"]==receipt["selected"],
            "Independent edge reader found a changed selection or budget")
    if receipt["recovery_exact"]:
        require(all(_exact(state,packet(root,receipt["states"][x])) for x in ("resume","completed")),
                "The saved exact recovery witness is invalid")
    model=p.model_factory(p.baseline(p.seed),p.config.arm,p.seed+7321)
    model.load_state_dict(state["selected"]["state"],strict=True)
    return p,witness,receipt,spec,model.to(device).eval()


@torch.inference_mode()
def evaluate_tile(p,model,witness,receipt,tile,band,execution,*,arm,operation=None,repeats=2):
    execution.sync()
    result=None
    times=[]
    for _ in range(repeats+1):
        if execution.device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(execution.device)
        start=perf_counter()
        actual,_,targets=p.confirmation(receipt["states"]["full"],tile,1,band)
        prep=perf_counter()-start
        outputs,graph,timing=p.registered_forward(model,actual,"devval",execution,edge=arm,region=operation)
        points={head:actual.lattice.project(pred).detach().cpu().numpy()[targets.cells.tolist()].tolist()
                for head,pred in zip(HEADS,outputs[:3],strict=True)}
        execution.sync()
        elapsed=perf_counter()-start
        proof=p.control(actual)
        row=dict(truth=[y for _,y in targets.points],band=band,block=tile,arms=arm,
            heads=points,graph=graph,regions=proof.topology,grid=actual.receipt["grid_shape"],
            spectral=actual.receipt["spectral"],ids=list(actual.view.bands.band_ids),boundary=witness[str(tile)],
            target_digest=targets.summary()["point_set"],geometry_hash=proof.fingerprint)
        if result is None:
            result=row
        else:
            require(row==result,"Repeated matched graph inference changed predictions")
            times.append(dict(total_seconds=elapsed,prepare_seconds=prep,forward_seconds=timing["forward_seconds"],
                              graph_seconds=sum(timing["graph_seconds"]),
                              peak_cuda_bytes=torch.cuda.max_memory_allocated(execution.device)
                              if execution.device.type=="cuda" else None))
    return result|dict(times=times)


def confirm(root,out,producer,code,name,execution):
    plan,link=load(root,out,producer)
    results={}
    for arm in edge.ARMS:
        p,geometry,receipt,spec,model=restored(root,out,plan,link,name,arm,producer,execution.device)
        before=state_hash(model)
        require(not set(p.roles["evaluation"])&{row["tile"] for row in (*spec["training"],*spec["validation"])},
                "Confirmation geometry entered optimizer or selection")
        rows=[evaluate_tile(p,model,geometry,receipt,tile,band,execution,arm=arm)
              for tile,band in product(p.roles["evaluation"],edge.VIEWS)]
        frozen,regions=[],[]
        if arm=="fixed_none":
            for tile in p.roles["evaluation"]:
                for trial in edge.ARMS[1:]:
                    frozen.append(evaluate_tile(p,model,geometry,receipt,tile,edge.BAND,execution,arm=trial))
                for operation in (RegionAction("replace",0,1),RegionAction("replace",0,2),RegionAction("zero",0)):
                    regions.append(evaluate_tile(p,model,geometry,receipt,tile,edge.BAND,execution,arm=arm,operation=operation)
                                   |dict(operation=operation.operation,target=operation.target,donor=operation.donor))
        require(state_hash(model)==before,"Independent edge inference altered frozen weights")
        results[arm]=dict(selected=receipt["selected"],state=before,initial=spec["initialization"]["state"],
            training=digest(spec["training"]),validation=digest(spec["validation"]),manifest=digest(spec["inputs"]),
            degree_policy=arm,train_device=receipt["device"],budget=edge.STEPS,checkpoint=receipt["states"]["full"]["sha256"],
            recovery=receipt["recovery_exact"],rows=rows,frozen_deletions=frozen,region_replacements=regions)
    require(len({row["initial"] for row in results.values()})==len({row["training"] for row in results.values()})
            ==len({row["validation"] for row in results.values()})==len({row["manifest"] for row in results.values()})==1,
            "Matched graph arms changed unregistered training budget, support or initial model")
    require_frozen_code(code)
    return dict(schema=SCHEMA,scope="independent_synthetic_confirmation",case=name,producer=producer,
                consumer=code["commit"],registration=link["sha256"],arms=results,device=execution.numeric,
                real_updates=0,independent_real_accuracy=None,query_accessed=False,raw_gt_loaded=False)
