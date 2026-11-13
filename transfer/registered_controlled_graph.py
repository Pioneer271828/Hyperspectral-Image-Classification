"""Registration and process-separated optimizer intervals for controlled synthetic graphs."""

from copy import deepcopy
import io
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.controlled_signal_learning import (BATCH, EVALUATION, EVOLUTION, LEARNING_RATE, SEEDS, SIGNAL,
    STREAMS, TOPOLOGIES, VALIDATION_STEPS, WIDTH, Trajectory, geometry, support, topology)
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_training import CurriculumRun
from transfer.xuzhou_multiscale_audit import _exact


FORMAT="wp06_wp08_controlled_synthetic_v1"
MAXIMUM={"signal":14400,"evolution":4320}
STAGES=("full","prefix","resume","completed")
RNG_SEED=771


def roster():
    cases=[]
    for kind,tops,modes in (("signal",TOPOLOGIES,SIGNAL),("evolution",("bottleneck",),EVOLUTION)):
        for seed in SEEDS:
            for graph in tops:
                for mode in modes:
                    key=f"{kind}-{graph}-{seed}-{mode}"
                    cases.append(dict(key=key,kind=kind,topology=graph,seed=seed,mode=mode,
                                      budget=200 if kind=="signal" else 120))
    return cases


def runtime():
    if torch.get_num_threads()!=1 or torch.cuda.is_available() or torch.get_default_dtype()!=torch.float32:
        raise ValueError("Registered one-thread CPU FP32 execution required")
    return admission.runtime_identity()


def design():
    cases=roster()
    topologies={name:topology(name) for name in TOPOLOGIES}
    arms={}
    stream_ids={}
    for c in cases:
        if c["kind"]=="signal":
            top=topologies[c["topology"]]
            pairs=support(top,c["mode"],c["seed"])
            arms[c["key"]]=dict(pairs=pairs,**geometry(top,pairs))
        if c["seed"] not in stream_ids:
            from evaluation.controlled_signal_learning import streams
            stream_ids[c["seed"]]={k:tensor_hash(v) for k,v in streams(c["seed"]).items()}
    return dict(format=FORMAT,scope="synthetic_regional_graph_only",cases=cases,topologies=topologies,
                static_arms=arms,stream_hashes={str(k):v for k,v in stream_ids.items()},generator="numpy_SeedSequence_PCG64_uniform_v1",
                generator_roles=STREAMS,learning_rate=LEARNING_RATE,batch=BATCH,evaluation=EVALUATION,width=WIDTH,
                torch_step_seed=RNG_SEED,prefix=7,maximum=MAXIMUM,validation_steps={k:list(v) for k,v in VALIDATION_STEPS.items()},
                selection="first_min_validation_mse_every_tenth_update",confirmation="once_after_all_54_selected_states_frozen",
                real_updates=0,gt=0,query=0)


def register(out,code):
    require_frozen_code(code)
    root=external_directory(out)
    data=design()
    root.mkdir()
    return dev._json_new(root/"registration.json",dict(**data,root=str(root),commit=code["commit"],
        source_hash=code["source_hash"],runtime=runtime(),optimizer_updates=0))


def registration(pin,code):
    require_frozen_code(code)
    reg=admission.read(pin)
    root=external_directory(reg["root"])
    if (pin["path"]!=str(root/"registration.json") or reg["commit"]!=code["commit"] or reg["source_hash"]!=code["source_hash"]
            or reg["runtime"]!=runtime() or reg["optimizer_updates"]!=0
            or any(reg[k]!=v for k,v in design().items())):
        raise ValueError("Exact frozen producer, roster, stream and budget required")
    return root,reg


def _notice(root,pin,note):
    actual=admission.read(note)
    if (note["path"]!=str(root/"notice.json") or set(actual)!={"registration","maximum","cpu_workers","estimate"}
            or actual["registration"]!=pin or actual["maximum"]!=MAXIMUM
            or actual["cpu_workers"]!=1 or not actual["estimate"]):
        raise ValueError("Frozen exact budget announcement required")


def case_map(reg):
    return {c["key"]:c for c in reg["cases"]}


def intervals(case):
    budget=case["budget"]
    return dict(full=(0,budget,None),prefix=(0,7,None),resume=(7,budget,"prefix"),completed=(budget,budget,"resume"))


def _root_case(root,key,stage):
    path=root/"cases"/key/stage
    if path.resolve()!=path:
        raise ValueError("Stage cannot use a redirected path")
    return path


def receipt(root,pin,case,stage,prior):
    start,stop,_=intervals(case)[stage]
    path=_root_case(root,case["key"],stage)
    status=dev._pin(path/"status.json")
    value=admission.read(status)
    expected=dict(registration=pin,case=case,stage=stage,start=start,stop=stop,prior=prior)
    if (value["status"]!="passed" or value["claim"]["path"]!=str(path/"claim.json")
            or admission.read(value["claim"])!=expected or value["updates"]!=stop-start
            or value["packet"]["path"]!=str(path/"checkpoint.pt")):
        raise ValueError("Stage receipt, optimizer interval or claim differs")
    return status,value


def packet(status):
    return torch.load(io.BytesIO(dev._read(status["packet"])),weights_only=True,map_location="cpu")


def _fixed(root,pin,case):
    if case["mode"]!="fixed_final":
        return None,None
    donor_key=f"evolution-{case['topology']}-{case['seed']}-regenerate"
    donor=case|{"mode":"regenerate","key":donor_key}
    chain={}
    rows={}
    for stage,(_,_,previous) in intervals(donor).items():
        chain[stage],rows[stage]=receipt(root,pin,donor,stage,chain.get(previous))
    full=packet(rows["full"])
    if not all(_exact(full,packet(rows[s])) for s in ("resume","completed")):
        raise ValueError("Final graph donor must have passed exact continuous/resumed completion")
    route=full["route"]
    if route["step"]!=120 or rows["full"]["final_support"]!=route["selected"]:
        raise ValueError("Final graph donor support differs")
    return route["selected"],chain


def _validate_state(value,case,expected):
    if (value["case"]!=case or len(value["history"])!=expected or not torch.is_tensor(value["torch_rng"])
            or value["torch_rng"].dtype!=torch.uint8 or not all(math.isfinite(row["loss"]) for row in value["history"])):
        raise ValueError("Checkpoint case, steps or values differ")
    groups=value["optimizer"]["param_groups"]
    slots=value["optimizer"]["state"]
    if (len(groups)!=1 or groups[0]["lr"]!=LEARNING_RATE or set(slots)!=set(groups[0]["params"])
            or any(set(slot)!={"step","exp_avg","exp_avg_sq"} or slot["step"].item()!=expected
                   or not all(torch.isfinite(v).all() for v in slot.values()) or (slot["exp_avg_sq"]<0).any() for slot in slots.values())):
        raise ValueError("Restored Adam steps or moments differ")
    states=[value["weights"]["graph"]]
    if value["selected_state"] is not None:
        states.append(value["selected_state"]["graph"])
    if any(any(t.is_floating_point() and not torch.isfinite(t).all() for t in x.values()) for x in states):
        raise ValueError("Nonfinite recovered model state")
    for x in states:
        for name,v in x.items():
            if name.endswith("running_mean") and torch.count_nonzero(v):
                raise ValueError("BatchNorm running mean changed")
            if name.endswith("running_var") and not torch.equal(v,torch.ones_like(v)):
                raise ValueError("BatchNorm running variance changed")
            if name.endswith("num_batches_tracked") and v.item()!=0:
                raise ValueError("BatchNorm training leaked into the task")


def execute_case(root,reg,pin,case,stage):
    start,stop,previous=intervals(case)[stage]
    key=case["key"]
    path=_root_case(root,key,stage)
    if path.exists():
        raise FileExistsError("Previously claimed optimizer interval")
    prior=None
    value=None
    if previous is not None:
        earlier=intervals(case)[previous][2]
        witness=None
        if earlier is not None:
            witness,_=receipt(root,pin,case,earlier,None)
        prior,old=receipt(root,pin,case,previous,witness)
        value=packet(old)
        _validate_state(value,case,start)
    fixed,donor=_fixed(root,pin,case)
    begin=perf_counter()
    trajectory=Trajectory(case,fixed=fixed)
    if value is not None:
        trajectory.restore(value)
    else:
        torch.manual_seed(RNG_SEED+case["seed"])
    path.mkdir(parents=True)
    claim=dev._json_new(path/"claim.json",dict(registration=pin,case=case,stage=stage,start=start,stop=stop,prior=prior))
    try:
        for _ in range(start,stop):
            trajectory.step()
        result=trajectory.packet()
        _validate_state(result,case,stop)
        sha=CurriculumRun._write(result,path/"checkpoint.pt")
        return dev._json_new(path/"status.json",dict(status="passed",claim=claim,updates=stop-start,
            packet=dict(path=str(path/"checkpoint.pt"),sha256=sha),final_support=trajectory.support_at(),donor=donor,
            cost=dict(trajectory.timing,**({} if trajectory.router is None else trajectory.router.cost)),
            wall_seconds=perf_counter()-begin,real_updates=0,query=0,gt=0))
    except BaseException as e:
        path_failure=path/"failed.pt"
        sha=CurriculumRun._write(trajectory.packet(),path_failure)
        dev._json_new(path/"status.json",dict(status="failed",claim=claim,error=str(e),observed_steps=len(trajectory.history),
            packet=dict(path=str(path_failure),sha256=sha),real_updates=0,query=0,gt=0))
        raise


def run_phase(pin,note,code,stage,group):
    root,reg=registration(pin,code)
    _notice(root,pin,note)
    if stage not in STAGES or group not in ("main","fixed_final"):
        raise ValueError("Invalid optimizer stage or roster")
    log=root/"phase-receipts"
    log.mkdir(exist_ok=True)
    proof=log/f"{group}-{stage}.json"
    if proof.exists():
        raise FileExistsError("Optimizer phase already completed")
    if stage!="full":
        preceding=STAGES[STAGES.index(stage)-1]
        if not (log/f"{group}-{preceding}.json").is_file():
            raise ValueError("Required earlier independent process receipt is missing")
    if group=="fixed_final" and not (log/"main-completed.json").is_file():
        raise ValueError("Freeze evolving donors before fixed-graph retraining")
    started=perf_counter()
    cases={}
    for c in reg["cases"]:
        if (c["mode"]=="fixed_final")!=(group=="fixed_final"):
            continue
        cases[c["key"]]=execute_case(root,reg,pin,c,stage)
        print(f"{group} {stage} {c['key']} complete",flush=True)
    require_frozen_code(code)
    return dev._json_new(proof,dict(registration=pin,notice=note,stage=stage,group=group,cases=cases,
        updates=sum(admission.read(p)["updates"] for p in cases.values()),seconds=perf_counter()-started,
        real_updates=0,query=0,gt=0))
