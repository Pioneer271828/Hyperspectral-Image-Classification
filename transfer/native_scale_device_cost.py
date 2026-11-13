"""Bounded real development cost and same-weight numerical comparisons."""

import io
from time import perf_counter

import numpy as np
import torch

from transfer import development_bank as bank, development_bank_custody as custody
from transfer import p4_cuda_qualification as qual, p4_runtime_device as p4
from transfer import registered_development as transport
from transfer.development_block_probe import VIEWS
from transfer.development_blocks import read_pin
from transfer.g3_training import state_hash
from transfer.native_scale_campaign import read
from transfer.native_scale_device import arithmetic
from transfer.native_scale_device_campaign import SCHEMA, _original
from transfer.native_scale_observation import FACTORS, forward, from_bank, head_hashes
from transfer.spatial_multiscale import initialize


WARMUP, REPEATS = 2,5
COST_ARMS = ("off","content","joint")


def _summary(samples, pixels):
    values = np.array(samples,dtype=np.float64)
    return dict(p50_seconds=float(np.median(values)),p95_seconds=float(np.percentile(values,95)),
                native_pixels_per_second=float(pixels/np.median(values)),samples=values.tolist())


def _probe(opened,row,band,factor,model,execution,producer):
    per = []
    memory = execution.device.type=="cuda"
    if memory:
        torch.cuda.reset_peak_memory_stats(execution.device)
        initial_bytes = torch.cuda.memory_allocated(execution.device)
    for repeat in range(WARMUP+REPEATS):
        execution.sync()
        tick = perf_counter()
        actual,_ = from_bank(opened,row["role"],row["id"],VIEWS[band],factor,producer)
        materialize = perf_counter()-tick
        phases = {}
        out,graph = forward(model,actual,producer,row["role"],timed=phases,execution=execution)
        execution.sync()
        total = perf_counter()-tick
        if repeat>=WARMUP:
            building = sum(phases["graph_seconds"])
            network = phases["network_with_graph_seconds"]
            per.append(dict(materialization=materialize,input_preparation=phases["input_prepare_seconds"],
                            host_to_device=phases["transfer_seconds"],graph=building,
                            network_without_graph=max(0.,network-building),network_with_graph=network,
                            end_to_end_with_trace=total))
    pixels = actual.receipt["pixels"]
    arrays = {name:p.detach().cpu().numpy() for name,p in zip(("fused","graph","cnn"),out[:3],strict=True)}
    arrays.update({key:data for name,pair in zip(("q","a0"),execution.graphs,strict=True)
                   for key,data in ((name+"_indices",pair[0]),(name+"_values",pair[1]))})
    peaks = None if not memory else dict(initial_allocated_bytes=initial_bytes,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(execution.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(execution.device))
    return dict(observation=actual.receipt,heads=head_hashes(out,actual),graph=graph,
                timings={key:_summary([x[key] for x in per],pixels) for key in per[0]},peaks=peaks), arrays


@torch.no_grad()
def cost(root,geometry,out,producer,protocol,uuid):
    roster = read(geometry,root)
    if roster["schema"]!=SCHEMA or roster["producer"]!=producer:
        raise ValueError("Native cost geometry identity differs")
    out.mkdir()
    refs, failures = [], []
    with arithmetic(protocol,uuid) as execution:
        execution.capture_values = True
        for scene,item in roster["scenes"].items():
            prior_pin,prior = _original(root,scene)
            if roster["banks"][scene]!={"pin":prior["bank"],"prior":prior_pin}:
                raise ValueError("Native cost source bank differs")
            intake = custody.PROVIDERS[scene](root)
            opened = bank.Bank(prior["bank"],root,intake.release)
            original = intake.baseline
            baseline_state = state_hash(original)
            arms = ("off",) if scene=="paviaU" else COST_ARMS
            models = {arm:initialize(original,arm,7321).to(execution.device) for arm in arms}
            initial = {arm:state_hash(model) for arm,model in models.items()}
            rows = opened.rows
            by_tile = {(row["context"]["tile"],row["view"]):row for row in prior["rows"]}
            tiles = sorted(rows) if scene=="paviaU" else item["representatives"]
            for tile in tiles:
                bands = ("full",) if scene=="paviaU" and tile not in item["representatives"] else VIEWS
                for band in bands:
                    for factor in FACTORS:
                        for arm,model in models.items():
                            row,arrays = _probe(opened,rows[tile],band,factor,model,execution,producer)
                            old = by_tile.get((tile,band))
                            if uuid is None and factor==1 and old and arm in old["probabilities"]:
                                if row["heads"]!=old["probabilities"][arm]:
                                    failures.append([scene,tile,band,factor,arm,"historical_probability"])
                            name = f"{scene}-{tile}-{band}-{factor}-{arm}.npz"
                            with (out/name).open("xb") as stream:
                                np.savez(stream,**arrays)
                            refs.append(dict(scene=scene,tile=tile,band=band,factor=factor,arm=arm,
                                             arrays=transport._pin(out/name),**row))
            opened.verify_authority(prior["bank"])
            if (state_hash(original)!=baseline_state or custody.protect(intake)<1
                    or any(state_hash(models[arm])!=state for arm,state in initial.items())):
                raise ValueError("Native cost modified a frozen model or source artifact")
    result = transport._json_new(out/"cost.json",dict(schema=SCHEMA,producer=producer,geometry=geometry,
        numeric=execution.numeric,kernels=dict(execution.calls),rows=refs,warmup=WARMUP,repeats=REPEATS,
        failures=failures,scope_restored=True,real_updates=0,raw_gt=0,query=0,new_labels=0,
        timing_contract="host materialization and transfer; synchronized network/graph; final trace D2H in end-to-end"))
    if failures:
        raise ValueError("Factor 1 CPU failed historical real equivalence")
    return result


def _arrays(pin,root):
    with np.load(io.BytesIO(read_pin(pin,root)),allow_pickle=False) as arrays:
        return {name:arrays[name] for name in arrays.files}


def compare_cost(root,cpu_pin,gpu_pin,out,producer):
    cpu,gpu = [read(p,root) for p in (cpu_pin,gpu_pin)]
    if (cpu["numeric"]["protocol"],gpu["numeric"]["protocol"])!=(p4.CPU,p4.CUDA_FP64_WEIGHT):
        raise ValueError("Native cost comparison requires CPU and strict CUDA")
    if (cpu["geometry"]!=gpu["geometry"] or cpu["producer"]!=producer or gpu["producer"]!=producer
            or cpu["failures"] or gpu["failures"]):
        raise ValueError("Native CPU/CUDA comparison scope differs")
    rows,failures = [], []
    fields = ("scene","tile","band","factor","arm","observation")
    for a,b in zip(cpu["rows"],gpu["rows"],strict=True):
        if any(a[k]!=b[k] for k in fields):
            raise ValueError("Native cost observations differ")
        x,y = _arrays(a["arrays"],root),_arrays(b["arrays"],root)
        comparisons = {key:qual.compare(y[key],x[key]) for key in ("fused","graph","cnn","q_values","a0_values")
                       if x[key].shape==y[key].shape}
        class_changes = {key:int((x[key].argmax(1)!=y[key].argmax(1)).sum()) for key in ("fused","graph","cnn")}
        structure = all(np.array_equal(x[key],y[key]) for key in ("q_indices","a0_indices"))
        if (not structure or len(comparisons)!=5 or any(class_changes.values())
                or not all(item["compatible"] for item in comparisons.values())):
            failures.append([a[k] for k in fields[:-1]])
        rows.append(dict(identity=[a[k] for k in fields[:-1]],comparisons=comparisons,
                         class_changes=class_changes,sparse_structure_exact=structure))
    return transport._json_new(out,dict(schema=SCHEMA,producer=producer,cpu=cpu_pin,gpu=gpu_pin,rows=rows,
        failures=failures,tolerances=dict(rtol=qual.RTOL,atol=qual.ATOL),real_updates=0,query=0))
