"""Bounded native-scale arithmetic registration, recovery and audit."""

from collections import Counter
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from joint_training import _run_lock
from model.research_config import digest
from transfer import native_scale_training as train, p4_cuda_qualification as qual, p4_runtime_device as p4
from transfer import registered_development as transport
from transfer.development_blocks import array_hash
from transfer.native_scale_campaign import packet, read
from transfer.native_scale_device import arithmetic, input_key
from transfer.native_scale_fixture import Fixture
from transfer.native_scale_observation import FACTORS
from transfer.spatial_multiscale import ARMS
from transfer.xuzhou_multiscale_audit import _exact
from scripts.run_spatial_multiscale import BANK_PRODUCER, NICK, REAL


SCHEMA = "wp07-native-device-engineering-v1"
PHASES = ("full","prefix","resume","completed")
DEVICES = (p4.CPU,p4.CUDA_FP64_WEIGHT)
EXPECTED_UPDATES = 1152


def cases():
    return tuple(f"{classes}-{seed}-{arm}" for classes in (9,22) for seed in train.SEEDS for arm in ARMS)


def _original(root,scene):
    ref = dict(path=str(root/f"wp02-bank-review-fix01/{NICK[scene]}-replay06.json"),sha256=REAL[scene])
    prior = read(ref,root)
    if prior["scene"]!=scene or prior["producer"]!=BANK_PRODUCER or prior["real_updates"]:
        raise ValueError("Real development bank has the wrong origin")
    return ref,prior


def geometry(root,out,producer):
    shapes = {(1,1),(1,7),(7,1),(3,5)}
    banks, scenes = {}, {}
    for scene in REAL:
        ref,prior = _original(root,scene)
        pin = prior["bank"]
        root_ = Path(pin["root"])
        registration = read(dict(path=str(root_/"registration.json"),sha256=pin["registration"]),root)
        roster = read(dict(path=str(root_/"bank.json"),sha256=pin["sha256"]),root)
        released = {member["id"] for member in roster["members"]}
        rows = [r for r in registration["rows"] if r["id"] in released]
        if len(rows)!=sum(entry["publishable_blocks"] for entry in prior["coverage"].values()):
            raise ValueError("Real native shape roster differs")
        groups = Counter()
        for row in rows:
            y0,y1,x0,x1 = row["box"]
            for f in FACTORS:
                shape = ((y1-y0+f-1)//f,(x1-x0+f-1)//f)
                shapes.add(shape)
                groups[(row["role"],f,*shape)] += 1
        banks[scene] = dict(pin=pin,prior=ref)
        scenes[scene] = dict(blocks=len(rows),shapes=[[role,f,h,w,n] for (role,f,h,w),n in sorted(groups.items())],
                             representatives=prior["representatives"])
    return transport._json_new(out,dict(schema=SCHEMA,producer=producer,banks=banks,scenes=scenes,
                              shapes=[list(x) for x in sorted(shapes)],thin_controls=[[1,1],[1,7],[7,1],[3,5]]))


def _operator_rows(shapes,target):
    rows = {}
    rng = np.random.default_rng(qual.FIXTURE_SEED)
    # Both native synthetic stride 4 and real strides sqrt(50)/10.
    for stride in (4,50**.5,10):
        for h,w in shapes:
            # The shared qualification builder expects integer cell indices.
            # Derive the exact real grid first, then reuse its COO builder.
            gh,gw = ceil(h/stride),ceil(w/stride)
            q = qual.neighbor_indices(h,w,gh,gw,1)
            a = qual.neighbor_indices(gh,gw,gh,gw,2)
            n,m = h*w,gh*gw
            grid = (gh,gw)
            sparse = (("assignment",q,(n,m),64),("assignment_transposed",q[::-1].copy(),(m,n),64),
                      ("adjacency64",a,(m,m),64),("adjacency128",a,(m,m),128))
            key = f"{stride:g}/{h}x{w}"
            rows[f"{key}/adaptive_pool"] = qual.evaluate_pool(rng,h,w,64,grid,target)
            for name,indices,shape,channels in sparse:
                rows[f"{key}/{name}"] = qual.evaluate_sparse(rng,indices,shape,channels,target,p4.CUDA_FP64_WEIGHT)
    for name,indices,shape,channels in qual.adversarial_fixtures():
        rows[name] = qual.evaluate_sparse(rng,indices,shape,channels,target,p4.CUDA_FP64_WEIGHT)
    return rows


def qualify(root,geometry_pin,out,producer,protocol,uuid):
    roster = read(geometry_pin,root)
    if roster["producer"]!=producer or roster["schema"]!=SCHEMA:
        raise ValueError("Qualification geometry has the wrong producer")
    started = perf_counter()
    with arithmetic(protocol,uuid) as execution:
        rows = _operator_rows(roster["shapes"],execution.device.type)
    expected = dict(adaptive_avg_pool2d=3*len(roster["shapes"]),sparse_mm=12*len(roster["shapes"])+2)
    failed = [name for name,row in rows.items() if not qual.check_comparisons(name,row)]
    if execution.device.type=="cuda" and dict(execution.calls)!=expected:
        failed.append("interception_count")
    result = transport._json_new(out,dict(schema=SCHEMA,producer=producer,geometry=geometry_pin,
        numeric=execution.numeric,scope_restored=True,tolerances=dict(rtol=qual.RTOL,atol=qual.ATOL),
        fixture_count=len(rows),rows=rows,failures=failed,kernels=dict(execution.calls),seconds=perf_counter()-started,
        updates=0,real_updates=0))
    if failed:
        raise ValueError(f"Native operator tolerance failed; retained {result['path']}: {failed}")
    return result


def _frozen_inputs(specs,producer):
    inputs, streams = {}, {}
    for spec in specs:
        c = spec["classes"]
        for role,events in (("devtrain",spec["training"]),("devval",spec["validation"])):
            if (c,role) not in streams:
                streams[c,role] = Fixture(c,role)
            for event in events:
                key = input_key(c,role,event)
                if key in inputs:
                    continue
                actual,labels,targets = streams[c,role].materialize(event["tile"],event["factor"],event["band"],
                                                                     event["profile"],producer)
                inputs[key] = dict(observation=actual.receipt["observation"],labels=array_hash(labels),
                                   area=int(targets.area.sum()),histogram=array_hash(targets.histogram))
    return inputs


def register(root,qualification,out,producer,protocol,uuid):
    qa = read(qualification,root)
    if qa["producer"]!=producer or qa["schema"]!=SCHEMA or qa["failures"]:
        raise ValueError("A passing native geometry qualification is required")
    with arithmetic(protocol,uuid) as execution:
        if execution.numeric!=qa["numeric"]:
            raise ValueError("Qualification device/environment differs")
        raw = [(name,train.contract(int(c),int(s),arm,producer,numeric=execution.numeric))
               for name in cases() for c,s,arm in (name.split("-"),)]
        inputs = _frozen_inputs([spec for _,spec in raw],producer)
        numeric = dict(execution.numeric,input_roster=digest(inputs))
        out.mkdir()
        hashes = {name:transport._json_new(out/f"{name}.json",dict(spec,numeric=numeric)) for name,spec in raw}
        return transport._json_new(out/"matrix.json",dict(schema=SCHEMA,producer=producer,
            qualification=qualification,numeric=numeric,inputs=inputs,cases=hashes,updates=EXPECTED_UPDATES,
            real_updates=0,confirmation_reads=0))


def matrix(root,pin,producer):
    result = read(pin,root)
    if (result["schema"]!=SCHEMA or result["producer"]!=producer or set(result["cases"])!=set(cases())
            or digest(result["inputs"])!=result["numeric"]["input_roster"]
            or (result["updates"],result["real_updates"],result["confirmation_reads"])!=(EXPECTED_UPDATES,0,0)):
        raise ValueError("Native device matrix registration differs")
    return result


def _admit(execution,reg):
    if execution.numeric!=reg["numeric"]:
        raise ValueError("Native registered device, protocol, environment or GPU UUID differs")


def stage(root,pin,case,phase,producer,protocol,uuid):
    reg = matrix(root,pin,producer)
    spec_pin = reg["cases"][case]
    spec = read(spec_pin,root)
    folder = Path(pin["path"]).parent/case
    with arithmetic(protocol,uuid,inputs=reg["inputs"]) as execution:
        _admit(execution,reg)
        folder.mkdir(exist_ok=True)
        with _run_lock(folder):
            parent = {"resume":"prefix","completed":"resume"}.get(phase)
            prior = None
            if parent:
                previous = read(transport._pin(folder/f"{parent}.json"),root)
                if (previous["case"],previous["phase"],previous["registration"],previous["numeric"])!=(
                        case,parent,spec_pin,reg["numeric"]):
                    raise ValueError("Native numerical predecessor differs")
                prior = packet(previous["checkpoint"],root)
            out = folder/f"{phase}.pt"
            info = train.execute(spec,phase,out,producer,prior,execution=execution)
            if uuid is not None and (bool(info["updates"])!=all(execution.calls.values())):
                raise ValueError("CUDA update stage did not exercise both registered kernel families")
            info.update(case=case,registration=spec_pin,checkpoint=transport._pin(out),numeric=execution.numeric,
                        kernel_calls=dict(execution.calls))
            return transport._json_new(folder/f"{phase}.json",info)


def _read_stages(root,pin,case,spec_pin,numeric):
    folder = Path(pin["path"]).parent/case
    rows = [read(transport._pin(folder/f"{phase}.json"),root) for phase in PHASES]
    if any((row["case"],row["phase"],row["registration"],row["numeric"])!=(case,phase,spec_pin,numeric)
           for row,phase in zip(rows,PHASES,strict=True)):
        raise ValueError("Stage evidence does not match the numerical contract")
    return rows


@torch.no_grad()
def audit(root,pin,out,producer,protocol,uuid):
    reg = matrix(root,pin,producer)
    paired, result = {}, {}
    started = perf_counter()
    with arithmetic(protocol,uuid,inputs=reg["inputs"]) as execution:
        _admit(execution,reg)
        for case,spec_pin in reg["cases"].items():
            spec = read(spec_pin,root)
            rows = _read_stages(root,pin,case,spec_pin,reg["numeric"])
            full,prefix,resumed,complete = [packet(row["checkpoint"],root) for row in rows]
            if ([row["updates"] for row in rows]!=[12,5,7,0] or not _exact(full,resumed)
                    or not _exact(full,complete) or not _exact(full["history"][:5],prefix["history"])
                    or not _exact(full["validation"][:1],prefix["validation"])):
                raise ValueError(f"Same-device native continuation failed for {case}")
            execution.validate_history(spec,full,12)
            run = train.Run(spec,producer,execution=execution)
            first = min(full["validation"],key=lambda row:row["fused_nll"])
            if first["step"]!=full["selected"]["step"]:
                raise ValueError("Native numerical first-min selection differs")
            schedule = [(r["observation"],r["labels"],r["histogram"],r["area"]) for r in full["history"]]
            if schedule!=paired.setdefault((spec["classes"],spec["seed"]),schedule):
                raise ValueError("Native paired input exposure differs")
            for weights,event in ((full["selected"]["state"],first),(full["model"],full["validation"][-1])):
                run.model.load_state_dict(weights,strict=True)
                if not _exact(run.evaluate(event["step"]),event):
                    raise ValueError(f"Reloaded three-head endpoint differs for {case}")
            result[case] = dict(stages=rows,history=full["history"],validation=full["validation"],
                                selected=first["step"],last=12,exact=True)
    return transport._json_new(out,dict(schema=SCHEMA,producer=producer,matrix=pin,numeric=reg["numeric"],
        cases=result,updates=EXPECTED_UPDATES,endpoint_replays=96,kernels=dict(execution.calls),
        scope_restored=True,seconds=perf_counter()-started,real_updates=0,confirmation_reads=0))
