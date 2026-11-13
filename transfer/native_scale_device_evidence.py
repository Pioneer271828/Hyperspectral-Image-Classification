"""Read-only, versioned evidence for native numerical engineering and WP12."""

import csv
import math
from pathlib import Path

import numpy as np
import torch

from research_runtime import hash_file, tensor_hash
from transfer import p4_cuda_qualification as qual, p4_runtime_device as p4
from transfer import registered_development as transport
from transfer.development_block_custody import historical, _protect
from transfer.development_blocks import inside
from transfer.native_scale_campaign import packet, read
from transfer.native_scale_device_campaign import EXPECTED_UPDATES, PHASES, SCHEMA, cases, matrix, operator_specs
from transfer.native_scale_device_cost import WARMUP, REPEATS, _arrays, _summary
from transfer.xuzhou_multiscale_audit import _exact


EVIDENCE = "wp07-native-device-evidence-v2"
HEADS = ("fused", "graph", "cnn")
COST_PHASES = ("materialization","input_preparation","host_to_device","graph","network_without_graph",
               "network_with_graph","end_to_end_with_trace")
IDENTITY = ("scene","tile","band","factor","arm")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def qualification(root, pin, producer, numeric=None, geometry=None):
    value = read(pin,root)
    g = read(value["geometry"],root)
    expected = {key for key,_ in operator_specs(g["shapes"])}
    _require(value["schema"]==g["schema"]==SCHEMA and value["producer"]==g["producer"]==producer
             and (geometry is None or value["geometry"]==geometry),
             "Qualification producer or geometry differs")
    actual = value["numeric"]
    _require(actual["protocol"] in (p4.CPU,p4.CUDA_FP64_WEIGHT)
             and (actual["device"],actual["physical_uuid"] is None)==(
                 ("cpu",True) if actual["protocol"]==p4.CPU else ("cuda:0",False))
             and actual["input_roster"] is None and value["scope_restored"]
             and actual["p4_arithmetic_sha256"]==hash_file(p4.__file__)
             and (numeric is None or dict(numeric,input_roster=None)==actual),
             "Qualification numerical environment differs")
    cuda = actual["protocol"]!=p4.CPU
    _require(not actual["deterministic_warn_only"] and bool(actual["deterministic_algorithms"])==cuda
             and (not cuda or p4.UUID.fullmatch(actual["physical_uuid"]))
             and value["tolerances"]==dict(rtol=qual.RTOL,atol=qual.ATOL)
             and (value["updates"],value["real_updates"])==(0,0),
             "Qualification arithmetic or tolerance differs")
    _require(set(value["rows"])==expected and value["fixture_count"]==len(expected),
             "Qualification fixture roster differs")
    oracle, native, direct = [], [], []
    for key,row in value["rows"].items():
        grads = qual.GRADIENTS["pool" if key.endswith("/adaptive_pool") else "sparse"]
        _require(set(row)=={"forward",*grads},"Qualification outputs are incomplete")
        for comparison in row.values():
            _require(set(comparison)=={"replacement_vs_math64","native_cpu32_vs_math64","replacement_vs_native_cpu32"},
                     "Qualification comparisons are incomplete")
            for item in comparison.values():
                ratio = item["max_tolerance_ratio"]
                _require(type(item["compatible"]) is bool and
                         item["compatible"]==(ratio is not None and math.isfinite(ratio) and 0<=ratio<=1),
                         "Qualification comparison failure bit differs")
        for failed,which in ((oracle,"replacement_vs_math64"),(native,"native_cpu32_vs_math64"),
                             (direct,"replacement_vs_native_cpu32")):
            if any(not item[which]["compatible"] for item in row.values()):
                failed.append(key)
    expected_calls = (dict(adaptive_avg_pool2d=3*len(g["shapes"]),sparse_mm=12*len(g["shapes"])+2)
                      if cuda else dict(adaptive_avg_pool2d=0,sparse_mm=0))
    if value["kernels"]!=expected_calls:
        oracle.append("interception_count")
    _require(set(value["oracle_failures"])==set(oracle)
             and set(value["failures"])==set(oracle+native+direct),"Qualification failure roster differs")
    return dict(pin=pin,geometry=value["geometry"],numeric=actual,operator_math="failed" if oracle else "passed",
                native_CPU_compatibility="failed" if native or direct else "passed",
                math_failures=oracle,native_failures=native,direct_failures=direct,
                all_comparisons_passed=not (oracle or native or direct))


def _check_complete_audit(root, audit, registered, producer):
    _require(audit["producer"]==producer and audit["schema"]==SCHEMA and audit["numeric"]==registered["numeric"]
             and (audit["updates"],audit["endpoint_replays"],audit["real_updates"],audit["confirmation_reads"])
                 ==(EXPECTED_UPDATES,96,0,0)
             and set(audit["cases"])==set(cases()),"Complete device audit requires all 48 cases and 96 endpoint replays")
    for name,case in audit["cases"].items():
        _require(case["exact"] is True and case["last"]==12 and len(case["stages"])==4
                 and [r["phase"] for r in case["stages"]]==list(PHASES)
                 and [r["updates"] for r in case["stages"]]==[12,5,7,0],
                 "Complete device audit has missing stages")
        folder = Path(audit["matrix"]["path"]).parent/name
        for row in case["stages"]:
            source = read(transport._pin(folder/f'{row["phase"]}.json'),root)
            _require(row==source and row["case"]==name and row["numeric"]==registered["numeric"]
                     and row["registration"]==registered["cases"][name],
                     "Device audit stage receipt differs")
            path = inside(row["checkpoint"]["path"],root)
            _require(hash_file(path)==row["checkpoint"]["sha256"],"Device audit checkpoint changed")
    return dict(status="complete",cases=48,endpoint_replays=96,updates=EXPECTED_UPDATES)


def _check_partial(root, value, registered, matrix_pin, producer):
    _require(value["producer"]==producer and value["registration"]==matrix_pin
             and value["numeric"]==registered["numeric"] and value["real_updates"]==value["confirmation_reads"]==0,
             "Partial device audit differs from its registration")
    exact, incomplete, missing = [],[],[]
    updates = 0
    for case in registered["cases"]:
        folder = Path(matrix_pin["path"]).parent/case
        stages = {}
        for phase,budget in zip(PHASES,(12,5,7,0),strict=True):
            path = folder/f"{phase}.json"
            if not path.is_file():
                continue
            row = read(transport._pin(path),root)
            _require(row["numeric"]==registered["numeric"] and row["registration"]==registered["cases"][case]
                     and (row["phase"],row["case"],row["updates"])==(phase,case,budget),
                     "Partial device stage identity differs")
            stages[phase] = packet(row["checkpoint"],root)
            updates += budget
        if len(stages)==4:
            full,prefix,resume,complete = (stages[phase] for phase in PHASES)
            _require(_exact(full,resume) and _exact(full,complete)
                     and _exact(full["history"][:5],prefix["history"])
                     and _exact(full["validation"][:1],prefix["validation"]),"Partial device exact evidence differs")
            exact.append(case)
        elif stages:
            incomplete.append(dict(case=case,phases=list(stages)))
        else:
            missing.append(case)
    _require(value["exact"]==exact and value["partial"]==incomplete and value["unstarted"]==missing
             and value["counts"]["updates"]==updates and value["formal_updates_remaining"]==EXPECTED_UPDATES-updates,
             "Partial device scope differs from completed artifacts")
    return dict(status="partial",cases=len(exact),endpoint_replays=0,updates=updates)


def read_audit(root, pin, producer, *, partial=False):
    record = read(pin,root)
    matrix_pin = record["registration"] if partial else record["matrix"]
    registered = matrix(root,matrix_pin,producer)
    proof = qualification(root,registered["qualification"],producer,registered["numeric"])
    _require(registered["qualification_all_comparisons_passed"] is proof["all_comparisons_passed"],
             "Device matrix hides qualification failure")
    if "qualification" in record:
        _require(record["qualification"]==proof,"Device audit qualification status differs")
    recovery = (_check_partial(root,record,registered,matrix_pin,producer) if partial
                else _check_complete_audit(root,record,registered,producer))
    return dict(source=pin,matrix=matrix_pin,qualification=proof,same_device_recovery=recovery,
                numerical=registered["numeric"],production_authorized=False)


def summarize_cost(samples, pixels):
    _require(type(pixels) is int and pixels>0 and isinstance(samples,list) and len(samples)==REPEATS
             and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in samples)
             and sorted(samples)[REPEATS//2]>0,"Cost samples are empty, incomplete or nonfinite")
    return _summary(samples,pixels)


def read_cost(root,pin,producer,audit):
    cost = read(pin,root)
    _require(cost["schema"]==SCHEMA and cost["producer"]==producer and not cost["failures"]
             and dict(audit["numerical"],input_roster=None)==cost["numeric"]
             and cost["geometry"]==audit["qualification"]["geometry"]
             and (cost["warmup"],cost["repeats"])==(WARMUP,REPEATS)
             and not any(cost[key] for key in ("real_updates","raw_gt","query","new_labels")),
             "Cost scope, protocol, geometry or producer differs")
    geometry = read(cost["geometry"],root)
    expected = set()
    for scene,data in geometry["scenes"].items():
        tiles = data["representatives"]
        if scene=="paviaU":
            # Read the pinned published roster to obtain both existing blocks.
            bank = geometry["banks"][scene]["pin"]
            path = Path(bank["root"])/"bank.json"
            tiles = [row["id"] for row in read(dict(path=str(path),sha256=bank["sha256"]),root)["members"]]
        for tile in tiles:
            bands = ("full",) if scene=="paviaU" and tile not in data["representatives"] else (
                "full","even_retained_positions")
            for band in bands:
                for factor in (1,2,4):
                    for arm in (("off",) if scene=="paviaU" else ("off","content","joint")):
                        expected.add((scene,tile,band,factor,arm))
    found = set()
    for row in cost["rows"]:
        key = tuple(row[k] for k in IDENTITY)
        obs = row["observation"]
        _require(key in expected and key not in found and obs["producer"]==producer
                 and obs["dataset"]==row["scene"] and obs["factor"]==row["factor"]
                 and obs["source"]["tile"]==row["tile"] and set(row["heads"])==set(HEADS)
                 and set(row["timings"])==set(COST_PHASES),"Cost observations or three-head scope differ")
        found.add(key)
        for timing in row["timings"].values():
            computed = summarize_cost(timing["samples"],obs["pixels"])
            _require(computed==timing,"Cost aggregate does not reproduce from samples")
    _require(found==expected,"Cost observation roster is incomplete")
    return cost


def compare(root,cpu,gpu):
    _require((cpu["numeric"]["protocol"],gpu["numeric"]["protocol"])==(p4.CPU,p4.CUDA_FP64_WEIGHT)
             and cpu["producer"]==gpu["producer"],"Cross-device comparison requires paired CPU and strict CUDA")
    a = {tuple(row[k] for k in IDENTITY):row for row in cpu["rows"]}
    b = {tuple(row[k] for k in IDENTITY):row for row in gpu["rows"]}
    _require(set(a)==set(b) and cpu["geometry"]==gpu["geometry"],"Cross-device cost scope differs")
    failures = []
    result = []
    for key,left in a.items():
        right = b[key]
        _require(left["observation"]==right["observation"],"Cross-device cost observation differs")
        x,y = (_arrays(row["arrays"],root) for row in (left,right))
        keys = {*HEADS,"q_indices","q_values","a0_indices","a0_values"}
        _require(set(x)==set(y)==keys,"Cross-device arrays omit a head or sparse operand")
        for row, arrays in ((left,x),(right,y)):
            _require(all(list(arrays[head].shape)==row["heads"][head]["shape"]
                         and tensor_hash(arrays[head])==row["heads"][head]["sha256"] for head in HEADS)
                     and all(tensor_hash(arrays[f"{name}_indices"])==row["graph"][f"{name}_indices"]
                             for name in ("q","a0")),"Cost array hashes differ from graph or probability evidence")
            q_shape = row["graph"]["q_shape"]
            for name,shape in (("q",q_shape),("a0",[q_shape[1],q_shape[1]])):
                graph = torch.sparse_coo_tensor(arrays[f"{name}_indices"],arrays[f"{name}_values"],shape).coalesce()
                _require(tensor_hash(graph)==row["graph"][name],"Cost graph values differ from graph receipt")
        checked = {name:qual.compare(y[name],x[name]) for name in (*HEADS,"q_values","a0_values")
                   if x[name].shape==y[name].shape}
        argmax = {head:int((x[head].argmax(1)!=y[head].argmax(1)).sum()) for head in HEADS}
        sparse = all(np.array_equal(x[name],y[name]) for name in ("q_indices","a0_indices"))
        if not sparse or len(checked)!=5 or any(argmax.values()) or not all(x["compatible"] for x in checked.values()):
            failures.append(key)
        result.append(dict(case=key,tolerance=checked,argmax_changes=argmax,sparse_structure_exact=sparse))
    return dict(status="failed" if failures else "passed",failures=failures,rows=result)


def publish(root,out,producer,consumer,cpu_audit_pin,cpu_cost_pin,*,gpu_audit_pin=None,gpu_cost_pin=None,partial=False):
    cpu = read_audit(root,cpu_audit_pin,producer)
    _require(cpu["numerical"]["protocol"]==p4.CPU,"CPU audit protocol differs")
    cpu_cost = read_cost(root,cpu_cost_pin,producer,cpu)
    gpu = None if gpu_audit_pin is None else read_audit(root,gpu_audit_pin,producer,partial=partial)
    if gpu is not None:
        _require(gpu["numerical"]["protocol"]==p4.CUDA_FP64_WEIGHT
                 and gpu["qualification"]["geometry"]==cpu["qualification"]["geometry"],
                 "CPU and CUDA audit geometry or protocol differ")
    _require(gpu_cost_pin is None or gpu is not None,"CUDA cost needs a bound CUDA audit")
    gpu_cost = None if gpu_cost_pin is None else read_cost(root,gpu_cost_pin,producer,gpu)
    parity = dict(status="not_run",rows=[],failures=[]) if gpu_cost is None else compare(root,cpu_cost,gpu_cost)
    old = read(historical(root,"wp12"),root)
    tables = {**old["preserved"],**old["tables"]}
    _protect(root,tables.values())
    _require(len(tables)==10,"WP12 historical table roster differs")
    out.mkdir()
    rows = []
    for audit,cost,cost_pin in ((cpu,cpu_cost,cpu_cost_pin),(gpu,gpu_cost,gpu_cost_pin)):
        if audit is None:
            continue
        q = audit["qualification"]
        state = dict(protocol=audit["numerical"]["protocol"],operator_math=q["operator_math"],
            native_CPU_compatibility=q["native_CPU_compatibility"],same_device_recovery=audit["same_device_recovery"]["status"],
            cross_device_forward=parity["status"],cost="measured" if cost else "not_run",
            qualification_sha256=q["pin"]["sha256"],geometry_sha256=q["geometry"]["sha256"],
            producer=producer,consumer=consumer,production_authorized=False)
        audit.update(state=state,cost=cost_pin)
        rows.append(state)
    with (out/"native-numerical-status.csv").open("x",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out/"native-numerical-cost.csv").open("x",newline="",encoding="utf-8") as stream:
        fields = [*rows[0],*IDENTITY,"phase","p50_seconds","p95_seconds","native_pixels_per_second","peak_bytes"]
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for audit,cost in ((cpu,cpu_cost),(gpu,gpu_cost)):
            if cost is None:
                continue
            for row in cost["rows"]:
                for phase,stats in row["timings"].items():
                    writer.writerow(dict(**audit["state"],**{key:row[key] for key in IDENTITY},phase=phase,
                        **{k:v for k,v in stats.items() if k!="samples"},
                        peak_bytes=None if row["peaks"] is None else row["peaks"]["peak_allocated_bytes"]))
    return transport._json_new(out/"evidence.json",dict(schema=EVIDENCE,producer=producer,consumer=consumer,
        cpu=cpu,cuda=gpu,cross_device_forward=parity,old_tables=tables,old_models=65,old_heads=193,
        real_updates=0,production_authorized=False,status=transport._pin(out/"native-numerical-status.csv"),
        cost=transport._pin(out/"native-numerical-cost.csv")))
