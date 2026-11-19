"""Registered cold, warm, resident and concurrent deployment measurements."""

from collections import defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
from statistics import fmean, median
import subprocess
import sys
from time import monotonic, perf_counter_ns, sleep

import torch

from evaluation.wp12_cost_primitives import timing_summary
from inference.native_deployment import Runtime, clock_ms, output_hash, prepare
from model.research_config import digest
from research_runtime import hash_file, require_frozen_code
from scripts import run_frozen_band_region as archive
from transfer import band_scale_real
from transfer.development_blocks import read_pin
from transfer.g3_training import state_hash
from transfer.native_coverage_queue import case_lock
from transfer.native_coverage_replay import EXPECTED_ARCHIVES, frozen_model
from transfer.spatial_coverage_protocol import bank_observation

SCHEMA="wp11_native_deployment_cost_v1"
CASES={"paviaU":"paviaU-64","xuzhou":"xuzhou","whu-honghu":"whu-honghu"}
SCOPES=("cold_process_first_prediction","warm_end_to_end","resident_forward","concurrent_throughput")
REPEATS=(3,6,6,4)
WARMUPS=2
WORKERS=2
TIMEOUT=120
LEGACY={
    "wp11_manifest":("wp11-unified-xuzhou01/registered/manifest.json","583e3949b6ed1aad09025b06984123d05654c88424d2598bc874f93f4df096fb"),
    "wp11_static":("wp11-matched-xuzhou-cost02/static-cost.json","1ded1f9d5a70e2f8dfc92851f64a5c5b079afbfc9ca0d4a53f7873bb8a5b9b9d"),
    "issue249":("issue249-post-drain-audit01/receipt.json","22c593478f4badda8b8f54e61d7bc148dd261d666f73283b63ed71e8e11af8aa"),
    "wp12_frozen54":("wp12-cost-frozen54-independent01/receipt.json","efa741072cfeed09e69345181cac9a8ec39b3e042bc12aac8ad8e9a3bd216e35"),
}


def need(condition,message):
    if not condition:
        raise ValueError(message)


def document(root,pin):
    return json.loads(read_pin(pin,root))


def inherited(root):
    evidence={k:document(root,{"path":str(root/name),"sha256":sha}) for k,(name,sha) in LEGACY.items()}
    stat, mf, f54, old=(evidence[k] for k in ("wp11_static","wp11_manifest","wp12_frozen54","issue249"))
    need(mf["routing"] is None and mf["release"]==stat["release"] and mf["static_input"]==stat["input_hash"]
         and stat["optimizer_updates"]==stat["query"]==old["new_optimizer_updates"]==f54["new_query_reads"]==0
         and f54["case_count"]==54 and old["query_labels_accessed"] is False,"Archived deployment evidence differs")
    return dict(pins={k:sha for k,(_,sha) in LEGACY.items()},
                wp11_static=dict(status=stat["status"],scope=stat["scope"],samples=stat["summary"],
                                blocks=stat["blocks"],pixels=stat["pixels"],mode="legacy_cpu"),
                wp11_matched=dict(device="archived_cpu",adaptive_admitted=False,model_identifiers={
                    key:row["state"] for key,row in mf["models"].items()}),
                issue249=dict(archived_head=old["role"],checkpoint=old["primary_sha256"],selected=old["selected_step"],
                              match=False,reason="Different model/observation family; never paired with the archived WP11 full model."),
                frozen54=dict(cases=f54["case_count"],timed_samples=f54["timed_samples"],scope="legacy_gpu_resident_forward",
                              primary=f54["preregistered_benefit_conclusion"],device=f54["hardware"]["name"]))


def approved_archive(root, name):
    digest_value=EXPECTED_ARCHIVES[name]
    path=root/"wp05-wp06-wp07-band-scale-20261119"/(name+".json")
    record=document(root,dict(path=str(path),sha256=digest_value))
    need(record["case"]==name and record["role"]=="devval" and record["real_updates"]==0
         and record["raw_gt_loaded"] is record["query_accessed"] is False,"Frozen model archive not admissible")
    return record


def context(root,case, *, load=False):
    name=CASES[case]
    source, docs=archive.source_documents(root)
    previous=approved_archive(root,name)
    bank,intake=archive._freeze_bank(root,case,{"sources":source})
    need(bank.pin["sha256"]==previous["preflight_population"],"Frozen model and authorized observation Bank differ")
    model=None
    if load:
        model,_,identity,_,weights=frozen_model(root,name,bank,intake,docs)
        need(identity==previous["model"] and weights==previous["checkpoint"],"Frozen model or weight sidecar differs")
        model.eval().requires_grad_(False)
        model.backbone.adapter.cache_attention=False
        model.backbone.adapter.clear_attention_cache()
    return bank,previous,model


def area(row):
    y,z,x,w=row["box"]
    return (z-y)*(w-x)


def planned_inputs(bank,producer):
    rows=sorted((r for r in bank.role_rows("devval") if r["blocked"] is None),
                key=lambda r:(area(r),r["id"]))
    need(len(rows)>0,"No released label-blind workload")
    row=rows[(len(rows)-1)//2]
    requests=[]
    for band in ("full","central_gap"):
        actual=bank_observation(bank,row,producer,band)
        requests.append(dict(tile=row["id"],parent=digest([row["parent_tile"],row["parent_box"]]),
                             band=band,observation=actual.receipt["observation"],shape=list(actual.lattice.shape),
                             original_bands=bank.release.shape[-1],retained_bands=len(actual.view.bands.band_ids)))
    return requests


def register(root,out,code):
    out.mkdir()
    cases={}
    for scene in CASES:
        bank,model,_=context(root,scene)
        inputs=planned_inputs(bank,code["commit"])
        cases[scene]=dict(bank=bank.pin["sha256"],model=model["model"],weights=model["checkpoint"],
                          archive=EXPECTED_ARCHIVES[CASES[scene]],requests=inputs)
    history=inherited(root)
    plan=dict(schema=SCHEMA,producer=code["commit"],cases=cases,archive=history,
              repeats=dict(zip(SCOPES,REPEATS,strict=True)),warmups=WARMUPS,concurrent_workers=WORKERS,
              os_cache="normal_os_page_cache_uncontrolled_no_flush",graph_policy="rebuild",
              workload_rule="middle-area published devval block; full and central-gap views; no label-based choice",
              resolution="native",mode="full",device_limits=dict(min_cuda_free_mib=12288,max_cuda_utilization=60),
              estimated_seconds=dict(cpu=100,gpu=110),timeout_per_worker_seconds=TIMEOUT,
              real_measurement=True,labels_emitted=False,optimizer_updates=0,query_accessed=False)
    require_frozen_code(code)
    return archive._write(out/"measurement-plan.json",plan)


def read_plan(root,out,code,expected=None):
    pin=archive.pin(out/"measurement-plan.json")
    need(expected is None or expected==pin["sha256"],"Frozen measurement plan digest differs")
    plan=document(root,pin)
    need(plan["schema"]==SCHEMA and plan["producer"]==code["commit"] and plan["cases"].keys()==CASES.keys()
         and plan["repeats"]==dict(zip(SCOPES,REPEATS,strict=True)) and plan["warmups"]==WARMUPS
         and plan["concurrent_workers"]==WORKERS and plan["mode"]=="full" and plan["graph_policy"]=="rebuild",
         "Foreign deployment measurement plan or clock")
    return pin,plan


class Worker:
    def __init__(self,root,plan,case,device):
        started=perf_counter_ns()
        self.plan,self.case=plan,case
        self.record=plan["cases"][case]
        self.bank,previous,model=context(root,case,load=True)
        need(previous["model"]==self.record["model"] and previous["checkpoint"]==self.record["weights"]
             and self.bank.pin["sha256"]==self.record["bank"],"Case admission changed after registration")
        self.runtime=Runtime(device=device,full=model,identity={"full":self.record["model"]})
        self.load_ms=clock_ms(started)

    def ready(self):
        values=[]
        for i,row in enumerate(self.record["requests"]):
            tile=self.bank.role_row("devval",row["tile"])
            need(digest([tile["parent_tile"],tile["parent_box"]])==row["parent"]
                 and self.bank.release.shape[-1]==row["original_bands"],"Foreign published parent or band axis")
            actual=bank_observation(self.bank,tile,self.plan["producer"],row["band"])
            request=prepare(actual,self.plan["producer"],key=f"{self.case}/{i}",device=self.runtime.device,
                            original_band_count=row["original_bands"],expected=row["observation"])
            need(list(request.x.shape[:2])==row["shape"],"Prepared deployment work shape differs")
            values.append(request)
        return values

    def sample(self,warm,prepared=None):
        start=perf_counter_ns()
        previous=self.bank.reads
        if self.runtime.device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(self.runtime.device)
        requests=self.ready() if warm else prepared
        self.runtime.sync()
        prep=clock_ms(start) if warm else 0.
        output=self.runtime.run(requests,"full")
        if warm:
            identity=output_hash(output)
            latency=clock_ms(start)
        else:
            latency=output["elapsed_ms"]
            identity=output_hash(output)
        need(self.bank.reads-previous==(len(requests) if warm else 0) and
             output["counts"]["graph_builds"]==len(requests) and output["counts"]["light_calls"]==0,
             "Observed read or graph count differs from the measured scope")
        return dict(wall_ms=latency,prepare_ms=prep,forward_ms=output["elapsed_ms"],graph_ms=output["graph_ms"],
                    gpu_event_ms=output["gpu_event_ms"],graph_calls=output["counts"]["graph_builds"],
                    io_reads=self.bank.reads-previous,prediction=identity,
                    rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    cuda_peak_bytes=torch.cuda.max_memory_allocated(self.runtime.device) if self.runtime.device.type=="cuda" else None)


def internal(root,out,code,case,scope,device,*,plan_sha,sync_folder=None):
    _,plan=read_plan(root,out,code,plan_sha)
    clock=perf_counter_ns()
    worker=Worker(root,plan,case,device)
    if scope=="cold_process_first_prediction":
        sample=worker.sample(True)
        return dict(**sample,pid=os.getpid(),worker_measured_ms=clock_ms(clock),load_ms=worker.load_ms,
                    first_prediction_ns=perf_counter_ns())
    if scope!="concurrent_throughput" or sync_folder is None:
        raise ValueError("Unknown worker timing scope")
    for _ in range(WARMUPS):
        worker.sample(True)
    archive._write(sync_folder/f"ready-{os.getpid()}.json",{"pid":os.getpid()})
    deadline=monotonic()+TIMEOUT
    while not (sync_folder/"start.json").exists():
        if monotonic()>deadline:
            raise RuntimeError("Concurrent workers did not receive a start lease")
        sleep(.010)
    samples=[worker.sample(True) for _ in range(REPEATS[-1]//WORKERS)]
    return dict(pid=os.getpid(),samples=samples,last_prediction_ns=perf_counter_ns())


def worker_command(root,out,code,case,scope,device,plan_sha,folder=None):
    cmd=[sys.executable,"-B",str(Path(__file__).resolve().parents[1]/"scripts/run_wp11_execution.py"),
         "--expected-commit",code["commit"],"deployment-worker","--runs",str(root),"--out",str(out),
         "--case",case,"--scope",scope,"--device",device,"--plan-sha",plan_sha]
    if folder is not None:
        cmd.extend(("--sync-folder",str(folder)))
    return cmd


def external_cold(root,out,code,case,device,attempt,plan_sha):
    samples=[]
    for number in range(REPEATS[0]):
        command=worker_command(root,out,code,case,SCOPES[0],device,plan_sha)
        start=perf_counter_ns()
        try:
            proc=subprocess.run(command,capture_output=True,text=True,timeout=TIMEOUT)
        except subprocess.TimeoutExpired as error:
            (attempt/f"cold-{number}.log").write_text(str(error))
            raise
        wall=clock_ms(start)
        (attempt/f"cold-{number}.log").write_text(proc.stderr+proc.stdout)
        proc.check_returncode()
        record=json.loads(proc.stdout.strip().splitlines()[-1])
        samples.append(dict(**{k:v for k,v in record.items() if k!="wall_ms"},
                            inside_request_ms=record["wall_ms"],process_exit_ms=wall,
                            wall_ms=(record["first_prediction_ns"]-start)/1e6))
    need(len({r["pid"] for r in samples})==len(samples),"Cold trials must use fresh processes")
    return samples,None


def concurrent(root,out,code,case,device,attempt,plan_sha):
    processes=[]
    try:
        for i in range(WORKERS):
            with (attempt/f"concurrent-{i}.log").open("x") as log:
                proc=subprocess.Popen(worker_command(root,out,code,case,SCOPES[-1],device,plan_sha,attempt),
                                      stdout=log,stderr=subprocess.STDOUT)
            processes.append(proc)
        deadline=monotonic()+TIMEOUT
        while len(list(attempt.glob("ready-*.json")))!=WORKERS:
            if any(p.poll() is not None for p in processes) or monotonic()>deadline:
                raise RuntimeError("Concurrent workers failed before first synchronized request")
            sleep(.015)
        start=perf_counter_ns()
        archive._write(attempt/"start.json",{"workers":WORKERS})
        for proc in processes:
            if proc.wait(timeout=TIMEOUT):
                raise RuntimeError("Concurrent deployment worker failed")
        rows=[json.loads((attempt/f"concurrent-{i}.log").read_text().strip().splitlines()[-1]) for i in range(WORKERS)]
        need(len({r["pid"] for r in rows})==WORKERS,"Concurrent results did not run in independent processes")
        duration=(max(row["last_prediction_ns"] for row in rows)-start)/1e6
        return [dict(pid=row["pid"],**sample) for row in rows for sample in row["samples"]], duration
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)


def summary(samples,scope,overlap):
    if scope==SCOPES[0]:
        values=[s["wall_ms"] for s in samples]
        # Three fresh starts still permit descriptive interpolated percentiles.
        ordered=sorted(values)
        stats=dict(p50_ms=median(values),p95_ms=ordered[-2]+.9*(ordered[-1]-ordered[-2]),
                   mean_ms=fmean(values),wall_ms=values)
    else:
        stats=timing_summary([s["wall_ms"] for s in samples],repeats=len(samples),blocks=samples[0]["graph_calls"])
    stats.update(rss_peak_kib=max(s["rss_kib"] for s in samples),
                 cuda_process_peak_bytes=max((s["cuda_peak_bytes"] for s in samples if s["cuda_peak_bytes"] is not None),default=None),
                 graph_ms=sum(s["graph_ms"] for s in samples),wall_concurrent_ms=overlap,
                 concurrent_requests_per_second=None if overlap is None else len(samples)*1000/overlap)
    return stats


def stage(root,out,code,case,scope,device,device_id,environment,plan_sha):
    plan_pin,plan=read_plan(root,out,code,plan_sha)
    lineage=dict(plan=plan_pin["sha256"],case=case,scope=scope,device=device_id,environment=environment)
    directory=out/"stages"/device_id/case/scope
    directory.mkdir(parents=True,exist_ok=True)
    with case_lock(directory):
        receipt=directory/"accepted.json"
        if receipt.exists():
            value=document(root,archive.pin(receipt))
            need(value["lineage"]==lineage,"Previously measured scope belongs to a different device or plan")
            return archive.pin(receipt)
        attempt=directory/("attempt-"+str(len(list(directory.glob("attempt-*")))+1))
        attempt.mkdir()
        try:
            if scope==SCOPES[0]:
                samples,overlap=external_cold(root,out,code,case,device,attempt,plan_pin["sha256"])
            elif scope==SCOPES[-1]:
                samples,overlap=concurrent(root,out,code,case,device,attempt,plan_pin["sha256"])
            else:
                worker=Worker(root,plan,case,device)
                prepared=worker.ready()
                warm=scope==SCOPES[1]
                for _ in range(WARMUPS):
                    worker.sample(warm,prepared)
                samples=[worker.sample(warm,prepared) for _ in range(plan["repeats"][scope])]
                overlap=None
            need(len(samples)==plan["repeats"][scope] and len({r["prediction"] for r in samples})==1,
                 "Repeated measurement changed prediction bytes or sample budget")
            value=dict(lineage=lineage,attempt=attempt.name,samples=samples,summary=summary(samples,scope,overlap),
                       warmups=0 if scope==SCOPES[0] else WARMUPS,optimizer_updates=0,query_accessed=False)
            require_frozen_code(code)
            return archive._write(receipt,value)
        except Exception as error:
            archive._write(attempt/"failure.json",dict(lineage=lineage,error=type(error).__name__,reason=str(error)))
            raise


def device_environment(gpu):
    from transfer import p4_runtime_device as p4
    if gpu is None:
        return "cpu","cpu",dict(torch=str(torch.__version__),threads=torch.get_num_threads(),cuda_runtime=None)
    info=p4.gate(gpu)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,"Only the registered physical CUDA device may be visible")
    props=torch.cuda.get_device_properties(0)
    return "cuda:0",gpu,dict(torch=str(torch.__version__),threads=torch.get_num_threads(),
                             accelerator=props.name,driver=info["driver"],cuda_runtime=torch.version.cuda)


def run(root,out,code,gpu=None,*,plan_sha,limit=None):
    from transfer.p4_runtime_device import gate
    from transfer.shared_gpu_lease import physical_lease

    plan_pin,plan=read_plan(root,out,code,plan_sha)
    with (nullcontext() if gpu is None else physical_lease(gpu)):
        device,identity,env=device_environment(gpu)
        qualification=None if gpu is None else gate(gpu)
        finished=[]
        for case in plan["cases"]:
            for scope in SCOPES:
                if limit is not None and len(finished)>=limit:
                    return finished
                finished.append(stage(root,out,code,case,scope,device,identity,env,plan_pin["sha256"]))
                print(identity,case,scope,flush=True)
        directory=out/"stages"/identity
        with case_lock(directory):
            final=directory/"device.json"
            if not final.exists():
                archive._write(final,dict(plan=plan_pin["sha256"],environment=env,qualification=qualification,
                                         stages=finished,device=identity))
        return finished
