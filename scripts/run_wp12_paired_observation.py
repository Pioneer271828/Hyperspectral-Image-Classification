"""Prepare, execute and independently rebuild a fixed WP12 observation A/B campaign."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
from statistics import median
import sys
from tempfile import mkdtemp
from time import perf_counter_ns
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.wp12_frozen54_deployment_contract import COLD, WARM, PHASES, SETUP, TIMING, read

PLAN = "wp12_observation_paired_plan_v1"
AUTHORITY = "wp12_observation_paired_authority_v1"
GPU3 = "GPU-2cbff259-d7db-32cf-b267-8f296e34b1d5"
WORKER = "3ba683439ed6c5ea3ae219bea0840e41b738ffce"
ARMS = ("buffered", "sealed-mmap")
CLASSES = {"paviaU":9, "xuzhou":9, "whu-honghu":22}
E2E = "wp12-frozen54-end-to-end-20261121/review-validated"
SYNTHETIC = "wp12-verified-observation-io-20261121/fixtures"
SOURCES = {
    "real": (E2E+"/real-plan/measurement-plan.json", "c4b21696e5394f7426a38d0393f11cdcddeb8f55912ee4e32e87e504a8807f3f"),
    "compact": (E2E+"/engineering-plan-A/measurement-plan.json", "38113db777e6f7b043fdbf000fd2b40e01a176bfaa63d8262adaf9af699608ab"),
    "fixtures": (SYNTHETIC+"/fixtures.json", "65804dad57c9c90bc7320d5e76e0306c8f425b355c2ecff7898309275be3b04a"),
    "models": (SYNTHETIC+"/models.json", "bb113c2c8d08139a4c997fc633cc370471ee8bed1209b38bf6a2175c2480f5ef"),
}
POLICY = dict(
    order="AB_even_BA_odd", backends=ARMS, cpu_affinity=24, min_free_mib=12288, max_gpu_utilization=60,
    max_snapshot_bytes=600*2**20, max_rss_growth_kib=256*1024, campaign_timeout_seconds=900,
    worker_reply_timeout_seconds=120, warmups_per_arm=2, real_pairs_per_scene=12,
    minimum_faster_pairs=9,
    adoption="each_scene_pairs_order_p95_noise_cuda_rss_and_snapshot_gate",
    estimator="median_of_buffered_minus_mmap_pairs_must_exceed_both_arms_p95_minus_p50",
    rollback="retain_buffered_route_no_automatic_retry_or_default_change",
    estimate="4-8 minutes of GPU occupancy, resource waiting separate",
)


def require(condition, why):
    if not condition:
        raise ValueError(why)


def canonical(root, path):
    require(path.is_absolute() and path.resolve() == path and root in path.parents and ROOT not in path.parents,
            "Use an isolated canonical path under the authorized runs directory")
    return path


def source_pins(root, engineering):
    use = ("compact","fixtures","models") if engineering else ("real",)
    return {name:{"path":str(root/SOURCES[name][0]),"sha256":SOURCES[name][1]} for name in use}


def roster(root, engineering):
    pins = source_pins(root, engineering)
    documents = {name:json.loads(read(root,pin)) for name,pin in pins.items()}

    def job(source, cold_pairs, warm_pairs):
        scene = source["scene"]
        workload = source["workload"]
        return dict(scene=scene, classes=CLASSES[scene], seed=source["seed"], arm=source["arm"],
                    checkpoint=source["checkpoint"], sidecar=source["sidecar"], workload=workload,
                    expected_ordered_scores=source["legacy_output"], cold_pairs=cold_pairs, warm_pairs=warm_pairs)

    if engineering:
        compact = documents["compact"]
        require(compact["engineering"] is True and compact["cases"]["engineering"]["workload"]["work_items"] == 36,
                "Compact aperture differs")
        result = {"compact":job(compact["cases"]["engineering"], 2, 2)}
        for scene, fixture in documents["fixtures"].items():
            weights = documents["models"][scene]
            result[scene] = job(dict(scene=scene, seed=64, arm="engineering_shape", legacy_output=None,
                workload=dict(file=fixture["pin"], tensor_sha256=fixture["tree_sha256"], work_items=fixture["pixels"]),
                **weights),0,2)
    else:
        frozen = documents["real"]
        require(frozen["engineering"] is False and len(frozen["cases"]) == 54 and frozen["worker_commit"] == WORKER,
                "Expected the authorized immutable frozen roster")
        result = {scene:job(next(row for row in frozen["cases"].values() if
                                (row["scene"],row["seed"],row["arm"]) == (scene,64,"target_scratch_full/none")),
                            0,POLICY["real_pairs_per_scene"]) for scene in CLASSES}
        require(all(case["expected_ordered_scores"] for case in result.values()), "Expected original three-head identities")
    return result


def implementation_pins():
    paths = ("benchmark_wp12_observation_io.py", "wp12_observation_snapshot.py", Path(__file__).name)
    return {name:sha256((ROOT/"scripts"/name).read_bytes()).hexdigest() for name in paths}


def draft(root, path, code, runtime, engineering):
    canonical(root, path)
    output = canonical(root, path.with_name(path.name+"-execution"))
    cases = roster(root, engineering)
    timing = {scope:TIMING[scope] for scope in ((COLD,WARM) if engineering else (WARM,))}
    return dict(schema=PLAN, producer=code, engineering=engineering, gpu_uuid=GPU3, worker=str(runtime),
                worker_commit=WORKER, source_pins=source_pins(root, engineering), execution=str(output),
                backends=ARMS, implementation=implementation_pins(), timing=timing, policy=POLICY, cases=cases)


def authority(plan, ref, decision, reference):
    return dict(schema=AUTHORITY, decision=decision, plan_sha256=ref["sha256"], producer=plan["producer"],
                execution=plan["execution"], physical_uuid=GPU3, scopes=list(plan["timing"]),
                data_access="synthetic_observations" if plan["engineering"] else "sealed_observations_no_labels",
                approval_reference=reference)


def admitted(root, plan_path, plan_sha, code, runtime, permit):
    from scripts.benchmark_wp12_observation_io import require_clean

    path = canonical(root, plan_path)
    ref = dict(path=str(path), sha256=plan_sha)
    plan = json.loads(read(root, ref))
    require(type(plan["engineering"]) is bool and path.name=="paired-plan.json",
            "Expected the independent paired observation plan")
    expected = draft(root, path.parent, code, runtime, plan["engineering"])
    require(json.loads(json.dumps(expected)) == plan, "Plan differs from its frozen sources, budget or execution")
    require(permit is not None, "A separately supplied paired approval is required")
    permit_path = canonical(root, Path(permit["path"]))
    require(path.parent not in permit_path.parents and Path(plan["execution"]) not in permit_path.parents,
            "Approval must be outside the plan and execution directories")
    signed = json.loads(read(root, permit))
    reference = signed.get("approval_reference")
    require(isinstance(reference,str) and bool(reference.strip()) and signed == authority(plan,ref,"approved",reference),
            "Approval does not bind the exact paired plan, scope, physical GPU and destination")
    require_clean(ROOT,code)
    require_clean(runtime,WORKER)
    return ref, plan


def binding(plan_sha, approval_sha, case_key, scope, backend, attempt):
    return dict(plan_sha256=plan_sha, approval_sha256=approval_sha, case=case_key,
                scope=scope, backend=backend, attempt=attempt, gpu=GPU3)


def worker_context(args):
    root, path, sha, permit_path, permit_sha, key, attempt = args.paired_context
    root = Path(root).resolve(strict=True)
    _, plan = admitted(root, Path(path), sha, args.expected_commit, args.runtime,
                       {"path":permit_path,"sha256":permit_sha})
    job = plan["cases"][key]
    expected_path = Path(plan["execution"])/key/attempt
    require(args.gpu_uuid == GPU3 and args.scene == job["scene"] and args.out == expected_path
            and attempt.startswith("attempt-") and Path(attempt).name == attempt and expected_path.is_dir()
            and (args.worker != "cold" or job["cold_pairs"]>0),
            "Worker invocation is outside the approved paired scope")
    workload = job["workload"]
    item = dict(pin=workload["file"],tree_sha256=workload["tensor_sha256"],pixels=workload["work_items"],
                classes=job["classes"])
    model = {k:job[k] for k in ("checkpoint","sidecar")}
    context = dict(binding=binding(sha,permit_sha,key,args.worker,args.backend,attempt), legacy_output=job["expected_ordered_scores"],
                   max_snapshot_bytes=POLICY["max_snapshot_bytes"], warmups=POLICY["warmups_per_arm"], warm_pairs=job["warm_pairs"])
    return root, item, model, context


def quantify(pairs, *, evaluate):
    from evaluation.wp12_cost_primitives import timing_summary

    def summarize(x):
        return timing_summary(x,repeats=len(x),blocks=1)

    def method(side):
        rows = [pair[side] for pair in pairs]
        return dict(wall=summarize([r["wall_ms"] for r in rows]),
                    phases={p:summarize([r["phases"][p]["wall_ms"] for r in rows]) for p in PHASES},
                    cuda_allocated_bytes=max(r["cuda_peak_bytes"] for r in rows),
                    process_peak_rss_kib=max(r["process_peak_rss_kib"] for r in rows),
                    logical_snapshot_bytes=max(r["snapshot_bytes"] for r in rows))

    # The raw array is always stored in fixed A,B order, regardless of launch order.
    a, b = method(0), method(1)
    differences = [old["wall_ms"]-new["wall_ms"] for old,new in pairs]
    width = max(x["wall"]["p95_ms"]-x["wall"]["p50_ms"] for x in (a,b))
    p50_ab, p50_ba = median(differences[::2]),median(differences[1::2])
    faster = sum(x>0 for x in differences)
    gates = (dict(faster_pairs=faster>=POLICY["minimum_faster_pairs"], both_orders=p50_ab>0 and p50_ba>0,
                 exceeds_noise=median(differences)>width, p95=b["wall"]["p95_ms"]<=a["wall"]["p95_ms"],
                 cuda=b["cuda_allocated_bytes"]<=a["cuda_allocated_bytes"],
                 rss=b["process_peak_rss_kib"]-a["process_peak_rss_kib"]<=POLICY["max_rss_growth_kib"],
                 snapshot=b["logical_snapshot_bytes"]<=POLICY["max_snapshot_bytes"]) if evaluate else None)
    return dict(buffered=a, sealed_mmap=b, saved_pair_ms=differences, median_saved_ms=median(differences),
                ab_median_saved_ms=p50_ab,ba_median_saved_ms=p50_ba,noise_width_ms=width,faster_pairs=faster,
                gates=gates)


def rebuild(root, ref, approval_sha, plan, key, attempt, invocations):
    from evaluation.wp12_frozen54_deployment import elapsed, phase_end
    from model.research_config import digest

    job = plan["cases"][key]
    cold_count = 2*job["cold_pairs"]
    require(len(invocations)==cold_count+2,"Unexpected process roster")
    phases_by_scope = {"cold":[],"warm":[]}
    last_cold_exit = 0
    identities, pids = set(), set()
    for index, link in enumerate(invocations):
        require(link["path"]==str(attempt/f"worker-{index}-invocation.json"),"Foreign invocation receipt")
        process = json.loads(read(root,link))
        require(process["stdout"]["path"]==str(attempt/f"worker-{index}.jsonl"),"Foreign original worker transcript")
        rows = [json.loads(line) for line in read(root,process["stdout"]).splitlines()]
        scope = "cold" if index<cold_count else "warm"
        if scope=="warm":
            require(rows.pop(0)=={"ready":True},"Missing resident setup acknowledgement")
        times = process["started"],process["exited"]
        elapsed(*times)
        if scope=="cold":
            require(last_cold_exit<=times[0],"Cold process overlapped a previous process")
            last_cold_exit = times[1]
        require(type(process["pid"]) is int and process["pid"]>0 and process["pid"] not in pids,
                "Original worker process identity was reused")
        pids.add(process["pid"])
        backend = process["backend"]
        expected = binding(ref["sha256"],approval_sha,key,scope,backend,attempt.name)
        expected_count = 1 if scope=="cold" else POLICY["warmups_per_arm"]+job["warm_pairs"]
        require(backend in ARMS and process["scope"]==scope and len(rows)==expected_count
                and (scope=="cold" or times[0]>=last_cold_exit),
                "Worker scope, backend, lifecycle or request count differs")
        setup = rows[0]["setup"]
        previous = phase_end(setup, SETUP, *times, GPU3)
        for sequence, row in enumerate(rows):
            start, stop = row["request_started_ns"],row["first_prediction_ns"]
            warmup = scope=="warm" and sequence<POLICY["warmups_per_arm"]
            require(row["binding"]==expected and row["pid"]==process["pid"] and row["sequence"]==sequence
                    and row["role"]==("warmup" if warmup else "measured") and row["setup"]==setup
                    and previous<=start<stop<=times[1] and phase_end(row["phases"],PHASES,start,stop,GPU3)==stop,
                    "Worker chronology, scope or role differs")
            require(row["checkpoint_sha256"]==job["checkpoint"]["sha256"] and row["sidecar_sha256"]==job["sidecar"]["sha256"]
                    and row["observation_sha256"]==job["workload"]["file"]["sha256"]
                    and row["decoded_sha256"]==job["workload"]["tensor_sha256"]
                    and len(row["score_sha256"])==len(row["decision_sha256"])==3
                    and all(re.fullmatch("[a-f0-9]{64}", v) for v in (*row["score_sha256"],*row["decision_sha256"]))
                    and digest([{"tensor":value} for value in row["score_sha256"]])==row["combined_score_sha256"]
                    and (job["expected_ordered_scores"] is None or row["combined_score_sha256"]==job["expected_ordered_scores"]),
                    "Original checkpoint, observation or three-head output identity differs")
            require(type(row["process_peak_rss_kib"]) is int and row["process_peak_rss_kib"]>0
                    and (row["snapshot_bytes"]==0 if backend==ARMS[0] else 0<row["snapshot_bytes"]<=POLICY["max_snapshot_bytes"]),
                    "Unexpected RSS or snapshot memory evidence")
            identities.add((row["combined_score_sha256"],tuple(row["score_sha256"]),tuple(row["decision_sha256"])))
            phases = list(row["phases"].values()) + ([] if scope=="warm" else list(setup.values()))
            row.update(backend=backend, wall_ms=elapsed(times[0] if scope=="cold" else start,stop),
                       cuda_peak_bytes=max(p["cuda_peak_bytes"] for p in phases))
            phases_by_scope[scope].append(row)
            previous = stop

    require(len(identities)==1,"Cold or warm backends did not return identical three-head results")
    warmups = POLICY["warmups_per_arm"]
    out = {}
    for scope, rows in phases_by_scope.items():
        if not rows:
            require(scope=="cold" and job["cold_pairs"]==0,"Missing required request scope")
            continue
        rows.sort(key=lambda row:row["request_started_ns"])
        require(all(a["first_prediction_ns"]<=b["request_started_ns"] for a,b in zip(rows,rows[1:])),
                "Requests overlapped")
        pairs = []
        for index in range(0,len(rows),2):
            pair = rows[index:index+2]
            pair_number = index//2 - (warmups if scope=="warm" else 0)
            arms = ARMS if pair_number%2==0 else ARMS[::-1]
            require(tuple(r["backend"] for r in pair)==arms and all(r["role"]==
                    ("warmup" if pair_number<0 else "measured") for r in pair)
                    and (scope=="cold" or all(r["sequence"]==index//2 for r in pair)),
                    "Original requests differ from the registered AB/BA schedule")
            if pair_number>=0:
                pairs.append(pair if arms==ARMS else pair[::-1])
        out[scope] = quantify(pairs,evaluate=not plan["engineering"] and scope=="warm")
    return dict(attempt=attempt.name, invocations=invocations, summaries=out, digest=next(iter(identities))[0])


def capture(args, root, ref, plan, key, attempt, lease):
    from scripts import benchmark_wp12_observation_io as bench
    from scripts.run_frozen_band_region import _write, pin

    job = plan["cases"][key]
    context = [str(root),ref["path"],ref["sha256"],*args.approval,key,attempt.name]
    worker_args = SimpleNamespace(expected_commit=args.expected_commit, runtime=args.runtime, out=attempt,
                                  scene=job["scene"], gpu_uuid=GPU3, paired_context=context)
    invocations, transcripts = [], {}

    @contextmanager
    def start_worker(options, scope, backend, current_lease, error_stream):
        worker_id = len(invocations)
        invocations.append(None)
        output = attempt/f"worker-{worker_id}.jsonl"
        with output.open("x",encoding="utf-8",newline="") as stream:
            with bench.launch(options,scope,backend,current_lease,error_stream) as (proc, started):
                transcripts[proc] = stream
                try:
                    yield proc, started
                finally:
                    del transcripts[proc]
            stop = perf_counter_ns()
        invocations[worker_id] = _write(attempt/f"worker-{worker_id}-invocation.json",dict(
            scope=scope,backend=backend,pid=proc.pid,stdout=pin(output),started=started,exited=stop))

    with (attempt/"stderr.log").open("x") as stream:
        bench.collect(worker_args,lease,stream,cold_pairs=job["cold_pairs"],warm_pairs=job["warm_pairs"],
                      warmups=POLICY["warmups_per_arm"],start_worker=start_worker,
                      receive_sample=lambda proc:bench.receive(proc,transcripts[proc]))
    return rebuild(root,ref,args.approval[1],plan,key,attempt,invocations)


def dispatch(args, root, ref, plan):
    from evaluation.wp12_resident_cost import gpu_measurement_lease
    from scripts.run_frozen_band_region import _write, pin
    from transfer.native_coverage_queue import case_lock

    execution = Path(plan["execution"])
    results = {}
    if args.operation=="run":
        execution.mkdir(exist_ok=True)
    else:
        require(execution.is_dir(),"Paired audit requires an existing execution")
    with case_lock(execution):
        for key in plan["cases"]:
            stage = execution/key
            accepted = stage/"accepted.json"
            if accepted.exists():
                value = json.loads(read(root,pin(accepted)))
                name = value["attempt"]
                require(isinstance(name,str) and Path(name).name==name and name.startswith("attempt-"),
                        "Foreign accepted attempt")
                results[key] = rebuild(root,ref,args.approval[1],plan,key,stage/name,value["invocations"])
                require(results[key]==value,"Accepted summary differs from reconstructed original worker streams")
            elif args.operation=="audit":
                raise ValueError("Paired audit requires every accepted case")
            else:
                require(not stage.exists(),"Incomplete attempt needs review before any retry")
                with gpu_measurement_lease(GPU3) as lease:
                    stage.mkdir()
                    attempt = Path(mkdtemp(prefix="attempt-",dir=stage))
                    try:
                        results[key] = capture(args,root,ref,plan,key,attempt,lease)
                        _write(accepted,results[key])
                    except Exception as exc:
                        _write(attempt/"failure.json",dict(error=type(exc).__name__,message=str(exc)))
                        raise
    if args.operation=="audit":
        passed = not plan["engineering"] and all(all(row["summaries"]["warm"]["gates"].values()) for row in results.values())
        report = canonical(root,args.report)
        require(report.parent == args.out.parent and not report.exists(), "Use a fresh independent report destination")
        return _write(report, dict(schema=PLAN,status="audited", engineering=plan["engineering"], plan=ref,
            approval=dict(zip(("path","sha256"),args.approval,strict=True)),cases=results,
            outcome="engineering_only" if plan["engineering"] else ("eligible_for_review" if passed else "retain_buffered")))
    return {"accepted_cases":len(results),"execution":plan["execution"]}


def main():
    from scripts.benchmark_wp12_observation_io import require_clean
    from scripts.run_frozen_band_region import _write

    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("runs","runtime","out"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    commands = parser.add_subparsers(dest="operation",required=True)
    commands.add_parser("prepare").add_argument("--engineering",action="store_true")
    for name in ("run","audit"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--plan-sha",required=True)
        cmd.add_argument("--approval",nargs=2)
        if name=="audit":
            cmd.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    root = args.runs.resolve(strict=True)
    require_clean(ROOT,args.expected_commit)
    require_clean(args.runtime,WORKER)
    path = canonical(root,args.out)
    if args.operation=="prepare":
        plan = draft(root,path,args.expected_commit,args.runtime,args.engineering)
        require(not path.exists() and not Path(plan["execution"]).exists(),"Fresh plan and execution locations required")
        path.mkdir()
        ref = _write(path/"paired-plan.json",plan)
        _write(path/"application-template.json",authority(plan,ref,None,None))
        result = ref
    else:
        ref, plan = admitted(root,path/"paired-plan.json",args.plan_sha,args.expected_commit,args.runtime,
                             dict(zip(("path","sha256"),args.approval,strict=True)) if args.approval else None)
        require(os.sched_getaffinity(0)=={POLICY["cpu_affinity"]},"Expected the preregistered single CPU core")
        if args.operation=="run":
            def expire(*_):
                raise TimeoutError("Paired campaign exceeded the registered 15-minute limit")

            signal.signal(signal.SIGALRM,expire)
            signal.alarm(POLICY["campaign_timeout_seconds"])
        result = dispatch(args,root,ref,plan)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
