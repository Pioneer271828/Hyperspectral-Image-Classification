"""Prepare, execute and independently audit the additional frozen54 E2E scopes."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter_ns

from evaluation.wp11_deployment_cost import document, need
from evaluation.wp12_cost_primitives import timing_summary
from evaluation.wp12_release_results import frozen54, proof
from evaluation.wp12_resident_cost import gpu_measurement_lease
from research_runtime import code_snapshot, hash_file, require_frozen_code
from scripts.measure_wp12_historical_cost import preserved_runtime
from scripts.run_frozen_band_region import _write, pin
from scripts.wp12_frozen54_deployment_contract import (
    APPROVAL, COLD, WARM, COUNT, WARMUPS, TIMEOUT, PHASES, SETUP, PLAN, GPU, approval, read,
)
from transfer.native_coverage_queue import case_lock
from transfer.shared_gpu_lease import lease_path


LEGACY = "3ba683439ed6c5ea3ae219bea0840e41b738ffce"
ADAPTER = Path(__file__).resolve().parents[1] / "scripts/wp12_frozen54_deployment_worker.py"
BOUNDARY = {
    COLD: "process_spawn_to_first_three_head_host_decisions",
    WARM: "sealed_observation_read_to_three_head_host_decisions_model_resident",
    "os_page_cache": "uncontrolled_no_eviction",
    "graph": "reuse_frozen_prepared_Q_A_from_each_read_no_graph_build",
    "unit": "one_registered_whole_scene_observation",
    "arithmetic": "historical_ordered",
}


def artifact(root, path):
    need(path.is_absolute() and path == path.resolve() and root in path.parents,
         "Use a canonical destination under the authorized runs root")
    return path


def runtime(worker):
    worker_code = code_snapshot(worker)
    need(worker_code["commit"] == LEGACY and not worker_code["dirty"], "Original clean resident worker is required")
    return worker_code


def roster(root, worker_code):
    # The existing independent resident consumer verifies the accepted 54 and their linked accuracy.
    old, lifecycle = frozen54(root)
    reg = document(root, proof(root, "registration"))
    links = document(root, proof(root, "measurements"))
    freeze = reg["inputs"]["accuracy"]["freeze"]
    identity = preserved_runtime(document(root, freeze), worker_code)
    cases = {}
    for i, (row, result_pin) in enumerate(zip(reg["cases"], links, strict=True), 1):
        result = document(root, result_pin)
        case = f"{i:02d}-{row['scene']}-{row['seed']}-{row['arm'].replace('/', '-')}"
        cases[case] = {**row, "workload": reg["workloads"][row["scene"]], "legacy_cost": result_pin,
                       "legacy_output": result["output_sha256"]}
    need(len(cases) == len(old) == lifecycle["existing_accepted"] == 54, "Incomplete legacy pairing")
    return cases, freeze, identity, old


def draft(root, code, worker, execution, *, fixture=None):
    worker_code = runtime(worker)
    if fixture is None:
        cases, freeze, identity, _ = roster(root, worker_code)
    else:
        cases, freeze, identity = {"engineering": fixture}, None, None
    return header(code, worker, execution, cases, freeze, identity)


def launch(root, worker, operation, *, gpu="cpu", lease=None, attempt=None, label="worker"):
    command = [sys.executable, "-B", str(ADAPTER), "--runs", str(root), "--worker", LEGACY, *operation]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    start = perf_counter_ns()
    try:
        process = subprocess.run(command, cwd=worker, capture_output=True, text=True, env=env, timeout=TIMEOUT,
                                 pass_fds=() if lease is None else (lease.fileno(),))
    except subprocess.TimeoutExpired as error:
        if attempt is not None:
            (attempt / f"{label}.log").write_text(str(error))
        raise
    elapsed = (perf_counter_ns()-start)/1e6
    if attempt is not None:
        (attempt / f"{label}.log").write_text(process.stderr + process.stdout)
    if process.returncode:
        raise RuntimeError(f"Frozen deployment worker failed ({process.returncode}): {process.stderr[-1600:]}")
    return json.loads(process.stdout.strip().splitlines()[-1]), start, elapsed


def prepare(root, out, code, worker, execution, engineering):
    artifact(root, out)
    artifact(root, execution)
    need(not out.exists() and not execution.exists() and execution != out and out not in execution.parents,
         "Use distinct fresh plan and execution directories")
    worker = worker.resolve(strict=True)
    runtime(worker)
    fixture = None
    if engineering:
        out.mkdir()
        fixture, _, _ = launch(root, worker, ["fixture", "--out", str(out/"fixture")])
    plan = draft(root, code, worker, execution, fixture=fixture)
    require_frozen_code(code)
    if not engineering:
        out.mkdir()
    plan_pin = _write(out/"measurement-plan.json", plan)
    if not engineering:
        request = dict(schema=APPROVAL, decision=None, plan_sha256=plan_pin["sha256"],
                       execution=plan["execution"], producer=code["commit"], physical_uuid=None,
                       scopes=[COLD,WARM], data_access="sealed_observations_no_labels", approval_reference=None)
        _write(out/"application-template.json", request)
    return plan_pin


def admitted(root, out, code, sha, permit, device):
    path = artifact(root, out/"measurement-plan.json")
    ref = dict(path=str(path), sha256=sha)
    plan = json.loads(read(root, ref))
    need(plan["schema"] == PLAN and plan["producer"] == code["commit"]
         and type(plan["engineering"]) is bool, "Foreign deployment plan")
    approval(root, plan, ref, permit, device)
    execution = artifact(root, Path(plan["execution"]))
    worker = Path(plan["worker"])
    if plan["engineering"]:
        job = plan["cases"]["engineering"]
        need(job["workload"]["work_items"] == 36 and job["arm"] == "engineering_fixture"
             and job["legacy_cost"] is job["legacy_output"] is None
             and [Path(job[key]["path"]) for key in ("checkpoint", "sidecar")]
                 == [out/"fixture"/name for name in ("fixture.pt","fixture.pt.json")]
             and Path(job["workload"]["file"]["path"]) == out/"fixture"/"observation.pt",
             "Engineering aperture permits only generated 6×6 fixtures")
        expected = draft(root, code, worker, execution, fixture=job)
        old = []
    else:
        cases, freeze, identity, old = roster(root, runtime(worker))
        expected = header(code, worker, execution, cases, freeze, identity)
    need(plan == expected, "Deployment plan changed registered model, observation, or timing boundary")
    return ref, plan, execution, worker, old


def header(code, worker, execution, cases, freeze, identity):
    """Compare a real plan to the independently rebuilt immutable registration."""
    return dict(schema=PLAN, producer=code["commit"], engineering=freeze is None, boundary=BOUNDARY,
                execution=str(execution), worker=str(worker), worker_commit=LEGACY,
                adapter_sha256=hash_file(ADAPTER), runtime=identity, freeze=freeze, cases=cases,
                repetitions=COUNT, warmups=WARMUPS, timeout_per_process_seconds=TIMEOUT,
                paired_execution="single_physical_device_serial_whole_campaign")


def stagesummary(samples):
    return dict(wall=timing_summary([s["wall_ms"] for s in samples], repeats=len(samples), blocks=1),
                by_phase={key: timing_summary([s["phases"][key]["wall_ms"] for s in samples],
                                                 repeats=len(samples), blocks=1) for key in PHASES},
                model_setup_by_phase={key: timing_summary([s["setup"][key]["wall_ms"] for s in samples],
                                                 repeats=len(samples), blocks=1) for key in SETUP}
                       if len({s["pid"] for s in samples}) == len(samples) else None,
                host_process_peak_rss_kib=max(s["rss_kib"] for s in samples),
                peak_allocated_cuda_bytes=max((s["cuda_peak_bytes"] for s in samples
                                               if s["cuda_peak_bytes"] is not None), default=None),
                graph_construction_ms=None, measured_skips=None)


def bound_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def audit_stage(record, lineage, job):
    scope, device = lineage["scope"], lineage["device"]
    samples = record["samples"]
    environment = record["environment"]
    need(record["lineage"] == lineage and len(samples) == COUNT[scope]
         and record["warmups"] == WARMUPS[scope] and record["batch"] == 1
         and record["optimizer_updates"] == 0 and record["query_accessed"] is False
         and environment["arithmetic"] == "ordered" and environment["threads"] == 1
         and (environment["device"] == "cpu" if device == "cpu" else
              environment["device"] == "cuda" and environment["gpu"]["uuid"] == device),
         "Stage scope, device or repetition count differs")
    need(len({s["pid"] for s in samples}) == (COUNT[COLD] if scope == COLD else 1),
         "Cold fresh processes or warm resident-model scope differs")
    expected = (job["checkpoint"]["sha256"], job["workload"]["file"]["sha256"],
                job["workload"]["tensor_sha256"], job["workload"]["work_items"])
    for sample in samples:
        need(tuple(sample[k] for k in ("checkpoint","observation","workload","pixels")) == expected,
             "Measured checkpoint or observation differs")
        need(sample["scope"] == scope and sample["device"] == device and sample["batch"] == 1
             and sample["observation_reads"] == 1 and sample["graph_builds"] == 0 and sample["skip_count"] is None
             and set(sample["phases"]) == set(PHASES) and set(sample["setup"]) == set(SETUP)
             and all(bound_time(sample[k]) for k in ("wall_ms","request_ms","process_exit_ms","unattributed_ms"))
             and (device == "cpu") == (sample["cuda_peak_bytes"] is None),
             "Measurement operations do not match the registered timing boundary")
        phase = list(sample["phases"].values()) + (list(sample["setup"].values()) if scope == COLD else [])
        need(all(bound_time(x["wall_ms"]) and ((device == "cpu" and x["cuda_peak_bytes"] is None)
                      or (device != "cpu" and type(x["cuda_peak_bytes"]) is int and x["cuda_peak_bytes"] >= 0)) for x in phase)
             and math.isclose(sum(x["wall_ms"] for x in phase) + sample["unattributed_ms"], sample["wall_ms"])
             and sample["process_exit_ms"] >= sample["wall_ms"] >= sample["request_ms"] > 0
             and (scope == COLD or sample["wall_ms"] == sample["request_ms"])
             and sample["cuda_peak_bytes"] == max((x["cuda_peak_bytes"] for x in phase
                                                  if x["cuda_peak_bytes"] is not None), default=None),
             "Stage wall clock, component times or peak memory differ")
    need(len({s["prediction"] for s in samples}) == len({s["probability_state"] for s in samples}) == 1
         and (job["legacy_output"] is None or samples[0]["probability_state"] == job["legacy_output"])
         and record["summary"] == stagesummary(samples),
         "Prediction lineage or independently recomputed statistics differ")
    return samples[0]["prediction"], samples[0]["probability_state"]


def capture(root, worker, plan_pin, permit, lineage, attempt, lease):
    scope, device = lineage["scope"], lineage["device"]
    args = ["measure", "--plan", plan_pin["path"], "--plan-sha", plan_pin["sha256"],
            "--case", lineage["case"], "--scope", scope, "--device", device]
    if permit is not None:
        args += ["--approval", permit["path"], permit["sha256"]]
    if lease is not None:
        args += ["--lease", str(lease.fileno()), "--lease-path", str(lease_path(device))]
    samples, environments = [], []
    for i in range(COUNT[COLD] if scope == COLD else 1):
        result, start, spent = launch(root, worker, args, gpu=device, lease=lease, attempt=attempt, label=f"worker-{i}")
        need(result["scope"] == scope and result["device"] == device and result["optimizer_updates"] == 0
             and result["query_accessed"] is False, "Worker escapes the approved scope")
        environments.append(result["environment"])
        for sample in result["samples"]:
            wall = (sample.pop("first_prediction_ns")-start)/1e6 if scope == COLD else sample["request_ms"]
            phases = list(sample["phases"].values()) + (list(result["setup"].values()) if scope == COLD else [])
            samples.append(dict(**sample, wall_ms=wall, process_exit_ms=spent, pid=result["pid"],
                                setup=result["setup"], scope=scope, device=device,
                                unattributed_ms=wall-sum(x["wall_ms"] for x in phases),
                                cuda_peak_bytes=max((x["cuda_peak_bytes"] for x in phases if x["cuda_peak_bytes"] is not None),
                                                    default=None)))
    need(all(v == environments[0] for v in environments), "Cold worker environment changed")
    return dict(lineage=lineage, samples=samples, warmups=WARMUPS[scope], batch=1,
                environment=environments[0], query_accessed=False, optimizer_updates=0,
                summary=stagesummary(samples))


def campaign(root, out, code, sha, permit, device, *, limit=None, report=None):
    ref, plan, execution, worker, old = admitted(root, out, code, sha, permit, device)
    dispatch = dict(plan=ref, approval=permit, device=device, source=code["commit"])
    if report is None:
        execution.mkdir(exist_ok=True)
        lease = case_lock(execution)
    else:
        artifact(root, report)
        need(not report.exists(), "Fresh audit result required")
        lease = nullcontext()
    with lease:
        binding = execution/"run.json"
        if not binding.exists() and report is None:
            _write(binding, dispatch)
        need(document(root, pin(binding)) == dispatch, "Execution already has a different approval or device")
        results, completed = {}, 0
        for case, job in plan["cases"].items():
            record = dict(scene=job["scene"], seed=job["seed"], arm=job["arm"], checkpoint=job["checkpoint"]["sha256"],
                          evaluation=job["evaluation_sha256"], workload=job["workload"]["tensor_sha256"], scopes={})
            results[case] = record
            outputs, environments = [], []
            for scope in (COLD, WARM):
                stage = execution/case/scope
                lineage = {**dispatch, "case":case, "scope":scope, "boundary":BOUNDARY[scope]}
                accepted = stage/"accepted.json"
                if not accepted.exists() and report is None:
                    if limit is not None and completed >= limit:
                        return dict(accepted_this_call=completed, completed=False)
                    stage.mkdir(parents=True, exist_ok=True)
                    attempt = Path(mkdtemp(prefix="attempt-", dir=stage))
                    try:
                        with (nullcontext() if device == "cpu" else gpu_measurement_lease(device)) as fd:
                            value = capture(root, worker, ref, permit, lineage, attempt, fd)
                        audit_stage(value, lineage, job)
                        require_frozen_code(code)
                        _write(accepted, value)
                    except Exception as error:
                        _write(attempt/"failure.json", dict(type=type(error).__name__, message=str(error), lineage=lineage))
                        raise
                    completed += 1
                proof = pin(accepted)
                value = document(root, proof)
                outputs.append(audit_stage(value, lineage, job))
                environments.append(value["environment"])
                record["scopes"][scope] = dict(receipt=proof, statistics=value["summary"])
            need(outputs[0] == outputs[1] and environments[0] == environments[1],
                 "Cold and warm used different environments or returned different ordered outputs")
        require_frozen_code(code)
    if report is None:
        return dict(accepted_this_call=completed, completed=True)
    old_map = {(v["scene"], v["seed"], f'{v["method"]}/{v["arm"]}'):v for v in old}
    for row in results.values():
        if old_map:
            native = old_map[row["scene"], row["seed"], row["arm"]]
            need((row["checkpoint"], row["evaluation"], row["workload"])
                 == (native["model"], native["protocol"], native["workload"]), "Historical accuracy/coverage is from a foreign pair")
            row["historical"] = native
    return _write(report, dict(status="audited", **dispatch, engineering=plan["engineering"], cases=results,
                               new_optimizer_updates=0, query_accessed=False, new_resident_measurements=0))
