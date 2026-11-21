"""Prepare, execute and independently audit the additional frozen54 E2E scopes."""

from contextlib import nullcontext
import json
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
    APPROVAL, COLD, WARM, COUNT, WARMUPS, TIMEOUT, PHASES, SETUP, PLAN, TIMING, approval, read, worker_binding,
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
    stop = perf_counter_ns()
    if attempt is not None:
        (attempt / f"{label}.log").write_text(process.stderr + process.stdout)
    if process.returncode:
        raise RuntimeError(f"Frozen deployment worker failed ({process.returncode}): {process.stderr[-1600:]}")
    output = process.stdout.splitlines(keepends=True)[-1]
    if attempt is None:
        return json.loads(output)
    payload = attempt/f"{label}.json"
    with payload.open("x", encoding="utf-8", newline="") as stream:
        stream.write(output)
    return _write(attempt/f"{label}-invocation.json",
                  dict(stdout=pin(payload), process_started_ns=start, process_exited_ns=stop))


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
        fixture = launch(root, worker, ["fixture", "--out", str(out/"fixture")])
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
                adapter_sha256=hash_file(ADAPTER), runtime=identity, freeze=freeze, cases=cases, timing=TIMING,
                repetitions=COUNT, warmups=WARMUPS, timeout_per_process_seconds=TIMEOUT,
                paired_execution="single_physical_device_serial_whole_campaign")


def stagesummary(samples, scope):
    return dict(wall=timing_summary([s["wall_ms"] for s in samples], repeats=len(samples), blocks=1),
                by_phase={key: timing_summary([s["phases"][key]["wall_ms"] for s in samples],
                                                 repeats=len(samples), blocks=1) for key in PHASES},
                model_setup_by_phase={key: timing_summary([s["setup"][key]["wall_ms"] for s in samples],
                                                 repeats=len(samples), blocks=1) for key in SETUP}
                       if scope == COLD else None,
                host_process_peak_rss_kib=max(s["rss_kib"] for s in samples),
                peak_allocated_cuda_bytes=max((s["cuda_peak_bytes"] for s in samples
                                               if s["cuda_peak_bytes"] is not None), default=None),
                graph_construction_ms=None, measured_skips=None)


def elapsed(start, stop):
    need(type(start) is int and type(stop) is int and 0 <= start < stop, "Invalid monotonic clock interval")
    return (stop-start)/1e6


def phase_end(phases, names, previous, deadline, device):
    need(set(phases) == set(names), "Worker timing phases differ")
    for name in names:
        item = phases[name]
        start, stop, peak = item["start_ns"], item["finish_ns"], item["cuda_peak_bytes"]
        need(elapsed(start, stop) == item["wall_ms"] and previous <= start < stop <= deadline
             and (peak is None if device == "cpu" else type(peak) is int and peak >= 0),
             "Worker phase clock or device peak differs")
        previous = stop
    return previous


def worker_rows(raw, binding, spawned, exited, job):
    scope, device = binding["scope"], binding["device"]
    environment = raw["environment"]
    need(raw["binding"] == binding and type(raw["pid"]) is int and raw["pid"] > 0
         and raw["optimizer_updates"] == 0 and raw["query_accessed"] is False,
         "Worker plan, attempt, scope or device binding differs")
    need(raw["timing"] == TIMING[scope] and not {"scope","excluded"}.intersection(environment),
         "Cold or warm timing boundary differs from the plan")
    need(environment["arithmetic"] == "ordered" and environment["threads"] == 1
         and (environment["device"] == "cpu" and "gpu" not in environment if device == "cpu" else
              environment["device"] == "cuda" and environment["gpu"]["uuid"] == device),
         "Worker hardware or arithmetic differs")
    warmups = WARMUPS[scope]
    timed = 1 if scope == COLD else COUNT[WARM]
    need(len(raw["requests"]) == warmups+timed, "Worker warmup or measured repetition count differs")
    lifetime = elapsed(spawned, exited)
    previous = phase_end(raw["setup"], SETUP, spawned, exited, device)
    expected = (job["checkpoint"]["sha256"], job["workload"]["file"]["sha256"],
                job["workload"]["tensor_sha256"], job["workload"]["work_items"])
    outputs, samples = set(), []
    for i, sample in enumerate(raw["requests"]):
        role = "warmup" if i < warmups else "measured"
        start, stop = sample["request_started_ns"], sample["first_prediction_ns"]
        request = elapsed(start, stop)
        need(type(sample["sequence"]) is int and sample["sequence"] == i and sample["role"] == role
             and previous <= start < stop <= exited, "Worker request sequence, role or interval overlaps")
        need(phase_end(sample["phases"], PHASES, start, stop, device) == stop,
             "Request did not stop at synchronized host materialization")
        need(tuple(sample[k] for k in ("checkpoint","observation","workload","pixels")) == expected,
             "Measured checkpoint or observation differs")
        need(sample["batch"] == sample["observation_reads"] == 1 and sample["graph_builds"] == 0
             and sample["skip_count"] is None and type(sample["rss_kib"]) is int and sample["rss_kib"] > 0,
             "Measurement operations do not match the registered timing boundary")
        previous = stop
        outputs.add((sample["prediction"], sample["probability_state"]))
        if role == "measured":
            wall_start = spawned if scope == COLD else start
            phases = [sample["phases"][k] for k in PHASES]
            if scope == COLD:
                phases += [raw["setup"][k] for k in SETUP]
            remainder_ns = stop-wall_start-sum(p["finish_ns"]-p["start_ns"] for p in phases)
            samples.append(dict(**sample, wall_ms=elapsed(wall_start, stop), request_ms=request,
                                process_started_ns=spawned, process_exited_ns=exited, process_exit_ms=lifetime,
                                pid=raw["pid"], setup=raw["setup"], scope=scope, device=device,
                                invocation=binding["invocation"], unattributed_ms=remainder_ns/1e6,
                                cuda_peak_bytes=max((p["cuda_peak_bytes"] for p in phases
                                                    if p["cuda_peak_bytes"] is not None), default=None)))
    need(len(outputs) == 1 and (job["legacy_output"] is None
                               or next(iter(outputs))[1] == job["legacy_output"]),
         "Worker prediction lineage differs from the ordered workload")
    return samples


def rebuild_stage(root, stage, lineage, job, attempt, invocations):
    scope, device = lineage["scope"], lineage["device"]
    count = COUNT[COLD] if scope == COLD else 1
    need(isinstance(attempt, str) and Path(attempt).name == attempt and attempt.startswith("attempt-")
         and len(invocations) == count, "Missing or foreign worker attempt")
    directory = stage/attempt
    samples, environments, pids = [], [], set()
    previous_exit = 0
    for i, link in enumerate(invocations):
        need(link["path"] == str(directory/f"worker-{i}-invocation.json"), "Foreign worker invocation path")
        envelope = json.loads(read(root, link))
        content = envelope["stdout"]
        need(content["path"] == str(directory/f"worker-{i}.json"), "Foreign worker stdout path")
        raw = json.loads(read(root, content))
        spawned, exited = envelope["process_started_ns"], envelope["process_exited_ns"]
        need(previous_exit <= spawned and raw["pid"] not in pids, "Cold worker process was reused or overlapped")
        binding = worker_binding(lineage["plan"]["sha256"], lineage["case"], scope, device, attempt, i)
        samples.extend(worker_rows(raw, binding, spawned, exited, job))
        environments.append(raw["environment"])
        previous_exit = exited
        pids.add(raw["pid"])
    need(all(v == environments[0] for v in environments)
         and len({(s["prediction"],s["probability_state"]) for s in samples}) == 1,
         "Worker environment or ordered outputs changed across subprocesses")
    return dict(lineage=lineage, attempt=attempt, invocations=invocations, samples=samples,
                warmups=WARMUPS[scope], batch=1, timing=TIMING[scope], environment=environments[0],
                query_accessed=False, optimizer_updates=0, summary=stagesummary(samples, scope))


def audit_stage(root, stage, record, lineage, job):
    need(record["lineage"] == lineage and record["timing"] == TIMING[lineage["scope"]],
         "Accepted scope or timing boundary differs from the plan")
    expected = rebuild_stage(root, stage, lineage, job, record["attempt"], record["invocations"])
    need(record == expected, "Accepted rows or summary differ from pinned original worker requests")
    sample = expected["samples"][0]
    return sample["prediction"], sample["probability_state"]


def capture(root, worker, plan_pin, permit, lineage, job, attempt, lease):
    scope, device = lineage["scope"], lineage["device"]
    args = ["measure", "--plan", plan_pin["path"], "--plan-sha", plan_pin["sha256"],
            "--case", lineage["case"], "--scope", scope, "--device", device, "--attempt", attempt.name]
    if permit is not None:
        args += ["--approval", permit["path"], permit["sha256"]]
    if lease is not None:
        args += ["--lease", str(lease.fileno()), "--lease-path", str(lease_path(device))]
    invocations = [launch(root, worker, [*args, "--invocation", str(i)], gpu=device, lease=lease,
                          attempt=attempt, label=f"worker-{i}") for i in range(COUNT[COLD] if scope == COLD else 1)]
    return rebuild_stage(root, attempt.parent, lineage, job, attempt.name, invocations)


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
                            value = capture(root, worker, ref, permit, lineage, job, attempt, fd)
                        require_frozen_code(code)
                        _write(accepted, value)
                    except Exception as error:
                        _write(attempt/"failure.json", dict(type=type(error).__name__, message=str(error), lineage=lineage))
                        raise
                    completed += 1
                proof = pin(accepted)
                value = document(root, proof)
                outputs.append(audit_stage(root, stage, value, lineage, job))
                environments.append(value["environment"])
                record["scopes"][scope] = dict(receipt=proof, timing=value["timing"], statistics=value["summary"])
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
