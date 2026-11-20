"""Time the first verified result of original, audited WP10 CPU packages."""

from time import perf_counter_ns

ENTERED_PYTHON = perf_counter_ns()

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
from statistics import median
import subprocess
import sys

PROGRAM = Path(__file__).resolve()
ROOT = PROGRAM.parents[1]
LEGACY = "2e5807e46d1c37163fbe8aed4fe54cd62eed139c"
MANIFEST_SHA = "a0cae3f8723902555820964e88184f476b11da5a4a01e19e92d67f0a1a7dae6b"
COST_SHA = "2d4b0618e6c5548f68ce1b3a9c3b8076ba4b2688512a8211482264f1882c2e9d"
SCOPE = "parent_launch_to_first_verified_full_three_head_result"
CPU_ENV = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")


def require_checkout(directory, commit):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()

    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Clean checkout at {commit} required: {directory}")


def pin(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def read(ref):
    data = Path(ref["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError(f"Input digest changed: {ref['path']}")
    return json.loads(data)


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def worker(legacy, manifest, name, cpu):
    sys.path.insert(0, str(legacy))
    import torch
    from compression.precision_startup import _memory_kib
    from inference.wp10_low_rank_execution import _outputs, _scan, load_manifest, package
    from research_runtime import code_snapshot
    from transfer.container_runtime import training_container_kind

    torch.set_num_threads(1)
    code = code_snapshot()
    if (not training_container_kind() or torch.cuda.is_available() or code["commit"] != LEGACY or code["dirty"]
            or os.sched_getaffinity(0) != {cpu} or torch.get_num_threads() != 1):
        raise ValueError("The original clean CPU runtime and pinned CPU are required")
    initialized = perf_counter_ns()
    reg, tiles = load_manifest(manifest, code)
    inputs_ready = perf_counter_ns()
    entry = reg["cases"][name]
    model = package(entry)
    model_ready = perf_counter_ns()
    values, factors = _scan(model, reg, tiles, evidence=True)
    forward_done = perf_counter_ns()
    memory = _memory_kib()
    hashes = _outputs(values, reg)
    verified = perf_counter_ns()
    print(json.dumps(dict(
        case=name, manifest=manifest, cpu=cpu, threads=1, pid=os.getpid(), torch=str(torch.__version__),
        tiles=reg["tiles"], output_hashes=hashes, state=entry["package"]["state"],
        package_bytes=entry["package"]["bytes"], parameters=entry["package"]["parameters"],
        factor_calls=factors, memory_after_forward=memory, query=0, gt=0, optimizer_updates=0, calibration=0,
        clocks_ns=dict(python=ENTERED_PYTHON, initialized=initialized, inputs=inputs_ready,
                       model=model_ready, forward=forward_done, verified=verified))), flush=True)


def identity_matches(row, known, pid, cpu):
    fields = ("case", "manifest", "tiles", "output_hashes", "state", "package_bytes", "parameters", "factor_calls")
    if (any(row[key] != known[key] for key in fields) or row["pid"] != pid
            or row["cpu"] != cpu or row["threads"] != 1
            or any(row[key] != 0 for key in ("query", "gt", "optimizer_updates", "calibration"))):
        raise ValueError("First result differs from the original package, input, output, or access scope")


def first_result(command, legacy, prefix, check, result_timeout=180, exit_timeout=90):
    stdout = bytearray()
    recorded = prefix.with_suffix(".stdout.log")
    with prefix.with_suffix(".stderr.log").open("xb") as stderr:
        started = perf_counter_ns()
        deadline = started + result_timeout * 1_000_000_000
        try:
            with subprocess.Popen(command, cwd=legacy, env={**os.environ, **CPU_ENV},
                                  stdout=subprocess.PIPE, stderr=stderr) as child:
                try:
                    with selectors.DefaultSelector() as watcher:
                        watcher.register(child.stdout, selectors.EVENT_READ)
                        while b"\n" not in stdout:
                            remaining = max(0, (deadline - perf_counter_ns()) / 1e9)
                            if not watcher.select(remaining):
                                raise TimeoutError("Worker timed out before a complete first result")
                            chunk = os.read(child.stdout.fileno(), 4096)
                            if not chunk or len(stdout) + len(chunk) > 65536:
                                raise RuntimeError("Worker did not publish one complete first result")
                            stdout.extend(chunk)
                    line, _, rest = bytes(stdout).partition(b"\n")
                    row = json.loads(line)
                    check(row, child.pid)
                    accepted = perf_counter_ns()
                    tail, _ = child.communicate(timeout=exit_timeout)
                    exited = perf_counter_ns()
                    stdout.extend(tail)
                    if rest or tail or child.returncode:
                        raise RuntimeError(f"Worker did not exit cleanly after its first result: {child.returncode}")
                    return row, started, accepted, exited
                except BaseException:
                    if child.poll() is None:
                        child.kill()
                    tail, _ = child.communicate()
                    stdout.extend(tail)
                    raise
        finally:
            with recorded.open("xb") as stream:
                stream.write(stdout)


def timing(row, started, accepted, exited):
    checkpoints = ("python", "initialized", "inputs", "model", "forward", "verified")
    clocks = [started, *(row["clocks_ns"][name] for name in checkpoints), accepted, exited]
    if any(type(ns) is not int for ns in clocks) or sorted(clocks) != clocks or accepted == started:
        raise ValueError("First-result clock ordering differs from the declared window")
    names = ("launch_to_python", "initialize", "authenticate_and_prepare_input", "load_authenticated_package",
             "first_full_graph_and_forward", "validate_and_hash_outputs", "serialize_deliver_and_verify", "exit_after_result")
    stages = dict(zip(names, ((b - a) / 1e6 for a, b in zip(clocks, clocks[1:])), strict=True))
    return dict(first_result_ms=(accepted - started) / 1e6, subprocess_exit_ms=(exited - started) / 1e6,
                start_ns=started, accepted_ns=accepted, exit_ns=exited, stages_ms=stages)


def baseline(runs):
    manifest = dict(path=str(runs / "wp10-low-rank-frozen-execution01/manifest.json"), sha256=MANIFEST_SHA)
    cost = dict(path=str(runs / "wp10-low-rank-cpu-cost01/receipt.json"), sha256=COST_SHA)
    reg, old = read(manifest), read(cost)
    resident = read(old["resident"])
    if (old["status"] != "passed_low_rank_cpu_cost" or old["commit"] != LEGACY or old["manifest"] != manifest
            or reg["commit"] != LEGACY or reg["data_protocol"] != "wp10_xuzhou_k5_v5"
            or reg["observation_role"] != "devval_image_only" or len(reg["cases"]) != 18
            or len(reg["tiles"]) != 5 or sum(row["pixels"] for row in reg["tiles"]) != 3920):
        raise ValueError("Original audited 18-package development workload required")
    audit = read(reg["audit"])
    runtime = dict(path=str(runs / "wp10-wp12-low-rank-proof02/commands/real-cpu-commands/runtime.json"),
                   sha256="ff300de9f986483fec5e04dae5bff226387f8b439a7207860a333a2ccf696628")
    cpu = read(runtime)["pinned_cpu"]
    old_resident = {}
    for group in resident.values():
        for arm, name in group["cases"].items():
            old_resident.setdefault(name, {})[group["kind"] + ":" + ",".join(group["cases"])] = (
                group["timing"]["summary"][arm]["wall_p50_ms"])
    for name, rows in old["startup"].items():
        selected = audit["report"][name]["selected"]
        known = [dict(zip(("fused", "graph", "cnn"), block, strict=True)) for block in selected["probabilities"]]
        if len(rows) != 3 or any(row["output_hashes"] != known for row in rows):
            raise ValueError("Prior cost and independently audited output hashes disagree")
    if set(old["startup"]) != set(reg["cases"]) or set(old_resident) != set(reg["cases"]):
        raise ValueError("Historical evidence does not cover each timed package")
    pins = [manifest, cost, runtime, reg["resident_inputs"], reg["request"], reg["registration"], reg["audit"],
            old["resident"], *(entry["package"][kind] for entry in reg["cases"].values() for kind in ("weights", "sidecar"))]
    return manifest, cpu, pins, old, old_resident


def registration(runs, legacy, commit):
    manifest, cpu, pins, old, old_resident = baseline(runs)
    if cpu not in os.sched_getaffinity(0):
        raise ValueError(f"Original pinned CPU {cpu} is unavailable")
    seeds = [64, 512, 2048]
    arms = ["dense", "svd-r16", "svd-r8", "svd-r32", "random-r16", "full-rank"]
    orders = []
    for trial, order in enumerate((arms, arms[::-1], arms[3:] + arms[:3])):
        orders.extend(dict(repeat=trial, case=f"s{seed}-{arm}") for seed in seeds[trial:] + seeds[:trial]
                      for arm in order)
    if {item["case"] for item in orders} != set(old["startup"]):
        raise ValueError("Execution order differs from the frozen packages")
    plan = dict(commit=commit, legacy_commit=LEGACY, legacy_checkout=str(legacy), scope=SCOPE,
                manifest=manifest, source_pins=pins, cpu=cpu, repeats=3, warmups=0,
                result_timeout=180, exit_timeout=90, filesystem_cache="uncontrolled", input_role="devval_image_only",
                blocks=5, pixels=3920, schedules=orders, original_startup_seconds=old["startup_seconds"],
                estimate_seconds=[360, 480])
    return plan, old, old_resident


def execute(plan, old, old_resident, out):
    os.sched_setaffinity(0, {plan["cpu"]})
    samples = {}
    for index, scheduled in enumerate(plan["schedules"]):
        name = scheduled["case"]
        ref = old["startup"][name][0]
        prefix = out / f"{index:02d}-{name}"
        cmd = [sys.executable, str(PROGRAM), "--worker", plan["legacy_checkout"], plan["manifest"]["path"],
               plan["manifest"]["sha256"], name, str(plan["cpu"])]
        write(prefix.with_suffix(".command.json"), cmd)
        try:
            row, started, accepted, exited = first_result(
                cmd, plan["legacy_checkout"], prefix, lambda value, pid: identity_matches(value, ref, pid, plan["cpu"]),
                plan["result_timeout"], plan["exit_timeout"])
            result = dict(**scheduled, index=index, result=row, timing=timing(row, started, accepted, exited))
        except BaseException as error:
            write(prefix.with_suffix(".failure.json"), dict(index=index, **scheduled, error=repr(error)))
            raise
        write(prefix.with_suffix(".sample.json"), result)
        samples.setdefault(name, []).append(result)
        if index in (1, 17, 35, 53):
            print(f"Verified {index + 1}/54 first results", flush=True)
    if len({s["result"]["pid"] for group in samples.values() for s in group}) != 54:
        raise ValueError("Expected 54 independent worker process identities")
    if any(pin(ref["path"]) != ref for ref in plan["source_pins"]):
        raise ValueError("Protected input changed during cost measurement")
    require_checkout(ROOT, plan["commit"])
    require_checkout(plan["legacy_checkout"], LEGACY)
    summary = {}
    for name, rows in samples.items():
        values = [row["timing"]["first_result_ms"] for row in rows]
        rss = [row["result"]["memory_after_forward"]["peak_rss_kib"] for row in rows]
        dense = samples[name.split("-")[0] + "-dense"]
        paired = [v / d["timing"]["first_result_ms"] for v, d in zip(values, dense, strict=True)]
        summary[name] = dict(first_result_ms=values, median_ms=median(values), range_ms=[min(values), max(values)],
                             paired_dense_ratios=paired, median_dense_ratio=median(paired), peak_rss_kib=rss,
                             package_bytes=rows[0]["result"]["package_bytes"], parameters=rows[0]["result"]["parameters"],
                             original_resident_medians_ms=old_resident[name],
                             original_load_median_ms=median(row["load_ms"] for row in old["startup"][name]),
                             original_first_sweep_median_ms=median(row["first_sweep_ms"] for row in old["startup"][name]))
    write(out / "receipt.json", dict(status="passed", registration=pin(out / "registration.json"), scope=SCOPE,
                                     cpu=plan["cpu"], first_results=54, summary=summary, query=0, gt=0,
                                     optimizer_updates=0, calibration=0))


def main():
    if len(sys.argv) == 7 and sys.argv[1] == "--worker":
        _, _, legacy, manifest, digest, name, cpu = sys.argv
        worker(Path(legacy), dict(path=manifest, sha256=digest), name, int(cpu))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--expected-commit", required=True)
    for name in ("runs", "legacy", "out"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--registration-sha256")
    args = parser.parse_args()
    require_checkout(ROOT, args.expected_commit)
    require_checkout(args.legacy, LEGACY)
    if args.out.parent != args.runs or args.runs.name != "runs" or ROOT in args.out.parents:
        raise ValueError("Use a new output directory immediately below the external project runs")
    plan, old, old_resident = registration(args.runs, args.legacy, args.expected_commit)
    if args.command == "prepare":
        args.out.mkdir()
        write(args.out / "registration.json", plan)
        print(json.dumps(pin(args.out / "registration.json")))
    else:
        ref = dict(path=str(args.out / "registration.json"), sha256=args.registration_sha256)
        if read(ref) != plan:
            raise ValueError("Exact preregistered workload and original affinity required")
        execute(plan, old, old_resident, args.out)


if __name__ == "__main__":
    main()
