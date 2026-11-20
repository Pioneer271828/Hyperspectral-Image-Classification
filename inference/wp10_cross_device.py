"""Compare the same frozen WP10 observation-to-output scope on explicit devices."""
from dataclasses import replace
import io
import os
from pathlib import Path
import resource
import subprocess
from time import perf_counter_ns

import numpy as np
import torch

from compression.low_rank_recovery import TARGET, factor_calls, factor_counts
from evaluation.wp12_cost_primitives import timing_summary
from inference import wp10_deployment_cost as warm, wp10_low_rank_execution as frozen
from inference.native_deployment import Runtime, clock_ms
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, p4_runtime_device as device, registered_development as dev
from transfer.shared_gpu_lease import physical_lease


SCHEMA = "wp10_end_to_end_device_v2"
ATOL, RTOL = 1e-6, 1e-5
FIRST_COUNT = 3


def result_dir(out, protected):
    out = external_directory(out)
    if out.exists() or any(out == Path(item["path"]) or out in Path(item["path"]).parents for item in protected):
        raise ValueError("New output directory separate from frozen inputs required")
    return out


def rank_plan(pin, code, *, cpu=False):
    require_frozen_code(code)
    reg = admission.read(pin)
    if (reg["kind"] != frozen.KIND or reg["policy"] != warm.PLAN or reg["commit"] != code["commit"]
            or reg["source_hash"] != code["source_hash"] or reg["seeds"] != list(warm.SEEDS)
            or reg["observation_role"] != "devval_image_only" or reg["data_protocol"] != "wp10_xuzhou_k5_v5"
            or reg["source_manifest"]["sha256"] != frozen.PUBLISHED_MANIFEST
            or reg["baseline_evidence"]["sha256"] != warm.REFERENCE_CPU_COST
            or (cpu and reg["runtime"] != frozen.runtime())):
        raise ValueError("Current sealed observation-only plan required")
    return reg


def rank_pairs():
    return {str(seed): (f"s{seed}-dense", f"s{seed}-svd-r16") for seed in warm.SEEDS}


def rank_inputs(reg, pin):
    return [pin, reg["resident_inputs"], *[entry["package"][key]
            for entry in reg["cases"].values() for key in ("weights", "sidecar")]]


def _regions(reg, runtime):
    start = perf_counter_ns()
    host = warm._requests(reg)
    prep = clock_ms(start)
    if runtime.device.type == "cpu":
        return host, prep, None
    runtime.sync()
    start = perf_counter_ns()
    inputs = [replace(item, x=item.x.to(runtime.device), waves=item.waves.to(runtime.device)) for item in host]
    runtime.sync()
    return inputs, prep, clock_ms(start)


def sample(runtime, reg):
    runtime.sync()
    start = perf_counter_ns()
    cuda = runtime.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(runtime.device)
    requests, prep, upload = _regions(reg, runtime)
    result = runtime.run(requests, "full")
    transfer = perf_counter_ns()
    values = [tuple(t.detach().to("cpu").contiguous() for t in row) for row in result["predictions"]]
    runtime.sync()
    download = clock_ms(transfer) if cuda else None
    materialization = perf_counter_ns()
    hashes = frozen._outputs(values, reg)
    output = clock_ms(materialization)
    wall = clock_ms(start)
    blocks = len(requests)
    if result["counts"]["full_calls"] != blocks or result["counts"]["graph_builds"] != blocks or result["counts"]["graph_reuses"]:
        raise ValueError("Full native three-head path must rebuild every graph")
    return dict(wall_ms=wall, read_verify_decode_ms=prep, host_to_device_ms=upload,
                forward_ms=result["elapsed_ms"], graph_ms=result["graph_ms"],
                graph_clock="cuda_event" if cuda else "host_wall",
                remaining_forward_ms=(result["gpu_event_ms"] if cuda else result["elapsed_ms"]) - result["graph_ms"],
                forward_event_ms=result["gpu_event_ms"], device_to_host_ms=download,
                output_materialization_and_hash_ms=output, graph_calls=blocks, blocks=blocks,
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
                hashes=hashes), values


def summary(samples, pixels):
    phases = ("wall_ms", "read_verify_decode_ms", "host_to_device_ms", "forward_ms", "forward_event_ms",
              "graph_ms", "remaining_forward_ms", "device_to_host_ms", "output_materialization_and_hash_ms")
    result = {name: timing_summary([row[name] for row in samples], repeats=len(samples), blocks=samples[0]["blocks"])
              for name in phases if samples[0][name] is not None}
    return dict(phases=result, sequential_pixels_per_second=pixels * len(samples) * 1000 / sum(r["wall_ms"] for r in samples),
                max_rss_kib=max(r["rss_peak_kib"] for r in samples),
                max_cuda_allocated_bytes=max((r["cuda_peak_allocated_bytes"] for r in samples
                                              if r["cuda_peak_allocated_bytes"] is not None), default=None),
                max_cuda_reserved_bytes=max((r["cuda_peak_reserved_bytes"] for r in samples
                                             if r["cuda_peak_reserved_bytes"] is not None), default=None))


def compare(reference, candidate):
    rows = {head: dict(elements=0, exceeded=0, max_abs=0., changed=0) for head in frozen.HEADS}
    for lhs, rhs in zip(reference, candidate, strict=True):
        for head, x, y in zip(frozen.HEADS, lhs, rhs, strict=True):
            a, b = x.numpy().astype(np.float64), y.numpy().astype(np.float64)
            if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Aligned finite three-head full outputs required")
            change = np.abs(a - b)
            row = rows[head]
            row["elements"] += a.size
            row["exceeded"] += int((change > ATOL + RTOL * np.abs(a)).sum())
            row["max_abs"] = max(row["max_abs"], float(change.max()))
            row["changed"] += int((a.argmax(1) != b.argmax(1)).sum())
    return rows


def driver_memory(gpu):
    proc_ids = {os.getpid()}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("NSpid:"):
            proc_ids.update(map(int, line.split()[1:]))
    try:
        text = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,used_gpu_memory",
                                        "--format=csv,noheader,nounits"], text=True)
        found = [int(fields[1]) for line in text.splitlines() if len(fields := line.split(",")) == 2
                 and fields[0].strip().isdigit() and int(fields[0]) in proc_ids and fields[1].strip().isdigit()]
    except subprocess.CalledProcessError:
        return None
    return found[0] if len(found) == 1 else None


def operators(runtime, reg, paths):
    outputs, hooks = {}, []

    def observe(path):
        def check(module, inputs, output):
            if inputs[0].device != runtime.device or output.device != runtime.device or output.dtype != torch.float32:
                raise ValueError("Point convolution ran on an unexpected device or dtype")
            outputs[path] = outputs.get(path, 0) + 1
        return check

    for path in paths:
        hooks.append(runtime.full.get_submodule(path).register_forward_hook(observe(path)))
    try:
        with factor_calls(runtime.full) as factors, torch.autograd.profiler.profile(use_cuda=False) as profile:
            sample(runtime, reg)
    finally:
        for hook in hooks:
            hook.remove()
    calls = factor_counts(runtime.full, len(reg["tiles"]))
    if outputs != dict.fromkeys(paths, len(reg["tiles"])) or factors != calls:
        raise ValueError("Native point convolution or factor call count changed")
    events = {r.key: r.count for r in profile.key_averages()
              if "conv2d" in r.key or "convolution" in r.key or "quantized::" in r.key}
    if not events:
        raise ValueError("No profiled convolution dispatch")
    return dict(device=str(runtime.device), point_calls=outputs, factor_calls=factors,
                convolution_dispatch=events, floating_heads=True)


def rank_reference(pin, code, out):
    reg = rank_plan(pin, code, cpu=True)
    protected = rank_inputs(reg, pin)
    out = result_dir(out, protected)
    values = {}
    for name, entry in reg["cases"].items():
        runtime = Runtime(device="cpu", full=frozen.package(entry), identity={"full": entry["package"]["state"]})
        row, data = sample(runtime, reg)
        if row["hashes"] != reg["proofs"][name]["output_hashes"]:
            raise ValueError("Reference CPU heads differ from the frozen bitwise baseline")
        values[name] = data
    require_frozen_code(code)
    admission.check_pins(protected)
    out.mkdir()
    torch.save(values, out / "probabilities.pt")
    return dev._json_new(out / "reference.json", dict(plan=pin, commit=code["commit"],
                        values=dev._pin(out / "probabilities.pt"), kind="wp10_bitwise_cpu_arrays"))


def accepted_arm(control, candidate, guards, evidence_passed=True):
    return candidate if (evidence_passed and all(row["passes_metric_guards"] for row in guards.values())) else control


def pair_run(reg, pairs, make, expected, proof, target, out, *, gpu=None):
    """Execute every diagnostic arm, then actually execute the protected route."""
    pixels = sum(row["pixels"] for row in reg["tiles"])
    cases, fallbacks, arrays = {}, {}, {}
    for group, names in pairs.items():
        models, first, sampled = {}, {k: [] for k in names}, {k: [] for k in names}
        series = {k: [] for k in names}
        orderings = []

        def fresh(name):
            start = perf_counter_ns()
            model = make(name)
            runtime = Runtime(device=target, full=model)
            runtime.sync()
            return runtime, clock_ms(start)

        def run(name, runtime):
            row, values = sample(runtime, reg)
            proof(name, row, values)
            return row, values

        for name in names:
            models[name], load = fresh(name)
            row, values = run(name, models[name])
            first[name].append(dict(row, model_load_and_device_ms=load))
            series[name].append(values)
        for _ in range(warm.WARMUPS):
            for name in names:
                run(name, models[name])
        for i in range(warm.REPEATS):
            order = names if i % 2 == 0 else names[::-1]
            orderings.append(list(order))
            for name in order:
                row, values = run(name, models[name])
                sampled[name].append(row)
                series[name].append(values)
        for name in names:
            dispatch = operators(models[name], reg, expected[name]["paths"])
            if expected[name]["int8"] and dispatch["convolution_dispatch"].get("onednn::qconv2d_pointwise") != len(reg["tiles"]):
                raise ValueError("Expected one true oneDNN integer point kernel per block")
            if not expected[name]["int8"] and any("qconv" in key or "quantized::" in key
                                                  for key in dispatch["convolution_dispatch"]):
                raise ValueError("FP32 path unexpectedly used a low-bit operator")
            cases[name] = dict(dispatch=dispatch, package_bytes=expected[name]["bytes"], state=expected[name]["state"],
                               samples=sampled[name], sample_summary=summary(sampled[name], pixels),
                               orderings=orderings, driver_process_memory_mib=driver_memory(gpu) if gpu else None)
        for i in range(1, FIRST_COUNT):
            for name in names if i % 2 == 0 else names[::-1]:
                runtime, load = fresh(name)
                row, values = run(name, runtime)
                first[name].append(dict(row, model_load_and_device_ms=load))
                series[name].append(values)
                del runtime
        control, candidate = names
        admitted = all(not row.get("exceeded", 0) and not row.get("changed", 0)
                       for trial in first[candidate] + sampled[candidate]
                       for row in trial.get("numeric", {}).values())
        effective = accepted_arm(control, candidate, expected[candidate]["guards"], admitted)
        fallback_row, values = run(effective, models[effective])
        fallbacks[group] = dict(requested=candidate, executed=effective, sample=fallback_row)
        arrays[group + "/protected"] = values
        models.clear()
        for name in names:
            cases[name].update(first=first[name], first_summary=summary(first[name], pixels))
            arrays[name] = series[name]
    torch.save(arrays, out / "full-probabilities.pt")
    return dict(cases=cases, protected_routes=fallbacks, outputs=dev._pin(out / "full-probabilities.pt"),
                pixels=pixels, blocks=len(reg["tiles"]), first_scope="fresh_model_reload_same_process",
                first_samples=FIRST_COUNT, warm_samples=warm.REPEATS, warmups=warm.WARMUPS)


def cuda_measure(pin, reference_pin, code, out, gpu):
    with physical_lease(gpu):
        admitted = device.gate(gpu)
        with device.arithmetic(device.CUDA_FP64_WEIGHT, gpu, resource_admission=admitted) as runtime:
            reg = rank_plan(pin, code)
            reference = admission.read(reference_pin)
            if (reference["kind"] != "wp10_bitwise_cpu_arrays" or reference["commit"] != code["commit"]
                    or reference["plan"] != pin):
                raise ValueError("Expected the same plan's bitwise CPU probability reference")
            values = torch.load(io.BytesIO(dev._read(reference["values"])), map_location="cpu", weights_only=True)
            if values.keys() != reg["cases"].keys() or any(frozen._outputs(data, reg) != reg["proofs"][name]["output_hashes"]
                                                         for name, data in values.items()):
                raise ValueError("CPU comparison arrays changed from the accepted original predictions")
            protected = rank_inputs(reg, pin) + [reference_pin, reference["values"]]
            out = result_dir(out, protected)
            out.mkdir()

            def make(name):
                return frozen.package(reg["cases"][name])

            def prove(name, row, data):
                row["numeric"] = compare(values[name], data)

            expected = {name: dict(state=entry["package"]["state"], bytes=entry["package"]["bytes"],
                                   paths=[TARGET], int8=False, guards=reg["proofs"][name]["guards"])
                        for name, entry in reg["cases"].items()}
            measured = pair_run(reg, rank_pairs(), make, expected, prove, "cuda:0", out, gpu=gpu)
            admission.check_pins(protected)
            require_frozen_code(code)
            return dev._json_new(out / "receipt.json", dict(
                **measured, schema=SCHEMA, kind="wp10_low_rank_cuda", commit=code["commit"], plan=pin,
                reference=reference_pin, source_manifest=reg["source_manifest"],
                resident_input_sha256=reg["resident_inputs"]["sha256"],
                device=runtime, resource_admission=admitted, tolerance=dict(atol=ATOL, rtol=RTOL),
                unsupported_on_cuda=["frozen_onednn_u8s8_cpu_only", "frozen_native_quantized_cpu_only",
                                     "ternary_packed_decode_fp32_no_qualified_cuda_kernel"],
                optimizer_updates=0, calibration_updates=0, gt=0, query=0, predictive_label_reads=0))
