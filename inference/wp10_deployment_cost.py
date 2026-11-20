"""Measure the missing warm end-to-end scope on the frozen WP10 release."""

import os
from pathlib import Path
import platform
import resource
from time import perf_counter_ns

import torch

from compression.low_rank import LowRankMatrix
from compression.low_rank_recovery import TARGET, factor_calls, factor_counts
from evaluation.wp12_cost_primitives import timing_summary
from evaluation.wp12_low_rank_evidence import verify_guard
from inference import wp10_low_rank_execution as frozen
from inference.native_deployment import Request, Runtime, clock_ms
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev


REFERENCE_CPU_COST = "2d4b0618e6c5548f68ce1b3a9c3b8076ba4b2688512a8211482264f1882c2e9d"
SCOPE = "warm_authenticated_observation_to_three_head_cpu_probabilities"
PLAN = "wp10_label_free_warm_plan_v1"
SEEDS = (64, 512, 2048)
WARMUPS = 2
REPEATS = 8


def reference_hashes(reference, name, entry, reg):
    rows = reference["startup"][name]
    expected = rows[0]["output_hashes"]
    if len(rows) != 3 or any(
            row["output_hashes"] != expected or row["tiles"] != reg["tiles"]
            or row["state"] != entry["package"]["state"] or row["package_bytes"] != entry["package"]["bytes"]
            for row in rows):
        raise ValueError("Original CPU acceptance does not bind this package and observation")
    return expected


def protection(reg, name, dense, audited, limits):
    base = audited["report"][dense]["selected"]
    recorded = audited["report"][name]["selected"]
    guard = verify_guard(base, recorded, reg["class_ids"], audited["small_classes"], limits)
    if guard != audited["guards"][name]["selected"]:
        raise ValueError("Archived all-head protection does not match its independent audit")
    return guard


def _requests(reg):
    return [
        Request(str(tile.tile_id), reg["dataset"], observation["image_sha256"],
                reg["resident_inputs"]["sha256"], "original_full", tile.image, tile.wavelengths,
                None, None, 100, 1.0, True)
        for tile, observation in zip(frozen.load_tiles(reg), reg["tiles"], strict=True)
    ]


def _sample(runtime, reg, expected):
    start = perf_counter_ns()
    inputs = _requests(reg)
    prep = clock_ms(start)
    result = runtime.run(inputs, "full")
    materialization = perf_counter_ns()
    outputs = frozen._outputs(result["predictions"], reg)
    hash_ms = clock_ms(materialization)
    wall = clock_ms(start)
    if (outputs != expected or result["counts"]["full_calls"] != len(inputs)
            or result["counts"]["graph_builds"] != len(inputs) or result["counts"]["graph_reuses"]):
        raise ValueError("Warm inference disagrees with the independently accepted three-head output or graph count")
    return dict(
        wall_ms=wall, read_verify_decode_ms=prep, forward_ms=result["elapsed_ms"],
        graph_ms=result["graph_ms"], forward_excluding_graph_ms=result["elapsed_ms"] - result["graph_ms"],
        output_materialization_and_hash_ms=hash_ms, host_to_device_ms=None,
        graph_calls=result["counts"]["graph_builds"], blocks=len(inputs),
        rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, cuda_peak_allocated_bytes=None)


def _operator_evidence(runtime, reg, expected):
    layer = runtime.full.get_submodule(TARGET)
    factor = type(layer) is LowRankMatrix
    weights = {"right": layer.right, "left": layer.left} if factor else {"dense": layer.weight}
    outputs = []

    def observe(_, arguments, output):
        if (arguments[0].device.type != "cpu" or output.device.type != "cpu"
                or arguments[0].dtype != torch.float32 or output.dtype != torch.float32):
            raise ValueError("Point-convolution runtime changed dtype or device")
        outputs.append(str(output.dtype))

    handle = layer.register_forward_hook(observe)
    try:
        with factor_calls(runtime.full) as factors, torch.autograd.profiler.profile(use_cuda=False) as profile:
            _sample(runtime, reg, expected)
    finally:
        handle.remove()
    kernels = {event.key: event.count for event in profile.key_averages()
               if "convolution" in event.key or "conv2d" in event.key or "quantized::" in event.key}
    calls = factor_counts(runtime.full, len(reg["tiles"]))
    if (not kernels or len(outputs) != len(reg["tiles"]) or factors != calls
            or any(value.device.type != "cpu" or value.dtype != torch.float32 for value in weights.values())
            or (not factor and (type(layer) is not torch.nn.Conv2d or layer.kernel_size != (1, 1)))):
        raise ValueError("Expected actual native CPU FP32 point convolutions")
    evidence = {part: dict(dtype="torch.float32", weight_shape=list(value.shape),
                           kernel_size=[1, 1], calls=calls.get(part, len(outputs)))
                for part, value in weights.items()}
    return dict(path="two_factor_fp32" if factor else "dense_fp32",
                device="cpu", input_dtype="torch.float32", output_dtype="torch.float32",
                point_convolutions=evidence, full_forward_convolution_events=kernels)


def _summary(samples, pixels):
    phases = ("wall_ms", "read_verify_decode_ms", "forward_ms", "graph_ms",
              "forward_excluding_graph_ms", "output_materialization_and_hash_ms")
    result = {name: timing_summary([sample[name] for sample in samples], repeats=len(samples),
                                   blocks=samples[0]["blocks"]) for name in phases}
    active_seconds = sum(row["wall_ms"] for row in samples) / 1000
    return dict(phases=result, sequential_pixels_per_second=pixels * len(samples) / active_seconds,
                sequential_blocks_per_second=sum(row["blocks"] for row in samples) / active_seconds,
                max_rss_kib=max(row["rss_peak_kib"] for row in samples))


def prepare(pin, reference_pin, code, out):
    if reference_pin["sha256"] != REFERENCE_CPU_COST:
        raise ValueError("Use the independently accepted WP10 CPU cost receipt")
    reference = admission.read(reference_pin)
    reg, tiles = frozen.load_manifest(pin, code, archived=True)
    if (reference["manifest"] != pin or reference["registration"] != reg["registration"]
            or reference["audit"] != reg["audit"] or reference["status"] != "passed_low_rank_cpu_cost"
            or any(reference[key] != 0 for key in ("query", "gt", "optimizer_updates"))):
        raise ValueError("CPU proof, release and audit must identify the same deployment")
    registration, audited = (admission.read(reg[key]) for key in ("registration", "audit"))
    protected = [pin, reference_pin, reg["resident_inputs"], reg["registration"], reg["audit"]]
    pixels = sum(row["pixels"] for row in reg["tiles"])
    if reg["data_protocol"] != "wp10_xuzhou_k5_v5" or (len(tiles), pixels) != (5, 3920):
        raise ValueError("Expected the same five frozen Xuzhou blocks and 3,920 pixels")
    cases, proofs = {}, {}
    for seed in SEEDS:
        dense = f"s{seed}-dense"
        for name in (dense, f"s{seed}-svd-r16"):
            cases[name] = entry = reg["cases"][name]
            protected.extend(entry["package"][key] for key in ("weights", "sidecar"))
            guards = protection(reg, name, dense, audited, registration["protection"]["limits"])
            proofs[name] = dict(
                guards=guards, output_hashes=reference_hashes(reference, name, entry, reg))
    admission.check_pins(protected)
    require_frozen_code(code)
    out = external_directory(out)
    if out.exists() or any(out == Path(item["path"]) or out in Path(item["path"]).parents for item in protected):
        raise ValueError("New independent warm deployment plan directory required")
    out.mkdir()
    fields = ("dataset", "data_protocol", "class_ids", "resident_inputs", "tiles", "runtime", "audit")
    return dev._json_new(out / "plan.json", dict(
        **{key: reg[key] for key in fields}, kind=frozen.KIND, policy=PLAN, commit=code["commit"],
        source_hash=code["source_hash"], source_manifest=pin, manifest_producer=reg["commit"],
        baseline_evidence=reference_pin, cases=cases, proofs=proofs, seeds=list(SEEDS),
        small_classes=audited["small_classes"], observation_role="devval_image_only"))


def measure(pin, code, out):
    require_frozen_code(code)
    reg = admission.read(pin)
    if (reg["kind"] != frozen.KIND or reg["policy"] != PLAN or reg["commit"] != code["commit"]
            or reg["source_hash"] != code["source_hash"] or reg["seeds"] != list(SEEDS)
            or reg["observation_role"] != "devval_image_only" or reg["runtime"] != frozen.runtime()
            or reg["source_manifest"]["sha256"] != frozen.PUBLISHED_MANIFEST
            or reg["baseline_evidence"]["sha256"] != REFERENCE_CPU_COST):
        raise ValueError("Sealed label-free warm plan for the actual deployment code required")
    tiles = frozen.load_tiles(reg)
    pixels = sum(row["pixels"] for row in reg["tiles"])
    if reg["data_protocol"] != "wp10_xuzhou_k5_v5" or (len(tiles), pixels) != (5, 3920):
        raise ValueError("Expected the released WP10 observation geometry")
    protected = [pin, reg["resident_inputs"], *[
        entry["package"][key] for entry in reg["cases"].values() for key in ("weights", "sidecar")]]
    out = external_directory(out)
    if out.exists() or any(out == Path(item["path"]) or out in Path(item["path"]).parents for item in protected):
        raise ValueError("New independent warm deployment output directory required")
    out.mkdir()
    cases, fallbacks, orderings = {}, {}, {}
    for seed in SEEDS:
        dense, low_rank = names = (f"s{seed}-dense", f"s{seed}-svd-r16")
        guards = {name: reg["proofs"][name]["guards"] for name in names}
        expected = {name: reg["proofs"][name]["output_hashes"] for name in names}
        models = {name: Runtime(device="cpu", full=frozen.package(reg["cases"][name]),
                                identity={"full": reg["cases"][name]["package"]["state"]}) for name in names}
        samples = {name: [] for name in names}
        for _ in range(WARMUPS):
            for name in names:
                _sample(models[name], reg, expected[name])
        orderings[str(seed)] = []
        for i in range(REPEATS):
            order = names if i % 2 == 0 else names[::-1]
            orderings[str(seed)].append(list(order))
            for name in order:
                samples[name].append(_sample(models[name], reg, expected[name]))
        for name in names:
            package = reg["cases"][name]["package"]
            cases[name] = dict(
                case=reg["cases"][name]["case"], package_state=package["state"],
                package_bytes=package["bytes"], package_sha256=package["weights"]["sha256"],
                parameters=package["parameters"], guard=guards[name],
                output_hashes=expected[name], operator_evidence=_operator_evidence(models[name], reg, expected[name]),
                samples=samples[name], summary=_summary(samples[name], pixels))
        effective = low_rank if all(r["passes_metric_guards"] for r in guards[low_rank].values()) else dense
        _sample(models[effective], reg, expected[effective])
        fallbacks[str(seed)] = dict(
            requested=low_rank, executed=effective, state=reg["cases"][effective]["package"]["state"],
            output_hashes=expected[effective],
            failed_heads=[head for head, values in guards[low_rank].items() if not values["passes_metric_guards"]])
    admission.check_pins(protected)
    require_frozen_code(code)
    return dev._json_new(out / "receipt.json", dict(
        status="passed", schema="wp10_warm_native_cost_v1", scope=SCOPE,
        execution_commit=code["commit"], plan=pin, manifest=reg["source_manifest"],
        manifest_producer=reg["manifest_producer"], baseline_evidence=reg["baseline_evidence"],
        audit=reg["audit"], data_protocol=reg["data_protocol"],
        resident_input_sha256=reg["resident_inputs"]["sha256"], pixels=pixels, blocks=len(tiles),
        class_ids=reg["class_ids"], small_classes=reg["small_classes"],
        device=dict(kind="cpu", torch=str(torch.__version__), host_arch=platform.machine(),
                    threads=torch.get_num_threads(), affinity=sorted(os.sched_getaffinity(0)), engine=torch.backends.quantized.engine),
        warmups=WARMUPS, repeats=REPEATS, orderings=orderings, cases=cases, fallbacks=fallbacks,
        optimizer_updates=0, calibration_updates=0, gt=0, query=0, predictive_label_reads=0))
