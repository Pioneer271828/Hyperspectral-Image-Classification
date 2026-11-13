"""Replay authenticated WP10 static INT8 endpoints on native and optional CPU kernels."""

from argparse import ArgumentParser
from contextlib import ExitStack
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns, process_time_ns

import numpy as np
import torch

from compression.g4_cpu_cost import _p95
from compression.int8_torchao_onednn import convert_backend, load_backend, register_backend
from compression.int8_torchao_reference import TORCHAO_COMMIT, TorchAOAffine, compare_layer
from compression.precision_cost import ResidentPrecisionTile, _state_identity, _tile_identity, prepare_resident_precision_tiles
from compression.precision_recovery import ARMS, HEADS
from compression.precision_recovery_audit import _guard, load_package
from compression.precision_startup import _memory_kib
from compression.registered_precision_recovery import _inputs, runtime
from evaluation.classification_metrics import metrics_from_confusion
from research_runtime import ROOT, code_snapshot, external_directory, require_frozen_code, tensor_hash, write_json
from transfer import matched_route_admission as admission, registered_development as dev


PRODUCER = "9dea199653079c9c430264b72ac968f1fefd7d7e"
PINS = {
    "registration": ("registration.json", "086eeb643eb1c0b2cea4299732299810bf8a18d9d6da0cba1c8eb2dc538d291d"),
    "audit": ("audit.json", "9588a95ca3db5ae461e686ae04ec75041f8108de075a248facea20b1d6b8bbb5"),
    "batch": ("batch-complete.json", "68087a7193b3f44e62c54b9a5b0c46b14288fb782da48c639dd8b25ca53b6195"),
    "cost": ("cost/receipt.json", "54c77e62530e575a7713b16be1aaad0a63f843e69664abd3615b5616cd77970d"),
}
INT8_ARMS = ("qat-first", "qat-second", "qat-both")


def measure(fn):
    wall, cpu = perf_counter_ns(), process_time_ns()
    value = fn()
    return value, dict(wall_ms=(perf_counter_ns()-wall)/1e6, process_cpu_ms=(process_time_ns()-cpu)/1e6)


def authenticate(root, code):
    pins = {key: dict(path=str(root / path), sha256=sha) for key, (path, sha) in PINS.items()}
    reg, audit, batch, cost = (admission.read(pins[k]) for k in PINS)
    runtime_paths = ("compression/int8.py", "compression/int8_model.py", "compression/int8_checkpoint.py",
                     "compression/precision_recovery.py", "compression/precision_recovery_audit.py",
                     "compression/registered_precision_recovery.py", "model", "loadData", "configs",
                     "transfer/curriculum_training.py")
    subprocess.run(["git", "diff", "--exit-code", PRODUCER, code["commit"], "--", *runtime_paths], cwd=ROOT, check=True)
    if (reg["commit"] != PRODUCER or reg["runtime"] != runtime() or reg["scope"] != "wp08_xuzhou_off"
            or (reg["budget"], reg["prefix"], reg["maximum_updates"], reg["calibration_tiles"]) != (155, 7, 2170, [92, 8])
            or reg["registration_optimizer_updates"] != 0 or reg["validation_fit_reads"] != 0
            or audit["status"] != "passed_precision_recovery_independent" or audit["registration"] != pins["registration"]
            or audit["completion"] != pins["batch"] or audit["actual_updates"] != 2170 or audit["independent_updates"] != 0
            or batch["registration"] != pins["registration"] or batch["updates"] != 2170
            or cost["registration"] != pins["registration"] or cost["audit"] != pins["audit"]
            or set(audit["report"]) != set(ARMS) or any(row[field] for row in (audit, batch, cost) for field in ("gt", "query"))):
        raise ValueError("Original completed WP10 155-step study required")
    _, source, scene, record, source_info, protected, plan, calibration = _inputs(reg["request"])
    if (source_info != reg["source"] or protected != reg["protected"] or calibration != [92, 8]
            or plan.to_dict() != reg["exposure"] or scene.dataset != "xuzhou"
            or len(record["validation_support"]["indices"]) != 45 or len(record["train_support"]["indices"]) != 45):
        raise ValueError("Historical environment, role or scope differs")
    extra = [*protected.values(), cost["resident"]]
    packages = {}
    for arm, row in audit["report"].items():
        p = row["selected"]["package"]
        if (row["selected"]["step"] != 155 or row["updates"] != 310
                or p["weights"]["path"] != str(root / "deployment" / arm / "selected.pt")):
            raise ValueError("Historical package identity differs")
        extra.extend(p[k] for k in ("weights", "sidecar") if p[k] is not None)
        if arm in ("fp32", *INT8_ARMS):
            packages[arm] = p
    admission.check_pins([*pins.values(), *extra])
    return pins, extra, reg, audit, cost, source, scene, record, packages


def locations(tiles, scene, record):
    width = scene.plan["shape"][1]
    labels = dict(zip(record["validation_support"]["indices"], record["validation_support"]["labels"], strict=True))
    roles = {row["id"]: row for row in record["roles"] if row["role"] == "devval"}
    ordered, points, used = [], [], set()
    for tile in tiles:
        y0, y1, x0, x1 = roles[tile.tile_id]["box"]
        block = sorted(i for i in labels if y0 <= i//width < y1 and x0 <= i%width < x1)
        if used.intersection(block):
            raise ValueError("Duplicate V5 point")
        used.update(block)
        ordered.extend(labels[i] for i in block)
        points.append([(i//width-y0)*(x1-x0)+i%width-x0 for i in block])
    if used != labels.keys() or sorted(ordered) != sorted(list(range(1, 10))*5):
        raise ValueError("Expected the original 45 development points")
    return ordered, points


@torch.inference_mode()
def sweep(model, tiles, dataset="xuzhou"):
    return [model(dataset, tile.image, wavelengths=tile.wavelengths, superpixel_scale=100,
                  return_graph_info=True)[:3] for tile in tiles]


def hashes(values):
    return [[tensor_hash(p) for p in row] for row in values]


def metrics(values, locations, classes):
    targets, points = locations
    quality, predictions = {}, {}
    for col, head in enumerate(HEADS):
        chosen = [classes[i] for row, local in zip(values, points, strict=True)
                  for i in row[col][local].argmax(1).tolist()]
        matrix = np.zeros((len(classes), len(classes)), np.int64)
        for a, b in zip(targets, chosen, strict=True):
            matrix[classes.index(a), classes.index(b)] += 1
        quality[head] = metrics_from_confusion(matrix, classes)
        predictions[head] = chosen
    return dict(metrics=quality, predictions=predictions, targets=targets, probability_hashes=hashes(values))


def compare_original(row, expected):
    if (row["probability_hashes"] != expected["probabilities"] or row["predictions"] != expected["predictions"]
            or any(any(v != expected["metrics"][head][key] for key, v in values.items())
                   for head, values in row["metrics"].items())):
        raise ValueError("Replayed native model differs from accepted audit")


def check_layers(native, optional, arm, tiles, oracle):
    captured = {path: [] for path in ARMS[arm][1]}
    with ExitStack() as stack:
        for path, rows in captured.items():
            stack.enter_context(native.get_submodule(path).register_forward_pre_hook(
                lambda _layer, args, rows=rows: rows.append(args[0].detach().clone())))
        base = sweep(native, tiles)
    diagnostics = {}
    for path, inputs in captured.items():
        if len(inputs) != len(tiles):
            raise ValueError("Each declared layer must execute exactly once per block")
        legacy, proposed = native.get_submodule(path), optional.get_submodule(path)
        counters = dict(elements=0, input_code_mismatches=0, oracle_code_mismatches=0,
                        backend_code_mismatches=0, backend_max_code_error=0, oracle_max_code_error=0,
                        max_dequantized_error=0., max_accumulator=0)
        for x in inputs:
            a, b, acc, codes0, reference, y, y_ref = compare_layer(legacy, x, oracle)
            codes1 = proposed.codes(x)
            if codes1.shape != codes0.shape or codes1.dtype != torch.uint8:
                raise ValueError("oneDNN changed the output layout contract")
            counters["elements"] += codes0.numel()
            counters["input_code_mismatches"] += int((a != b).count_nonzero())
            counters["oracle_code_mismatches"] += int((codes0 != reference).count_nonzero())
            counters["backend_code_mismatches"] += int((codes0 != codes1).count_nonzero())
            counters["backend_max_code_error"] = max(counters["backend_max_code_error"],
                                                       int((codes1.int()-codes0.int()).abs().max()))
            counters["oracle_max_code_error"] = max(counters["oracle_max_code_error"],
                                                      int((reference.int()-codes0.int()).abs().max()))
            counters["max_dequantized_error"] = max(counters["max_dequantized_error"], float((y-y_ref).abs().max()))
            counters["max_accumulator"] = max(counters["max_accumulator"], int(acc.abs().max()))
        diagnostics[path] = counters
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities) as profile:
        actual = sweep(optional, tiles)
    dispatch = {e.key: e.count for e in profile.key_averages() if "onednn::qconv" in e.key}
    if dispatch.get("onednn::qconv2d_pointwise") != len(tiles)*len(captured):
        raise ValueError("Actual U8/S8 oneDNN dispatch did not occur")
    return base, actual, diagnostics, dispatch


def compare_probabilities(expected, actual):
    result = {}
    for col, head in enumerate(HEADS):
        errors = []
        for old, new in zip(expected, actual, strict=True):
            a, b = old[col].double(), new[col].double()
            if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Finite matched three-head probabilities required")
            delta = (b-a).abs()
            bound = 1e-6 + 1e-5*a.abs()
            errors.append((a.numel(), int((delta > bound).count_nonzero()), float(delta.max()),
                           float((delta/bound).max())))
        result[head] = dict(elements=sum(v[0] for v in errors), exceeds=sum(v[1] for v in errors),
                            max_abs=max(v[2] for v in errors), max_ratio=max(v[3] for v in errors))
    return result


def pair(arms, repeats=12):
    a, b = arms
    for _ in range(3):
        for method in arms.values():
            method()
    rows = []
    for i in range(repeats):
        order = [a, b] if i % 2 == 0 else [b, a]
        rows.append(dict(order=order, timing={name: measure(arms[name])[1] for name in order}))
    return dict(samples=rows, warmups=3, summary={name: {
        "p50_ms": median(row["timing"][name]["wall_ms"] for row in rows),
        "p95_ms": _p95([row["timing"][name]["wall_ms"] for row in rows])} for name in arms})


def startup(req):
    model, load = measure(lambda: load_package(req["package"]))
    oracle, attestation = (None, None)
    conversion = None
    if req["mode"] == "onednn":
        oracle, attestation = measure(lambda: TorchAOAffine(req["provider"]))
        # The sidecar is authenticated before replacing any native module.
        model, conversion = measure(lambda: load_backend(req["backend"], oracle, enabled=True, native=model))
    import io
    tiles, read = measure(lambda: [ResidentPrecisionTile(**row) for row in torch.load(
        io.BytesIO(dev._read(req["resident"])), weights_only=True, map_location="cpu")])
    values, first = measure(lambda: sweep(model, tiles))
    return dict(pid=os.getpid(), load_native=load, attestation=attestation, validate_and_convert=conversion,
                input_file_read=read, first_sweep=first, rss=_memory_kib(), output_hashes=hashes(values))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--startup":
        print(json.dumps(startup(json.loads(sys.argv[2]))))
        return
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for name in ("campaign", "torchao-source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        raise ValueError("Expected exact source")
    require_frozen_code(code)
    runtime()
    out = external_directory(args.out)
    out.mkdir()
    scope, scope_timing = measure(lambda: authenticate(args.campaign.resolve(), code))
    pins, protected, reg, old, cost, source, scene, record, packages = scope
    tiles, prep = measure(lambda: prepare_resident_precision_tiles(scene, reg["validation"]["tiles"]))
    workload = _tile_identity(tiles)
    if len(tiles) != 5 or sum(t["pixels"] for t in workload) != 3920:
        raise ValueError("Original five-block scope required")
    loc = locations(tiles, scene, record)
    classes = record["classes"]
    source.backbone.adapter.cache_attention = False
    origin = metrics(sweep(source, tiles), loc, classes)
    compare_original(origin, old["source"])
    native = {arm: load_package(p) for arm, p in packages.items()}
    native_scores = {arm: metrics(sweep(model, tiles), loc, classes) for arm, model in native.items()}
    for arm, row in native_scores.items():
        compare_original(row, old["report"][arm]["selected"])
    provider, attestation = measure(lambda: TorchAOAffine(args.torchao_source))
    results, loaded, pins_backend = {}, {}, {}
    for arm in INT8_ARMS:
        model, conversion = measure(lambda: convert_backend(native[arm], provider, enabled=True))
        base, values, layers, dispatch = check_layers(native[arm], model, arm, tiles, provider)
        row = metrics(values, loc, classes)
        guards = _guard(native_scores["fp32"], row, classes, old["small_classes"])
        paired_guard = _guard(native_scores[arm], row, classes, old["small_classes"])
        continuity = compare_probabilities(base, values)
        admissible = all(v["exceeds"] == 0 for v in continuity.values()) and all(
            v["passes_metric_guards"] for v in paired_guard.values()) and all(
            v["backend_max_code_error"] <= 1 for v in layers.values())
        loaded[arm] = model
        if admissible:
            pin = register_backend(packages[arm], out / (arm + "-backend.json"), provider, native=native[arm])
            if hashes(sweep(load_backend(pin, provider, enabled=True), tiles)) != row["probability_hashes"]:
                raise ValueError("Reconstructed backend changed the output")
            pins_backend[arm] = pin
        results[arm] = dict(row, layers=layers, dispatch=dispatch, versus_native_probability=continuity,
                            versus_same_budget_fp32_guards=guards, versus_original_guard=_guard(
                                origin, row, classes, old["small_classes"]), versus_native_guards=paired_guard,
                            eligible=admissible, conversion=conversion, original_package_bytes=packages[arm]["bytes"])
        print("arm", arm, "eligible", admissible, flush=True)
    old_guards = {arm: _guard(native_scores["fp32"], native_scores[arm], classes, old["small_classes"]) for arm in INT8_ARMS}
    historical = {row["arm"]: row["heads"] for row in old["decisions"]["selected"]["qat"]["trials"]}
    if old_guards != historical or old_guards["qat-both"]["cnn"]["passes_metric_guards"]:
        raise ValueError("Historical QAT guard or failed CNN class lost")
    for arm in INT8_ARMS:
        for path, info in reg["initial"][arm]["calibration_batches"].items():
            if info != 2 or path not in ARMS[arm][1]:
                raise ValueError("Expected the original two frozen training observations")
    if any(_state_identity(model) != packages[k]["state"] for k, model in native.items()):
        raise ValueError("Native checkpoints changed during comparison")
    chosen = {
        "first": results["qat-first"]["eligible"] and results["qat-first"]["versus_same_budget_fp32_guards"]["fused"]["passes_metric_guards"],
        "second": results["qat-second"]["eligible"] and results["qat-second"]["versus_same_budget_fp32_guards"]["fused"]["passes_metric_guards"],
    }
    both = all(chosen.values()) and results["qat-both"]["eligible"] and results["qat-both"][
        "versus_same_budget_fp32_guards"]["fused"]["passes_metric_guards"]
    fallback = "qat-both" if both else next((f"qat-{name}" for name, allowed in chosen.items() if allowed), "fp32")
    receipt = dict(commit=code["commit"], provider=TORCHAO_COMMIT, scope=pins, protected=len(protected)+len(pins),
        packaged_endpoints=packages, original_dense_package={k: reg["source"][k] for k in ("weights", "sidecar")},
        old_ternary_authenticated_only=True,
        calibration_role="devtrain_frozen", calibration_tiles=reg["calibration_tiles"],
        calibration_timing={k: reg["initial"][k]["preparation_seconds"] for k in INT8_ARMS},
        historical_precision_steps=reg["budget"], historical_training_updates=reg["maximum_updates"],
        historical_native_decision=old["decisions"]["selected"]["qat"]["arm"], original=origin, native=native_scores,
        native_guards=old_guards, candidate=results, backend_pins=pins_backend, optional_fusion_guard_decision=fallback,
        evaluation_tiles=workload, scope_authentication=scope_timing, canonical_input_preparation=prep,
        provider_attestation=attestation, calibration_new_calls=0, query=0, gt=0, optimizer_updates=0)
    write_json(out / "quality.json", receipt)
    resident_path = out / "inputs.pt"
    torch.save([dict(tile_id=t.tile_id, image=t.image, wavelengths=t.wavelengths) for t in tiles], resident_path)
    resident_pin = dev._pin(resident_path)
    timing = {arm: pair({"native": lambda: sweep(native[arm], tiles), "onednn": lambda: sweep(loaded[arm], tiles)})
              for arm in INT8_ARMS}
    dense_controls = {arm: pair({"same_budget_fp32": lambda: sweep(native["fp32"], tiles),
                                 "onednn": lambda: sweep(loaded[arm], tiles)}) for arm in INT8_ARMS}
    workers = {}
    for arm, backend in pins_backend.items():
        rows = []
        for order in (("native", "onednn"), ("onednn", "native")):
            samples = {}
            for mode in order:
                request = dict(mode=mode, package=packages[arm], backend=backend,
                               provider=str(args.torchao_source), resident=resident_pin)
                child, elapsed = measure(lambda: subprocess.run([sys.executable, "-m",
                    "scripts.review_wp10_int8_torchao", "--startup", json.dumps(request)],
                    cwd=ROOT, capture_output=True, text=True, check=True))
                sample = json.loads(child.stdout)
                expected = native_scores[arm] if mode == "native" else results[arm]
                if sample["output_hashes"] != expected["probability_hashes"]:
                    raise ValueError("Isolated process disagrees with corresponding resident path")
                samples[mode] = dict(sample, process_wall=elapsed, stderr=child.stderr)
            rows.append(dict(order=list(order), samples=samples))
        workers[arm] = rows
    admission.check_pins([*pins.values(), *protected, resident_pin])
    require_frozen_code(code)
    write_json(out / "cost.json", dict(resident=timing, versus_same_budget_fp32=dense_controls, isolated_startup=workers,
        package_bytes={arm: dict(native=packages[arm]["bytes"], with_backend_sidecar=packages[arm]["bytes"] +
                                Path(pin["path"]).stat().st_size) for arm, pin in pins_backend.items()},
        historical_cost=cost["startup_summary"],
        new_updates=0, includes_graph_and_all_three_heads=True, device="cpu", threads=torch.get_num_threads(),
        affinity=sorted(os.sched_getaffinity(0))))


if __name__ == "__main__":
    main()
