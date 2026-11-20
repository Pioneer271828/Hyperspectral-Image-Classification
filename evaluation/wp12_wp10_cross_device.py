"""Independently reconcile current device output and sample scopes before WP12 publication."""
import csv
import io
import math
import re
import subprocess

import torch

from evaluation.wp12_wp10_deployment import ARCHIVES
from inference import wp10_cross_device as cross, wp10_low_rank_execution as frozen, wp10_onednn_deployment as onednn
from research_runtime import ROOT
from transfer import matched_route_admission as admission, p4_runtime_device as device, registered_development as dev


def arrays(pin):
    return torch.load(io.BytesIO(dev._read(pin)), map_location="cpu", weights_only=True)


def _same_inputs(packet, kind, code):
    zeros = ("gt", "query", "optimizer_updates", "calibration_updates", "predictive_label_reads")
    tested = packet["commit"]
    if (packet["schema"] != cross.SCHEMA or packet["kind"] != kind or not re.fullmatch("[0-9a-f]{40}", tested)
            or (packet["blocks"], packet["pixels"], packet["warm_samples"], packet["first_samples"],
                packet["warmups"], packet["first_scope"]) != (5, 3920, 8, 3, 2, "fresh_model_reload_same_process")
            or any(packet[key] != 0 for key in zeros)):
        raise ValueError("Foreign device sample scope or new optimization/label access")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", tested, code["commit"]], check=True)
    paths = (*frozen.RUNTIME_PATHS, "inference/native_deployment.py", "inference/wp10_cross_device.py",
             "inference/wp10_deployment_cost.py", "inference/wp10_onednn_deployment.py",
             "compression/int8_torchao_onednn.py", "compression/int8_torchao_reference.py",
             "scripts/run_frozen_cpu_deployment.py", "transfer/p4_runtime_device.py", "transfer/shared_gpu_lease.py")
    subprocess.run(["git", "-C", str(ROOT), "diff", "--exit-code", tested, code["commit"], "--", *paths], check=True)
    reg = admission.read(packet["plan"])
    if reg["commit"] != tested or reg["resident_inputs"]["sha256"] != packet["resident_input_sha256"]:
        raise ValueError("Plan and measured bytes differ")
    return reg, arrays(packet["outputs"])


def _numbers(ref, actual):
    out = [{key: 0 for key in ("elements", "exceeded", "max_abs", "changed")} for _ in frozen.HEADS]
    for old, new in zip(ref, actual, strict=True):
        for r, x, y in zip(out, old, new, strict=True):
            a, b = x.double(), y.double()
            error = (a - b).abs()
            r["elements"] += error.numel()
            r["exceeded"] += int((error > a.abs().mul(cross.RTOL).add(cross.ATOL)).count_nonzero().item())
            r["max_abs"] = max(r["max_abs"], float(error.max().item()))
            r["changed"] += int((a.argmax(1) != b.argmax(1)).count_nonzero().item())
    return dict(zip(frozen.HEADS, out, strict=True))


def _validate_samples(row, values, reg, cuda):
    first, warm = row["first"], row["samples"]
    if (len(first) != 3 or len(warm) != 8 or row["first_summary"] != cross.summary(first, 3920)
            or row["sample_summary"] != cross.summary(warm, 3920)):
        raise ValueError("Observed phase summary differs from the complete sample roster")
    samples = [first[0], *warm, *first[1:]]
    for sample, output in zip(samples, values, strict=True):
        if (frozen._outputs(output, reg) != sample["hashes"] or sample["blocks"] != 5 or sample["graph_calls"] != 5
                or sample["graph_clock"] != ("cuda_event" if cuda else "host_wall")
                or sample["wall_ms"] < sample["forward_ms"]):
            raise ValueError("Measured full output or graph clocks differ")
        phases = ("host_to_device_ms", "device_to_host_ms", "forward_event_ms", "cuda_peak_allocated_bytes",
                  "cuda_peak_reserved_bytes")
        if any((sample[key] is not None) != cuda for key in phases):
            raise ValueError("Timing or memory scope belongs to another device")
        forward = sample["forward_event_ms"] if cuda else sample["forward_ms"]
        if not math.isclose(forward, sample["remaining_forward_ms"] + sample["graph_ms"]):
            raise ValueError("Graph and network times must use the same clock")
    return samples


def _row(name, group, device_kind, scope, row, quality, guards, *, checked=None):
    return dict(case=name, group=group, device=device_kind, scope=scope, package_bytes=row["package_bytes"],
                head_quality=quality, protection=guards, numerical_check=checked,
                first_summary=row["first_summary"], warm_summary=row["sample_summary"], dispatch=row["dispatch"])


def _cuda(pin, cpu, code):
    run = admission.read(pin)
    reg, measured = _same_inputs(run, "wp10_low_rank_cuda", code)
    gpu = run["device"]["physical_uuid"]
    if (not device.UUID.fullmatch(gpu) or run["resource_admission"]["physical_uuid"] != gpu
            or not device.eligible(run["resource_admission"]) or run["device"]["device"] != "cuda:0"
            or run["device"]["deterministic_algorithms"] is not True
            or run["device"]["protocol"] != device.CUDA_FP64_WEIGHT or run["device"]["tf32"] is not False
            or run["tolerance"] != dict(atol=cross.ATOL, rtol=cross.RTOL)
            or run["source_manifest"] != reg["source_manifest"]
            or reg["resident_inputs"]["sha256"] != cpu["rank_resident_identity"]
            or set(run["cases"]) != {row["name"] for row in cpu["matched_rows"]}):
        raise ValueError("CUDA UUID, established tolerance or paired CPU observations differ")
    ref = admission.read(run["reference"])
    if ref["plan"] != run["plan"] or ref["commit"] != run["commit"] or ref["kind"] != "wp10_bitwise_cpu_arrays":
        raise ValueError("Bitwise CPU anchor does not bind the measured plan")
    references = arrays(ref["values"])
    rows, qualification = [], {}
    for old in cpu["matched_rows"]:
        name = old["name"]
        row = run["cases"][name]
        entry = reg["cases"][name]["package"]
        if (row["state"] != old["package_state"] or row["package_bytes"] != old["package_bytes"]
                or entry["state"] != old["package_state"] or entry["weights"]["sha256"] != old["package_sha256"]
                or frozen._outputs(references[name], reg) != old["bitwise_cpu_outputs"]
                or row["dispatch"]["device"] != "cuda:0" or row["dispatch"]["floating_heads"] is not True):
            raise ValueError("CUDA case displaced the same approved package or FP32 execution")
        samples = _validate_samples(row, measured[name], reg, True)
        totals = {head: dict(elements=0, exceeded=0, changed=0, max_abs=0.) for head in frozen.HEADS}
        for sample, values in zip(samples, measured[name], strict=True):
            numeric = _numbers(references[name], values)
            if sample["numeric"] != numeric:
                raise ValueError("Independent CUDA probability and decision comparison differs")
            for head, values in numeric.items():
                for key in ("elements", "exceeded", "changed"):
                    totals[head][key] += values[key]
                totals[head]["max_abs"] = max(totals[head]["max_abs"], values["max_abs"])
        qualification[name] = all(not (r["exceeded"] or r["changed"]) for r in totals.values())
        consistency = dict(bitwise=all(s["hashes"] == old["bitwise_cpu_outputs"] for s in samples), heads=totals)
        rows.append(_row(name, old["seed"], "cuda", "wp10_xuzhou_k5_v5",
                         row, old["quality"] if all(r["changed"] == 0 for r in totals.values()) else None,
                         old["guard"], checked=consistency))
    for seed, (dense, candidate) in cross.rank_pairs().items():
        expected_row = next(row for row in cpu["matched_rows"] if row["name"] == candidate)
        selected = cross.accepted_arm(dense, candidate, expected_row["guard"], qualification[candidate])
        actual = run["protected_routes"][seed]
        values = measured[seed + "/protected"]
        if ((actual["requested"], actual["executed"]) != (candidate, selected)
                or actual["sample"]["hashes"] != frozen._outputs(values, reg)
                or actual["sample"]["numeric"] != _numbers(references[selected], values)):
            raise ValueError("Actual protected CUDA execution disagrees with its guard or complete output")
    return dict(receipt_sha256=pin["sha256"], physical_uuid=gpu, execution_commit=run["commit"],
                device=run["device"], admission=run["resource_admission"], protected_routes=run["protected_routes"],
                unsupported_on_cuda=run["unsupported_on_cuda"], rows=rows)


def _onednn(pin, code):
    run = admission.read(pin)
    reg, measured = _same_inputs(run, onednn.PLAN, code)
    if (reg["kind"] != onednn.PLAN or reg["quality"] != run["quality"] or run["quality"]["sha256"] != ARCHIVES["onednn_quality"]
            or reg["resident_inputs"]["sha256"] != onednn.OBSERVATION or reg["backend"]["sha256"] != onednn.BACKEND
            or run["device"]["kind"] != "cpu" or set(run["cases"]) != {onednn.CONTROL, onednn.CANDIDATE}):
        raise ValueError("Unknown precision package, CPU backend or observation identity")
    quality = admission.read(run["quality"])
    if (reg["guards"] != quality["candidate"][onednn.CANDIDATE]["versus_same_budget_fp32_guards"]
            or reg["rejected"] != quality["candidate"][onednn.REJECTED]["versus_same_budget_fp32_guards"]):
        raise ValueError("Original all-head precision guards changed")
    rows = []
    for name, row in run["cases"].items():
        integer = name == onednn.CANDIDATE
        truth = quality["candidate"][name] if integer else quality["native"][name]
        hashes = [dict(zip(frozen.HEADS, values, strict=True)) for values in truth["probability_hashes"]]
        samples = _validate_samples(row, measured[name], reg, False)
        dispatch = row["dispatch"]["convolution_dispatch"].get("onednn::qconv2d_pointwise", 0)
        if (row["state"] != quality["packaged_endpoints"][name]["state"]
                or reg["packages"][name] != quality["packaged_endpoints"][name]
                or any(sample["hashes"] != hashes for sample in samples)
                or row["dispatch"]["device"] != "cpu" or dispatch != (5 if integer else 0)):
            raise ValueError("oneDNN end-to-end run differs from its attested frozen package")
        guards = truth["versus_same_budget_fp32_guards"] if integer else None
        rows.append(_row(name, "qat", "cpu", "wp10_xuzhou_qat_v5", row, truth["metrics"], guards))
    seen = run["protected_routes"]["qat"]
    rejected = run["rejected_route"]
    if (seen["executed"] != onednn.CANDIDATE or seen["sample"]["hashes"] != frozen._outputs(measured["qat/protected"], reg)
            or seen["sample"]["hashes"] != reg["hashes"][onednn.CANDIDATE]
            or rejected["requested"] != onednn.REJECTED or rejected["executed"] != onednn.CONTROL
            or rejected["sample"]["hashes"] != reg["hashes"][onednn.CONTROL]):
        raise ValueError("Protected oneDNN and actual FP32 fallback differ")
    return dict(receipt_sha256=pin["sha256"], execution_commit=run["commit"], rows=rows,
                admissible=seen, rejected_route=rejected)


def publish(pins, report, code, out):
    result = {name: _cuda(pin, report, code) if name == "cuda" else _onednn(pin, code)
              for name, pin in pins.items()}
    values = []
    for row in report["matched_rows"]:
        for head in frozen.HEADS:
            quality, guard = row["quality"][head], row["guard"][head]
            warm = row["warm_cpu"]
            values.append(dict(cohort=row["scope"], device="cpu", group=row["seed"], case=row["name"],
                               head=head, oa=quality["oa"], aa=quality["aa"], case_bytes=row["package_bytes"],
                               guards_passed=guard["passes_metric_guards"],
                               warm_p50_ms=warm["phases"]["wall_ms"]["p50_ms"],
                               warm_p95_ms=warm["phases"]["wall_ms"]["p95_ms"],
                               first_scope="new_process_to_verified_output",
                               first_p50_ms=row["reused_cold_cpu"]["median_ms"], first_p95_ms=None,
                               sequential_pixels_per_second=warm["sequential_pixels_per_second"],
                               max_rss_kib=warm["max_rss_kib"], max_cuda_allocated_bytes=None,
                               max_cuda_reserved_bytes=None))
    for key, group in result.items():
        for row in group["rows"]:
            for head in frozen.HEADS:
                quality = row["head_quality"][head] if row["head_quality"] else None
                guards = row["protection"][head] if row["protection"] else None
                warm, first = row["warm_summary"], row["first_summary"]
                values.append(dict(cohort=row["scope"], device=row["device"], group=row["group"], case=row["case"],
                                   head=head, oa=None if quality is None else quality["oa"],
                                   aa=None if quality is None else quality["aa"], case_bytes=row["package_bytes"],
                                   guards_passed=None if guards is None else guards["passes_metric_guards"],
                                   warm_p50_ms=warm["phases"]["wall_ms"]["p50_ms"],
                                   warm_p95_ms=warm["phases"]["wall_ms"]["p95_ms"],
                                   first_scope="fresh_model_reload_same_process",
                                   first_p50_ms=first["phases"]["wall_ms"]["p50_ms"],
                                   first_p95_ms=first["phases"]["wall_ms"]["p95_ms"],
                                   sequential_pixels_per_second=warm["sequential_pixels_per_second"],
                                   max_rss_kib=warm["max_rss_kib"],
                                   max_cuda_allocated_bytes=warm["max_cuda_allocated_bytes"],
                                   max_cuda_reserved_bytes=warm["max_cuda_reserved_bytes"]))
    return result, values


def write_csv(out, rows):
    with (out / "device-quality-cost.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
