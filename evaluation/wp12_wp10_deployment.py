"""Publish matched WP10 quality and deployment cost without pooling different scopes."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess

from inference import wp10_low_rank_execution as frozen
from inference.wp10_deployment_cost import (PLAN, REFERENCE_CPU_COST, REPEATS, SEEDS, SCOPE, WARMUPS,
                                           _summary, protection, reference_hashes)
from research_runtime import ROOT, code_snapshot, external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.container_runtime import training_container_kind


ARCHIVES = {
    "cold": "2e432c52d4caea3e0d92e7beddf7e2d472599a990bc59555d003a64bb9220471",
    "cuda": "00dda97a864550b8c5be63cc66f141f7b0ad2f201119473c50845fa67a58c184",
    "cuda_audit": "9b9fdee08e9d28f84229925d529b33e86b9efe1901aa306b74df0f036300286a",
    "onednn_quality": "18887f7a1a7b3bfb3aacced82e020b9298e90562017afb1b2926ab44edac751b",
    "onednn_cost": "9a5bcb86d914a294d52256dc18f8c024ded07683f862e047d2aadf4eb18dc2c3",
    "wp11_cpu": "b7e941fffa8a92ece74a9969bb19e1b4fb7c058c9318eb9aa2914b50e6ce4fb5",
    "wp11_cuda": "370dc4c931073b95e4cfe75cdea3f8624c39bde9e86c43d5019445b17eee84e0",
}


def _matches(blob, values, message):
    if any(blob.get(k) != value for k, value in values.items()):
        raise ValueError(message)


def _warm(pins, code):
    warm = admission.read(pins["warm"])
    _matches(warm, dict(
        status="passed", schema="wp10_warm_native_cost_v1", scope=SCOPE, data_protocol="wp10_xuzhou_k5_v5",
        manifest_producer=frozen.PUBLISHED_MANIFEST_COMMIT, blocks=5, pixels=3920, warmups=WARMUPS, repeats=REPEATS,
        optimizer_updates=0, calibration_updates=0, gt=0, query=0, predictive_label_reads=0),
        "Warm scope is not the accepted label-blind WP10 comparison")
    if (warm["device"]["kind"] != "cpu" or warm["device"]["threads"] != 1 or len(warm["device"]["affinity"]) != 1
            or warm["manifest"]["sha256"] != frozen.PUBLISHED_MANIFEST
            or warm["baseline_evidence"]["sha256"] != REFERENCE_CPU_COST):
        raise ValueError("Warm cost evidence has a foreign CPU or source identity")
    tested = warm["execution_commit"]
    if not isinstance(tested, str) or re.fullmatch("[0-9a-f]{40}", tested) is None:
        raise ValueError("Warm evidence must identify its complete execution commit")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", tested, code["commit"]], check=True)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--exit-code", tested, code["commit"], "--",
                    *frozen.RUNTIME_PATHS, "inference/native_deployment.py", "inference/wp10_deployment_cost.py",
                    "inference/wp10_low_rank_execution.py", "scripts/run_frozen_cpu_deployment.py"], check=True)
    manifest = admission.read(warm["manifest"])
    baseline = admission.read(warm["baseline_evidence"])
    plan = admission.read(warm["plan"])
    if (warm["audit"] != manifest["audit"] or baseline["manifest"] != warm["manifest"]
            or baseline["audit"] != warm["audit"] or baseline["registration"] != manifest["registration"]
            or warm["resident_input_sha256"] != manifest["resident_inputs"]["sha256"]
            or warm["class_ids"] != manifest["class_ids"]):
        raise ValueError("Warm result displaced the original approved observation or score cohort")
    _matches(plan, dict(policy=PLAN, commit=tested, source_manifest=warm["manifest"],
                        baseline_evidence=warm["baseline_evidence"], audit=warm["audit"],
                        resident_inputs=manifest["resident_inputs"], tiles=manifest["tiles"]),
             "Measured runtime was not the sealed label-free plan")
    registration, audit = (admission.read(manifest[k]) for k in ("registration", "audit"))
    return warm, manifest, plan, baseline, registration, audit


def _roster(warm):
    expected = {}
    for seed in SEEDS:
        pair = [f"s{seed}-dense", f"s{seed}-svd-r16"]
        expected[str(seed)] = [pair if i % 2 == 0 else pair[::-1] for i in range(REPEATS)]
    if (warm["orderings"] != expected or set(warm["fallbacks"]) != set(expected)
            or set(warm["cases"]) != {name for groups in expected.values() for name in groups[0]}):
        raise ValueError("Expected the three original paired seeds and balanced order")
    return {int(seed): schedule[0] for seed, schedule in expected.items()}


def _samples(row, blocks, pixels):
    names = ("wall_ms", "read_verify_decode_ms", "forward_ms", "graph_ms",
             "forward_excluding_graph_ms", "output_materialization_and_hash_ms")
    samples = row["samples"]
    if len(samples) != REPEATS:
        raise ValueError("Warm measurement sample count changed")
    for sample in samples:
        if (sample["blocks"] != blocks or sample["graph_calls"] != blocks or sample["host_to_device_ms"] is not None
                or sample["cuda_peak_allocated_bytes"] is not None
                or any(not math.isfinite(sample[key]) or sample[key] < 0 for key in names)
                or not math.isclose(sample["forward_ms"], sample["graph_ms"] + sample["forward_excluding_graph_ms"])
                or sample["wall_ms"] < sample["forward_ms"]):
            raise ValueError("Warm CPU stages must be finite and obey the native graph scope")
    if row["summary"] != _summary(samples, pixels):
        raise ValueError("Warm summary differs from raw samples")


def _cases(warm, manifest, plan, reference, registration, audit, cold, cuda):
    rows = []
    for seed, (dense, candidate) in _roster(warm).items():
        for name in (dense, candidate):
            selected = manifest["cases"][name]
            package = selected["package"]
            row = warm["cases"][name]
            guards = protection(manifest, name, dense, audit, registration["protection"]["limits"])
            hashes = reference_hashes(reference, name, selected, manifest)
            score = audit["report"][name]["selected"]["metrics"]
            _matches(row, dict(
                case=selected["case"], package_state=package["state"], package_bytes=package["bytes"],
                package_sha256=package["weights"]["sha256"], parameters=package["parameters"],
                output_hashes=hashes, guard=guards),
                "Warm predictions or reported quality differ from the matched parent package")
            if (plan["proofs"][name] != dict(guards=guards, output_hashes=hashes)
                    or plan["cases"][name] != selected):
                raise ValueError("Sealed observation-only plan displaced a package or protection decision")
            _samples(row, warm["blocks"], warm["pixels"])
            cold_case = cold["summary"][name]
            gpu_case = cuda["cases"][name]
            if (cold_case["package_bytes"] != package["bytes"] or cold_case["parameters"] != package["parameters"]
                    or len(cold_case["first_result_ms"]) != 3 or gpu_case["state"] != package["state"]
                    or gpu_case["targets"] != audit["report"][name]["selected"]["targets"]):
                raise ValueError("Archived cold or CUDA case differs in package or V5 score order")
            rows.append(dict(name=name, seed=seed, arm=selected["case"]["arm"], scope=warm["data_protocol"],
                             parameters=package["parameters"], package_bytes=package["bytes"],
                             quality=score, guard=guards, actual_operators=row["operator_evidence"],
                             warm_cpu=row["summary"], reused_cold_cpu=cold_case,
                             reused_cuda=dict(
                                 timing_scope="historical_first_instrumented_resident_sweep",
                                 first_sweep_ms=gpu_case["first_instrumented_sweep"]["wall_ms"],
                                 peak_allocated_delta_bytes=gpu_case["peak_allocated_delta"],
                                 changes_vs_cpu=gpu_case["changed_predictions"],
                                 probability_delta_vs_cpu=gpu_case["cpu_cuda_max_probability_delta"])))
        protection_row = warm["cases"][candidate]["guard"]
        failed = [head for head, val in protection_row.items() if not val["passes_metric_guards"]]
        effective = dense if failed else candidate
        fallback = warm["fallbacks"][str(seed)]
        _matches(fallback, dict(requested=candidate, executed=effective, state=warm["cases"][effective]["package_state"],
                                output_hashes=warm["cases"][effective]["output_hashes"]),
                 "Recorded protection path did not execute the selected release")
        if set(fallback["failed_heads"]) != set(failed):
            raise ValueError("Recorded protection lost a failing head")
    return rows


def build(pins, code):
    if pins.keys() != ARCHIVES.keys() | {"warm"} or any(pins[k]["sha256"] != v for k, v in ARCHIVES.items()):
        raise ValueError("Use the published archived evidence and one approved warm receipt")
    archived = {name: admission.read(pins[name]) for name in ARCHIVES}
    warm, manifest, plan, reference, reg, audit = _warm(pins, code)
    cold, cuda, independent = (archived[k] for k in ("cold", "cuda", "cuda_audit"))
    cold_plan = admission.read(cold["registration"])
    _matches(cold, dict(status="passed", first_results=54, optimizer_updates=0, gt=0, query=0),
             "Cold archive did not accept all original fresh processes")
    if cold_plan["manifest"] != warm["manifest"]:
        raise ValueError("Cold archive belongs to a foreign observation or weight cohort")
    _matches(cuda, dict(status="completed_forward_under_strict_settings", optimizer_updates=0, gt=0, query=0,
                        bitwise_repeat_passed=False, numerical_repeat_passed=True, v5_repeat_classification_passed=True),
             "Archived CUDA numerical scope or strict qualification differs")
    _matches(independent, dict(receipt=pins["cuda"], exceeded=0, pairs=38, elements=4021920),
             "Historical CUDA output tolerance was not independently accepted")
    quality, cost = (archived[key] for key in ("onednn_quality", "onednn_cost"))
    _matches(quality, dict(query=0, gt=0, optimizer_updates=0, calibration_new_calls=0),
             "oneDNN historical evaluation exceeded the frozen label and update scope")
    _matches(cost, dict(new_updates=0, includes_graph_and_all_three_heads=True, device="cpu"),
             "oneDNN historical cost scope differs")
    low_bit = {name: dict(
        operators=row["dispatch"], guards=row["versus_same_budget_fp32_guards"],
        quality=row["metrics"], costs=cost["resident"][name]["summary"],
        package_bytes=cost["package_bytes"][name])
               for name, row in quality["candidate"].items()}
    for name in ("wp11_cpu", "wp11_cuda"):
        if archived[name]["status"] != "deployment_engineering_accepted":
            raise ValueError("WP11 receipt lacks deployment acceptance")
    return dict(
        status="passed", kind="wp10_wp12_native_deployment_review", consumer_commit=code["commit"],
        measured_warm_commit=warm["execution_commit"], input_sha256={k: v["sha256"] for k, v in pins.items()},
        new_warm_cpu=dict(scope=SCOPE, device=warm["device"], samples_per_case=REPEATS,
                          blocks=warm["blocks"], pixels=warm["pixels"], fallbacks=warm["fallbacks"]),
        matched_rows=_cases(warm, manifest, plan, reference, reg, audit, cold, cuda),
        reused_cuda=dict(device=cuda["runtime_device"], runtime_commit=cuda["commit"],
                         numerical_passed=True, bitwise_passed=False, independent_pairs=independent["pairs"],
                         input_copy_ms=cuda["resident_input_copy"]["wall_ms"]),
        reused_low_bit_cpu=dict(scope="separate_wp10_xuzhou_int8_qat_recovery",
                                onednn=low_bit, ternary="historical_storage_decode_then_float_compute",
                                cuda_int8=None, cuda_ternary=None),
        reused_wp11=dict(cpu_sha256=ARCHIVES["wp11_cpu"], cuda_sha256=ARCHIVES["wp11_cuda"],
                         scope="fixed_and_dynamic_matched_wp11_separate_requests"),
        new_cuda=None, speedup_across_scopes=None, optimizer_updates=0, raw_gt_reads=0, query_reads=0)


def _csv(out, report):
    rows = []
    for row in report["matched_rows"]:
        for head, quality in row["quality"].items():
            warm = row["warm_cpu"]
            rows.append(dict(
                case=row["name"], seed=row["seed"], arm=row["arm"], scope=row["scope"], head=head,
                oa=quality["oa"], aa=quality["aa"], nll=quality["nll"],
                guard_passed=row["guard"][head]["passes_metric_guards"],
                failed_recall_classes=json.dumps(row["guard"][head]["failed_recall_classes"]),
                new_small_class_errors=json.dumps(row["guard"][head]["new_small_class_errors"], sort_keys=True),
                package_bytes=row["package_bytes"], parameters=row["parameters"],
                warm_cpu_p50_ms=warm["phases"]["wall_ms"]["p50_ms"],
                warm_cpu_p95_ms=warm["phases"]["wall_ms"]["p95_ms"],
                warm_cpu_pixels_per_second=warm["sequential_pixels_per_second"],
                reused_cpu_cold_p50_ms=row["reused_cold_cpu"]["median_ms"],
                reused_cuda_first_resident_ms=row["reused_cuda"]["first_sweep_ms"]))
    with (out / "quality-cost.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("warm", *ARCHIVES):
        parser.add_argument("--" + key.replace("_", "-"), nargs=2, required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if not training_container_kind() or not __debug__ or code["commit"] != args.expected_commit:
        raise ValueError("Expected an exact authorized LXC review")
    require_frozen_code(code)
    pins = {name: dict(zip(("path", "sha256"), getattr(args, name), strict=True)) for name in ("warm", *ARCHIVES)}
    out = external_directory(args.out)
    if out.exists() or any(out == Path(pin["path"]).parent or out in Path(pin["path"]).parents for pin in pins.values()):
        raise ValueError("Use a new output directory separate from the source evidence")
    report = build(pins, code)
    require_frozen_code(code)
    out.mkdir()
    _csv(out, report)
    return dev._json_new(out / "receipt.json", report)


if __name__ == "__main__":
    main()
