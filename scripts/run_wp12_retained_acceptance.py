"""Assemble retained-observation receipts and run the bounded holder engineering proof."""

import argparse
from collections import Counter
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from evaluation.frozen_training_stages import HEADS, VIEWS, file_pin, json_pin, require
from evaluation.retained_spectral_evidence import LAYERS
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from scripts.run_wp12_retained_evidence import CASES, SCHEMA as NATIVE
from transfer import native_independent_release as independent, wp02_sensor_sidecar as optical
from transfer import wp06_center_spectral as center
from transfer.development_bank_training import _baseline
from transfer.g3_training import state_hash
from transfer.native_scale_observation import forward
from transfer.spatial_coverage_protocol import bank_view

SCHEMA = "wp12_retained_observation_acceptance_v1"
GPU_CASES = CASES[2:3] + CASES[4:]
FUTURE = dict(status="awaiting_independent_release_and_training_authorization",
              arms=["native", "native_with_retained_center_residual"],
              paired_seeds=[64, 512, 2048],
              same=["support", "calibration", "evaluation", "init", "trainable_parameter_count",
                    "optimizer_updates", "arithmetic", "view_column_order", "native_graph_geometry",
                    "preprocessing", "prediction_and_selection_rule"],
              views=list(VIEWS), ordinary_aggregate_min_delta=-.005, ordinary_focus_min_recall_delta=-.02,
              honghu_small_class_ids=[15, 21], honghu_additional_same_point_errors=0,
              matched_cost_stages=["released_input_fetch", "statistics_transform", "band_selection",
                    "native_lattice", "host_to_device", "center_projection_if_used", "native_QA",
                    "VCA_CNN_graph_three_heads", "prediction_projection_and_output_synchronization"],
              report_cost=["preparation", "forward", "total_p50", "total_p95", "peak_memory",
                           "checkpoint_or_package_bytes", "calibration_and_training_cost"],
              real_quality=None, production_cost=None, deployed=False)


def close_scores(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def optical_example(root, out, bank, actual, code, role):
    base = bank.release.bands
    source = out/"synthetic-source.json"
    width = len(base.band_ids)
    ids = [100+i*2 for i in range(width)]
    deleted = [101]
    measurements = [.45 + .02*i for i in range(width)]
    write_json(source, dict(scope="synthetic_table_format_example", wavelengths_um=measurements))
    doc = dict(schema=optical.SCHEMA, scene=bank.release.scene, sensor="synthetic engineering sensor",
        bank_sha256=bank.pin["sha256"], base_band_sha256=base.fingerprint,
        source=dict(version="engineering-fixture-v1", publisher="project synthetic fixture",
                    uri="https://example.invalid/synthetic-optical-fixture", terms="synthetic engineering use",
                    evidence=file_pin(source)),
        columns=[dict(delivered_id=key, detector_id=detector, center=value)
                 for key, detector, value in zip(base.band_ids, ids, measurements, strict=True)],
        detector_axis=sorted(ids + deleted), removed_detector_ids=deleted, unit="um", fwhm=[.003]*width,
        srf=[dict(wavelengths=[i-.002, i, i+.002], response=[.1,1.,.1]) for i in measurements])
    sidecar, approval = out/"sidecar.json", out/"optical-approval.json"

    def admission():
        write_json(sidecar, doc)
        write_json(approval, dict(schema=optical.APPROVAL, sidecar_sha256=hash_file(sidecar),
            bank_sha256=bank.pin["sha256"], scene=bank.release.scene,
            permission="read_only_optical_metadata", mode="synthetic"))
        return optical.admit(root, bank, file_pin(sidecar), file_pin(approval))

    first = admission()
    updated = first.view(actual, code, role)
    require(updated.observation is actual.view.observation and updated.bands.band_ids == actual.view.bands.band_ids
            and updated.bands.wavelength_quality == "verified" and updated.bands.fwhm_unit == "nm"
            and updated.bands.fingerprint != actual.view.bands.fingerprint, "Verified metadata view changed observations")
    require(np.allclose(first.measured.wavelengths, np.asarray(measurements)*1000)
            and np.allclose(first.measured.fwhm, 3.), "Physical unit normalization differs")
    doc["source"]["version"] = "engineering-fixture-v2"
    second = admission()
    require(first.measured.wavelengths == second.measured.wavelengths
            and first.measured.fingerprint != second.measured.fingerprint, "Optical release identities collided")
    failures = {}
    for name, key, value in (
            ("unit", "unit", "auto"),
            ("reordered_columns", "columns", list(reversed(doc["columns"]))),
            ("wrong_bank", "bank_sha256", "0"*64),
            ("wrong_base", "base_band_sha256", "0"*64),
            ("unreconciled_removal", "removed_detector_ids", []),
            ("incomplete_srf", "srf", doc["srf"][:-1])):
        previous = doc[key]
        doc[key] = value
        try:
            admission()
        except ValueError as exc:
            failures[name] = str(exc)
        finally:
            doc[key] = previous
    require(len(failures) == 6, "A malformed optical table passed admission")
    admission()
    return dict(scope="synthetic", first_fingerprint=first.measured.fingerprint,
        second_fingerprint=second.measured.fingerprint, sensor_width=width,
        geometries_equal=True, retained_bytes_equal=True, rejected=failures,
        sidecar=hash_file(sidecar), approval=hash_file(approval))


def engineering(root, out, code, device):
    from evaluation.wp11_deployment_cost import gpu_scope

    uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device == "cuda":
        require(uuid is not None and uuid.startswith("GPU-"), "Pin a single physical GPU UUID")
    with (nullcontext((None, None, None)) if device == "cpu" else gpu_scope(uuid)) as (resource, numeric, _):
        out.mkdir(parents=True)
        history = independent.audit_existing(root, out/"historical", code)
        verified = independent.synthetic_demo(root, out/"holder-fixture", code)
        fixture = out/"holder-fixture"
        manifest = json_pin(root, file_pin(fixture/"manifest.json"))
        bank = independent.bank_from_entry(root, manifest["candidate"])
        row = bank.role_row("devval", manifest["roles"]["evaluation"][0])
        role = row["role"]
        model = _baseline(bank.release, 64).to(device).eval().requires_grad_(False)
        state = state_hash(model)
        checkpoint = out/"native.pt"
        save_model_checkpoint(model, checkpoint)
        restored = load_model_checkpoint(checkpoint, device).requires_grad_(False)
        restored.backbone.adapter.cache_attention = False
        require(state_hash(restored) == state, "Compatible checkpoint does not round-trip")
        axis = center.CenterAxis.from_bank(bank)
        rows = {}
        for band in VIEWS:
            events = {}
            for name, arm in (("native", model), ("center", model), ("restored_center", restored), ("restored_native", restored)):
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
                actual, _, targets = bank_view(bank, row, code["commit"], band, factor=1)
                prep = perf_counter()-start
                with torch.inference_mode():
                    if "center" in name:
                        prediction, graph, timing = center.predict(arm, actual, code["commit"], role, axis)
                    else:
                        timing = {}
                        prediction, graph = forward(arm, actual, code["commit"], role, timed=timing)
                    projected = [actual.lattice.project(p).cpu() for p in prediction[:3]]
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = perf_counter()-start
                events[name] = (projected, graph, dict(prepare_seconds=prep, total_seconds=elapsed,
                     graph_seconds=sum(timing["graph_seconds"]),
                     peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None))
            native, graph, _ = events["native"]
            bypass = events["center"][0]
            for i in range(len(HEADS)):
                close_scores(events["restored_native"][0][i], native[i])
                close_scores(events["restored_center"][0][i], bypass[i])
                if i > 0:
                    close_scores(native[i], bypass[i])
            require(all(value[1] == graph for value in events.values())
                    and not torch.equal(native[0], bypass[0]) and len(targets.cells) > 0,
                    "Center smoke failed to isolate the fused head on the same graph")
            rows[band] = dict(width=len(actual.view.bands.band_ids), q_shape=graph["q_shape"],
                          fused_changed=True, other_heads_exact=True, restored_both_exact=True,
                          elapsed={arm: row[2] for arm, row in events.items()})
        retained, _, _ = bank_view(bank, row, code["commit"], "central_gap", factor=1)
        sidecar = optical_example(root, out, bank, retained, code["commit"], role)
        require(state_hash(model) == state_hash(restored) == state, "Read-only bypass changed the checkpoint")
        result = dict(schema=SCHEMA, scope="synthetic_engineering_only", commit=code["commit"], device=device,
                      admission=resource, numerical=numeric, history=history["sha256"],
                      independent_fixture=verified["sha256"], optical=sidecar, axis=axis.fingerprint,
                      checkpoint=file_pin(checkpoint)["sha256"], checkpoint_bytes=checkpoint.stat().st_size,
                      parameters=sum(p.numel() for p in model.parameters()), views=rows,
                      optimizer_updates=0, query_accessed=False, raw_gt_loaded=False)
        require_frozen_code(code)
        write_json(out/"engineering.json", result)


def metric_summary(truth, rows, classes, focus):
    prediction = [row["predicted"] for row in rows]
    result = classification_metrics(truth, prediction, range(1, classes+1))
    by_class = {}
    for label in focus:
        observed = [row for y, row in zip(truth, rows, strict=True) if y == label]
        errors = Counter(row["predicted"] for row in observed if row["predicted"] != label)
        competitors = Counter(row["highest_other"] for row in observed)
        margins = [row["margin"] for row in observed if row["margin"] is not None]
        by_class[str(label)] = dict(n=len(observed), correct=len(observed)-sum(errors.values()),
                error_classes=dict(errors.most_common()), highest_other=dict(competitors.most_common()),
                fixed_rival=observed[0]["fixed_rival"] if observed else None,
                mean_fixed_margin=float(np.mean(margins)) if margins else None)
    return dict(oa=result["oa"], aa=result["aa"], macro_f1=result["macro_f1"], per_class=result["per_class"],
                focus=by_class)


def metrics(view):
    all_heads = {f"native_{key}": value for key, value in view["native"].items()}
    for layer in LAYERS:
        for method, rows in view["readout"]["layers"][layer].items():
            all_heads[f"{layer}_{method}"] = rows
    return all_heads


def weight_fraction(rows, field):
    variances = [row["pixels"]*row[field] for row in rows]
    totals = [row["pixels"]*row["total"] for row in rows]
    return sum(variances) / sum(totals) if sum(totals) else None


def device_compare(a, b):
    require((a["checkpoint"], a["state"], a["authority"]) == (b["checkpoint"], b["state"], b["authority"]),
            "CPU/GPU model or Bank differs")
    result = {}
    for band in VIEWS:
        x, y = [case["views"][band] for case in (a, b)]
        require(x["truth"] == y["truth"] and x["band_ids"] == y["band_ids"],
                "CPU/GPU label or observation scope differs")
        graph_key = lambda row: (row["pixels"], row["regions"], row["nonempty_regions"], row["nnz"])
        require(all([graph_key(r) for r in x["roles"][role]["q"]] == [graph_key(r) for r in y["roles"][role]["q"]]
                    for role in ("devtrain", "devval")), "CPU/GPU Q support differs")
        maximum = {}
        for head in HEADS:
            p, q = [np.asarray([row[head] for row in view["native_probabilities"]]) for view in (x, y)]
            require(np.array_equal(p.argmax(1), q.argmax(1)) and np.allclose(p, q, atol=5e-5, rtol=1e-5),
                    "CPU/GPU native head differs")
            maximum[head] = float(np.max(np.abs(p-q)))
        result[band] = dict(native_changed=0, maximum_absolute=maximum,
            support_only_changed={key: sum(i["predicted"] != j["predicted"] for i, j in zip(left, right, strict=True))
            for (key, left), right in zip(metrics(x).items(), metrics(y).values(), strict=True) if not key.startswith("native_")})
    return result


def read_cases(root, directories, commit_list, acceptable, device):
    cases = {}
    for directory, expected in zip(directories, commit_list, strict=True):
        require(root in directory.resolve(strict=True).parents, "Case directories must stay inside authorized runs")
        for path in sorted(directory.glob("*.json")):
            pin = file_pin(path)
            receipt = json_pin(root, pin)
            name = receipt["case"]
            require(receipt["schema"] == NATIVE and receipt["producer"] == expected and name in acceptable
                    and name not in cases and receipt["result"]["case"] == name and receipt["device"] == device
                    and receipt["optimizer_updates"] == 0 and receipt["raw_gt_loaded"] is receipt["query_accessed"] is False,
                    "Incomplete or foreign native receipt")
            cases[name] = (receipt, pin["sha256"])
    require(set(cases) == set(acceptable), "Missing or duplicate cases in acceptance assembly")
    return cases


def assemble(root, out, code, cpu, gpu, fixture):
    out.mkdir(parents=True)
    demo = json_pin(root, fixture)
    require(demo["schema"] == SCHEMA and demo["commit"] == code["commit"]
            and demo["scope"] == "synthetic_engineering_only"
            and demo["optimizer_updates"] == 0 and demo["raw_gt_loaded"] is demo["query_accessed"] is False,
            "Engineering receipt differs")
    gpu_checks = {case: device_compare(cpu[case][0]["result"], gpu[case][0]["result"]) for case in GPU_CASES}
    summary = {}
    fields = ("scene", "case", "stage", "view", "point_ordinal", "truth", "readout",
              "predicted", "highest_other", "fixed_rival", "margin", "true_class_fitted")
    with (out/"same-points.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_out = csv.DictWriter(stream, fieldnames=fields)
        csv_out.writeheader()
        for case in CASES:
            item, checksum = cpu[case]
            scene = item["result"]
            summary[case] = dict(scene=scene["scene"], step=scene["stage"], cpu_sha256=checksum,
                    cpu_commit=item["producer"], gpu_sha256=gpu[case][1] if case in gpu else None,
                    gpu_commit=gpu[case][0]["producer"] if case in gpu else None,
                    focus=scene["focus"], state=scene["state"], authority=scene["authority"],
                    checkpoint=scene["checkpoint"], archived_sha256=scene["archive_sha256"], views={})
            base = metrics(scene["views"]["full"])
            for band in VIEWS:
                view = scene["views"][band]
                truth = view["truth"]
                require(truth == scene["views"]["full"]["truth"], "Devval labels changed across view")
                q = view["roles"]["devval"]["q"]
                quality = {}
                for readout, rows in metrics(view).items():
                    item = metric_summary(truth, rows, scene["classes"], scene["focus"])
                    changes = Counter((label, a["predicted"], b["predicted"])
                                      for label, a, b in zip(truth, base[readout], rows, strict=True))
                    item["same_point_transitions"] = [list(key) + [count] for key, count in sorted(changes.items())]
                    quality[readout] = item
                    csv_out.writerows(dict(scene=scene["scene"], case=case, stage=scene["stage"], view=band,
                         point_ordinal=i, truth=label, readout=readout, **row)
                         for i, (label, row) in enumerate(zip(truth, rows, strict=True), 1))
                summary[case]["views"][band] = dict(band_ids=view["band_ids"], quality=quality,
                     role_population={role: values["label_counts"] for role, values in view["roles"].items()},
                     moments={role: values["moments"] for role, values in view["roles"].items()},
                     q=dict(blocks=len(q), single_region=sum(row["nonempty_regions"] == 1 for row in q),
                         single_hard_region=sum(row["occupied_hard_regions"] == 1 for row in q),
                         within_region_fraction=weight_fraction(q, "within_region"),
                         between_region_fraction=weight_fraction(q, "between_region"), per_block=q),
                     historical_comparison=view["archived_comparison"],
                     fit_classes=view["readout"]["fit_class_ids"], fit_missing=view["readout"]["classes_missing_from_fit"])
    require_frozen_code(code)
    write_json(out/"wp12-retained-acceptance.json", dict(schema=SCHEMA, commit=code["commit"], cases=summary,
        device_checks=gpu_checks, device_admission={case: gpu[case][0]["admission"] for case in GPU_CASES},
        historical_static=next(iter(cpu.values()))[0]["static"], engineering=fixture["sha256"],
        proposed_registered_gate=FUTURE, point_csv_sha256=hash_file(out/"same-points.csv"),
        optimizer_updates=0, query_accessed=False, raw_gt_loaded=False, independent_real_outcome=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("engineer", "assemble"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu", action="append", type=Path, default=[])
    parser.add_argument("--cpu-commit", action="append", default=[])
    parser.add_argument("--gpu", action="append", type=Path, default=[])
    parser.add_argument("--gpu-commit", action="append", default=[])
    parser.add_argument("--engineering-receipt", type=Path)
    args = parser.parse_args()
    root = args.runs.resolve(strict=True)
    out = external_directory(args.out)
    require(root in out.parents, "WP12 outputs must be outside checkout in the authorized run area")
    code = code_snapshot()
    require(code["commit"] == args.expected_commit, "Expected complete source commit differs")
    require_frozen_code(code)
    if args.action == "engineer":
        engineering(root, out, code, args.device)
    else:
        cpu = read_cases(root, args.cpu, args.cpu_commit, CASES, "cpu")
        gpu = read_cases(root, args.gpu, args.gpu_commit, GPU_CASES, "cuda")
        assemble(root, out, code, cpu, gpu, file_pin(args.engineering_receipt.resolve(strict=True)))


if __name__ == "__main__":
    main()
