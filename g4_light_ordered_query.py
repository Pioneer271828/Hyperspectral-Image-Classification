"""Issue249 final ordered scoring, after authenticated all54 freeze."""
import argparse
import os
from pathlib import Path
import sys
import time

import torch

from evaluation.g4_target import predict_query, query_metrics
from g4_light_frozen_boundary import POLICY as FREEZE_POLICY, case_directory, model_binding
from g4_light_full_target import FullLightTargetRuntime, read_registration, require_gpu, roles
from g4_light_paired_analysis import PLAN_SHA256, analyze_rows
from g4_matched_target_analysis import FT, validate_rows
from g4_matched_target_evaluation import (QUERY_POLICY as REFERENCE_POLICY, new_output, read_query_labels,
                                           require_frozen_boundary)
from g4_matched_target_execution import case_data, unchanged_producer
from g4_support_context import check_files, read
from joint_baseline import NAMES
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.research_config import digest
from research.reproduction.m4g4b.none_source_campaign import wait_gpu
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from target_initialization import state_id


POLICY = "g4_light_issue249_all54_ordered_query_v1"
REGISTRATION_SHA256 = "b3cca8f1028004da30177704b107e959cd2fa47aa7892390522e4589fe39e735"
REFERENCE_FREEZE_SHA256 = "7dd7e87334766d74005d14345e994fc2e700e109f6d8141b462004831496b48c"
REFERENCE_QUERY_SHA256 = "33d7d17130ee94834b9ab3bbc21e3e36917f3a638beb4e1dc188c99d64f81b54"


def light_inputs(freeze, code):
    keys = roles()
    if (freeze["policy"] != FREEZE_POLICY or freeze["status"] != "all54_frozen_before_query"
            or freeze["experiment_issue"] != 249 or freeze["registration_sha256"] != REGISTRATION_SHA256
            or freeze["query_labels_accessed"] is not False or freeze["new_optimizer_updates"] != 0
            or freeze["training_optimizer_updates"] != 81000 or [m["role"] for m in freeze["models"]] != list(map(list, keys))):
        raise ValueError("Exact Issue249 completed all54 freeze required")
    root = external_directory(freeze["training_root"])
    boundary, protected = freeze["boundary_files"], freeze["protected_files"]
    acceptance = str(root / "training-acceptance.json")
    status = str(root / "status.json")
    if boundary.get(acceptance) != freeze["acceptance_sha256"] or status not in boundary:
        raise ValueError("Pinned live campaign completion and acceptance required")
    check_files(boundary)
    if read(status)["status"] != "completed":
        raise ValueError("Issue249 must be completed before query")
    accepted = read(acceptance)
    producer = freeze["producer"]
    if (accepted["status"] != "passed" or accepted["query_labels_accessed"] is not False
            or accepted["optimizer_updates"] != 81000 or accepted["protected_files"] != protected
            or accepted["registration_sha256"] != REGISTRATION_SHA256 or accepted["code"] != producer
            or accepted["audit_receipts"] != {k: v for k, v in boundary.items() if k not in (acceptance, status)}):
        raise ValueError("Complete training acceptance differs from frozen inputs")
    unchanged_producer(producer, freeze["code"])
    unchanged_producer(freeze["code"], code)
    registration, pins = read_registration(freeze["registration_path"], REGISTRATION_SHA256, producer)
    if (registration["experiment_issue"] != 249 or registration["analysis_plan"]["sha256"] != PLAN_SHA256
            or registration["reference_launch"] != freeze["reference_launch"]
            or external_directory(registration["out_dir"]) != root
            or any(protected.get(k) != v for k, v in pins.items())):
        raise ValueError("Light freeze does not match its preregistration")
    check_files(protected)
    launch = read(registration["reference_launch"]["path"])
    unchanged_producer(launch["code"], producer)
    reference = {(c["target"], c["seed"], c["mode"]): c for c in launch["cases"] if c["method"] == FT}
    gpus = registration["gpu_ids"]
    for index, (key, model) in enumerate(zip(keys, freeze["models"])):
        target, seed, _, mode = key
        folder = case_directory(root, key)
        audit_path = str(root / "audits" / (folder.name + ".json"))
        audit_sha256 = boundary[audit_path]
        audit = read(audit_path)
        bound = model_binding(root, key, gpus[index % len(gpus)], reference[target, seed, mode],
                              audit, protected, producer, REGISTRATION_SHA256)
        raw = str(folder / f"raw-validation-{bound['selected_step']:06d}.pt")
        if (model != {**bound, "audit_path": audit_path, "audit_sha256": audit_sha256}
                or raw not in audit["artifacts"]):
            raise ValueError("Frozen model, selected raw state or audit binding differs")
    return root, launch, reference


def reference_inputs(freeze, query, launch_sha256, code):
    if (freeze["launch_sha256"] != launch_sha256 or query["policy"] != REFERENCE_POLICY
            or query["status"] != "passed" or query["freeze_sha256"] != REFERENCE_FREEZE_SHA256
            or query["optimizer_updates"] != 0 or query["query_labels_accessed"] is not True):
        raise ValueError("Exact Issue184 ordered reference cohort required")
    unchanged_producer(freeze["code"], code)
    unchanged_producer(query["code"], code)
    require_frozen_boundary(freeze)
    validate_rows(query["models"])
    weights = {m["name"]: m["primary_sha256"] for m in freeze["models"]}
    if any(weights.get(row["name"]) != row["checkpoint_sha256"] for row in query["models"]):
        raise ValueError("Reference metric rows do not match frozen weights")


def matched_releases(reference_freeze, rows, references, data_root, prepared, prepared_sha256):
    releases, pins = {}, {}
    for target in NAMES:
        case = next(c for c in references.values() if c["target"] == target)
        release, raw_pins = read_query_labels(reference_freeze, case, data_root, prepared, prepared_sha256)
        for row in rows:
            if row["target"] == target and digest({**release, "checkpoint_sha256": row["checkpoint_sha256"]}) != row["release_sha256"]:
                raise ValueError(f"Ordered query pixel or label identity differs from Issue184 for {target}")
        releases[target] = release
        pins.update(raw_pins)
    return releases, pins


def score_model(root, reference, frozen, release, protected):
    target, seed, method, mode = frozen["role"]
    folder = case_directory(root, (target, seed, method, mode))
    primary = folder / "primary.pt"
    raw = folder / f"raw-validation-{frozen['selected_step']:06d}.pt"
    pins = {**frozen["model_files"], str(raw): protected[str(raw)]}
    check_files(pins)
    items, _, guard = case_data(reference)
    device = "cuda:0"
    runtime = FullLightTargetRuntime(reference, method, items, folder, REGISTRATION_SHA256,
                                     frozen["physical_gpu"], device=device)
    moved = {name: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()} for name, item in items.items()}
    model = runtime.load_selected(primary, moved, raw, device=device)
    before = state_id(model)
    checkpoint_sha256 = pins[str(primary)]
    with OrderedValidation():
        probabilities = predict_query(model, moved[target], release["indices"])
    metrics, branches = query_metrics(release["labels"], probabilities)
    if state_id(model) != before:
        raise ValueError("Query inference changed selected parameters or buffers")
    guard()
    check_files(pins)
    return {"name": folder.name, "target": target, "seed": seed, "method": method, "mode": mode,
            "metrics": metrics, "branches": branches, "checkpoint_sha256": checkpoint_sha256,
            "release_sha256": digest({**release, "checkpoint_sha256": checkpoint_sha256})}


def evaluate(args, code):
    light, old_freeze, old_query, prepared = [Path(getattr(args, key)).resolve(strict=True)
                                             for key in ("freeze", "reference_freeze", "reference_query", "prepared_manifest")]
    pins = {str(light): args.freeze_sha256, str(old_freeze): REFERENCE_FREEZE_SHA256,
            str(old_query): REFERENCE_QUERY_SHA256, str(prepared): args.prepared_sha256}
    if len(pins) != 4:
        raise ValueError("Separate frozen, reference and prepared input files required")
    check_files(pins)
    freeze = read(light)
    root, launch, references = light_inputs(freeze, code)
    baseline, query = read(old_freeze), read(old_query)
    reference_inputs(baseline, query, freeze["reference_launch"]["sha256"], code)
    out = new_output(args.out, launch, (root, light.parent, old_freeze.parent, old_query.parent,
                                        prepared.parent, Path(args.data_root).resolve(strict=True)))
    protected = freeze["protected_files"]
    if any(out == Path(path).resolve() or out in Path(path).resolve().parents for path in protected):
        raise ValueError("Query output overlaps an accepted input")
    require_gpu(args.gpu)
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    releases, label_pins = matched_releases(baseline, query["models"], references, args.data_root,
                                            prepared, args.prepared_sha256)
    rows = []
    for model in freeze["models"]:
        wait_gpu(args.gpu, out)
        check_files(pins)
        target, seed, _, mode = model["role"]
        row = score_model(root, references[target, seed, mode], model, releases[target], protected)
        rows.append(row)
        write_json(out / (model["name"] + ".json"), row)
        torch.cuda.empty_cache()
    analysis = analyze_rows(rows, query["models"], plan_sha256=PLAN_SHA256)
    check_files({**pins, **label_pins})
    check_files(freeze["boundary_files"])
    check_files(protected)
    require_frozen_code(code)
    receipt = {"policy": POLICY, "status": "passed", "code": code, "freeze_sha256": args.freeze_sha256,
               "reference_query_sha256": REFERENCE_QUERY_SHA256, "prepared_sha256": args.prepared_sha256,
               "models": rows, "analysis": analysis, "query_labels_accessed": True, "optimizer_updates": 0,
               "physical_gpu": args.gpu, "elapsed_seconds": time.monotonic() - started}
    write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "freeze", "freeze-sha256", "reference-freeze", "reference-query",
                 "data-root", "prepared-manifest", "prepared-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if (sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit
            or args.gpu < 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu)):
        raise RuntimeError("Clean expected Linux commit and one explicit physical GPU required")
    evaluate(args, code)
    print(Path(args.out).resolve() / "receipt.json")


if __name__ == "__main__":
    main()
