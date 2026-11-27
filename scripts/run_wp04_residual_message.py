"""Bound the WP04 residual/message experiment and preflight the released real inputs."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from g4_residual_message_target import ARMS, PARAMETERS, POLICY, SEEDS, TARGET, ResidualMessageRuntime, require, settings
from g4_support_context import OBS_KEYS, check_files, overlay_identity, read, validate_overlay
from g4_support_target import settings as support_settings
from joint_baseline import _item_fingerprint
from joint_training import run_training
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, external_directory, require_frozen_code
from target_bn_calibration import observation_id
from target_initialization import METHODS, state_id


PORTABLE = "44c999462209e8527cdc5d423f8da768e9950d2f95e91da5e9fce32e3b65ef08"
SOURCE = "2dbd07d0aa8a2d56613da8ac64f4c4b941dd6b566bbdc9c799ebcb5584c6edd6"
HEADS = ("fused", "graph", "cnn")


def publish(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def portable(root):
    metadata_path = root / "portable-manifest.json"
    check_files({str(metadata_path): PORTABLE})
    meta = read(metadata_path)
    roles = meta["context"]
    names = {"original-manifest.json", "observation.pt", "observation-record.json", "train-dev-overlay.pt"}
    require(meta["source_registration_sha256"] == SOURCE and set(meta["files"]) == set(meta["original_paths"]) == names
            and meta["query"] is False and meta["raw_labels"] is False and meta["optimizer_updates"] == 0,
            "Unexpected portable source or access policy")
    pins = {str(root / name): sha for name,sha in meta["files"].items()}
    pins[str(metadata_path)] = PORTABLE
    check_files(pins)
    catalog, record = [read(root / name) for name in ("original-manifest.json", "observation-record.json")]
    require(roles["manifest_sha256"] == meta["files"]["original-manifest.json"]
            and catalog["execution_enabled"] is False and catalog["query_labels_accessed"] is False
            and catalog["test_scoring"] is False and roles["query_labels_accessed"] is False,
            "Portable registration differs from the approved source role")
    registered = [r for r in catalog["contexts"] if r["name"] == roles["context"]]
    require(len(registered) == 1, "Ambiguous portable roles")
    registry = registered[0]
    original = catalog["observations"][TARGET]["input"]
    require((original["observation_sha256"], original["record_sha256"], registry["sha256"]) == tuple(
            meta["files"][m] for m in ("observation.pt", "observation-record.json", "train-dev-overlay.pt"))
            and original["observation"] == meta["original_paths"]["observation.pt"]
            and original["record"] == meta["original_paths"]["observation-record.json"],
            "Portable bytes are not the catalogued observation and supervision")
    observed, overlay = [torch.load(root / name, weights_only=True, map_location="cpu") for name in
                        ("observation.pt", "train-dev-overlay.pt")]
    require(set(observed) == OBS_KEYS | {"name", "height", "width", "bands", "class_num"}
            and observed["name"] == TARGET and observed["class_num"] == 9
            and [observed["height"], observed["width"]] == registry["shape"]
            and _item_fingerprint(observed) == record["prepared_fields"]
            and digest(record["prepared_fields"]) == registry["graph_identity"] == roles["graph_identity"],
            "Portable observation geometry or graph differs")
    observation_id({k: observed[k] for k in OBS_KEYS})
    validate_overlay(overlay, TARGET, 1, registry["shape"])
    require(overlay_identity(overlay) == registry["overlay_identity"]
            and (registry["phase_hashes"], registry["family_hash"]) == (roles["phase_hashes"], roles["family_hash"]),
            "Portable labels are outside the frozen development overlay")
    observed.update(overlay)
    require(digest(_item_fingerprint(observed)) == meta["item_sha256"], "Portable prepared item hash differs")
    check_files(pins)
    return meta, {TARGET: observed}, pins


def options(meta, arm, seed, device="cpu"):
    context = {**meta["context"], "model_seed": seed, "head_seed": seed+1000}
    bound = {"context": context, "item_sha256": meta["item_sha256"]}
    original = support_settings(TARGET, METHODS[0], "none", model_seed=seed, support_seed=512,
                                per_class=1, binding=bound, device=device)
    return settings(original, arm)


def registration(meta, code):
    cases = [{"seed": s, "arm": a, "head_seed": s+1000,
              "cpu_settings_sha256": digest(options(meta, a, s))} for s in SEEDS for a in ARMS]
    return {"schema": POLICY, "code": {k: code[k] for k in ("commit","source_hash")},
            "portable_manifest_sha256": PORTABLE, "input_sha256": meta["item_sha256"],
            "support_context": meta["context"], "parameters": PARAMETERS, "cases": cases,
            "arms": {k: list(v) for k,v in ARMS.items()}, "main_updates": 144,
            "recovery_check_updates": 144, "steps_whole_prefix_resume_completed": [12, 2, 10, 0],
            "selection": "first minimum fused development NLL at the original scheduled steps",
            "contrast": "(R-cross - R-local) - (Bcap-cross - Bcap-local)",
            "place": "Xuzhou K1/V5 global transductive context; CPU preflight",
            "interpretation": "fixed Q/A, equal parameter count; message alignment and residual operations have different costs",
            "train_query": False, "authorized_in_this_delivery": False}


def preflight(meta, items, code, out, reg_sha, pins):
    start = perf_counter()
    out.mkdir(mode=0o700, exist_ok=False)
    metrics, cases = [], []
    target_item = items[TARGET]
    graph = observation_id({k: target_item[k] for k in OBS_KEYS})
    for seed in SEEDS:
        initial = {}
        for arm in ARMS:
            tick = perf_counter()
            opt = options(meta, arm, seed)
            runtime = ResidualMessageRuntime(opt, items, out / f"{seed}-{arm}")
            model = runtime.build_model(ResearchConfig.from_dict(opt["config"]), {TARGET: 9})
            state = state_id(model)
            leaf = out / f"{seed}-{arm}"
            leaf.mkdir(mode=0o700)
            raw = leaf / "raw.pt"
            candidate, actual = runtime.validation_candidate(model, items, raw)
            facts, probabilities = runtime.validation_snapshot(candidate, items)
            require(state_id(model) == state and actual["observation"] == graph
                    and actual["optimizer_updates"] == 0, "Preflight changed the raw checkpoint or original graph")
            values = probabilities[TARGET]["val"]
            labels = target_item["val_labels"].numpy()
            for head in HEADS:
                p = values[head].numpy().astype(np.float64)
                guess = p.argmax(1)
                cm = np.bincount(9*labels+guess, minlength=81).reshape(9,9)
                score = facts["metrics"][TARGET]["val"][head]
                nll = float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-8, 1)).mean())
                require(cm.tolist() == score["confusion"] and abs(nll-score["nll"]) < 1e-6,
                        "Independent real frozen score differs")
                metrics.extend({"seed": seed, "arm": arm, "head": head, "class": cls+1, "support": int(cm[cls].sum()),
                                "correct": int(cm[cls,cls]), "oa": int(np.trace(cm))/len(labels),
                                "aa": float(np.mean(cm.diagonal()/cm.sum(1))), "nll": nll} for cls in range(9))
            torch.save(values, leaf / "dev-probabilities.pt")
            initial[arm] = runtime.binding
            cases.append({"seed": seed, "arm": arm, "binding": runtime.binding,
                          "source_state_sha256": state, "calibrated_state_sha256": state_id(candidate),
                          "dev_probability_sha256": file_hash(leaf / "dev-probabilities.pt"), "seconds": perf_counter()-tick})
        for key in ("shared_sha256","message_sha256"):
            require(len({b[key] for b in initial.values()}) == 1, "Paired common/message initialization differs")
        for family in ("R","Bcap"):
            require(initial[family+"-cross"]["residual_sha256"] == initial[family+"-local"]["residual_sha256"],
                    "Residual and ordinary control weights differ within message pair")
    with (out / "real-frozen-metrics.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    check_files(pins)
    require_frozen_code(code)
    publish(out / "receipt.json", {"status": "passed", "commit": code["commit"], "registration_sha256": reg_sha,
              "dev_labels": 45, "train_labels": 9, "parameters": PARAMETERS, "case_count": len(cases), "cases": cases,
              "model_forwards": {"ordered_calibration": 12, "calibrated_snapshot": 12},
              "elapsed_seconds": perf_counter()-start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "real_optimizer_updates": 0, "query_access": False, "dev_metrics_sha256": file_hash(out / "real-frozen-metrics.csv")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "preflight", "run-case"))
    parser.add_argument("--material", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--registration-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--stop-after", type=int, choices=(2,))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-real-registry", action="store_true")
    args = parser.parse_args()
    code = code_snapshot()
    require(sys.platform == "linux" and __debug__ and code["commit"] == args.expected_commit and not code["dirty"],
            "Clean expected Linux WP04 checkout required")
    require_frozen_code(code)
    torch.set_num_threads(4 if args.action == "preflight" else 1)
    material_root = args.material.resolve(strict=True)
    meta, items, pins = portable(material_root)
    current = registration(meta, code)
    if args.action == "register":
        require(not any((args.registration_sha256, args.out, args.arm, args.seed, args.stop_after,
                         args.resume, args.execute_real_registry)), "Registration contains execution options")
        path = external_directory(args.registration)
        require(not path.is_relative_to(material_root), "Output cannot enter the portable source directory")
        publish(path, current)
        print(file_hash(path))
        return
    require(args.registration_sha256 and args.out, "Locked registration and external output required")
    check_files({str(args.registration): args.registration_sha256})
    require(read(args.registration) == current, "Live program and inputs differ from registered run")
    directory = external_directory(args.out)
    require(not directory.is_relative_to(material_root) and not material_root.is_relative_to(directory),
            "Output must be separate from portable inputs")
    if args.action == "preflight":
        require(not any((args.arm, args.seed, args.stop_after, args.resume, args.execute_real_registry)),
                "Preflight is zero-update all-arm evaluation")
        preflight(meta, items, code, directory, args.registration_sha256, pins)
    else:
        require(args.execute_real_registry and args.arm and args.seed,
                "A real case requires registered arm, seed, and a separate execution decision")
        opt = options(meta, args.arm, args.seed)
        runtime = ResidualMessageRuntime(opt, items, directory)
        def guard():
            require_frozen_code(code)
            check_files({**pins, str(args.registration): args.registration_sha256})
        provenance = {"scope": POLICY, "portable_manifest_sha256": PORTABLE,
                      "registration_sha256": args.registration_sha256, "query_labels_accessed": False}
        result = run_training(opt, items, directory, provenance, code, device="cpu", runtime_adapter=runtime,
                              finish_guard=guard, resume=args.resume, stop_after=args.stop_after)
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
