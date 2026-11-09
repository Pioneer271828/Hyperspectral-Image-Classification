"""Immutable raw-state, selection and LR audit after the selected-state gate."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from evaluation.g4_target import check_files, independent_trainval, load_case, verify_training
from joint_calibrated_selection import validation_candidate
from joint_training import load_plan, schedule_for
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, write_json
from research.reproduction.m4g4b.none_selected_compatibility import (
    selected_conclusion, verify_source_implementation)
from source_data_context import load_source_context
from source_pretrain import SourcePretrainRuntime, select_case
from target_initialization import METHODS, state_id

SELECTED_PRODUCER = "139087a6383adcc199612b4c6f8213ba07b42b39"


def first_selected(events, role):
    if (role not in ("source", "target") or not events
            or any(b["step"] <= a["step"] for a, b in zip(events, events[1:]))
            or any(not math.isfinite(e[k]) for e in events for k in ("val_loss", "macro_oa"))):
        raise ValueError("Finite chronological source/target events required")
    return (max(events, key=lambda e: e["macro_oa"]) if role == "source"
            else min(events, key=lambda e: e["val_loss"]))["step"]


def replay_lr(settings, events, trace):
    """Only scheduler.step; the dummy optimizer never takes an update."""
    first_selected(events, "target")
    if ([r["step"] for r in trace] != list(range(1, len(trace) + 1))
            or any(e["step"] not in range(1, len(trace) + 1) for e in events)):
        raise ValueError("Events must belong to a complete update trace")
    training = settings["config"]["training"]
    optimizer = torch.optim.SGD([
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": training["lr"]},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": training["head_lr"]}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
        factor=settings["scheduler"]["factor"], patience=settings["scheduler"]["patience"],
        threshold=settings["scheduler"]["threshold"], threshold_mode="rel")
    lookup = {e["step"]: e for e in events}
    rates = []
    for step in range(1, len(trace) + 1):
        # Training logs rates before the validation at this same update.
        rates.append([float(g["lr"]) for g in optimizer.param_groups])
        if step in lookup:
            scheduler.step(lookup[step]["val_loss"])
    return rates


def trajectory_decision(settings, old, new, trace, role, selected_step):
    if [e["step"] for e in old] != [e["step"] for e in new]:
        raise ValueError("Identical validation-event coverage required")
    old_rates = replay_lr(settings, old, trace)
    if old_rates != [r["learning_rates"] for r in trace] or first_selected(old, role) != selected_step:
        raise ValueError("Original scheduler/selection does not reproduce frozen trace")
    new_rates = replay_lr(settings, new, trace)
    mismatches = [i + 1 for i, (a, b) in enumerate(zip(old_rates, new_rates)) if a != b]
    selected = first_selected(new, role)
    return {"original_trace_reproduced": True, "original_selected_step": selected_step,
            "ordered_selected_step": selected, "selected_step_unchanged": selected == selected_step,
            "learning_rates_unchanged": not mismatches, "lr_mismatch_update_steps": mismatches,
            "historical_lr_transitions": [i + 1 for i in range(1, len(old_rates))
                                          if old_rates[i] != old_rates[i - 1]]}


def close_scalar(old, new):
    if not math.isfinite(old) or not math.isfinite(new):
        raise ValueError("Finite metric required")
    return abs(new - old) <= 1e-6 + 1e-5 * abs(old)


def metrics_compatible(old, new):
    failures = []
    for key in ("val_loss", "macro_oa"):
        if not close_scalar(old[key], new[key]):
            failures.append(key)
    if set(old["metrics"]) != set(new["metrics"]):
        raise ValueError("Task coverage changed")
    max_nll_error = 0.
    for name in old["metrics"]:
        for phase in ("train", "val"):
            for head in ("fused", "cnn", "graph"):
                a, b = old["metrics"][name][phase][head], new["metrics"][name][phase][head]
                key = name + "/" + phase + "/" + head
                if a["confusion"] != b["confusion"] or a["recall"] != b["recall"]:
                    failures.append(key + "/classification_counts")
                if not all(close_scalar(a[k], b[k]) for k in ("nll", "oa", "aa")):
                    failures.append(key + "/scalar_metrics")
                max_nll_error = max(max_nll_error, abs(a["nll"] - b["nll"]))
    return {"passes_metrics_gate": not failures, "failures": failures,
            "max_head_nll_abs_error": max_nll_error}


def audit_trajectory(model, runtime, items, folder, result, out, role):
    old = json.loads((folder / "history.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    ordered, records = [], []
    metrics = recalls = 0
    candidate_fn = getattr(runtime, "validation_candidate", validation_candidate)
    for event in old:
        step = event["step"]
        path = folder / f"raw-validation-{step:06d}.pt"
        raw_sha = result["artifacts"].get(path.name)
        if not raw_sha or file_hash(path) != raw_sha:
            raise ValueError("Historical raw validation snapshot changed")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        before = state_id(model)
        with OrderedValidation():
            # This owned working copy may be replaced; historical raw is immutable.
            copy_path = out / "calibration-raw-copy.pt"
            candidate, _ = candidate_fn(model, items, copy_path)
            if state_id(model) != before or file_hash(copy_path) != raw_sha:
                raise ValueError("Calibration mutated raw model/input serialization")
            calibrated = state_id(candidate)
            current, values = runtime.validation_snapshot(candidate, items)
            if state_id(candidate) != calibrated:
                raise ValueError("Validation mutated calibrated candidate")
        for name, item in items.items():
            reference = item
            if "train_labels" not in item:
                reference = {"class_num": item["class_num"]}
                for phase in ("train", "val"):
                    gt = item[phase + "_samples_gt"]
                    reference[phase + "_labels"] = gt[gt > 0].long() - 1
            oracle = independent_trainval(values[name], reference, current["metrics"][name])
            metrics += oracle["head_metrics"]
            recalls += oracle["class_recalls"]
        ordered.append({"step": step, **current})
        records.append({"step": step, "raw_sha256": raw_sha,
                        "original_macro_oa": event["macro_oa"],
                        "original_val_loss": event["val_loss"],
                        "ordered": current, **metrics_compatible(event, current)})
        if len(records) % 10 == 0:
            print("NONE_TRAJECTORY_EVENT", role, folder.name, len(records), flush=True)
        del candidate, values
    decision = trajectory_decision(result["settings"], old, ordered, trace, role, result["selected_step"])
    passed = (decision["selected_step_unchanged"] and decision["learning_rates_unchanged"]
              and all(r["passes_metrics_gate"] for r in records))
    write_json(out / "events.json", records)
    return {**decision, "passes_trajectory_gate": passed, "validation_events": len(old),
            "metrics_gate_failures": sum(not r["passes_metrics_gate"] for r in records),
            "head_metrics": metrics, "class_recalls": recalls,
            "events_sha256": file_hash(out / "events.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target-root", "target-launch-sha256", "target-acceptance-sha256",
                 "source-root", "source-launch-sha256", "selected-receipt", "selected-sha256",
                 "out", "expected-commit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    current = code_snapshot()
    if current["dirty"] or current["commit"] != args.expected_commit:
        raise ValueError("Clean frozen trajectory verifier required")
    selected_path = Path(args.selected_receipt).resolve(strict=True)
    if file_hash(selected_path) != args.selected_sha256:
        raise ValueError("Selected-state audit receipt changed")
    selected = json.loads(selected_path.read_text())
    scode = selected["audit_code"]
    if (selected["status"] != "audit_completed"
            or selected_conclusion(selected["cases"]) != "selected_gate_passed_trajectory_audit_required"
            or selected["new_optimizer_updates"] != 0 or selected["query_labels_accessed"] is not False
            or selected["head_metrics"] != 216 or selected["class_recalls"] != 2880
            or scode["commit"] != SELECTED_PRODUCER or scode["dirty"]
            or digest(scode["files"]) != scode["source_hash"]
            or any(current["files"].get(k) != v for k, v in scode["files"].items())):
        raise ValueError("Complete selected-state audit and unchanged additive verifier required")
    target_root, source_root = Path(args.target_root).resolve(strict=True), Path(args.source_root).resolve(strict=True)
    out = external_directory(args.out)
    if out.parent != target_root.parent or out in (target_root, source_root, selected_path.parent):
        raise ValueError("Fresh sibling trajectory output required")
    registration_path = selected_path.parent / "registration.json"
    registration = json.loads(registration_path.read_text())
    for key in ("target_launch_sha256", "target_acceptance_sha256", "source_launch_sha256"):
        if registration[key] != getattr(args, key):
            raise ValueError("Selected and trajectory phases use different inputs")
    stage = verify_training(target_root, args.target_launch_sha256, args.target_acceptance_sha256, current)
    if file_hash(source_root / "launch.json") != args.source_launch_sha256:
        raise ValueError("Source launch changed")
    launch = json.loads((source_root / "launch.json").read_text())
    implementation = verify_source_implementation(launch["code"], current)
    roots = (source_root, target_root, selected_path.parent)
    pins = {str(p): file_hash(p) for root in roots for p in root.rglob("*") if p.is_file()}
    historical_paths = set(pins)
    out.mkdir(exist_ok=False)
    write_json(out / "registration.json", {"audit_code": current, "selected_sha256": args.selected_sha256,
        "selected_registration_sha256": file_hash(registration_path),
        "source_implementation_proof": implementation, "inputs": registration,
        "roles": 27, "validation_events_per_role": 51, "validation_events_total": 1377,
        "new_optimizer_updates": 0, "query_labels_accessed": False})
    started = time.monotonic()
    rows = []
    for case in launch["cases"]:
        select_case(launch, launch["code"], case["held_out"], case["seed"])
        context = load_source_context(launch["prepared"], launch["data_root"],
            held_out=case["held_out"], consumer=launch["code"],
            manifest_sha256=load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"])
        if context.record != case["source_context"]:
            raise ValueError("Source context differs")
        pins.update(context.record["protected_files"])
        folder = Path(case["out"])
        result = json.loads((folder / "result.json").read_text())
        if (folder.resolve() != source_root / "cases" / case["name"]
                or result["settings"] != case["settings"] or result["start_code"] != launch["code"]
                or result["optimizer_steps"] != 1002 or result["status"] != "completed"
                or result["identity"]["prepared"] != context.record or result["test_scoring"] is not False):
            raise ValueError("Frozen source result differs")
        for name, sha in result["artifacts"].items():
            if Path(name).name != name or file_hash(folder / name) != sha:
                raise ValueError("Source artifact differs")
        trace = json.loads((folder / "trace.json").read_text())
        if [r["dataset"] for r in trace] != schedule_for(case["settings"], task_names=context.sources):
            raise ValueError("Source exposure differs")
        gpu = {n: {k: v.cuda() if torch.is_tensor(v) else v for k, v in item.items()}
               for n, item in context.items.items()}
        runtime = SourcePretrainRuntime()
        model = runtime.load_selected(folder / "primary.pt", gpu,
            folder / f'raw-validation-{result["selected_step"]:06d}.pt', device="cuda")
        owned = out / ("source-" + case["name"])
        owned.mkdir()
        row = {"role": "source", "name": case["name"], "target": case["held_out"], "seed": case["seed"],
               **audit_trajectory(model, runtime, gpu, folder, result, owned, "source")}
        if row["validation_events"] != 51:
            raise ValueError("Incomplete full source trajectory")
        rows.append(row)
        write_json(owned / "result.json", row)
        write_json(out / "progress.json", rows)
        print("NONE_TRAJECTORY_SOURCE", case["name"], row["passes_trajectory_gate"], flush=True)
        del model, runtime, gpu, context
        torch.cuda.empty_cache()
    for entry in stage["rows"]:
        case = entry["case"]
        if case["method"] not in METHODS[:2]:
            continue
        model, runtime, items = load_case(entry, device="cuda")
        pins.update(case["provenance"]["protected_files"])
        owned = out / ("target-" + case["name"])
        owned.mkdir()
        row = {"role": "target", "name": case["name"], "target": case["target"], "seed": case["seed"],
               "method": case["method"],
               **audit_trajectory(model, runtime, items, Path(case["out"]), entry["result"], owned, "target")}
        if row["validation_events"] != 51:
            raise ValueError("Incomplete full target trajectory")
        rows.append(row)
        write_json(owned / "result.json", row)
        write_json(out / "progress.json", rows)
        print("NONE_TRAJECTORY_TARGET", case["name"], row["passes_trajectory_gate"], flush=True)
        del model, runtime, items
        torch.cuda.empty_cache()
    check_files(pins)
    if ({str(p) for root in roots for p in root.rglob("*") if p.is_file()} != historical_paths
            or code_snapshot() != current):
        raise ValueError("Protected file set/code changed")
    selected_conclusion([{**r, "passes_selected_gate": r["passes_trajectory_gate"]} for r in rows])
    eligible = all(r["passes_trajectory_gate"] for r in rows)
    write_json(out / "receipt.json", {"status": "audit_completed", "audit_code": current,
        "historical_none_eligible_for_reuse": eligible,
        "conclusion": "historical_none_trainval_compatible" if eligible else "historical_none_reuse_rejected",
        "selected_sha256": args.selected_sha256, "cases": rows, "trajectory_audit_completed": True,
        "head_metrics": sum(r["head_metrics"] for r in rows),
        "class_recalls": sum(r["class_recalls"] for r in rows),
        "validation_events": sum(r["validation_events"] for r in rows),
        "protected_files_and_inputs": len(pins), "new_optimizer_updates": 0,
        "query_labels_accessed": False, "elapsed_seconds": time.monotonic() - started,
        "limits": "Train/val arithmetic, selection and LR compatibility only; no new query/accuracy claim"})
    print("NONE_TRAJECTORY_CONCLUSION", eligible, flush=True)


if __name__ == "__main__":
    main()
