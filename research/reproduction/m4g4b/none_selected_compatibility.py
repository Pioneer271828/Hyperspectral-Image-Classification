"""Zero-update selected-state gate for historical none versus ordered validation.

A positive selected-state gate is deliberately insufficient for reuse: historical
selection and scheduler trajectories must be audited separately. A negative gate
rejects reuse without changing or retraining any historical model.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from evaluation.g4_target import check_files, independent_trainval, load_case, verify_training
from joint_calibrated_selection import validation_candidate
from joint_training import _tree_fingerprint, load_plan, schedule_for
from loadData.dataset_registry import file_hash
from m4_ordered_source import OrderedValidation
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, write_json
from source_data_context import load_source_context
from source_pretrain import SourcePretrainRuntime, select_case
from target_initialization import METHODS, state_id

SOURCE_PRODUCER = "9c66e20f85d26e88ecb9e6790d8c1e2c62ae8ed1"
SOURCE_JOINT = "7166ab50e740f2374c284aafcd50b55c295865f0ef3466f1a7203b5c85575eecf"
TARGET_JOINT = "bd251819f990c344fbd0e307db7c315c844863b50b16ff31022937adc6e4bd4e9"


def nontraining_body(text):
    module = ast.parse(text)
    module.body = [node for node in module.body
                   if not isinstance(node, ast.FunctionDef) or node.name != "run_training"]
    return ast.dump(module, include_attributes=False)


def verify_source_implementation(producer, current):
    """Explicit historical target-adapter change; audit helpers stay identical."""
    if (producer["dirty"] or producer["commit"] != SOURCE_PRODUCER
            or producer["source_hash"] != digest(producer["files"])):
        raise ValueError("Source producer identity differs")
    changes = {k: [v, current["files"].get(k)] for k, v in producer["files"].items()
               if current["files"].get(k) != v}
    if changes != {"joint_training.py": [SOURCE_JOINT, TARGET_JOINT]}:
        raise ValueError("Unregistered source implementation difference")
    original = subprocess.check_output(["git", "show", SOURCE_PRODUCER + ":joint_training.py"])
    present = Path(__file__).resolve().parents[3] / "joint_training.py"
    if (hashlib.sha256(original.replace(b"\r\n", b"\n")).hexdigest() != SOURCE_JOINT
            or hashlib.sha256(present.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != TARGET_JOINT
            or nontraining_body(original.decode()) != nontraining_body(present.read_text())):
        raise ValueError("Source audit imports/helpers differ from historical implementation")
    if any(hasattr(SourcePretrainRuntime(), name) for name in ("optimizer", "selection_score",
           "validation_candidate", "calibration_metadata_key", "checkpoint_metadata")):
        raise ValueError("Legacy source would use a new target adapter hook")
    return {"registered_file_changes": changes, "all_other_producer_files_identical": True,
            "joint_module_except_run_training_ast_identical": True,
            "historical_change": "69a24ba/f6ac647 target runtime hooks; legacy source uses original defaults",
            "run_training_called_by_this_audit": False}


def compare(reference, current):
    if (reference.shape != current.shape or reference.ndim != 2 or not reference.numel()
            or not torch.isfinite(reference).all() or not torch.isfinite(current).all()):
        raise ValueError("Finite equally shaped probability matrices required")
    delta = (current - reference).abs()
    return {"max_abs": float(delta.max()), "unequal": int(torch.count_nonzero(delta)),
            "over_tolerance": int((~torch.isclose(current, reference, rtol=1e-5, atol=1e-6)).sum()),
            "decision_changes": int((current.argmax(1) != reference.argmax(1)).sum())}


def selected_conclusion(rows):
    targets = {(t, s, m) for t in ("paviaU", "xuzhou", "whu-honghu")
               for s in (64, 512, 2048) for m in METHODS[:2]}
    sources = {(t, s) for t in ("paviaU", "xuzhou", "whu-honghu") for s in (64, 512, 2048)}
    actual_targets = [(r["target"], r["seed"], r["method"]) for r in rows if r["role"] == "target"]
    actual_sources = [(r["target"], r["seed"]) for r in rows if r["role"] == "source"]
    if (len(rows) != 27 or len(actual_targets) != 18 or set(actual_targets) != targets
            or len(actual_sources) != 9 or set(actual_sources) != sources):
        raise ValueError("Nine sources and eighteen distinct target roles required")
    if any(not r["passes_selected_gate"] for r in rows):
        return "historical_none_reuse_rejected"
    return "selected_gate_passed_trajectory_audit_required"


def audit_selected(model, runtime, items, folder, result, out):
    """Reload original selected weights and recalibrate raw weights independently."""
    progress = json.loads((folder / "progress.json").read_text())
    state = torch.load(folder / progress["state_file"], map_location="cpu", weights_only=True)
    saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
    history = json.loads((folder / "history.json").read_text())
    if (state["cursor"] != result["optimizer_steps"] or state["best_step"] != result["selected_step"]
            or state["start_code"] != result["start_code"] or state["identity"] != result["identity"]
            or state["history"] != history or state["trace"] != json.loads((folder / "trace.json").read_text())
            or _tree_fingerprint(saved) != _tree_fingerprint(state["best_probabilities"])
            or _tree_fingerprint(model.state_dict()) != _tree_fingerprint(state["best_model"])
            or any(_tree_fingerprint(state[k]) != state["state_fingerprints"][k]
                   for k in ("model", "optimizer", "scheduler"))):
        raise ValueError("Original selected state/probabilities differ")
    expected = result["selected_validation"]["metrics"]
    metrics = recalls = 0
    for name, item in items.items():
        reference = item
        if "train_labels" not in item:
            reference = {"class_num": item["class_num"]}
            for phase in ("train", "val"):
                gt = item[phase + "_samples_gt"]
                reference[phase + "_labels"] = gt[gt > 0].long() - 1
        oracle = independent_trainval(saved[name], reference, expected[name])
        metrics += oracle["head_metrics"]
        recalls += oracle["class_recalls"]
    raw_path = folder / f'raw-validation-{result["selected_step"]:06d}.pt'
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    primary_state = _tree_fingerprint(model.state_dict())
    records = []
    device = next(model.parameters()).device
    for mode in ("saved_calibration_ordered_forward", "ordered_recalibration_and_forward"):
        first = None
        for repeat in range(3):
            reloaded = load_model_checkpoint(folder / "primary.pt", device=device)
            if _tree_fingerprint(reloaded.state_dict()) != primary_state:
                raise ValueError("Independent checkpoint reload differs")
            with OrderedValidation():
                if mode == "ordered_recalibration_and_forward":
                    reloaded.load_state_dict(raw, strict=True)
                    reloaded.eval()
                    before = state_id(reloaded)
                    path = out / f"recalibration-raw-{repeat}.pt"
                    candidate_fn = getattr(runtime, "validation_candidate", validation_candidate)
                    candidate, _ = candidate_fn(reloaded, items, path)
                    if state_id(reloaded) != before or file_hash(path) != file_hash(raw_path):
                        raise ValueError("Calibration mutated raw state or serialization differs")
                else:
                    candidate = reloaded
                before = state_id(candidate)
                _, values = runtime.validation_snapshot(candidate, items)
                if state_id(candidate) != before:
                    raise ValueError("Validation mutated candidate")
            if first is None:
                first = copy.deepcopy(values)
            comparisons = []
            for name in saved:
                for phase in ("train", "val"):
                    for head in ("fused", "cnn", "graph"):
                        current = values[name][phase][head]
                        comparisons.append({"dataset": name, "phase": phase, "head": head,
                            "vs_saved": compare(saved[name][phase][head], current),
                            "vs_first": compare(first[name][phase][head], current)})
            records.append({"mode": mode, "repeat": repeat, "comparisons": comparisons})
            del reloaded, candidate, values
        del first
    passed = all(c["vs_saved"]["over_tolerance"] == c["vs_saved"]["decision_changes"] == 0
                 and c["vs_first"]["unequal"] == 0
                 for r in records for c in r["comparisons"])
    if _tree_fingerprint(model.state_dict()) != primary_state:
        raise ValueError("Original selected model changed")
    return {"passes_selected_gate": passed, "selected_step": result["selected_step"],
            "head_metrics": metrics, "class_recalls": recalls, "comparisons": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target-root", "target-launch-sha256", "target-acceptance-sha256",
                 "source-root", "source-launch-sha256", "out", "expected-commit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    current = code_snapshot()
    if current["dirty"] or current["commit"] != args.expected_commit:
        raise ValueError("Clean preregistered audit code required")
    target_root = Path(args.target_root).resolve(strict=True)
    source_root = Path(args.source_root).resolve(strict=True)
    out = external_directory(args.out)
    if out.parent != target_root.parent or out in (source_root, target_root):
        raise ValueError("Fresh sibling audit directory required")
    stage = verify_training(target_root, args.target_launch_sha256, args.target_acceptance_sha256, current)
    if file_hash(source_root / "launch.json") != args.source_launch_sha256:
        raise ValueError("Source batch identity differs")
    launch = json.loads((source_root / "launch.json").read_text())
    producer = launch["code"]
    implementation = verify_source_implementation(producer, current)
    pins = {str(p): file_hash(p) for root in (source_root, target_root)
            for p in root.rglob("*") if p.is_file()}
    out.mkdir(exist_ok=False)
    write_json(out / "registration.json", {"audit_code": current,
        "source_implementation_proof": implementation,
        "source_launch_sha256": args.source_launch_sha256,
        "target_launch_sha256": args.target_launch_sha256,
        "target_acceptance_sha256": args.target_acceptance_sha256,
        "rtol": 1e-5, "atol": 1e-6, "classification_changes": 0,
        "new_optimizer_updates": 0, "query_labels_accessed": False,
        "role_count": 27, "selected_gate_is_sufficient_for_reuse": False})
    started = time.monotonic()
    rows = []
    for case in launch["cases"]:
        select_case(launch, producer, case["held_out"], case["seed"])
        context = load_source_context(launch["prepared"], launch["data_root"],
            held_out=case["held_out"], consumer=producer,
            manifest_sha256=load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"])
        if context.record != case["source_context"]:
            raise ValueError("Source context changed")
        pins.update(context.record["protected_files"])
        folder = Path(case["out"])
        if folder.resolve() != source_root / "cases" / case["name"]:
            raise ValueError("Source case escapes pinned root")
        result = json.loads((folder / "result.json").read_text())
        if (result["status"] != "completed" or result["optimizer_steps"] != 1002
                or result["start_code"] != producer or result["settings"] != case["settings"]
                or result["identity"]["prepared"] != context.record
                or result["test_scoring"] is not False):
            raise ValueError("Source result identity differs")
        for name, sha in result["artifacts"].items():
            if Path(name).name != name or file_hash(folder / name) != sha:
                raise ValueError("Source artifact differs")
        history = json.loads((folder / "history.json").read_text())
        trace = json.loads((folder / "trace.json").read_text())
        if (len(history) != 51 or len(trace) != 1002
                or max(history, key=lambda h: h["macro_oa"])["step"] != result["selected_step"]
                or [r["dataset"] for r in trace] != schedule_for(case["settings"], task_names=context.sources)):
            raise ValueError("Source selection/exposure differs")
        gpu = {n: {k: v.cuda() if torch.is_tensor(v) else v for k, v in item.items()}
               for n, item in context.items.items()}
        runtime = SourcePretrainRuntime()
        model = runtime.load_selected(folder / "primary.pt", gpu,
            folder / f'raw-validation-{result["selected_step"]:06d}.pt', device="cuda")
        owned = out / ("source-" + case["name"])
        owned.mkdir()
        row = {"role": "source", "target": case["held_out"], "seed": case["seed"],
               "name": case["name"], **audit_selected(model, runtime, gpu, folder, result, owned)}
        rows.append(row)
        write_json(owned / "result.json", row)
        write_json(out / "progress.json", rows)
        print("NONE_SELECTED_SOURCE", case["name"], row["passes_selected_gate"], flush=True)
        del model, runtime, gpu, context
        torch.cuda.empty_cache()
    for entry in stage["rows"]:
        case = entry["case"]
        if case["method"] not in METHODS[:2]:
            continue
        model, runtime, items = load_case(entry, device="cuda")
        pins.update(case["provenance"]["protected_files"])
        folder = Path(case["out"])
        owned = out / ("target-" + case["name"])
        owned.mkdir()
        row = {"role": "target", "target": case["target"], "seed": case["seed"],
               "method": case["method"], "name": case["name"],
               **audit_selected(model, runtime, items, folder, entry["result"], owned)}
        rows.append(row)
        write_json(owned / "result.json", row)
        write_json(out / "progress.json", rows)
        print("NONE_SELECTED_TARGET", case["name"], row["passes_selected_gate"], flush=True)
        del model, runtime, items
        torch.cuda.empty_cache()
    check_files(pins)
    after = {str(p) for root in (source_root, target_root) for p in root.rglob("*") if p.is_file()}
    if after != {p for p in pins if source_root in Path(p).parents or target_root in Path(p).parents}:
        raise ValueError("Historical artifact file set changed")
    if code_snapshot() != current:
        raise ValueError("Audit code changed")
    conclusion = selected_conclusion(rows)
    write_json(out / "receipt.json", {"status": "audit_completed", "conclusion": conclusion,
        "historical_none_eligible_for_reuse": False, "trajectory_audit_completed": False,
        "cases": rows, "audit_code": current, "protected_files_and_inputs": len(pins),
        "new_optimizer_updates": 0, "query_labels_accessed": False,
        "head_metrics": sum(r["head_metrics"] for r in rows),
        "class_recalls": sum(r["class_recalls"] for r in rows),
        "elapsed_seconds": time.monotonic() - started})
    print("NONE_SELECTED_CONCLUSION", conclusion, flush=True)


if __name__ == "__main__":
    main()
