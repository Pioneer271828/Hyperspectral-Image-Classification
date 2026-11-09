"""Full54 train/validation evidence and independent zero-update verification."""
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch
from evaluation.g4_target import independent_trainval
from g4_matched_target_execution import case_data, unchanged_producer
from g4_matched_target_study import EXPECTED, grid, initialize, runtime, select_case, settings
from joint_training import _tree_fingerprint, schedule_for
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research.reproduction.m4g4b.none_trajectory_compatibility import replay_lr
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from target_initialization import state_id

ACCEPTANCE_POLICY = "g4_matched_full54_training_acceptance_v2"
AUDIT_POLICY = "g4_matched_full54_independent_audit_v2"
REQUIRED = {"identity.json", "progress.json", "primary.pt", "primary.pt.json", "last.pt", "last.pt.json",
            "selected-probabilities.pt", "trace.json", "history.json"}


def read(path):
    return json.loads(Path(path).read_text())


def files(root):
    return {str(p): file_hash(p) for p in Path(root).rglob("*") if p.is_file()}


def check_files(pins):
    if any(file_hash(p) != h for p, h in pins.items()):
        raise ValueError("Frozen full target/source/input artifact changed")


def inspect_case(case, producer, launch_sha256, registration_sha256):
    """Check all raw events/selection/LR/state and artifacts before model inference."""
    folder = Path(case["out"])
    result, progress, history, trace = [read(folder / p) for p in ("result.json", "progress.json", "history.json", "trace.json")]
    expected = settings(case["target"], case["seed"], case["method"], case["mode"], device=case["settings"]["config"]["device"])
    if (result["status"] != "completed" or progress["status"] != "completed" or result["start_code"] != producer
            or result["settings"] != case["settings"] or case["settings"] != expected
            or result["optimizer_steps"] != 1500 or result["test_scoring"] is not False
            or result["schedule_sha256"] != digest(schedule_for(expected, task_names=(case["target"],)))
            or result["per_dataset_updates"] != {case["target"]: 1500}
            or [r["step"] for r in trace] != list(range(1, 1501))
            or any(r["dataset"] != case["target"] for r in trace)
            or [h["step"] for h in history] != list(range(1, 1472, 30)) + [1500]
            or result["selected_step"] != min(history, key=lambda h: h["val_loss"])["step"]
            or replay_lr(expected, history, trace) != [r["learning_rates"] for r in trace]):
        raise ValueError("Full1500/51 events/first-min/LR/producer contract differs")
    provenance = result["identity"]["prepared"]
    binding = {"protocol": EXPECTED["protocol"], "launch_sha256": launch_sha256, "case": case["name"],
               "registration_sha256": registration_sha256, "initial_state_sha256": case["initial_state_sha256"]}
    if (provenance.get("matched_full_target_launch") != binding or provenance.get("m4_mode") != case["mode"]
            or provenance.get("source_spec") != case["source_spec"]
            or any(provenance.get(k) != v for k, v in case["target_context"].items() if k != "protected_files")
            or any(provenance["protected_files"].get(p) != h for p, h in case["target_context"]["protected_files"].items())):
        raise ValueError("Full target label/source/initial/launch provenance differs")
    required = REQUIRED | {f'raw-validation-{h["step"]:06d}.pt' for h in history}
    if not required <= set(result["artifacts"]):
        raise ValueError("Complete51 raw events and canonical target artifacts required")
    pins = {str(folder / "result.json"): file_hash(folder / "result.json")}
    for name, h in result["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe full target artifact path")
        pins[str(folder / name)] = h
    pins.update(provenance["protected_files"])
    check_files(pins)
    if (Path(progress["state_file"]).name != progress["state_file"] or progress["state_file"] not in result["artifacts"]
            or file_hash(folder / progress["state_file"]) != progress["state_sha256"]):
        raise ValueError("Authoritative full target state changed")
    state = torch.load(folder / progress["state_file"], map_location="cpu", weights_only=True)
    saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    if (state["cursor"] != 1500 or state["identity"] != result["identity"] or state["start_code"] != producer
            or state["trace"] != trace or state["history"] != history or state["best_step"] != result["selected_step"]
            or state["best_score"] != -min(h["val_loss"] for h in history)
            or _tree_fingerprint(state["model"]) != _tree_fingerprint(last)
            or _tree_fingerprint(state["best_probabilities"]) != _tree_fingerprint(saved)
            or any(_tree_fingerprint(state[k]) != state["state_fingerprints"][k] for k in ("model", "optimizer", "scheduler"))):
        raise ValueError("Full target authoritative model/optimizer/selection state differs")
    selected = next(h for h in history if h["step"] == result["selected_step"])
    for key in ("val_loss", "macro_oa"):
        np.testing.assert_allclose(result["selected_validation"][key], selected[key], rtol=1e-5, atol=1e-6)
    if result["selected_validation"]["metrics"] != selected["metrics"]:
        raise ValueError("Selected full target event metrics differ")
    return result, history, trace, saved, last, pins


def audit_case(case, producer, launch_sha256, registration_sha256, *, device="cuda"):
    folder, before = Path(case["out"]), files(case["out"])
    result, history, trace, saved, last, pins = inspect_case(case, producer, launch_sha256, registration_sha256)
    items, _, guard = case_data(case)
    target = case["target"]
    checks = independent_trainval(saved[target], items[target], result["selected_validation"]["metrics"][target])
    moved = {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()} for n, item in items.items()}
    adapter = runtime(case, items)
    fingerprints = []
    for _ in range(3):
        model = adapter.load_selected(folder / "primary.pt", moved,
            folder / f'raw-validation-{result["selected_step"]:06d}.pt', device=device)
        before_model = state_id(model)
        _, actual = adapter.validation_snapshot(model, moved)
        for phase in ("train", "val"):
            for head in ("fused", "graph", "cnn"):
                old, new = saved[target][phase][head], actual[target][phase][head]
                torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                if not torch.equal(old.argmax(1), new.argmax(1)):
                    raise ValueError("Full target decisions changed")
        fingerprints.append(_tree_fingerprint(actual))
        if state_id(model) != before_model:
            raise ValueError("Frozen full target mutated during inference")
        del model
    if set(fingerprints) != {_tree_fingerprint(saved)}:
        raise ValueError("Ordered full target repeats are not elementwise exact")
    initial, receipt = initialize(target, case["seed"], case["method"], case["mode"], source_spec=case["source_spec"])
    if receipt != case["initialization"]:
        raise ValueError("Full target frozen initialization changed")
    changed = {}
    prefixes = {"adapter": "backbone.adapter.", "cnn": "backbone.CNN_Branch.", "graph": "backbone.HGNN_Branch."}
    if case["mode"] != "none":
        prefixes["m4"] = "backbone.region_residual."
    for group, prefix in prefixes.items():
        changed[group] = sum(int(torch.count_nonzero(last[n] - p.detach())) for n, p in initial.named_parameters() if n.startswith(prefix))
        if changed[group] <= 0:
            raise ValueError("Missing actual full target parameter changes: " + group)
    if any(not math.isfinite(r["loss"]["total"]) or any(not math.isfinite(v) for v in r["gradient_norms"].values()) for r in trace):
        raise ValueError("Nonfinite full target updates")
    positive = {g: sum(r["gradient_norms"][g] > 0 for r in trace) for g in ("adapter", "graph")}
    if case["mode"] != "none":
        positive["m4"] = sum(r["m4_parameter_gradient_norm"] > 0 for r in trace)
    if any(v <= 0 for v in positive.values()):
        raise ValueError("Full target trainable mechanism never participated")
    guard()
    check_files(pins)
    if before != files(folder):
        raise ValueError("Independent full target audit changed original files")
    return {"name": case["name"], "status": "passed", "updates": 1500, "validation_events": 51,
        "selected_step": result["selected_step"], **checks, "exact_reload_repeats": 3,
        "changed_parameter_elements": changed, "positive_gradient_steps": positive,
        "loss_first30": statistics.mean(r["loss"]["total"] for r in trace[:30]),
        "loss_last30": statistics.mean(r["loss"]["total"] for r in trace[-30:]),
        "selected_validation": result["selected_validation"], "last_validation": history[-1],
        "lr_change_steps": [r["step"] for old, r in zip(trace, trace[1:]) if old["learning_rates"] != r["learning_rates"]],
        "result_sha256": file_hash(folder / "result.json"), "primary_sha256": file_hash(folder / "primary.pt"),
        "sidecar_sha256": file_hash(folder / "primary.pt.json"), "query_labels_accessed": False}


def full_acceptance(root, launch_sha256, acceptance_sha256, consumer):
    root = external_directory(root)
    if file_hash(root / "launch.json") != launch_sha256 or file_hash(root / "training-acceptance.json") != acceptance_sha256:
        raise ValueError("Pinned complete full54 training acceptance required")
    launch, accepted = read(root / "launch.json"), read(root / "training-acceptance.json")
    unchanged_producer(launch["code"], consumer)
    select_case(launch, launch["code"], *grid()[0], phase="full_campaign")
    rows = accepted["models"]
    if (Path(launch["out_dir"]).resolve() != root or read(root / "status.json")["status"] != "completed"
            or accepted["policy"] != ACCEPTANCE_POLICY or accepted["status"] != "passed"
            or accepted["code"] != launch["code"] or accepted["launch_sha256"] != launch_sha256
            or accepted["registration_sha256"] != launch["registration_sha256"]
            or accepted["optimizer_updates"] != 81000 or accepted["query_labels_accessed"] is not False
            or accepted["test_scoring"] is not False or len(rows) != 54
            or len({r["name"] for r in rows}) != 54 or {r["name"] for r in rows} != {c["name"] for c in launch["cases"]}
            or sum(r["head_metrics"] for r in rows) != 324 or sum(r["class_recalls"] for r in rows) != 4320
            or sum(r["exact_reload_repeats"] for r in rows) != 162
            or file_hash(root / "protected-files.json") != accepted["protected_manifest_sha256"]):
        raise ValueError("All54 accepted1500 models and independent coverage required")
    pins = read(root / "protected-files.json")
    pins.update({str(root / "training-acceptance.json"): acceptance_sha256,
                 str(root / "protected-files.json"): accepted["protected_manifest_sha256"],
                 str(root / "status.json"): file_hash(root / "status.json")})
    for case in launch["cases"]:
        row = next(r for r in rows if r["name"] == case["name"])
        if (row["status"] != "passed" or row["updates"] != 1500 or row["validation_events"] != 51
                or row["completed_resume_updates"] != 0 or row["head_metrics"] != 6
                or row["class_recalls"] != 6 * DATASETS[case["target"]][2] or row["exact_reload_repeats"] != 3):
            raise ValueError("Accepted full target case coverage or completed recovery differs")
        for name, key in (("result.json", "result_sha256"), ("primary.pt", "primary_sha256"), ("primary.pt.json", "sidecar_sha256")):
            pins[str(Path(case["out"]) / name)] = row[key]
        _, _, _, _, _, artifacts = inspect_case(case, launch["code"], launch_sha256, launch["registration_sha256"])
        pins.update(artifacts)
    check_files(pins)
    return launch, accepted, pins
