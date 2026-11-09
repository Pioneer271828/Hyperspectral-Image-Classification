"""Read-only paired CPU forward costs for a pinned G4 short-recovery run."""
import copy
import os
from pathlib import Path
import json
import statistics
import time

import torch

from joint_baseline import forward
from joint_training import _tree_fingerprint
from loadData.dataset_registry import file_hash
from research_runtime import write_json
from target_bn_calibration import OBS_KEYS

from .g4_cnn_stage1 import _mapped_geometry


def _p95(samples):
    return statistics.quantiles(samples, n=100, method="inclusive")[94]


def _timed_forward(model, target, observation):
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    outputs = forward(model, target, observation)
    wall_ms = (time.perf_counter_ns() - start) / 1e6
    cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
    del outputs
    return {"wall_ms": wall_ms, "process_cpu_ms": cpu_ms}


def _check_outputs(model, target, observation, item):
    shape = (item["height"] * item["width"], item["class_num"])
    outputs = forward(model, target, observation)
    if len(outputs) < 3 or any(tuple(head.shape) != shape or not torch.isfinite(head).all() for head in outputs[:3]):
        raise ValueError("Expected three finite full-scene probability heads")
    return _tree_fingerprint(tuple(head[item["val_indices"]] for head in outputs[:3]))


def benchmark_recovery_pair(reference, items, recovery, out, *, expected_receipt_sha256, guard,
                            warmups=2, pairs=12):
    """Time native resident-input inference on CPU, with alternating arm order."""
    if type(warmups) is not int or warmups < 1 or type(pairs) is not int or pairs < 4 or pairs % 2:
        raise ValueError("Positive warm-up and an even number of at least four pairs required")
    if next(reference.parameters()).device.type != "cpu" or torch.get_num_threads() != 1:
        raise ValueError("Matched CPU cost requires a single CPU thread")
    recovery = Path(recovery)
    receipt_path, plan_path = recovery / "receipt.json", recovery / "plan.json"
    guard()
    if file_hash(receipt_path) != expected_receipt_sha256:
        raise ValueError("Recovery receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt["status"] != "passed" or receipt["query_labels_accessed"] is not False
            or receipt["same_data_budget_and_selection_rule"] is not True
            or file_hash(plan_path) != receipt["plan_sha256"]):
        raise ValueError("Passing frozen recovery plan required")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    target = plan["target_context"]["target"]
    if tuple(items) != (target,):
        raise ValueError("Exact recovery target required")
    item = items[target]
    input_sha = _tree_fingerprint(item)
    reference_sha = _tree_fingerprint(reference.state_dict())
    if input_sha != plan["input_sha256"] or reference_sha != plan["dense_initial_sha256"]:
        raise ValueError("Recovery data or reference initialization differs")
    compact, _, _ = _mapped_geometry(reference, plan["removed"])
    models = {"dense": copy.deepcopy(reference), "compact": compact}
    pins = {receipt_path: expected_receipt_sha256, plan_path: receipt["plan_sha256"]}
    details = {}
    for role, model in models.items():
        arm = receipt["arms"][role]
        path = recovery / role / "selected-state.pt"
        pins[path] = arm["selected_file_sha256"]
        if file_hash(path) != pins[path] or arm["exact_selected_reload"] is not True:
            raise ValueError("Exact recovery-selected model required")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        params = sum(parameter.numel() for parameter in model.parameters())
        if params != arm["parameters"]:
            raise ValueError("Recovery-selected parameter count differs")
        state = model.state_dict()
        details[role] = {"parameters": params, "state_tensor_bytes": sum(t.numel() * t.element_size() for t in state.values()),
                         "selected_step": arm["selected_step"], "state_sha256": _tree_fingerprint(state)}
    observation = {key: item[key] for key in OBS_KEYS}
    out = Path(out)
    out.mkdir(exist_ok=False)
    order = [list(models) if i % 2 == 0 else list(reversed(models)) for i in range(pairs)]
    timing_plan = {"target": target, "input_sha256": input_sha, "scene_shape": [item["height"], item["width"]],
                   "scope": "resident_cpu_full_scene_native_three_head_forward", "threads": torch.get_num_threads(),
                   "affinity": sorted(os.sched_getaffinity(0)), "torch_version": str(torch.__version__),
                   "warmups_per_arm": warmups, "order": order, "models": details,
                   "diagnostic_geometry_only": plan["diagnostic_geometry_only"],
                   "pins": {str(path): sha for path, sha in pins.items()}}
    write_json(out / "plan.json", timing_plan)
    samples = []
    with torch.inference_mode():
        before = {role: _check_outputs(model, target, observation, item) for role, model in models.items()}
        for _ in range(warmups):
            for model in models.values():
                forward(model, target, observation)
        for roles in order:
            samples.append({role: _timed_forward(models[role], target, observation) for role in roles})
        after = {role: _check_outputs(model, target, observation, item) for role, model in models.items()}
    if (before != after or _tree_fingerprint(item) != input_sha
            or _tree_fingerprint(reference.state_dict()) != reference_sha
            or any(_tree_fingerprint(model.state_dict()) != details[role]["state_sha256"] for role, model in models.items())
            or any(file_hash(path) != sha for path, sha in pins.items())):
        raise RuntimeError("Frozen benchmark inputs, states or outputs changed")
    guard()
    summary = {}
    for role in models:
        wall = [pair[role]["wall_ms"] for pair in samples]
        cpu = [pair[role]["process_cpu_ms"] for pair in samples]
        summary[role] = {"wall_p50_ms": statistics.median(wall), "wall_p95_ms": _p95(wall),
                         "process_cpu_p50_ms": statistics.median(cpu)}
    ratios = [pair["dense"]["wall_ms"] / pair["compact"]["wall_ms"] for pair in samples]
    result = {"status": "passed", "plan_sha256": file_hash(out / "plan.json"), "samples": samples,
              "summary": summary, "median_paired_dense_over_compact": statistics.median(ratios),
              "validation_output_sha256": before, "optimizer_updates": 0, "query_labels_accessed": False}
    write_json(out / "receipt.json", result)
    return result
