"""Paired CPU engineering recovery with the existing G4 target arithmetic."""
import copy
import math
from pathlib import Path

import torch

from joint_training import _tree_fingerprint, schedule_for, validation_epochs
from loadData.dataset_registry import file_hash
from m4_full_target_runtime import TARGET_ARITHMETIC
from m4_ordered_source import OrderedValidation
from model.research_config import ResearchConfig
from research_runtime import write_json
from target_initialization import METHODS
from target_training import TargetRuntime, save_tensor_mapping, target_settings

from .compact_g4_checkpoint import load_compact_initialization
from .g4_cnn_stage1 import _mapped_geometry


def _run_arm(model, runtime, config, item, schedule, events, out):
    out.mkdir()
    model = copy.deepcopy(model)
    optimizer = runtime.optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", threshold_mode="rel", **runtime.settings["scheduler"])
    updates = []
    history = []
    best = None

    def count_update(*_):
        updates.append(len(updates) + 1)

    optimizer.register_step_post_hook(count_update)
    parameter = model.backbone.CNN_Branch[0].point_conv.weight
    initial_parameter = parameter.detach().clone()
    trace = []
    for step, target in enumerate(schedule, 1):
        row = runtime.train_step_at(step - 1, model, optimizer, target, item, config)
        if len(updates) != step:
            raise RuntimeError("Exactly one optimizer update required per schedule entry")
        trace.append({"step": step, **row})
        if step not in events:
            continue
        model.eval()
        with OrderedValidation():
            candidate, calibration = runtime.validation_candidate(
                model, runtime.items, out / f"raw-validation-{step:06d}.pt")
            metric, probabilities = runtime.validation_snapshot(candidate, runtime.items)
        nll = metric["val_loss"]
        if not math.isfinite(nll):
            raise ValueError("Nonfinite validation NLL")
        scheduler.step(nll)
        history.append({"step": step, "calibration": calibration, **metric})
        if best is None or nll < best["val_loss"]:
            best = {"step": step, "val_loss": nll, "probabilities": _tree_fingerprint(probabilities),
                    "state": candidate.state_dict()}
    if int(optimizer.state[parameter]["step"]) != len(schedule) or torch.equal(parameter, initial_parameter):
        raise RuntimeError("CNN recovery weights did not execute the recorded budget")
    selected = out / "selected-state.pt"
    save_tensor_mapping(best["state"], selected)
    model.load_state_dict(torch.load(selected, map_location="cpu", weights_only=True), strict=True)
    with OrderedValidation():
        metric, probabilities = runtime.validation_snapshot(model, runtime.items)
    if metric["val_loss"] != best["val_loss"] or _tree_fingerprint(probabilities) != best["probabilities"]:
        raise RuntimeError("Selected state failed exact CPU validation reload")
    write_json(out / "trace.json", trace)
    write_json(out / "validation.json", history)
    return {"parameters": sum(p.numel() for p in model.parameters()), "updates": len(updates),
            "cnn_adam_updates": int(optimizer.state[parameter]["step"]),
            "optimizer_step_ids": updates, "validation_steps": [row["step"] for row in history],
            "validation_fused_nll": [row["val_loss"] for row in history], "selected_step": best["step"],
            "selected_fused_nll": best["val_loss"], "selected_file_sha256": file_hash(selected),
            "selected_probabilities_sha256": best["probabilities"], "exact_selected_reload": True,
            "learning_rates": [row["learning_rates"] for row in trace]}


def run_recovery_pair(reference, items, provenance, out, *, removed, expected_reference_state, guard,
                      compact_initialization=None):
    """Exercise a pinned compact artifact or an explicitly diagnostic channel removal."""
    target = provenance["target"]
    if provenance["query_labels_accessed"] is not False or next(reference.parameters()).device.type != "cpu":
        raise ValueError("Sanitized CPU recovery inputs required")
    settings = target_settings(target, METHODS[1], support_seed=provenance["support_seed"],
                               per_class=provenance["support_per_class"])
    runtime = TargetRuntime(settings, items, out, source_spec={"already_initialized": True})
    item = items[target]
    config = ResearchConfig.from_dict(settings["config"])
    schedule = schedule_for(settings, task_names=(target,))
    events = [epoch + 1 for epoch in validation_epochs(settings)]
    if len(schedule) != settings["optimizer_steps"] or set(reference.heads) != {target}:
        raise ValueError("Target and recovery schedule must match the reference")
    reference_state = _tree_fingerprint(reference.state_dict())
    if reference_state != expected_reference_state:
        raise ValueError("Pinned dense initialization differs")
    guard()
    item_state = _tree_fingerprint(item)
    compact, retained, _ = _mapped_geometry(reference, removed)
    compact_state = _tree_fingerprint(compact.state_dict())
    if compact_initialization is not None:
        path, file_sha, mapping_sha = compact_initialization
        compact, mapping = load_compact_initialization(
            path, expected_sha256=file_sha, expected_mapping_sha256=mapping_sha)
        if mapping["reference_state"] != reference_state or _tree_fingerprint(compact.state_dict()) != compact_state:
            raise ValueError("Compact file does not match the dense source and declared channel mapping")
        compact.training_config = copy.deepcopy(reference.training_config)
    out = Path(out)
    out.mkdir(exist_ok=False)
    common = {"settings": settings, "validation_arithmetic": TARGET_ARITHMETIC,
              "schedule": schedule, "validation_steps": events, "target_context": provenance,
              "input_sha256": item_state, "dense_initial_sha256": reference_state,
              "compact_initial_sha256": compact_state, "retained": retained, "removed": list(removed),
              "compact_file_sha256": None if compact_initialization is None else compact_initialization[1],
              "diagnostic_geometry_only": compact_initialization is None}
    write_json(out / "plan.json", common)
    arms = {}
    for role, model in (("dense", reference), ("compact", compact)):
        guard()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings["model_seed"])
            arms[role] = _run_arm(model, runtime, config, item, schedule, events, out / role)
        if _tree_fingerprint(item) != item_state or _tree_fingerprint(reference.state_dict()) != reference_state:
            raise RuntimeError("Protected inputs or reference state changed")
        guard()
    if (arms["dense"]["learning_rates"] != arms["compact"]["learning_rates"]
            or any(arm["selected_step"] != events[min(range(len(events)), key=arm["validation_fused_nll"].__getitem__)]
                   for arm in arms.values())):
        raise RuntimeError("Recovery learning rates or first-min selection diverged from the common rules")
    receipt = {"status": "passed", "plan_sha256": file_hash(out / "plan.json"), "arms": arms,
               "input_sha256": item_state, "same_data_budget_and_selection_rule": True,
               "query_labels_accessed": False, "optimizer_state_transferred": False}
    write_json(out / "receipt.json", receipt)
    return receipt
