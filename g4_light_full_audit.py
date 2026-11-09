"""Independent train/validation acceptance of a completed light adaptation case."""
import math
from pathlib import Path

import torch

from evaluation.g4_target import independent_trainval
from g4_light_full_target import POLICY, UPDATES, prepare_case
from g4_light_target import HEAD_ONLY
from g4_matched_target_audit import REQUIRED, files, read
from joint_training import _tree_fingerprint, schedule_for, validation_epochs
from model.research_config import digest
from research_runtime import tensor_hash
from target_initialization import state_id


def audit_schedule(settings, history, trace):
    event_steps = [epoch + 1 for epoch in validation_epochs(settings)]
    if ([row["step"] for row in history] != event_steps
            or [row["step"] for row in trace] != list(range(1, UPDATES + 1))
            or any(row["dataset"] != settings["target"] or not math.isfinite(row["loss"]["total"])
                   or not row["frozen_parameters_unchanged"] or not row["frozen_buffers_unchanged"]
                   or any(not math.isfinite(v) for v in row["gradient_norms"].values()) for row in trace)
            or any(not math.isfinite(row["val_loss"]) or row["selection_score"] != -row["val_loss"] for row in history)):
        raise ValueError("Light adaptation update, validation, or freeze trace differs")
    config = settings["config"]["training"]
    rates = [config["head_lr"]] if settings["method"] == HEAD_ONLY else [config["lr"], config["head_lr"]]
    optimizer = torch.optim.SGD([{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": rate} for rate in rates])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", threshold_mode="rel", **settings["scheduler"])
    events = {row["step"]: row for row in history}
    for row in trace:
        if [group["lr"] for group in optimizer.param_groups] != row["learning_rates"]:
            raise ValueError("Learning-rate trace differs from validation replay")
        if row["step"] in events:
            scheduler.step(events[row["step"]]["val_loss"])
    return min(history, key=lambda row: row["val_loss"]), scheduler.state_dict()


def audit_case(registration_path, registration_sha256, code, key, *, device="cuda:0"):
    items, provenance, adapter, finish = prepare_case(registration_path, registration_sha256, code, key, device=device)
    target, _, method, _ = key
    folder, settings = adapter.directory, adapter.settings
    before = files(folder)
    result, progress, history, trace = [read(folder / name) for name in
                                       ("result.json", "progress.json", "history.json", "trace.json")]
    schedule_sha = digest(schedule_for(settings, task_names=(target,)))
    identity = {"protocol": POLICY, "settings_hash": digest(settings), "settings": settings,
                "execution_device": str(device), "consumer_source_hash": code["source_hash"],
                "prepared": provenance, "schedule_sha256": schedule_sha}
    if (result["status"] != "completed" or progress["status"] != "completed" or progress["cursor"] != UPDATES
            or result["start_code"] != code or result["identity"] != identity or read(folder / "identity.json") != identity
            or result["settings"] != settings or result["schedule_sha256"] != schedule_sha
            or result["optimizer_steps"] != UPDATES or result["per_dataset_updates"] != {target: UPDATES}
            or result["test_scoring"] is not False or result["execution"]["device"] != str(device)
            or (str(device).startswith("cuda") and result["execution"]["visible_cuda_devices"] != str(provenance["physical_gpu"]))):
        raise ValueError("Completed light adaptation identity or budget differs")
    selected, scheduler = audit_schedule(settings, history, trace)
    if result["selected_step"] != selected["step"] or result["selected_validation"] != {
            k: v for k, v in selected.items() if k not in ("epoch", "step", "selection_score")}:
        raise ValueError("Selected light checkpoint differs from the first minimum validation NLL")
    required = REQUIRED | {f'raw-validation-{h["step"]:06d}.pt' for h in history}
    if method == HEAD_ONLY:
        required |= {"feature-bank.json", "feature-bank.pt"}
    artifacts = result["artifacts"]
    if (not required <= artifacts.keys() or progress["state_file"] != f"state-{UPDATES:06d}.pt"
            or artifacts.get(progress["state_file"]) != progress["state_sha256"]
            or any(Path(name).name != name or before.get(str(folder / name)) != sha for name, sha in artifacts.items())):
        raise ValueError("Light target artifacts or authoritative state hashes differ")
    state, saved, last = [torch.load(folder / name, map_location="cpu", weights_only=True) for name in
                          (progress["state_file"], "selected-probabilities.pt", "last.pt")]
    if (state["identity"] != identity or state["start_code"] != code or state["cursor"] != UPDATES
            or state["trace"] != trace or state["history"] != history or state["best_step"] != selected["step"]
            or state["best_score"] != -selected["val_loss"] or state["scheduler"] != scheduler
            or _tree_fingerprint(state["model"]) != _tree_fingerprint(last)
            or _tree_fingerprint(state["best_probabilities"]) != _tree_fingerprint(saved)
            or any(_tree_fingerprint(state[k]) != state["state_fingerprints"][k] for k in ("model", "optimizer", "scheduler"))):
        raise ValueError("Light target authoritative checkpoint state differs")
    checks = independent_trainval(saved[target], items[target], result["selected_validation"]["metrics"][target])
    moved = {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()} for n, item in items.items()}
    raw = folder / f'raw-validation-{selected["step"]:06d}.pt'
    primary = folder / "primary.pt"
    for repeat in range(3):
        model = adapter.load_selected(primary, moved, raw, device=device)
        old = state_id(model)
        _, probabilities = adapter.validation_snapshot(model, moved)
        if _tree_fingerprint(probabilities) != _tree_fingerprint(saved) or state_id(model) != old:
            raise ValueError("Independent light target reload is not exact and read-only")
        if repeat < 2:
            del model
    frozen_names = {name for name, p in model.named_parameters() if not p.requires_grad}
    if (digest({name: tensor_hash(last[name]) for name in frozen_names}) != adapter.parameter_contract["frozen_parameter_sha256"]
            or digest({name: tensor_hash(last[name]) for name, _ in model.named_buffers()}) != adapter.optimizer_buffers
            or _tree_fingerprint(model.state_dict()) != _tree_fingerprint(state["best_model"])):
        raise ValueError("Frozen parameters, optimizer-phase buffers, or selected weights differ")
    changes = sum(row["adapter_changed_elements"] > 0 for row in trace)
    if (changes == 0) != (method == HEAD_ONLY) or not any(row["changed_trainable_elements"] > 0 for row in trace):
        raise ValueError("Declared light trainable mechanism did not participate")
    finish()
    if before != files(folder):
        raise ValueError("Independent audit mutated the training tree")
    return {"status": "passed", "name": folder.name, "role": list(key), "physical_gpu": provenance["physical_gpu"],
            "updates": UPDATES, "selected_step": selected["step"], "validation_events": len(history),
            **checks, "exact_reload_repeats": 3, "adapter_changed_steps": changes,
            "result_sha256": before[str(folder / "result.json")], "primary_sha256": before[str(primary)],
            "sidecar_sha256": before[str(primary) + ".json"], "protected_inputs": provenance["protected_files"],
            "artifacts": before, "new_optimizer_updates": 0, "query_labels_accessed": False}
