"""Independent train/validation evidence for an ordered none source case; no updates."""
import json
from pathlib import Path

import torch
from evaluation.g4_target import independent_trainval
from g4_none_source_launch import BoundNoneSourceRuntime, launch_binding
from joint_training import _tree_fingerprint, schedule_for, validation_epochs
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig, digest
from target_initialization import state_id


def audit_case(launch, case, context, launch_sha256, *, device="cuda"):
    folder = Path(case["out"])
    result = json.loads((folder / "result.json").read_text())
    settings = case["settings"]
    budget = settings["optimizer_steps"]
    binding = launch_binding(launch, case, launch_sha256)
    provenance = {**context.record, "none_source_launch": binding}
    schedule = schedule_for(settings, task_names=context.sources)
    if (result["status"] != "completed" or result["optimizer_steps"] != budget
            or result["start_code"] != launch["code"] or result["settings"] != settings
            or result["identity"]["prepared"] != provenance or result["identity"]["settings_hash"] != digest(settings)
            or result["identity"]["consumer_source_hash"] != launch["code"]["source_hash"]
            or result["identity"]["execution_device"] != launch["execution_device"]
            or result["schedule_sha256"] != digest(schedule) or result["test_scoring"] is not False):
        raise ValueError("Source execution identity differs")
    for name, sha in result["artifacts"].items():
        if Path(name).name != name or file_hash(folder / name) != sha:
            raise ValueError("Source artifact changed")
    progress = json.loads((folder / "progress.json").read_text())
    if Path(progress["state_file"]).name != progress["state_file"] or file_hash(folder / progress["state_file"]) != progress["state_sha256"]:
        raise ValueError("Unsafe or changed authoritative state")
    state = torch.load(folder / progress["state_file"], map_location="cpu", weights_only=True)
    saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
    trace, history = state["trace"], state["history"]
    if (progress["status"] != "completed" or progress["cursor"] != budget or state["cursor"] != budget
            or state["identity"] != result["identity"] or state["start_code"] != launch["code"]
            or len(trace) != budget or [r["step"] for r in trace] != list(range(1, budget + 1))
            or [r["dataset"] for r in trace] != schedule
            or [r["step"] for r in history] != [(e + 1) * settings["steps_per_epoch"] for e in validation_epochs(settings)]
            or trace != json.loads((folder / "trace.json").read_text())
            or history != json.loads((folder / "history.json").read_text())
            or state["best_step"] != max(history, key=lambda h: h["macro_oa"])["step"]
            or state["best_step"] != result["selected_step"]
            or _tree_fingerprint(saved) != _tree_fingerprint(state["best_probabilities"])):
        raise ValueError("Source state/schedule/first-max selection differs")
    for name in ("model", "optimizer", "scheduler"):
        if _tree_fingerprint(state[name]) != state["state_fingerprints"][name]:
            raise ValueError("Authoritative source state fingerprint differs")
    if _tree_fingerprint(torch.load(folder / "last.pt", map_location="cpu", weights_only=True)) != _tree_fingerprint(state["model"]):
        raise ValueError("Last source model differs")
    selected = next(h for h in history if h["step"] == state["best_step"])
    metrics = recalls = 0
    for name, item in context.items.items():
        reference = {"class_num": item["class_num"]}
        for phase in ("train", "val"):
            gt = item[phase + "_samples_gt"]
            reference[phase + "_labels"] = gt[gt > 0].long() - 1
        checked = independent_trainval(saved[name], reference, selected["metrics"][name])
        metrics += checked["head_metrics"]
        recalls += checked["class_recalls"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(case["seed"])
        initial = ResearchConfig.from_dict(settings["config"]).build_model(context.class_counts)
    if state_id(initial) != case["initial_state_sha256"]:
        raise ValueError("Frozen initial source model differs")
    changed = {}
    for group, prefix in {"adapter": "backbone.adapter.", "cnn": "backbone.CNN_Branch.",
                          "graph": "backbone.HGNN_Branch."}.items():
        changed[group] = sum(int(torch.count_nonzero(state["model"][n] - p.detach()))
                             for n, p in initial.named_parameters() if n.startswith(prefix))
        if changed[group] <= 0:
            raise ValueError("Source module has no actual parameter update")
    del initial
    # Source draws are with replacement: a branch need only receive gradients on
    # steps that actually selected its source, while shared branches update every step.
    if any(r["gradient_norms"]["graph"] <= 0 or r["graph_changed_elements"] <= 0 for r in trace):
        raise ValueError("Source graph branch did not participate in all updates")
    moved = {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
             for n, item in context.items.items()}
    runtime = BoundNoneSourceRuntime(binding)
    for _ in range(3):
        model = runtime.load_selected(folder / "primary.pt", moved,
            folder / f'raw-validation-{state["best_step"]:06d}.pt', device=device)
        if _tree_fingerprint(model.state_dict()) != _tree_fingerprint(state["best_model"]):
            raise ValueError("Source selected state differs")
        before = state_id(model)
        values = runtime.validation_snapshot(model, moved)[1]
        for name in saved:
            for phase in ("train", "val"):
                for head in saved[name][phase]:
                    old, new = saved[name][phase][head], values[name][phase][head]
                    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                    if not torch.equal(new.argmax(1), old.argmax(1)):
                        raise ValueError("Source decisions changed")
        if _tree_fingerprint(values) != _tree_fingerprint(saved) or state_id(model) != before:
            raise ValueError("Exact source reload/read-only inference differs")
        del model
    return {"name": case["name"], "updates": budget, "selected_step": state["best_step"],
            "source_validation_macro_oa": selected["macro_oa"], "metrics": metrics, "recalls": recalls,
            "exact_reload_repeats": 3, "positive_graph_gradient_steps": budget, "changed_parameters": changed,
            "loss_first30": sum(r["loss"]["total"] for r in trace[:30]) / len(trace[:30]),
            "loss_last30": sum(r["loss"]["total"] for r in trace[-30:]) / len(trace[-30:]),
            "last_validation_gain": history[-1]["macro_oa"] - history[-min(6, len(history))]["macro_oa"],
            "train_fused_oa": {n: selected["metrics"][n]["train"]["fused"]["oa"] for n in context.sources},
            "val_heads": {n: selected["metrics"][n]["val"] for n in context.sources},
            "result_sha256": file_hash(folder / "result.json")}
