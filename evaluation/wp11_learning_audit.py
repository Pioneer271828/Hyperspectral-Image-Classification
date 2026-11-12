"""Independently reconcile matched stage packets and publish both frozen models."""

from copy import deepcopy
import io
from pathlib import Path

import torch

from evaluation.wp11_confirmation import HISTORY_POLICY, validate_full_history
from inference.trained_light import export_trained_light, load_trained_light
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from research_runtime import write_json
from transfer.g3_training import state_hash
from transfer.registered_development import _load_source, _pin, _read
from transfer.registered_wp11_learning import (ARMS, RELEASE, completed_segment, load_registration, read, verify, verify_notice)
from transfer.wp11_matched_learning import (count_graphs, frozen_full, full_predictions, light_predictions,
                                            require_exposure, restore, validation_values)


def exact(left, right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def packet(receipt):
    return torch.load(io.BytesIO(_read(receipt["checkpoint"])), map_location="cpu", weights_only=True)


def trajectory(root, reg, reg_pin, note, source, scene, record, plan, arm):
    receipts = {stage: completed_segment(root, reg, reg_pin, note, arm, stage) for stage in reg["stages"]}
    states = {stage: packet(receipt) for stage, receipt in receipts.items()}
    baseline = states["continuous"]
    for stage in ("resume", "completed"):
        if not exact(states[stage], baseline):
            raise ValueError(f"{arm} {stage} differs from the continuous run")
    prefix = states["prefix"]["history"]
    if prefix != baseline["history"][:len(prefix)]:
        raise ValueError("Frozen prefix history differs from the continuous run")
    expected_graphs = 0 if arm == "light" else 1 + sum(
        any(row["role"] == "devval" and row["box"][0] <= index // scene.plan["shape"][1] < row["box"][1]
            and row["box"][2] <= index % scene.plan["shape"][1] < row["box"][3]
            for index in record["validation_support"]["indices"])
        for row in record["roles"])
    for stage, receipt in receipts.items():
        if (receipt["graph_calls"] != expected_graphs * receipt["updates"]
                or receipt["prefix"] != plan.progress(receipt["stop"])
                or receipt["history_hash"] != digest(states[stage]["history"])
                or receipt["selected_step"] != states[stage]["selected_step"]
                or receipt["source_state"] != state_hash(source)):
            raise ValueError("Executed graph, exposure or training evidence differs")
    light_rows = {stage: read(receipt["light_evidence"]) for stage, receipt in receipts.items()}
    if arm == "light":
        if (light_rows["continuous"] != light_rows["prefix"] + light_rows["resume"] or light_rows["completed"]
                or len(light_rows["continuous"]) != plan.updates):
            raise ValueError("Light gradient/update evidence differs after recovery")
        evidence = []
        for row, actual in zip(baseline["history"], light_rows["continuous"], strict=True):
            parameters = actual["parameters"]
            if (actual["step"] != row["step"] or any(p["gradient_l1"] is None for p in parameters.values())
                    or sum(p["gradient_l1"] for p in parameters.values()) != row["head_gradient_l1"]):
                raise ValueError("Per-head gradient evidence differs")
            evidence.append(parameters)
    else:
        if any(light_rows.values()):
            raise ValueError("Unexpected light-head evidence in full-model trajectory")
        evidence = [row["parameter_evidence"] for row in baseline["history"]]
    parameter_summary = {
        name: dict(gradient_steps=sum(row[name]["gradient_l1"] is not None for row in evidence),
                   nonzero_gradient_steps=sum((row[name]["gradient_l1"] or 0.) > 0. for row in evidence),
                   changed_steps=sum(row[name]["changed"] > 0 for row in evidence),
                   changed_elements=sum(row[name]["changed"] for row in evidence))
        for name in evidence[0]}
    last = receipts["continuous"]["checkpoint"]
    run = restore(arm, source, scene, record, plan, last, reg["contracts"][arm])
    require_exposure(run, plan)
    if run.completed != plan.updates:
        raise ValueError("Only complete matched trajectories can be frozen")
    return run, last, parameter_summary, receipts


def loss_summary(history):
    values = [row["loss"] for row in history]
    k = min(32, len(values))
    return dict(first=values[0], last=values[-1], first_mean=sum(values[:k])/k,
                last_mean=sum(values[-k:])/k, window=k)


def audit(reg_pin, notice, code, out):
    root, reg, source, scene, record, plan, pins = load_registration(reg_pin, code)
    verify_notice(notice, reg_pin, reg)
    if out.exists():
        raise ValueError("New external freeze directory required")
    verify([notice, *pins], code, out)
    trajectories = {arm: trajectory(root, reg, reg_pin, notice, source, scene, record, plan, arm) for arm in ARMS}
    light, light_checkpoint, light_parameters, light_stages = trajectories["light"]
    full, full_checkpoint, full_parameters, full_stages = trajectories["full"]
    selected_model = frozen_full(full, True)
    last_model = frozen_full(full, False)
    with count_graphs(source) as graphs:
        light_states = {"initial": source.cnn_heads[scene.dataset].state_dict(), "selected": light.selected_head,
                        "last": light.head.state_dict()}
        light_metrics = {name: light_predictions(light, state) for name, state in light_states.items()}
    if graphs["count"] or state_hash(source) != reg["source_state"]:
        raise ValueError("Light endpoint evaluation changed the source or constructed a graph")
    full_metrics = {name: full_predictions(model, scene, record) for name, model in
                    (("initial", source), ("selected", selected_model), ("last", last_model))}
    for run, results, head in ((light, light_metrics, "light"), (full, full_metrics, "fused")):
        for name, index in (("selected", run.selected_step - 1), ("last", -1)):
            if validation_values(results[name][head]) != run.history[index]["validation"]:
                raise ValueError("Independent endpoint differs from the original V5 validation")
    request = read(reg["request"])
    source_spec = request["inputs"]
    old_history = read(source_spec["full_history"])
    original_training = source_spec["development"]
    full_usage = deepcopy(old_history["tile_usage"])
    full_usage["training"] = sorted(set(full_usage["training"]) | {row["tile_id"] for row in full.history})
    label_use = dict(scene=scene.dataset, development=original_training, source_history=source_spec["full_history"],
                     source_history_evidence=source_spec["history_evidence"], source_full_usage=old_history["tile_usage"],
                     fit_tiles=old_history["tile_usage"]["fit"],
                     training_indices=record["train_support"]["indices"],
                     training_support_hash=digest(record["train_support"]),
                     selection_indices=record["validation_support"]["indices"],
                     selection_support_hash=digest(record["validation_support"]),
                     optimizer_tiles=sorted({row["tile_id"] for row in full.history}),
                     selection_tiles=full_usage["selection"], read_calibration=False, read_confirmation=False,
                     original_gt_read=False, query=0)
    out.mkdir()
    light_path = out / "light.pt"
    light_digest = export_trained_light(source, scene, record, light_checkpoint["path"], light_path,
                                       expected_training_sha256=light_checkpoint["sha256"],
                                       expected_contract_hash=light.contract_hash)
    light_spec = load_trained_light(source, light_path, expected_sha256=light_digest,
                                   expected_training_sha256=light_checkpoint["sha256"],
                                   expected_contract_hash=light.contract_hash).specification
    full_path = out / "full.pt"
    selected_model.training_config.update(wp11_matched=dict(source_state=reg["source_state"],
                      registration=reg_pin["sha256"], training_sha256=full_checkpoint["sha256"],
                      completed=plan.updates, selected_step=full.selected_step,
                      selection=full.spec["selection"], label_use_hash=digest(label_use)))
    save_model_checkpoint(selected_model, full_path)
    full_pin, full_meta = _pin(full_path), _pin(Path(str(full_path) + ".json"))
    reloaded = _load_source(full_pin, full_meta)
    if (state_hash(reloaded) != state_hash(selected_model)
            or full_predictions(reloaded, scene, record) != full_metrics["selected"]):
        raise ValueError("Frozen full reload differs")
    evidence = dict(status="audited", policy=RELEASE, commit=code["commit"], registration=reg_pin,
                    notice=notice, source_state=reg["source_state"], plan=plan.plan_hash,
                    full_state=state_hash(reloaded), label_use=label_use, selected_steps=dict(light=light.selected_step,
                    full=full.selected_step), endpoints=dict(light=light_metrics, full=full_metrics),
                    exact_continuous_resume_completed=True, actual_updates=reg["total_updates"],
                    audit_optimizer_updates=0, original_gt_read=False, query=0,
                    stages=dict(light=light_stages, full=full_stages), loss=dict(light=loss_summary(light.history),
                    full=loss_summary(full.history)), parameters=dict(light=light_parameters, full=full_parameters))
    evidence_path = out / "audit.json"
    write_json(evidence_path, evidence)
    history = dict(policy=HISTORY_POLICY, scope="all_same_scene_ancestry", tile_usage=full_usage,
                   full_state=state_hash(reloaded), scene_identity=old_history["scene_identity"],
                   history_evidence_sha256=_pin(evidence_path)["sha256"])
    validate_full_history(reloaded, scene, record, history)
    write_json(out / "full-history.json", history)
    release = dict(policy=RELEASE, commit=code["commit"], registration=reg_pin,
                   development=original_training, source_state=reg["source_state"], full_state=state_hash(reloaded),
                   source_weights=source_spec["full_weights"], source_metadata=source_spec["full_metadata"],
                   source_history=source_spec["full_history"], source_evidence=source_spec["history_evidence"],
                   full_weights=full_pin, full_metadata=full_meta, full_history=_pin(out / "full-history.json"),
                   history_evidence=_pin(evidence_path), light=_pin(light_path),
                   light_training_sha256=light_checkpoint["sha256"], light_contract_hash=light.contract_hash,
                   light_spec_sha256=digest(light_spec))
    verify([notice, *pins], code)
    write_json(out / "release.json", release)
    return release
