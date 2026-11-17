"""Freeze mixed-producer endpoints, then independently replay the twelve cells."""

from collections import Counter
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compression.precision_recovery_audit import HEADS, _guard
from compression.registered_precision_recovery import stages
from evaluation.classification_metrics import metrics_from_confusion
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_factorial_real import BUDGET, CASES, FORMAT, NEW_UPDATES, ORDER, PREFIX, REFERENCED_UPDATES, RealFactorialRun, inputs, new_run
from transfer.curriculum_observation_admission import CONDITIONS, audited_forward
from transfer.curriculum_observation_audit import _pack
from transfer.curriculum_factorial_controls import unchanged
from transfer.curriculum_real import verify_notice
from transfer.curriculum_real_audit import audit_exposure
from transfer.curriculum_support_review import _validation_members
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import read_stage
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact, _packet, _verify_history


POINTS = ("selected", "last")
METRICS = ("oa", "aa", "kappa", "macro_f1", "nll")
BASELINES = ("none", "s_only", "o_only")
SNAPSHOT = ("metrics", "predictions", "targets", "scores", "forwards")


def _verify_forward(history, policy, plan, scene, record, cache):
    frozen = policy.to_dict()["training"]
    for step, (row, r) in enumerate(zip(history, frozen, strict=True), 1):
        tile = scene.plan["tiles"][plan.event(step)["tile_id"]]
        key = tile["id"], r["record_hash"]
        if key not in cache:
            actual = policy.view(scene, record, tile, phase="training", step=step)
            cache[key] = tensor_hash(actual.observation.spectral.copy())
        evidence = row["forward_evidence"]
        if (row["observation"] != policy.training_record(step) or evidence["spectral_sha256"] != cache[key]
                or evidence["band_ids"] != r["selected_band_ids"] or evidence["metadata_sha256"] != r["selected_metadata_hash"]
                or evidence["band_count"] != len(r["selected_positions"]) or evidence["calls"] != dict(adapter=1, denoise=1, graph=1)
                or len(evidence["three_heads"]) != 3 or not evidence["graph_info"]):
            raise ValueError("New factorial actual input, band record or learned graph differs")
    for group in ("adapter", "cnn", "graph", "heads"):
        if not any(row["gradient_l1"][group] > 0 and row["changed_parameters"][group] > 0 for row in history):
            raise ValueError(f"No positive gradient and actual parameter change in {group}")


def _joint_case(root, reg, pin, complete, name, plan, trial, source, scene, record, cache, out):
    packets, status_rows, receipts = {}, {}, {}
    intervals = stages(BUDGET, PREFIX)
    for stage, (_, end, prior) in intervals.items():
        row, receipt = read_stage(root, pin, name, stage, receipts.get(prior), arms=reg["contracts"], stages=intervals)
        if receipt != complete["attempts"][name][stage]:
            raise ValueError("New factorial status differs from the frozen completion")
        status_rows[stage], receipts[stage] = row, receipt
        packets[stage] = _packet(row, reg["contracts"][name], end, packet_format=RealFactorialRun.FORMAT)
    full = packets["full"]
    if (not _exact(full["history"][:PREFIX], packets["prefix"]["history"])
            or not all(_exact(full[k], packets[stage][k]) for stage in ("resume", "completed") for k in SCIENCE)):
        raise ValueError("New factorial authenticated restore differs from continuous recovery")
    history = full["history"]
    chosen = _verify_history(full, plan)
    teacher = admission.read(reg["teacher"])
    applied = audit_exposure(history, plan, teacher, ORDER, trial[1].to_dict())
    _verify_forward(history, trial[2], plan, scene, record, cache)
    points = {}
    folder = out/name
    folder.mkdir()
    for point, step, key in (("selected", chosen, "selected_state"), ("last", BUDGET, "state")):
        package = _pack(source, full[key], folder/f"{point}.pt")
        if package["state"] != history[step-1]["state_hash"]:
            raise ValueError("New endpoint differs from its executed checkpoint")
        points[point] = dict(step=step, package=package)
    return dict(producer=reg["commit"], new=True, history=history, history_sha256=digest(history),
                endpoints=points, applied=applied, attempts=receipts, stage_costs={stage:{
                    k: row[k] for k in ("new_updates", "timing", "restore_or_prepare_seconds", "update_loop_seconds",
                                         "checkpoint_finish_seconds")} for stage, row in status_rows.items()})


@torch.no_grad()
def replay(model, scene, record, policy, condition):
    started = perf_counter()
    members = _validation_members(scene, record)
    width = scene.plan["shape"][1]
    groups = {}
    for index, label, tile in members:
        groups.setdefault(tile, []).append((index, label))
    scores, targets, blocks = [], [], []
    predictions = {h: [] for h in HEADS}
    probabilities = {h: [] for h in HEADS}
    matrices = {h: np.zeros((scene.classes, scene.classes), dtype=np.int64) for h in HEADS}
    nll = Counter()
    observation = policy.to_dict()["validation"][condition]
    before = state_hash(model)
    for tid, rows in groups.items():
        tile = scene.plan["tiles"][tid]
        y0, _, x0, x1 = tile["box"]
        local = [(i//width-y0)*(x1-x0)+i%width-x0 for i, _ in rows]
        truth = np.array([label-1 for _, label in rows])
        targets.extend((truth+1).tolist())
        blocks.append(dict(tile=tid, indices=[i for i, _ in rows]))
        view = policy.view(scene, record, tile, phase="validation", condition=condition)
        result, evidence = audited_forward(model, scene, record, tile, view, observation, phase="validation")
        scores.append(evidence["three_heads"])
        for head, tensor in zip(HEADS, result[:3], strict=True):
            p = tensor.detach().cpu().numpy()[local]
            hard = p.argmax(1)
            np.add.at(matrices[head], (truth, hard), 1)
            predictions[head].extend((hard+1).tolist())
            probabilities[head].extend(p.tolist())
            nll[head] -= float(np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-8)).sum())
    quality = {}
    for head in HEADS:
        score = metrics_from_confusion(matrices[head], record["classes"])
        score.update(nll=nll[head]/len(targets), zero_recall=[v["class_id"] for v in score["per_class"] if v["recall"] == 0.])
        quality[head] = score
    if state_hash(model) != before:
        raise ValueError("Read-only factorial condition changed a frozen state")
    return dict(condition_record=observation["record_hash"], forwards=len(groups), labels=len(targets),
                metrics=quality, predictions=predictions, targets=targets, scores=scores,
                blocks=blocks, probabilities=probabilities, seconds=perf_counter()-started)


def _freeze_ready(witness, registration, completion, cases):
    expected = {c["case"] for c in CASES}
    if (witness["registration"] != registration or witness["completion"] != completion or set(cases) != expected
            or set(witness["cases"]) != expected or witness["optimizer_updates"] != 0
            or any(witness["cases"][k] != cases[k]["endpoints"] or set(witness["cases"][k]) != set(POINTS) for k in expected)):
        raise ValueError("All twelve selected and last models must be frozen before replay")


def _cases_replay(cases, witness_pin, reg_pin, end_pin, reg, source, scene, record, policy):
    _freeze_ready(admission.read(witness_pin), reg_pin, end_pin, cases)
    snapshots = {c: replay(source, scene, record, policy, c) for c in CONDITIONS}
    for name, entry in cases.items():
        for point, end in entry["endpoints"].items():
            package = end["package"]
            model = dev._load_source(package["weights"], package["sidecar"])
            if state_hash(model) != package["state"]:
                raise ValueError("Frozen factorial package state differs")
            end["conditions"] = {c: replay(model, scene, record, policy, c) for c in CONDITIONS}
            actual = end["conditions"]["full"]
            event = entry["history"][end["step"]-1]["validation"]
            fused = actual["metrics"]["fused"]
            if (fused["confusion_matrix"] != event["confusion"] or fused["oa"] != event["oa"]
                    or not math.isclose(fused["nll"], event["nll"], abs_tol=1e-7)):
                raise ValueError("Frozen factorial replay differs from the original selection event")
            if not entry["new"] and digest({k:actual[k] for k in SNAPSHOT}) != end["full_sha256"]:
                raise ValueError("Referenced control full replay changed after the common freeze")
            if any(v["condition_record"] != reg["validation_identity"][c] for c, v in end["conditions"].items()):
                raise ValueError("Frozen factorial validation condition differs")
    return snapshots


def audit(pin, completion, campaign, out, code):
    started = perf_counter()
    require_frozen_code(code)
    reg = admission.read(pin)
    root = external_directory(reg["root"])
    if (reg["format"] != FORMAT or pin["path"] != str(root/"registration.json")
            or reg["maximum_updates"] != NEW_UPDATES or reg["historical_control_updates"] != REFERENCED_UPDATES
            or reg["comparison_updates"] != NEW_UPDATES + REFERENCED_UPDATES
            or (reg["budget"], reg["prefix"]) != (BUDGET, PREFIX)
            or any(reg[k] for k in ("optimizer_updates", "gt", "query"))):
        raise ValueError("Exact real execution registration required for read-only audit")
    unchanged(reg["commit"], code, ("transfer/curriculum_factorial_real.py", "transfer/curriculum_factorial_controls.py",
                                    "transfer/curriculum_factorial.py", "scripts/run_real_curriculum_factorial.py"))
    _, s, o, source, scene, record, plans, trials = inputs(reg["request"], code)
    if (reg["all_cases"] != list(CASES) or reg["teacher"] != s["teacher"] or reg["validation"] != s["validation"]
            or reg["support"] != s["support"] or reg["source"] != s["source"]
            or reg["validation_identity"] != o["validation_identity"] or reg["protection"] != s["protection"]
            or reg["contracts"] != {name: new_run(source, scene, record, plans[name.split("-")[0][1:]], trial).contract_hash
                                     for name, trial in trials.items()}):
        raise ValueError("Frozen factorial contract differs in the independent audit")
    end, run = (admission.read(p) for p in (completion, campaign))
    if (completion["path"] != str(root/"batch-complete.json") or end["registration"] != pin or end["updates"] != NEW_UPDATES
            or run["status"] != "passed" or run["producer"] != reg["commit"]
            or run["registration"] != pin["sha256"] or run["completion"] != completion["sha256"]
            or run["notice"] != end["notice"]["sha256"] or run["new_updates"] != NEW_UPDATES
            or any(run[k] for k in ("gt", "query")) or any(end[k] for k in ("optimizer_updates", "gt", "query"))):
        raise ValueError("Exact completed factorial campaign required")
    verify_notice(root, reg, pin, end["notice"])
    controls = admission.read(reg["controls"])
    if (controls["commit"] != reg["commit"] or controls["source_hash"] != reg["source_hash"]
            or controls["request"] != reg["request"] or controls["status"] != "passed"):
        raise ValueError("Original exact control authentication required for audit")
    out = external_directory(out)
    if out.exists() or out == root or out in root.parents or root in out.parents:
        raise ValueError("Fresh audit directory separate from all optimizer attempts required")
    out.mkdir()
    history_cache = {}
    audits = {}
    cases = {}
    for c in CASES:
        name, seed = c["case"], c["seed"]
        if name in trials:
            row = _joint_case(root, reg, pin, end, name, plans[str(seed)], trials[name], source, scene, record, history_cache, out)
        else:
            ref = controls["cases"][name]
            if ref["audit"]["path"] not in audits:
                audits[ref["audit"]["path"]] = admission.read(ref["audit"])
            history = audits[ref["audit"]["path"]]["cases"][ref["original_case"]]["history"]
            if digest(history) != ref["history_sha256"]:
                raise ValueError("Referenced historical history digest changed")
            row = dict(producer=ref["producer"], new=False, original_case=ref["original_case"],
                       history=history, history_sha256=ref["history_sha256"], endpoints=ref["endpoints"])
        cases[name] = dict(case=c, **row)
    frozen = dev._json_new(out/"frozen-endpoints.json", dict(registration=pin, completion=completion,
        cases={name:v["endpoints"] for name,v in cases.items()}, optimizer_updates=0))
    _, _, policy, _ = next(iter(trials.values()))
    source_views = _cases_replay(cases, frozen, pin, completion, reg, source, scene, record, policy)
    sample = admission.read(admission.read(reg["request"])["sample"]["registration"])
    if any(source_views["full"][k] != sample["source_replay"][k] for k in ("metrics", "scores", "forwards")):
        raise ValueError("Original factorial source full validation changed")
    small = sorted(c for c,n in Counter(v["label"] for v in reg["support"]).items() if n <= reg["protection"]["small_support_max"])
    guards = {}
    for case in reg["cases"]:
        seed, name = case["seed"], case["case"]
        candidate = cases[name]["endpoints"]
        guards[name] = {point:{c:{b:_guard(cases[f"s{seed}-{b}"]["endpoints"][point]["conditions"][c],
                                           candidate[point]["conditions"][c], record["classes"], small)
                                   for b in BASELINES} for c in CONDITIONS} for point in POINTS}
    snapshots = [v for row in cases.values() for end in row["endpoints"].values() for v in end["conditions"].values()]
    endpoint_pins = [v["package"][k] for row in cases.values() for v in row["endpoints"].values() for k in ("weights", "sidecar")]
    admission.check_pins([pin, completion, campaign, frozen, *endpoint_pins, *reg["protected"]])
    require_frozen_code(code)
    return dev._json_new(out/"audit.json", dict(status="passed_real_mixed_factorial", adapter=code["commit"],
        producer=reg["commit"], registration=pin, completion=completion, campaign=campaign, frozen=frozen,
        cases=cases, source=source_views, guards=guards, small_classes=small, optimizer_updates=0, gt=0, query=0,
        new_updates=NEW_UPDATES, source_forwards=sum(v["forwards"] for v in source_views.values()),
        source_label_accesses=sum(v["labels"] for v in source_views.values()),
        endpoint_forwards=sum(v["forwards"] for v in snapshots), endpoint_label_accesses=sum(v["labels"] for v in snapshots),
        replay_seconds=math.fsum(v["seconds"] for v in [*source_views.values(), *snapshots]),
        independent_training_input_replays=len(history_cache), seconds=perf_counter()-started))
