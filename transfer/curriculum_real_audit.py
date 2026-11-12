"""Independently reconcile CUR-S states, applied weights and released endpoints."""

from collections import Counter, defaultdict
from copy import deepcopy
import math
from pathlib import Path
from struct import pack, unpack
from time import perf_counter

import numpy as np
import torch

from compression.precision_recovery_audit import _guard, independent_predictions, save_package
from compression.registered_precision_recovery import stages
from loadData.spatial_protocol import forward_partition
from loadData.spatial_tiles import observe_tile
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_real import RegisteredSampleRun, load_registration
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import read_stage
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact, _packet, _verify_history


def fp32(value):
    return unpack("f", pack("f", value))[0]


def class_ranks(teacher, arm, seed):
    support = teacher["support"]
    rows = list(zip(support["indices"], support["labels"], teacher["losses"], strict=True))
    ranks = {}
    for cls in teacher["classes"]:
        members = [(idx, loss) for idx, label, loss in rows if label == cls]
        if arm in ("easy_to_hard", "hard_to_easy"):
            sign = 1 if arm == "easy_to_hard" else -1
            members.sort(key=lambda p: (sign * p[1], p[0]))
        if arm == "random":
            order = np.random.default_rng(int(digest({"seed": seed, "class": cls})[:16], 16)).permutation(len(members))
            members = [members[j] for j in order]
        ranks.update({i: j for j, (i, _) in enumerate(members)})
    return ranks


def audit_exposure(history, plan, teacher, arm, pacing):
    support = teacher["support"]
    labels = dict(zip(support["indices"], support["labels"], strict=True))
    counts = Counter(labels.values())
    ranks = class_ranks(teacher, arm, pacing["seed"])
    visits = Counter()
    weights = defaultdict(list)
    classes, anchors, tiles = Counter(), Counter(), Counter()
    minimum, deviation = 1., 0.
    for step, row in enumerate(history, 1):
        ids = plan.event(step)["support_ids"]
        ratio = min(1., pacing["start_fraction"] + (1 - pacing["start_fraction"]) * (step - 1) / (pacing["full_at"] - 1))
        by_class = defaultdict(list)
        for i in ids:
            by_class[labels[i]].append(i)
        expected = {}
        for cls, members in by_class.items():
            threshold = math.ceil(counts[cls] * ratio)
            raw = [1. if arm in ("off", "balanced") or ranks[i] < threshold else pacing["minimum_weight"] for i in members]
            scale = len(members) / math.fsum(raw)
            expected.update({i: value * scale for i, value in zip(members, raw, strict=True)})
        classes.update(by_class.keys())
        anchors[row["anchor"]] += 1
        tiles[row["tile_id"]] += 1
        visits.update(ids)
        actual_by_class = defaultdict(list)
        for i, declared, applied in zip(ids, row["weights"], row["applied_weights"], strict=True):
            if not math.isclose(declared, expected[i], rel_tol=1e-12, abs_tol=1e-12) or applied != fp32(expected[i]):
                raise ValueError("Independently recomputed effective loss weight differs")
            if not applied > 0:
                raise ValueError("Nonpositive applied weight")
            weights[i].append(applied)
            actual_by_class[labels[i]].append(applied)
            minimum = min(minimum, applied)
        for cls, values in actual_by_class.items():
            error = abs(math.fsum(values) - len(by_class[cls]))
            if error > len(values) * 2e-7:
                raise ValueError("Class loss mass no longer matches the unweighted support count")
            deviation = max(deviation, error)
    native = plan.exposure()
    if ({str(i): visits[i] for i in labels} != native["support_visits"]
            or {str(c): classes[c] for c in counts} != native["class_blocks"]
            or {str(c): anchors[c] for c in counts} != native["anchor_counts"]
            or {str(t): tiles[t] for t in plan.to_dict()["sampler"]["tile_ids"]} != native["tile_counts"]):
        raise ValueError("Raw class-first event accounting differs")
    by_support = {str(i): dict(raw=visits[i], effective=math.fsum(weights[i])) for i in labels}
    by_class = {str(c): dict(raw=sum(visits[i] for i in labels if labels[i] == c),
               effective=math.fsum(by_support[str(i)]["effective"] for i in labels if labels[i] == c),
               participating_steps=classes[c]) for c in counts}
    if any(by_class[str(c)]["raw"] != native["class_label_visits"][str(c)] for c in counts):
        raise ValueError("Raw class label visits differ")
    return dict(by_support=by_support, by_class=by_class, total_label_visits=sum(visits.values()),
                minimum_applied_weight=minimum, maximum_class_mass_roundoff=deviation)


@torch.no_grad()
def train_support_predictions(model, scene, record, ownership):
    """Replay independently by released global coordinate, with no other role."""
    before = state_hash(model)
    width = scene.plan["shape"][1]
    groups = defaultdict(list)
    for member in ownership:
        groups[member["tile"]].append(member)
    output = {}
    for tile_id, members in sorted(groups.items()):
        tile = scene.plan["tiles"][tile_id]
        y0, y1, x0, x1 = tile["box"]
        if record["roles"][tile_id]["role"] != "devtrain":
            raise ValueError("Train support may only use its authenticated devtrain core")
        block = observe_tile(scene.cube, scene.plan, scene.stats, tile)
        fused = forward_partition(model, scene.dataset, block, np.asarray(scene.bands.wavelengths, dtype=np.float32),
                                  50 if scene.dataset == "paviaU" else 100)[0]
        for item in members:
            index, label = item["index"], item["label"]
            y, x = divmod(index, width)
            if not (y0 <= y < y1 and x0 <= x < x1) or index in output:
                raise ValueError("Malformed train support ownership")
            row = fused[(y-y0)*(x1-x0)+x-x0]
            p = float(row[label-1])
            if not torch.isfinite(row).all():
                raise ValueError("Nonfinite train support endpoint")
            output[index] = dict(tile=tile_id, label=label, prediction=int(row.argmax())+1, probability=p,
                                 nll=-math.log(max(1e-8, min(1., p))))
    support = record["train_support"]
    if {i: v["label"] for i, v in output.items()} != dict(zip(support["indices"], support["labels"], strict=True)):
        raise ValueError("Training support IDs or labels changed")
    if before != state_hash(model):
        raise ValueError("Frozen training probe changed model state")
    return output, len(groups)


def public(result, step, state, package=None):
    return dict(step=step, metrics=result["metrics"], scores=result["scores"], predictions=result["predictions"],
                targets=result["targets"], forwards=result["forwards"], state=state, package=package)


def _endpoint(model, source, scene, record, state, step, history, path):
    model.load_state_dict(state, strict=True)
    identity = state_hash(model)
    value = independent_predictions(model, scene, record)
    package = None
    if step:
        selected = history[step-1]
        measured = value["metrics"]["fused"]
        recorded = selected["validation"]
        if (identity != selected["state_hash"] or measured["confusion_matrix"] != recorded["confusion"]
                or measured["oa"] != recorded["oa"] or not math.isclose(measured["nll"], recorded["nll"], abs_tol=1e-7)):
            raise ValueError("Replay differs from the original full V5 selection")
        package = save_package(model, "fp32", path)
        package["state"] = identity
        restored = dev._load_source(package["weights"], package["sidecar"])
        if state_hash(restored) != identity or source.export_config() != restored.export_config():
            raise ValueError("Curriculum endpoint package did not preserve the frozen state")
    return public(value, step, identity, package)


def audit(pin, completion, code):
    started = perf_counter()
    root, reg, source, scene, record, plans, trials = load_registration(pin, code)
    end = admission.read(completion)
    if (completion["path"] != str(root / "batch-complete.json") or end["registration"] != pin
            or end["updates"] != reg["maximum_updates"] or any(end[k] for k in ("optimizer_updates", "query", "gt"))):
        raise ValueError("Invalid completed sample curriculum")
    teacher = admission.read(reg["teacher"])
    independent, teacher_forwards = train_support_predictions(source, scene, record, reg["support"])
    for index, expected in zip(teacher["support"]["indices"], teacher["losses"], strict=True):
        if not math.isclose(expected, independent[index]["nll"], rel_tol=2e-7, abs_tol=1e-7):
            raise ValueError("Independent teacher NLL replay differs")
    initial = public(independent_predictions(source, scene, record), 0, state_hash(source))
    if any(initial[k] != reg["source_replay"][k] for k in ("metrics", "scores", "forwards")):
        raise ValueError("Frozen initial source predictions changed")
    out = root / "endpoints"
    out.mkdir()
    intervals = stages(reg["budget"], reg["prefix"])
    reports, controls = {}, {}
    updates, val_forwards = 0, initial["forwards"]
    validation_checks = 0
    for name, (case, policy, _) in trials.items():
        packets, statuses, pins = {}, {}, {}
        for stage, (_, stop, preceding) in intervals.items():
            status, receipt = read_stage(root, pin, name, stage, pins.get(preceding), arms=trials, stages=intervals)
            if end["attempts"][name][stage] != receipt:
                raise ValueError("Independent stage does not match completion")
            packets[stage] = _packet(status, reg["contracts"][name], stop, packet_format=RegisteredSampleRun.FORMAT)
            pins[stage], statuses[stage] = receipt, status
        full = packets["full"]
        seed, arm = case["seed"], case["arm"]
        if (any(not _exact(full[k], packets[other][k]) for other in ("resume", "completed") for k in SCIENCE)
                or not _exact(full["history"][:reg["prefix"]], packets["prefix"]["history"])):
            raise ValueError("Continuous and authenticated curriculum resume differ")
        if arm == "off":
            controls[seed] = {k: digest(full[k]) if k in ("history", "progress", "selected_step") else full[k] for k in SCIENCE}
        elif arm == "balanced":
            other = controls.pop(seed)
            if any(not _exact(other[k], digest(full[k]) if k in ("history", "progress", "selected_step") else full[k]) for k in SCIENCE):
                raise ValueError("The required off/balanced degeneracy differs")
        plan = plans[str(seed)]
        selected = _verify_history(full, plan)
        history = full["history"]
        weights = audit_exposure(history, plan, teacher, arm, reg["pacing"])
        folder = out / name
        folder.mkdir()
        model = deepcopy(source).eval()
        points = {point: _endpoint(model, source, scene, record, full[key], step, history, folder / f"{point}.pt")
                  for point, key, step in (("selected", "selected_state", selected), ("last", "state", reg["budget"]))}
        val_forwards += sum(v["forwards"] for v in points.values())
        count = sum(s["new_updates"] for s in statuses.values())
        updates += count
        validations = sum(s["timing"]["full_v5"]["calls"] for s in statuses.values())
        validation_checks += validations
        reports[name] = dict(case=case, steps=reg["budget"], updates=count, initial=initial, **points, exposure=weights,
                             policy_hash=None if policy is None else policy.policy_hash, history=history,
                             validation_checks=validations, same_recovery_fields=list(SCIENCE),
                             stages={s: {k: row[k] for k in ("new_updates", "timing", "restore_or_prepare_seconds",
                                      "update_loop_seconds", "checkpoint_finish_seconds")} for s, row in statuses.items()})
        print(f"{name}: exact recovery, effective loss mass and three-head endpoints", flush=True)
    if controls or updates != reg["maximum_updates"]:
        raise ValueError("Incomplete seed controls or optimizer budget")
    small = [cls for cls, count in Counter(r["label"] for r in reg["support"]).items() if count <= reg["protection"]["small_support_max"]]
    guards = {}
    for name, row in reports.items():
        baseline = reports[f"s{row['case']['seed']}-off"]
        guards[name] = {point: _guard(baseline[point], row[point], record["classes"], small)
                        for point in ("initial", "selected", "last")}
    admission.check_pins([pin, completion, reg["teacher"], *reg["protected"].values()])
    require_frozen_code(code)
    return dev._json_new(root / "audit.json", dict(status="passed_independent_curriculum", registration=pin,
        completion=completion, source=initial, teacher=reg["teacher"], teacher_independent_forwards=teacher_forwards,
        teacher_independent_labels=len(independent), independent_v5_forwards=val_forwards, stages_v5_checks=validation_checks,
        actual_updates=updates, cases=reports, guards=guards, small_classes=sorted(small), off_balanced_exact=True,
        train_labels_in_update_trajectories=2 * sum(r["exposure"]["total_label_visits"] for r in reports.values()),
        devval_labels_in_stage_checks=validation_checks * len(reg["validation"]["indices"]),
        original_gt_reads=0, query=0, optimizer_updates=0, seconds=perf_counter()-started))
