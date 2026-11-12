"""Independent endpoint and optimizer audit for the registered WP07 batch."""

from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import io
import math
from statistics import fmean
from time import perf_counter

import numpy as np
import torch

from evaluation.classification_metrics import metrics_from_confusion
from loadData.spatial_protocol import forward_partition
from loadData.spatial_tiles import observe_tile
from research_runtime import require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.g3_training import state_hash
from transfer.multiscale_run import MultiscaleRun
from transfer.xuzhou_multiscale import ARMS, BUDGET, SEED, STAGES, load_registration, read_stage


HEADS = ("fused", "graph", "cnn")
SCIENCE = ("state", "selected_state", "optimizer", "rng", "torch_rng", "history", "progress", "selected_step")


def _exact(left, right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and left.shape == right.shape and tensor_hash(left) == tensor_hash(right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _packet(status, contract, stop, *, packet_format=MultiscaleRun.FORMAT):
    packet = torch.load(io.BytesIO(dev._read(status["checkpoint"])), map_location="cpu", weights_only=True)
    if (packet["format"] != packet_format or packet["contract_hash"] != contract
            or status["contract"] != contract or len(packet["history"]) != stop):
        raise ValueError("Checkpoint identity differs from the registered interval")
    return packet


@torch.no_grad()
def _endpoint(model, state, scene, record):
    model.load_state_dict(state, strict=True)
    model.eval()
    model.backbone.adapter.cache_attention = False
    matrices = np.zeros((3, scene.classes, scene.classes), dtype=np.int64)
    nll = [0.] * 3
    identities = []
    width = scene.plan["shape"][1]
    release = record["validation_support"]
    published = dict(zip(release["indices"], release["labels"], strict=True))
    seen = set()
    for row in record["roles"]:
        if row["role"] != "devval":
            continue
        tile = scene.plan["tiles"][row["id"]]
        observation = observe_tile(scene.cube, scene.plan, scene.stats, tile)
        y0, y1, x0, x1 = row["box"]
        allowed = sorted(i for i in published if y0 <= i // width < y1 and x0 <= i % width < x1)
        if not allowed or seen.intersection(allowed):
            raise ValueError("Independent V5 coverage changed")
        seen.update(allowed)
        pos = [(i // width - y0) * (x1 - x0) + i % width - x0 for i in allowed]
        labels = [published[i] - 1 for i in allowed]
        binding = model.partition_bands(scene.bands) if hasattr(model, "partition_bands") else nullcontext()
        with binding:
            outputs = forward_partition(model, scene.dataset, observation,
                                        np.asarray(scene.bands.wavelengths, dtype=np.float32), 100)
        identities.append([tensor_hash(x) for x in outputs[:3]])
        for index, probabilities in enumerate(outputs[:3]):
            probs = probabilities.cpu().numpy()[pos]
            if not np.isfinite(probs).all():
                raise ValueError("Nonfinite independently replayed scores")
            for label, prediction in zip(labels, probs.argmax(1), strict=True):
                matrices[index, label, prediction] += 1
            nll[index] -= float(np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-8)).sum())
    if seen != published.keys() or len(seen) != 45 or not (matrices.sum(axis=2) == 5).all():
        raise ValueError("Independent endpoint omitted released V5 labels")
    result = {}
    for name, matrix, loss in zip(HEADS, matrices, nll, strict=True):
        metrics = metrics_from_confusion(matrix, record["classes"])
        metrics.update(nll=loss / len(seen), zero_recall=[row["class_id"] for row in metrics["per_class"]
                                                        if row["recall"] == 0.])
        result[name] = metrics
    return result, identities


def _distribution(values):
    return dict(min=min(values), max=max(values), mean=fmean(values))


def _summarize_training(history, plan, selected):
    losses = [row["loss"] for row in history]
    validation = [row["validation"]["nll"] for row in history]
    coverage = {}
    for step in (1, selected, plan.updates):
        exposure = plan.exposure(step)
        coverage[str(step)] = dict(visited_support=sum(n > 0 for n in exposure["support_visits"].values()),
                                   minimum_visits=min(exposure["support_visits"].values()),
                                   visited_classes=sum(n > 0 for n in exposure["class_blocks"].values()),
                                   visited_blocks=sum(n > 0 for n in exposure["tile_counts"].values()))
    return dict(loss=dict(first=losses[0], last=losses[-1], first16_mean=fmean(losses[:16]),
                          last16_mean=fmean(losses[-16:])),
                validation_nll=dict(first=validation[0], last=validation[-1], selected=validation[selected - 1],
                                    first16_mean=fmean(validation[:16]), last16_mean=fmean(validation[-16:])),
                coverage=coverage,
                gradient_l1={key: _distribution([r["gradient_l1"][key] for r in history])
                             for key in history[0]["gradient_l1"]},
                changed_elements={key: _distribution([r["changed_parameters"][key] for r in history])
                                  for key in history[0]["changed_parameters"]})


def _summarize(history, plan, selected):
    selectors = {}
    for name in history[0]["selectors"]:
        values = [row["selectors"][name] for row in history]
        gradients = [v["gradient_l1"] for v in values if v["gradient_l1"] is not None]
        selectors[name] = dict(gradient_steps=len(gradients), positive_gradient_steps=sum(v > 0 for v in gradients),
                               max_gradient_l1=max(gradients, default=0.),
                               update_steps=sum(v["changed"] > 0 for v in values),
                               changed_elements_over_updates=sum(v["changed"] for v in values))
    scales = {}
    for key in ("cnn_weights", "graph_weights", "graph_nnz"):
        scales[key] = [_distribution([r["scales"][key][i] for r in history])
                       for i in range(len(history[0]["scales"][key]))]
    for key in ("token_count", "assignment_nnz"):
        scales[key] = _distribution([r["scales"][key] for r in history])
    scales["cnn_dilations"] = history[0]["scales"]["cnn_dilations"]
    scales["graph_radii"] = history[0]["scales"]["graph_radii"]
    return dict(**_summarize_training(history, plan, selected), selectors=selectors, scales=scales)


def _verify_history(packet, plan):
    budget = plan.updates
    hist = packet["history"]
    if len(hist) != budget:
        raise ValueError("Incomplete production update history")
    seen = Counter()
    for number, row in enumerate(hist, 1):
        event = plan.event(number)
        if any(row[k] != v for k, v in event.items()) or not math.isfinite(row["loss"]):
            raise ValueError("Independent event sequence differs")
        seen.update(row["gradient_parameters"])
    if packet["progress"] != plan.progress(budget):
        raise ValueError("Independent terminal exposure differs")
    order = packet["spec"]["parameters"]
    optimizer = packet["optimizer"]
    slots = dict(zip(optimizer["param_groups"][0]["params"], order, strict=True))
    actual = {slots[i]: int(s["step"].item()) for i, s in optimizer["state"].items()}
    if actual != seen:
        raise ValueError("Adam update accounting differs from gradient participation")
    expected = min(range(budget), key=lambda i: hist[i]["validation"]["nll"]) + 1
    if packet["selected_step"] != expected:
        raise ValueError("Independent first-min selected step differs")
    return expected


def audit(registration, complete, code):
    started = perf_counter()
    root, value, baseline, scene, record, plan = load_registration(registration, code)
    evidence = admission.read(complete)
    start = admission.read(evidence["start"])
    if (complete["path"] != str(root / "batch-complete.json")
            or evidence["start"]["path"] != str(root / "batch-start.json") or start["registration"] != registration
            or evidence["registration"] != registration or set(evidence["attempts"]) != set(ARMS)
            or evidence["updates"] != value["maximum_updates"] or evidence["raw_gt"] or evidence["query"]):
        raise ValueError("Batch completion receipt does not authenticate all registered arms")
    results = {}
    baseline_packet = None
    for arm, trial in ARMS.items():
        arm_started = perf_counter()
        packets, receipts, elapsed = {}, {}, {}
        for stage, (_, stop, previous) in STAGES.items():
            status, receipt = read_stage(root, registration, arm, stage, receipts.get(previous))
            if receipt != evidence["attempts"][arm][stage]:
                raise ValueError("Independent stage receipt differs")
            receipts[stage] = receipt
            elapsed[stage] = {key: val for key, val in status.items() if key.endswith("_seconds") or key == "timing"}
            packets[stage] = _packet(status, value["contracts"][arm], stop)
        full, prefix, resumed, complete_packet = (packets[name] for name in STAGES)
        for restored in (resumed, complete_packet):
            if any(not _exact(full[key], restored[key]) for key in SCIENCE):
                raise ValueError("Continuous, resumed and completed scientific state differ")
        if not _exact(prefix["history"], full["history"][:7]):
            raise ValueError("Independent prefix mismatch")
        if arm == "off":
            baseline_packet = full
        elif arm == "cnn_fixed1" and any(not _exact(baseline_packet[key], full[key]) for key in SCIENCE):
            raise ValueError("Fixed dilation one changed baseline training")
        chosen = _verify_history(full, plan)
        model = deepcopy(baseline) if trial is None else trial.initialize(baseline, SEED).eval()
        endpoint = {}
        for name, field, step in (("selected", "selected_state", chosen), ("last", "state", BUDGET)):
            metrics, scores = _endpoint(model, full[field], scene, record)
            log = full["history"][step - 1]
            if (state_hash(model) != log["state_hash"] or metrics["fused"]["nll"] != log["validation"]["nll"]
                    or metrics["fused"]["confusion_matrix"] != log["validation"]["confusion"]):
                raise ValueError("Independent endpoint differs from training validation")
            endpoint[name] = dict(step=step, scores=scores, metrics=metrics)
        shared_count = sum(p.numel() for p in baseline.parameters())
        extra_count = sum(p.numel() for p in model.parameters()) - shared_count
        results[arm] = dict(endpoints=endpoint, summary=_summarize(full["history"], plan, chosen),
                            shared_parameters=shared_count, selector_parameters=extra_count,
                            per_stage=elapsed, wall_seconds=perf_counter() - arm_started,
                            replay_exact_fields=list(SCIENCE), updates=2 * BUDGET, completed_reload_updates=0)
        print(f"audited {arm} selected={chosen}", flush=True)
    for arm, result in results.items():
        protection = {}
        for name, endpoint in result["endpoints"].items():
            protection[name] = {}
            for head, metrics in endpoint["metrics"].items():
                baseline_metrics = results["off"]["endpoints"][name]["metrics"][head]
                deltas = {str(a["class_id"]): a["recall"] - b["recall"]
                          for a, b in zip(metrics["per_class"], baseline_metrics["per_class"], strict=True)}
                protection[name][head] = dict(deltas=deltas, passes=all(delta >= value["recall_floor_delta"]
                                                                        for delta in deltas.values()))
        result["recall_vs_off"] = protection
    admission.check_pins([*value["protected"].values(), value["source"], value["sidecar"], registration, complete])
    require_frozen_code(code)
    return dev._json_new(root / "audit.json", dict(status="passed", registration=registration, complete=complete,
        results=results, independent_forwards=len(ARMS) * 2 * 5, independent_updates=0, raw_gt=0, query=0,
        fixed1_equal_to_off=True, wall_seconds=perf_counter() - started))
