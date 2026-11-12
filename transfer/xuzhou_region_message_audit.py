"""Independently compare recovered states and replay the seven V5 endpoints."""

from copy import deepcopy
from time import perf_counter

from research_runtime import require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.g3_training import state_hash
from transfer.region_message_attention import TRACE_FIELDS
from transfer.region_message_run import ATTENTION, DIRECTIONS, RegionMessageRun, _group, initialize
from transfer.xuzhou_multiscale import read_stage
from transfer.xuzhou_multiscale_audit import (
    SCIENCE, _distribution, _endpoint, _exact, _packet, _summarize_training, _verify_history,
)
from transfer.xuzhou_region_message import ARMS, SEED, STAGES, load_registration


def _training_summary(history, plan, chosen):
    summary = _summarize_training(history, plan, chosen)
    parameters = {}
    for name in history[0]["message_parameters"]:
        values = [row["message_parameters"][name] for row in history]
        grads = [r["gradient_l1"] for r in values if r["gradient_l1"] is not None]
        parameters[name] = dict(group=_group(name), gradient_steps=len(grads),
                                nonzero_gradient_steps=sum(x > 0 for x in grads),
                                zero_gradient_steps=sum(x == 0 for x in grads),
                                missing_gradient_steps=len(history) - len(grads),
                                max_gradient_l1=max(grads, default=0.),
                                changed_steps=sum(r["changed"] > 0 for r in values),
                                changed_elements=sum(r["changed"] for r in values))
    trace = {}
    for branch in DIRECTIONS:
        if branch not in history[0]["message"]["branches"]:
            continue
        rows = [r["message"]["branches"][branch] for r in history]
        trace[branch] = {key: _distribution([row[key] for row in rows]) for key in TRACE_FIELDS}
        trace[branch]["k_clipped_steps"] = sum(0 < row["k"] < ATTENTION["top_k"] for row in rows)
    gates = {key: _distribution([r["message"]["gates"][key] for r in history])
             for key in history[0]["message"]["gates"]}
    return dict(**summary, trainable_message_parameters=parameters, message_access=trace, outer_gates=gates,
                graph={key: _distribution([r["message"][key] for r in history])
                       for key in ("regions", "assignment_nnz", "graph_nnz")})


def _pair(left, right, threshold):
    result = {}
    for endpoint, row in left["endpoints"].items():
        heads = {}
        for head, metrics in row["metrics"].items():
            ref = right["endpoints"][endpoint]["metrics"][head]
            deltas = {str(a["class_id"]): a["recall"] - b["recall"]
                      for a, b in zip(metrics["per_class"], ref["per_class"], strict=True)}
            heads[head] = dict(delta={k: None if metrics[k] is None or ref[k] is None else metrics[k] - ref[k]
                                      for k in ("oa", "aa", "kappa", "macro_f1", "nll")},
                               recall_delta=deltas, passes=all(value >= threshold for value in deltas.values()))
        result[endpoint] = heads
    return result


def audit(registration, complete, code):
    started = perf_counter()
    root, value, baseline, scene, record, plan = load_registration(registration, code)
    evidence = admission.read(complete)
    start = admission.read(evidence["start"])
    if (complete["path"] != str(root / "batch-complete.json")
            or evidence["start"]["path"] != str(root / "batch-start.json") or start["registration"] != registration
            or evidence["registration"] != registration or set(evidence["attempts"]) != set(ARMS)
            or evidence["updates"] != value["maximum_updates"] or evidence["raw_gt"] or evidence["query"]):
        raise ValueError("Complete receipt does not authenticate the registered message batch")
    results = {}
    val_blocks = sum(r["role"] == "devval" for r in record["roles"])
    proof_count = 0
    for arm in ARMS:
        arm_started = perf_counter()
        packets, receipts, elapsed = {}, {}, {}
        for stage, (begin, stop, previous) in STAGES.items():
            status, receipt = read_stage(root, registration, arm, stage, receipts.get(previous), arms=ARMS, stages=STAGES)
            if receipt != evidence["attempts"][arm][stage]:
                raise ValueError("Independent stage receipt differs")
            receipts[stage] = receipt
            time = status["timing"]
            updates = stop - begin
            validation = updates + (2 if previous is not None else 0)
            if (time["train_forward"]["calls"] != updates or time["train_backward"]["calls"] != updates
                    or time["validation"]["calls"] != validation
                    or time.get("train_graph_build", {"calls": 0})["calls"] != updates
                    or time["validation_graph_build"]["calls"] != validation * val_blocks):
                raise ValueError("Graph proof or execution call totals differ from registered updates")
            proof_count += updates + validation * val_blocks
            elapsed[stage] = {k: v for k, v in status.items() if k.endswith("_seconds") or k == "timing"}
            packets[stage] = _packet(status, value["contracts"][arm], stop, packet_format=RegionMessageRun.FORMAT)
        full, prefix, resumed, completed = (packets[name] for name in STAGES)
        for restored in (resumed, completed):
            if any(not _exact(full[key], restored[key]) for key in SCIENCE):
                raise ValueError("Continuous, resumed and completed scientific states differ")
        prefix_size = STAGES["prefix"][1]
        if not _exact(prefix["history"], full["history"][:prefix_size]):
            raise ValueError("Independent prefix differs")
        chosen = _verify_history(full, plan)
        model = deepcopy(baseline) if arm == "off" else initialize(baseline, arm, SEED)
        endpoints = {}
        for name, field, step in (("selected", "selected_state", chosen), ("last", "state", plan.updates)):
            metrics, scores = _endpoint(model, full[field], scene, record)
            log = full["history"][step - 1]
            if (state_hash(model) != log["state_hash"] or metrics["fused"]["nll"] != log["validation"]["nll"]
                    or metrics["fused"]["confusion_matrix"] != log["validation"]["confusion"]):
                raise ValueError("Independent native V5 differs from selection validation")
            endpoints[name] = dict(step=step, scores=scores, metrics=metrics)
        results[arm] = dict(endpoints=endpoints, summary=_training_summary(full["history"], plan, chosen),
                            initialization=value["initialization"][arm], per_stage=elapsed,
                            wall_seconds=perf_counter()-arm_started, replay_exact_fields=list(SCIENCE),
                            updates=2*plan.updates, completed_reload_updates=0)
        print(f"audited {arm} selected={chosen}", flush=True)
    comparisons = {kind: {f"{a}-{b}": _pair(results[a], results[b], value["recall_floor_delta"])
                          for a, b in pairs} for kind, pairs in value["comparisons"].items()}
    admission.check_pins([*value["protected"].values(), value["source"], value["sidecar"], registration, complete])
    require_frozen_code(code)
    return dev._json_new(root / "audit.json", dict(status="passed", registration=registration, complete=complete,
        results=results, comparisons=comparisons, same_forward_graphs_verified=proof_count,
        independent_forwards=len(ARMS)*2*val_blocks, independent_updates=0, raw_gt=0, query=0,
        wall_seconds=perf_counter()-started))
