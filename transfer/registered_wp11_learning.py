"""Two-arm WP11 registration over the audited isolated development source."""

from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import torch

from model.research_config import digest
from research_runtime import external_directory, hash_file, require_frozen_code, write_json
from scripts.run_wp11_light import load_inputs
from transfer.curriculum_exposure import ExposurePlan
from transfer.development_training import development_counts
from transfer.g3_training import state_hash
from transfer.registered_development import _pin, _read
from transfer.wp11_matched_learning import LR, SEED, advance, construct, require_exposure, restore


POLICY = "wp11-matched-development-v1"
RELEASE = "wp11-matched-model-release-v1"
REAL = "xuzhou_90_45"
ORIGIN = "b40a6be99c4016d96ce1592f9bf166ec97488b5592e46507a96518c0af71c558"
ARMS = ("light", "full")


def read(pin):
    return json.loads(_read(pin))


def segments(updates):
    prefix = min(updates - 1, 7)
    return {"continuous": [0, updates, None], "prefix": [0, prefix, None],
            "resume": [prefix, updates, "prefix"], "completed": [updates, updates, "resume"]}


def inputs(request):
    if set(request) != {"scope", "origin", "exposure", "inputs"}:
        raise ValueError("Exact WP11 source, scope and exposure request required")
    spec = request["inputs"]
    options = spec["training"]
    updates = options["updates"]
    if (options != dict(updates=updates, seed=SEED, lr=LR, class_balance=True)
            or type(updates) is not int or options["class_balance"] is not True):
        raise ValueError("Fixed class-balanced Adam and class-first seed required")
    auxiliary = [request["exposure"]]
    scope = request["scope"]
    if scope == REAL:
        origin = request["origin"]
        if updates != 512 or origin["sha256"] != ORIGIN:
            raise ValueError("Real Xuzhou budget and previously accepted origin are immutable")
        prior = read(origin)
        old_spec = deepcopy(spec)
        old_spec["training"]["updates"] = 12
        if prior["inputs"] != old_spec or (prior["optimizer_updates"], prior["query"]) != (0, 0):
            raise ValueError("Real source differs from the accepted zero-step registration")
        auxiliary.append(origin)
    elif scope != "synthetic" or request["origin"] is not None or not 2 <= updates <= 16:
        raise ValueError("Only the registered real source or a bounded synthetic fixture is supported")
    source, scene, record, pins = load_inputs(spec)
    if scope == REAL:
        if scene.dataset != "xuzhou" or development_counts(record) != (90, 45):
            raise ValueError("Original WP11 Xuzhou 90/45 roles required")
    elif scene.dataset == "xuzhou" or max(scene.plan["shape"]) > 160:
        raise ValueError("Use a small synthetic fixture for engineering acceptance")
    value = read(request["exposure"])
    plan = ExposurePlan(scene, record, value, expected_plan_hash=value["plan_hash"])
    if (plan.updates != updates or value["seed"] != SEED or value["minimum_support_visits"] != 1
            or value["minimum_class_blocks"] != {str(c): 1 for c in record["classes"]}):
        raise ValueError("Exposure budget, seed and coverage floor differ")
    plan.require_eligible()
    return source, scene, record, plan, [*pins, *auxiliary]


def verify(pins, code, out=None):
    require_frozen_code(code)
    for pin in pins:
        path = Path(pin["path"]).resolve()
        if out is not None and (path == out or out in path.parents):
            raise ValueError("New outputs must stay outside registered inputs")
        if hash_file(path) != pin["sha256"]:
            raise ValueError(f"Registered input changed: {path.name}")


def register(request_pin, out, code):
    request = read(request_pin)
    source, scene, record, plan, pins = inputs(request)
    verify([request_pin, *pins], code, out)
    runs = {arm: construct(arm, source, scene, record, plan) for arm in ARMS}
    if state_hash(runs["light"].head) != state_hash(source.cnn_heads[scene.dataset]):
        raise ValueError("Light initialization differs from the original CNN head")
    if state_hash(runs["full"].model) != state_hash(source):
        raise ValueError("Full initialization differs from the exact source")
    value = dict(policy=POLICY, commit=code["commit"], source_hash=code["source_hash"], request=request_pin,
                 source_state=state_hash(source), plan_hash=plan.plan_hash, updates_per_arm=plan.updates,
                 total_updates=4 * plan.updates, stages=segments(plan.updates),
                 contracts={arm: run.contract_hash for arm, run in runs.items()},
                 optimizer_updates=0, original_gt_read=False, query=0)
    out.mkdir()
    write_json(out / "registration.json", value)
    return value


def load_registration(reg_pin, code):
    value = read(reg_pin)
    fields = {"policy", "commit", "source_hash", "request", "source_state", "plan_hash", "updates_per_arm",
              "total_updates", "stages", "contracts", "optimizer_updates", "original_gt_read", "query"}
    if (set(value) != fields or value["policy"] != POLICY or value["commit"] != code["commit"]
            or value["source_hash"] != code["source_hash"] or value["optimizer_updates"] != 0
            or value["query"] != 0 or value["original_gt_read"] is not False):
        raise ValueError("Exact frozen code and zero-step matched registration required")
    root = external_directory(Path(reg_pin["path"]).parent)
    if Path(reg_pin["path"]).resolve() != root / "registration.json":
        raise ValueError("Use the original registered campaign directory")
    request = read(value["request"])
    source, scene, record, plan, pins = inputs(request)
    if (value["source_state"] != state_hash(source) or value["plan_hash"] != plan.plan_hash
            or value["updates_per_arm"] != plan.updates or value["total_updates"] != 4 * plan.updates
            or value["stages"] != segments(plan.updates) or set(value["contracts"]) != set(ARMS)):
        raise ValueError("Source, exposure or stage design differs from the frozen registration")
    return root, value, source, scene, record, plan, [reg_pin, value["request"], *pins]


def attempt_path(root, arm, stage):
    if arm not in ARMS or stage not in segments(2):
        raise ValueError("Unregistered arm or segment")
    path = root / arm / stage
    if path.resolve() != path:
        raise ValueError("Stage may not alias another directory")
    return path


def completed_segment(root, reg, reg_pin, notice_pin, arm, stage):
    path = attempt_path(root, arm, stage)
    receipt = read(_pin(path / "receipt.json"))
    start, stop, _ = reg["stages"][stage]
    if (receipt["status"] != "passed" or receipt["registration"] != reg_pin["sha256"]
            or receipt["notice"] != notice_pin["sha256"] or receipt["arm"] != arm or receipt["stage"] != stage
            or receipt["start"] != start or receipt["stop"] != stop or receipt["updates"] != stop - start):
        raise ValueError("Prior receipt does not describe this registered segment")
    for key, filename in (("checkpoint", "training.pt"), ("light_evidence", "light-parameters.json")):
        pin = receipt[key]
        if Path(pin["path"]).resolve() != path / filename or _pin(path / filename) != pin:
            raise ValueError("Prior segment artifact differs")
    return receipt


def verify_notice(pin, reg_pin, reg):
    note = read(pin)
    if (set(note) != {"registration", "commit", "total_updates", "device", "estimate", "basis"}
            or note["registration"] != reg_pin["sha256"] or note["commit"] != reg["commit"]
            or note["total_updates"] != reg["total_updates"] or note["device"] != "single_queue_cpu"
            or any(not isinstance(note[k], str) or not note[k].strip() for k in ("estimate", "basis"))):
        raise ValueError("Pinned pre-start budget, estimate and device notice required")


def execute(reg_pin, notice_pin, code, arm, stage):
    root, reg, source, scene, record, plan, pins = load_registration(reg_pin, code)
    verify_notice(notice_pin, reg_pin, reg)
    out = attempt_path(root, arm, stage)
    start, stop, parent = reg["stages"][stage]
    previous = None
    if parent is not None:
        previous = completed_segment(root, reg, reg_pin, notice_pin, arm, parent)
        run = restore(arm, source, scene, record, plan, previous["checkpoint"], reg["contracts"][arm])
        pins.extend(previous[key] for key in ("checkpoint", "light_evidence"))
    else:
        run = construct(arm, source, scene, record, plan)
    if run.contract_hash != reg["contracts"][arm] or run.completed != start:
        raise ValueError("Matched contract or stage cursor differs")
    require_exposure(run, plan)
    if torch.get_num_threads() != 1 or torch.cuda.is_available():
        raise ValueError("One CPU thread with CUDA hidden required")
    verify([notice_pin, *pins], code, out)
    out.parent.mkdir(exist_ok=True)
    out.mkdir()
    started = perf_counter()
    try:
        rows, graphs = advance(arm, run, plan, stop)
        require_exposure(run, plan)
        run.save(out / "training.pt")
        write_json(out / "light-parameters.json", rows)
        verify([notice_pin, *pins], code)
    except Exception as error:
        if arm == "full" and run.failed:
            run.save_failure(out / "diagnostic.pt")
        write_json(out / "receipt.json", dict(status="failed", arm=arm, stage=stage, start=start,
                   stop=stop, completed_observed=run.completed, error_type=type(error).__name__,
                   error=str(error), optimizer_attempt_may_be_partial=True, query=0))
        raise
    receipt = dict(status="passed", registration=reg_pin["sha256"], notice=notice_pin["sha256"], arm=arm,
                   stage=stage, start=start, stop=stop, updates=stop-start, selected_step=run.selected_step,
                   checkpoint=_pin(out / "training.pt"), light_evidence=_pin(out / "light-parameters.json"),
                   prefix=plan.progress(stop), history_hash=digest(run.history), graph_calls=graphs,
                   source_state=state_hash(source), wall_seconds=perf_counter()-started, query=0)
    write_json(out / "receipt.json", receipt)
    return receipt
