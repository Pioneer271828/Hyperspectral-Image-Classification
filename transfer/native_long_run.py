"""Append-only progress for the native matched coverage campaign."""

from contextlib import contextmanager
import io
from pathlib import Path
import re
import signal
from time import perf_counter

import torch

from evaluation.frozen_training_stages import file_pin, json_pin, pinned, require
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_scale_training as native
from transfer.development_bank import _atomic_new
from transfer.native_coverage_queue import case_lock
from transfer.native_long_campaign import (CHECKPOINT_INTERVAL, DESIGN, SCHEMA, STEPS, MatchedProvider,
                                          design, real_inputs, register_synthetic)

REFERENCE = "xuzhou-m64-none"


def register(root, out, code, *, request=None, approval=None, real=False):
    if real:
        queue = real_inputs(root, out, code, request, approval)
        require(not any(out == Path(entry["bank"]["root"]) or out in Path(entry["bank"]["root"]).parents
                        or Path(entry["bank"]["root"]) in out.parents
                        for entry in queue["inputs"].values()), "Execution output overlaps a publication")
        out.mkdir(exist_ok=True)
    else:
        require(request is approval is None, "Synthetic registration accepts no real approval")
        queue = register_synthetic(out, code)
    place = out/"campaign.json"
    if place.exists():
        require(json_pin(root, file_pin(place)) == queue, "Existing campaign registration differs")
        return file_pin(place)
    require_frozen_code(code)
    return _write(place, queue)


def campaign(root, out, code):
    plan_pin = file_pin(out/"campaign.json")
    plan = json_pin(root, plan_pin)
    require(plan["producer"] == code["commit"] and plan["source_hash"] == code["source_hash"]
            and plan["schema"] == SCHEMA and plan["cases"] == design() and plan["design"] == DESIGN
            and plan["optimization_updates"] == 12*STEPS and plan["scope"] in ("synthetic_only", "external_real")
            and plan["query_accessed"] is plan["raw_gt_loaded"] is False, "Frozen campaign source or design differs")
    if plan["scope"] == "external_real":
        require(plan == real_inputs(root, out, code, plan["request"], plan["approval"]),
                "Real campaign approval or released purpose differs")
    return plan, plan_pin


@contextmanager
def drain_on_signal():
    state = {"requested": None}

    def drain(signum, frame):
        state["requested"] = signal.Signals(signum).name

    previous = {sig: signal.signal(sig, drain) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield state
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def write_checkpoint(path, packet):
    with io.BytesIO() as buffer:
        torch.save(packet, buffer)
        _atomic_new(path, buffer.getvalue())
    return file_pin(path)


def packet(root, pin):
    return torch.load(pinned(root, pin), map_location="cpu", weights_only=True)


def checkpoint(place, step):
    return place/f"clock-{step:04d}.pt"


def checkpoints(root, place, contract, *, read_all=False, steps=STEPS):
    previous = None
    rows = []
    for path in sorted(place.glob("clock-*.pt")):
        require(re.fullmatch(r"clock-\d{4}\.pt", path.name) is not None, "Unexpected campaign checkpoint filename")
        pin = file_pin(path)
        item = packet(root, pin)
        info = item["journal"]
        step = int(path.stem.split("-")[-1])
        start = 0 if previous is None else previous["step"]
        require(item["training"]["registration"] == contract and item["training"]["cursor"] == step
                and info["registration"] == contract and info["start"] == start and info["step"] == step
                and info["updates"] == step-start and info["parent"] == (None if previous is None else previous["pin"]["sha256"])
                and (step == 0 if previous is None else start < step <= steps), "Checkpoint lineage/cursor differs")
        if rows and read_all:
            old = rows[-1]["item"]["training"]
            require(old["history"] == item["training"]["history"][:start]
                    and old["validation"] == item["training"]["validation"][:len(old["validation"])],
                    "Historical events differ across committed segments")
        previous = {**info, "pin": pin}
        rows.append(previous)
        if read_all:
            previous["item"] = item
    return rows


def append(place, run, parent, start, end, contract, timings, reason, started, *, cuda_peak=None):
    packet = run.pack()
    interval = {key: run.timing[key]-before for key, before in timings.items()}
    journal = dict(registration=contract, parent=parent, start=start, step=end, updates=end-start,
                   wall_seconds=perf_counter()-started, timing=interval, reason=reason,
                   model_state=state_hash(run.model), learning_rate=run.optimizer.param_groups[0]["lr"])
    if cuda_peak is not None:
        journal["cuda_peak"] = cuda_peak
    checkpoint_started = perf_counter()
    pin = write_checkpoint(checkpoint(place,end), dict(journal=journal, training=packet))
    return dict(**journal, pin=pin, serialization_seconds=perf_counter()-checkpoint_started)


def state_hash(model):
    return native.state_hash(model)


def execute(root, out, code, case, execution, *, until=STEPS, reference=False):
    plan, plan_pin = campaign(root, out, code)
    require(case in plan["cases"] and type(until) is int and 0 < until <= STEPS,
            "A registered case and bounded drain cursor are required")
    uuid = execution.numeric["physical_uuid"]
    if reference:
        require(plan["scope"] == "synthetic_only" and case == REFERENCE and until == STEPS,
                "Only the registered synthetic continuous reference is authorized")
    if plan["scope"] == "external_real":
        require(uuid is not None and uuid == plan["physical_devices"][case], "Real case physical device differs from approval")
    place = out/("reference" if reference else "cases")/case
    provider = MatchedProvider(root, code, plan, case)
    spec = provider.contract(provider.classes, provider.seed, "off", 1, code["commit"], numeric=execution.numeric)
    contract = digest(spec)
    place.parent.mkdir(exist_ok=True)
    with case_lock(place), drain_on_signal() as drain:
        reg = place/"registration.json"
        if reg.exists():
            require(json_pin(root, file_pin(reg)) == spec, "Cannot migrate or rebind an authoritative trajectory")
        else:
            _write(reg, spec)
        rows = checkpoints(root, place, contract)
        if rows and rows[-1]["step"] == STEPS:
            path = place/"receipt.json"
            if not path.exists():
                return finish(root, place, code, plan_pin, spec, rows, updates=0)
            receipt = json_pin(root, file_pin(path))
            require(receipt["registration"] == contract and receipt["checkpoint"] == rows[-1]["pin"],
                    "Completed native execution markers disagree")
            return dict(case=case, status="completed", updates_this_invocation=0, receipt=file_pin(path))
        require(not reference or not rows, "Continuous reference timed out or was interrupted")
        require(not (place/"receipt.json").exists(), "Incomplete execution cannot have a completed receipt")
        started = perf_counter()
        run = native.Run(spec, spec["producer"], execution=execution, released=provider)
        cursor = rows[-1]["step"] if rows else 0
        require(until >= cursor, "Cannot go backwards on an authoritative trajectory")
        if rows:
            run.restore(packet(root, rows[-1]["pin"])["training"], cursor)
            parent = rows[-1]["pin"]["sha256"]
        else:
            prior_timing = dict(run.timing)
            run.start()
            row = append(place, run, None, 0, 0, contract, prior_timing, "initial", started)
            rows.append(row)
            parent = row["pin"]["sha256"]
        start = cursor
        while cursor < until and drain["requested"] is None:
            limit = min(until, (cursor//CHECKPOINT_INTERVAL+1)*CHECKPOINT_INTERVAL)
            segment = cursor
            started, prior_timing = perf_counter(), dict(run.timing)
            for event in spec["training"][cursor:limit]:
                if drain["requested"]:
                    break
                run.update(event)
                cursor = event["step"]
            if cursor == segment:
                break
            provider.bank.verify_authority(provider.bank.pin)
            reason = drain["requested"] or ("finish" if cursor == STEPS else "drain_cursor" if cursor == until else "interval")
            row = append(place, run, parent, segment, cursor, contract, prior_timing, reason, started)
            rows.append(row)
            parent = row["pin"]["sha256"]
        if cursor == STEPS:
            return finish(root, place, code, plan_pin, spec, rows, updates=cursor-start)
        return dict(case=case, status="drained", cursor=cursor, updates_this_invocation=cursor-start,
                    checkpoint=rows[-1]["pin"], signal=drain["requested"])


def finish(root, place, code, plan_pin, spec, rows, *, updates):
    end = rows[-1]
    state = packet(root, end["pin"])["training"]
    require(state["cursor"] == STEPS and sum(row["updates"] for row in rows) == STEPS
            and {0, 750, STEPS} <= {row["step"] for row in rows}, "Physical endpoint or exposure is incomplete")
    require_frozen_code(code)
    receipt = dict(status="completed", campaign=plan_pin["sha256"], registration=digest(spec),
                   spec=file_pin(place/"registration.json"), checkpoint=end["pin"],
                   checkpoints=[row["pin"] for row in rows], updates=STEPS,
                   selected=state["selected"]["step"], device=spec["numeric"], query_accessed=False,
                   raw_gt_loaded=False, real_updates=spec["real_updates"])
    return dict(case=place.name, status="completed", updates_this_invocation=updates,
                receipt=_write(place/"receipt.json", receipt))


def freeze(root, out, code):
    plan, plan_pin = campaign(root, out, code)
    freezes = {}
    for case in plan["cases"]:
        folder = out/"cases"/case
        pin = file_pin(folder/"receipt.json")
        receipt = json_pin(root, pin)
        spec = json_pin(root, receipt["spec"])
        rows = checkpoints(root, folder, receipt["registration"])
        require(receipt["status"] == "completed" and receipt["campaign"] == plan_pin["sha256"]
                and receipt["registration"] == digest(spec) and receipt["updates"] == STEPS
                and receipt["checkpoint"] == rows[-1]["pin"] and receipt["checkpoints"] == [item["pin"] for item in rows]
                and receipt["query_accessed"] is receipt["raw_gt_loaded"] is False, "All twelve authoritative fits required")
        freezes[case] = pin
    value = dict(schema=SCHEMA, scope=plan["scope"], campaign=plan_pin, cases=freezes, updates=12*STEPS,
                 query_accessed=False, raw_gt_loaded=False)
    path = out/"freeze.json"
    if path.exists():
        require(json_pin(root,file_pin(path)) == value, "Frozen completion roster differs")
        return file_pin(path)
    require_frozen_code(code)
    return _write(path, value)
