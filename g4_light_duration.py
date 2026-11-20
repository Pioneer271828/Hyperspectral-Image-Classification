"""Forecast two- and three-slot WP03 schedules from pinned measured phases."""
from collections import defaultdict
import math
from pathlib import Path

from g4_light_engineering_registration import pinned
from g4_light_full_target import roles
from g4_light_real_campaign import BUDGET, historical
from g4_light_target import ADAPTER_HEADS
from g4_support_context import check_files

POLICY = "wp03_full54_duration_forecast_v1"
GPU_MATRIX_SHA = "26ca0184dac7303390e8ce7babdee7c1ff507cc8fd5d314f447df9cb86effdda"
UPDATE = "optimizer_update_with_invariant_checks"


def original_costs(inventory):
    durations = {}
    pins = inventory["checked_files"]
    for case in inventory["cases"]:
        if not case["level"].endswith("receipt_revalidated"):
            continue
        audit = pinned(dict(path=case["receipt"], sha256=case["receipt_sha256"]))
        paths = [path for path in audit["artifacts"] if Path(path).name == "adaptation-cost.json"]
        if len(paths) != 1 or audit["artifacts"][paths[0]] != pins.get(paths[0]):
            raise ValueError("Exactly one original independently pinned adaptation cost required")
        log = pinned(dict(path=paths[0], sha256=pins[paths[0]]))
        if log["deployment_speedup_claim"] is not False or log["scope"] != (
                "initialization_calibration_optimizer_and_validation_with_invariant_checks"):
            raise ValueError("Unexpected original timing scope")
        by_phase = defaultdict(list)
        for e in log["events"]:
            if not math.isfinite(e["seconds"]) or e["seconds"] < 0:
                raise ValueError("Invalid original adaptation duration")
            by_phase[e["phase"]].append(e["seconds"])
        if len(by_phase[UPDATE]) < BUDGET["per_case_updates"] or len(by_phase["validation_snapshot"]) < 51:
            raise ValueError("Original optimizer or validation phase evidence incomplete")
        durations[tuple(case["role"])] = {"update": math.fsum(by_phase[UPDATE]),
            "other": math.fsum(seconds for phase, values in by_phase.items() if phase != UPDATE for seconds in values),
            "source": paths[0]}
    if len(durations) != 36:
        raise ValueError("All 36 original historical cost rows are required")
    check_files({v["source"]: pins[v["source"]] for v in durations.values()})
    return durations


def bounds(rows, key):
    return [min(v[key] for v in rows), max(v[key] for v in rows)]


def slot_range(rows, key):
    return [math.fsum(row[key][i] for row in rows) for i in range(2)]


def schedule(rows, slots, *, balanced):
    queues = [[] for _ in range(slots)]
    for index, row in enumerate(sorted(rows, key=lambda r:r["phase_seconds"][1], reverse=True) if balanced else rows):
        slot = min(range(slots), key=lambda i: slot_range(queues[i], "phase_seconds")[1]) if balanced else index % slots
        queues[slot].append(row)
    ranges = [{"slot":index, "cases":[r["role"] for r in queue],
               "update_seconds":slot_range(queue, "update_seconds"),
               "phase_seconds":slot_range(queue, "phase_seconds")} for index,queue in enumerate(queues)]
    return {"queues": ranges,
            "wall_update_seconds": [max(q["update_seconds"][i] for q in ranges) for i in range(2)],
            "wall_phase_seconds": [max(q["phase_seconds"][i] for q in ranges) for i in range(2)]}


def forecast(registration, inventory, gpu_matrix):
    _, original = historical(registration, inventory)
    costs = original_costs(original)
    if gpu_matrix["sha256"] != GPU_MATRIX_SHA:
        raise ValueError("Use the verified package-23 engineering timing matrix")
    matrix = pinned(gpu_matrix)
    if matrix["scope"] != "synthesized_inputs_only":
        raise ValueError("Target-size timing must keep its synthetic scope")
    sizes = {mode: [matrix["samples"][f"honghu-{mode}-recompute"]["worker"]["wall_seconds"]]
             for mode in ("none","capacity_control","x_residual")}
    all_steps = [seconds for v in sizes.values() for seconds in v]
    if any(not math.isfinite(s) or s <= 0 for s in all_steps):
        raise ValueError("Positive measured target-size seconds required")
    rows = []
    for role in roles():
        scene, _, method, mode = role
        group = [row for r,row in costs.items() if (r[0],r[2]) == (scene,method)]
        if scene == "whu-honghu" and method == ADAPTER_HEADS:
            update = [s * BUDGET["per_case_updates"] for s in (min(all_steps),max(all_steps))]
            # Measured adapter validation/calibration from other scenes is only a proxy.
            other = bounds([row for r,row in costs.items() if r[2] == method], "other")
            basis = "synthetic_shape_optimizer_and_cross_scene_adapter_phase_proxy"
        elif role in costs:
            row = costs[role]
            update, other = [[row[k]]*2 for k in ("update","other")]
            basis = "original_same_role_measured_phases"
        else:
            update, other = [bounds(group,k) for k in ("update","other")]
            basis = "original_same_scene_and_method_range"
        rows.append({"role":list(role), "basis":basis, "update_seconds":update,
                     "other_phase_seconds":other, "phase_seconds":[a+b for a,b in zip(update,other,strict=True)]})
    scenes = {scene: {"original_cases":sum(r[0]==scene for r in costs),
                      "update_card_seconds":slot_range([r for r in rows if r["role"][0]==scene],"update_seconds"),
                      "phase_card_seconds":slot_range([r for r in rows if r["role"][0]==scene],"phase_seconds")}
              for scene in {role[0] for role in roles()}}
    return {"policy":POLICY, "inventory":inventory, "original":registration, "synthetic_gpu_matrix":gpu_matrix,
            "budget":BUDGET, "observed_original_cases":36, "historical_observations":[
                {"role":list(role), **row} for role,row in costs.items()],
            "per_case":rows, "scenes":scenes, "round_robin":{str(n):schedule(rows,n,balanced=False) for n in (2,3)},
            "balanced_projection":{str(n):schedule(rows,n,balanced=True) for n in (2,3)},
            "external_wait_seconds":None,
            "unmeasured":["new HongHu real adapter phase costs and native reread", "process startup and input hashing",
                          "persistence and independent review", "real CUDA co-tenant utilization and launch waits"]}
