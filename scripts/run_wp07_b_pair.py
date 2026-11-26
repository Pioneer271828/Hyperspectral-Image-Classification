"""Bounded Xuzhou WP07 pairing through the unchanged released-point worker.

This controller owns only its new output directory. Resource samples are
observations, as authorized on 2026-11-26; physical leases and arithmetic
attestation remain in the production worker. Each lane is a separate process.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from scripts import run_native_released_points as worker
from transfer import native_released_campaign as campaign, native_released_real as real
from transfer import p4_runtime_device as resource, registered_development as transport
from transfer.native_scale_campaign import packet, read
from transfer.native_scale_training import Run
from transfer.xuzhou_multiscale_audit import _exact

ARMS = ("off", "fixed", "mean", "content", "cnn_content", "graph_content")
SEEDS = (64, 512, 2048)
SCHEMA = "wp07_b_xuzhou_factor2_pair_v1"
POLICY = "record_resources_then_attempt_cuda_no_memory_or_utilization_threshold"


@contextmanager
def observation_only_resources():
    """Override admission only inside this dedicated, single-thread worker.

    The same production gate is called at the outer stage and inside P4
    arithmetic. Rebinding it here reaches both, while preserving UUID checking,
    CUDA allocation, deterministic kernels and physical leases. No source or
    global configuration file is changed; exceptions restore the function.
    """
    previous = resource.gate
    resource.gate = resource.snapshot
    try:
        yield
    finally:
        resource.gate = previous


def pin(path):
    return transport._pin(Path(path).resolve(strict=True))


def prepare(root, out, preflight, devices, code):
    require_frozen_code(code)
    request = read(preflight, root)
    campaign.require(request["scene"] == "xuzhou" and request["design"] == real.DUTY
                     and request["schema"] == "wp07_native_released_real_preflight_v1"
                     and request["bank"]["sha256"] == "c877232e6cc38e3f8be37eb1b147d69a7253edbeb665f98e5b8cd6acb8e5cb17"
                     and request["optimization_updates"] == 0
                     and request["raw_gt_loaded"] is request["query_accessed"] is False,
                     "Expected the existing real Xuzhou released-point duties preflight")
    campaign.require(out == out.resolve() and root in out.parents and not out.exists(), "Use a new external B directory")
    campaign.require(len(devices) == 2 and len(set(devices)) == 2
                     and all(resource.UUID.fullmatch(gpu) for gpu in devices), "Bind two distinct physical UUIDs")
    old = root / "wp07-native-duty01" / "real-execution"
    cases = []
    for seed in SEEDS:
        schedule = None
        for arm in ARMS:
            key = f"xuzhou/{seed}/2/{arm}"
            row = real.registered_case(request, key, engineering=False)
            campaign.require(row["planned_updates"] == 62 and row["events"] == [0, 31, 62]
                             and row["train_exposures_per_class"] == [10]*9,
                             "Historical Xuzhou budget or released exposure differs")
            schedule = row["schedule"] if schedule is None else schedule
            campaign.require(row["schedule"] == schedule, "Paired exposure differs")
            previous = old / key.replace("/", "-")
            campaign.require(not previous.exists(), "Original registered case has artifacts; review and reuse first")
            cases.append({"id": key, "preflight": preflight, "sha256": digest(row), "updates": 62,
                          "destination": str(out / "execution" / key.replace("/", "-")),
                          "gpu": devices[1 if seed == 512 else 0]})
    out.mkdir()
    (out / "execution").mkdir()
    (out / "approvals").mkdir()
    for case in cases:
        case["approval"] = transport._json_new(out / "approvals" / (case["id"].replace("/", "-") + ".json"), {
            "schema": real.RELEASE, "decision": "approved", "source_commit": code["commit"],
            "source_hash": code["source_hash"], "preflight": preflight, "case_sha256": case["sha256"],
            "destination": case["destination"], "optimizer_updates": 62, "physical_uuid": case["gpu"],
            "protocol": resource.CUDA_FP64_WEIGHT, "route": "split",
            "raw_gt_authorized": False, "query_authorized": False})
    return transport._json_new(out / "plan.json", {
        "schema": SCHEMA, "producer": code["commit"], "source_hash": code["source_hash"],
        "authorization": "2026-11-26 user-approved execution B WP07 development pairing",
        "cases": cases, "optimizer_updates": 1116, "scene": "xuzhou", "factor": 2,
        "seeds": list(SEEDS), "arms": list(ARMS), "resource_policy": POLICY,
        "primary_comparison": "content-minus-mean", "secondary_comparisons": ["mean-minus-off", "fixed-minus-off",
            "cnn_content-minus-off", "graph_content-minus-off", "content-minus-cnn_content", "content-minus-graph_content"],
        "selection": "first_min_released_point_fused_nll_at_0_31_62",
        "class_guard_recall_delta": -0.02, "effect_requires": "all three paired model seeds improve fused NLL and no per-head/class guard failure",
        "estimated_wall_minutes": [20, 40], "scope": "existing K5/V5 development release only; no independent confirmation",
        "raw_gt_authorized": False, "query_authorized": False})


def load_plan(root, plan_pin, code):
    require_frozen_code(code)
    plan = read(plan_pin, root)
    campaign.require(plan["schema"] == SCHEMA and plan["producer"] == code["commit"]
                     and plan["source_hash"] == code["source_hash"] and plan["resource_policy"] == POLICY
                     and plan["optimizer_updates"] == 1116
                     and plan["raw_gt_authorized"] is plan["query_authorized"] is False,
                     "B source, authority or budget differs")
    expected = {f"xuzhou/{s}/2/{a}" for s in SEEDS for a in ARMS}
    campaign.require(len(plan["cases"]) == len(expected) and {c["id"] for c in plan["cases"]} == expected,
                     "B case roster differs")
    for case in plan["cases"]:
        campaign.require(case["updates"] == 62, "B update budget differs")
        campaign.authorized_case(root, case, case["approval"], code, intake=None)
    return plan


def run_lane(root, plan_pin, gpu, code):
    plan = load_plan(root, plan_pin, code)
    cases = [c for c in plan["cases"] if c["gpu"] == gpu]
    campaign.require(cases, "GPU has no assigned cases")
    out = Path(plan_pin["path"]).parent
    with resource.lease(out / (gpu + ".lane.lock")), observation_only_resources():
        for case in cases:
            dest = Path(case["destination"])
            for phase in ("prefix", "resume", "completed"):
                if (dest / f"{phase}.json").exists():
                    # Verify committed transport before continuing. The unchanged
                    # next stage validates registration, cursor and full history.
                    campaign.receipt(root, dest, phase, pin(dest / "registration.json"))
                    continue
                started = perf_counter()
                resource_sample = resource.snapshot(gpu)
                value = worker.real_stage(root, dest, case["preflight"], case["approval"], case["id"],
                                          phase, code, gpu)
                print(json.dumps({"case": case["id"], "phase": phase, "receipt": value,
                    "wall_seconds": perf_counter()-started, "resource_policy": POLICY,
                    "resource_sample": resource_sample}), flush=True)
            audit_path = dest / "b-audit.json"
            if not audit_path.exists():
                started = perf_counter()
                audit = campaign.audit_case(root, case, case["approval"], code, intake=None, cli=worker)
                spec = read(pin(dest / "registration.json"), root)
                state = packet(read(pin(dest / "completed.json"), root)["checkpoint"], root)
                with worker.arithmetic(root, gpu) as execution:
                    replay = Run(spec, code["commit"], execution=execution,
                                 released=real.RealReleasedRun(root, case["preflight"], case["id"],
                                     case["approval"], dest, code))
                    replay.restore(state, 62)
                    last = replay.evaluate(62, detailed=True)
                campaign.require(last["fused_nll"] == state["validation"][-1]["fused_nll"], "Last endpoint replay differs")
                audit.update(last_head_metrics=campaign.scoring(last, 9, historical=state["validation"][-1]),
                             audit_wall_seconds=perf_counter()-started,
                             fresh_updates={name: sum(bool(event["fresh"][name]["changed"]) for event in state["history"])
                                            for name in spec["initialization"]["fresh"]})
                transport._json_new(audit_path, audit)
            print(json.dumps({"case": case["id"], "status": "audited", "audit": pin(audit_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--root", type=Path, required=True)
    modes = parser.add_subparsers(dest="action", required=True)
    p = modes.add_parser("prepare")
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", nargs=2, required=True)
    p = modes.add_parser("lane")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    campaign.require(code["commit"] == args.expected_commit, "Unexpected B commit")
    root = args.root.resolve(strict=True)
    if args.action == "prepare":
        result = prepare(root, args.out.resolve(), pin(args.preflight), args.gpus, code)
        print(json.dumps(result), flush=True)
    else:
        run_lane(root, pin(args.plan), args.gpu, code)


if __name__ == "__main__":
    main()
