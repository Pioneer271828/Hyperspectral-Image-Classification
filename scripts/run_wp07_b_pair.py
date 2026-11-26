"""Bounded Xuzhou WP07 pairing through the unchanged released-point worker.

This controller owns only its new output directory. Resource samples are
observations, as authorized on 2026-11-26; physical leases and arithmetic
attestation remain in the production worker. Each lane is a separate process.
"""
import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

CONTROLLER_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = Path(os.environ.get("WP07_B_FROZEN_WORKER", str(CONTROLLER_ROOT))).resolve(strict=True)
sys.path.insert(0, str(WORKER_ROOT))

import torch

from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from scripts import run_native_released_points as worker
from transfer import native_released_campaign as campaign, native_released_real as real
from transfer import p4_runtime_device as resource, registered_development as transport
from transfer.native_scale_campaign import packet, read
from transfer.native_scale_training import Run
from transfer.spatial_multiscale import initialize
from transfer.xuzhou_multiscale_audit import _exact

ARMS = ("off", "fixed", "mean", "content", "cnn_content", "graph_content")
SEEDS = (64, 512, 2048)
SCHEMA = "wp07_b_xuzhou_factor2_pair_v1"
POLICY = "record_resources_then_attempt_cuda_no_memory_or_utilization_threshold"


def missing_fallback(state, baseline, arm, seed):
    campaign.require(arm in ("mean", "content"), "Expected a trained fallback arm")
    alternate = initialize(baseline, "metadata" if arm == "mean" else "joint", seed)
    incompatible = alternate.load_state_dict(state, strict=False)
    campaign.require(not incompatible.unexpected_keys and incompatible.missing_keys
        and all(".condition_encoder." in key or ".condition_selector." in key
                or key.startswith("backbone.range_encoder.") or key.startswith("backbone.range_metadata.")
                for key in incompatible.missing_keys), "Fallback changes non-metadata state")
    return alternate


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
    campaign.require(os.environ.get("CUDA_VISIBLE_DEVICES") == gpu
                     and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == resource.WORKSPACE,
                     "Expose the assigned UUID and registered workspace")
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
                torch.cuda.synchronize()
                print(json.dumps({"case": case["id"], "phase": phase, "receipt": value,
                    "wall_seconds": perf_counter()-started, "resource_policy": POLICY,
                    "resource_sample": resource_sample,
                    "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "process_peak_reserved_bytes": torch.cuda.max_memory_reserved()}), flush=True)
            audit_path = dest / "b-audit.json"
            if not audit_path.exists():
                started = perf_counter()
                audit = campaign.audit_case(root, case, case["approval"], code, intake=None, cli=worker)
                spec = read(pin(dest / "registration.json"), root)
                state = packet(read(pin(dest / "completed.json"), root)["checkpoint"], root)
                fallback = None
                with worker.arithmetic(root, gpu) as execution:
                    replay = Run(spec, code["commit"], execution=execution,
                                 released=real.RealReleasedRun(root, case["preflight"], case["id"],
                                     case["approval"], dest, code))
                    replay.restore(state, 62)
                    last = replay.evaluate(62, detailed=True)
                    arm = case["id"].split("/")[-1]
                    if arm in ("mean", "content"):
                        # The checkpoint's own trained content/shared weights are
                        # copied; only unused metadata layers remain newly initialized.
                        chosen = state["selected"]
                        replay.model.load_state_dict(chosen["state"], strict=True)
                        original = replay.evaluate(chosen["step"], detailed=True)
                        alternate = missing_fallback(chosen["state"], replay._released.baseline(spec["seed"]),
                                                     arm, spec["module_seed"])
                        replay.model = alternate.to(replay.device)
                        observed = replay.evaluate(chosen["step"], detailed=True)
                        campaign.require(_exact(original, observed), "Actual missing-condition fallback differs")
                        fallback = {"from": "metadata" if arm == "mean" else "joint", "to": arm,
                                    "all_validation_rows_and_head_hashes_exact": True,
                                    "views": len(observed["rows"]), "optimizer_updates": 0}
                campaign.require(last["fused_nll"] == state["validation"][-1]["fused_nll"], "Last endpoint replay differs")
                audit.update(last_head_metrics=campaign.scoring(last, 9, historical=state["validation"][-1]),
                             audit_wall_seconds=perf_counter()-started,
                             missing_fallback=fallback,
                             fresh_updates={name: sum(bool(event["fresh"][name]["changed"]) for event in state["history"])
                                            for name in spec["initialization"]["fresh"]})
                transport._json_new(audit_path, audit)
            print(json.dumps({"case": case["id"], "status": "audited", "audit": pin(audit_path)}), flush=True)


def summarize(root, plan_pin, out, code):
    plan = load_plan(root, plan_pin, code)
    campaign.require(not out.exists() and root in out.parents, "Use a fresh external summary directory")
    audits = []
    for case in plan["cases"]:
        dest = Path(case["destination"])
        row = read(pin(dest / "b-audit.json"), root)
        campaign.require(row["case"] == case["id"] and row["optimizer_updates"] == 62
                         and row["completed_resume_updates"] == 0, "Incomplete B audit")
        audits.append(row)
    campaign.require(sum(r["optimizer_updates"] for r in audits) == 1116, "Incomplete B budget")
    for seed in SEEDS:
        group = [r for r in audits if r["case"].split("/")[1] == str(seed)]
        campaign.require(len({r["source_state"] for r in group}) == 1, "Paired source initializations differ")
    campaign.require(len({r["source_state"] for r in audits}) == 3, "Missing distinct seeded initializations")
    out.mkdir()
    campaign.export_csv(out / "selected-per-class.csv", audits, False)
    campaign.export_csv(out / "last-per-class.csv", [{**r, "head_metrics": r["last_head_metrics"],
                         "selected_step": 62} for r in audits], False)
    comparisons = [("content", "mean"), ("mean", "off"), ("fixed", "off"),
                   ("cnn_content", "off"), ("graph_content", "off"),
                   ("content", "cnn_content"), ("content", "graph_content")]
    rows, classes = [], []
    index = {r["case"]: r for r in audits}
    for seed in SEEDS:
        for candidate, baseline in comparisons:
            a, b = (index[f"xuzhou/{seed}/2/{arm}"] for arm in (candidate, baseline))
            for endpoint, field in (("selected", "head_metrics"), ("last", "last_head_metrics")):
                for head, bands in a[field].items():
                    for band, score in bands.items():
                        base = b[field][head][band]
                        campaign.require(score["count"] == base["count"] == 45
                                         and [p["support"] for p in score["per_class"]] == [5]*9,
                                         "B validation support differs")
                        changes = []
                        for item, other in zip(score["per_class"], base["per_class"], strict=True):
                            campaign.require(item["class_id"] == other["class_id"], "Class order differs")
                            change = item["recall"]-other["recall"]
                            changes.append(change)
                            classes.append(dict(seed=seed, candidate=candidate, baseline=baseline, endpoint=endpoint,
                                head=head, band=band, class_id=item["class_id"], support=item["support"],
                                candidate_recall=item["recall"], baseline_recall=other["recall"], recall_delta=change,
                                protected=change >= plan["class_guard_recall_delta"]))
                        rows.append(dict(seed=seed, candidate=candidate, baseline=baseline, endpoint=endpoint,
                            head=head, band=band, candidate_step=a["selected_step"] if endpoint == "selected" else 62,
                            baseline_step=b["selected_step"] if endpoint == "selected" else 62,
                            **{name+"_delta": score[name]-base[name] for name in ("oa", "aa", "nll")},
                            all_classes_protected=min(changes) >= plan["class_guard_recall_delta"]))
    for filename, records in (("paired-endpoints.csv", rows), ("paired-per-class.csv", classes)):
        with (out / filename).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    primary = [r for r in rows if r["candidate"] == "content" and r["baseline"] == "mean"
               and r["endpoint"] == "selected"]
    positive = all(r["all_classes_protected"] for r in primary) and all(
        r["nll_delta"] < 0 for r in primary if r["head"] == "fused")
    return transport._json_new(out / "summary.json", {"schema": SCHEMA, "producer": code["commit"],
        "plan": plan_pin, "status": "passed", "optimizer_updates": 1116, "audit_optimizer_updates": 0,
        "primary_effect_passed": positive, "primary_pairs": primary,
        "audits": [pin(Path(c["destination"]) / "b-audit.json") for c in plan["cases"]],
        "selected_steps": {r["case"]: r["selected_step"] for r in audits},
        "raw_gt_loaded": False, "query_accessed": False,
        "limitation": "V5 reused for selection and development reporting; each class has five points; three model seeds share one spatial/support release"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--controller-commit")
    parser.add_argument("--root", type=Path, required=True)
    modes = parser.add_subparsers(dest="action", required=True)
    p = modes.add_parser("prepare")
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", nargs=2, required=True)
    p = modes.add_parser("lane")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p = modes.add_parser("summarize")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    controller = code_snapshot(CONTROLLER_ROOT)
    campaign.require(not controller["dirty"] and (args.controller_commit is None
                     or controller["commit"] == args.controller_commit), "Unexpected or dirty B controller")
    campaign.require(code["commit"] == args.expected_commit, "Unexpected B commit")
    print(json.dumps({"controller_commit": controller["commit"], "controller_source_hash": controller["source_hash"],
                      "worker_commit": code["commit"], "worker_source_hash": code["source_hash"]}), flush=True)
    root = args.root.resolve(strict=True)
    if args.action == "prepare":
        result = prepare(root, args.out.resolve(), pin(args.preflight), args.gpus, code)
        print(json.dumps(result), flush=True)
    elif args.action == "lane":
        run_lane(root, pin(args.plan), args.gpu, code)
    else:
        print(json.dumps(summarize(root, pin(args.plan), args.out.resolve(), code)), flush=True)


if __name__ == "__main__":
    main()
