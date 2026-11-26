"""Register and run the admitted PaviaU frozen-Q graph-range comparison."""
import argparse
from collections import Counter
from io import BytesIO
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from scripts import run_native_released_points as worker
from scripts.run_wp07_b_pair import observation_only_resources
from transfer import wp07_graph_range as graph, wp07_graph_training as training
from transfer import p4_runtime_device as resource, registered_development as dev
from transfer.development_bank import _atomic_new
from transfer.g3_training import state_hash
from transfer.native_scale_campaign import read, packet
from transfer.xuzhou_multiscale_audit import _exact
from joint_training import _run_lock

GEOMETRY_SHA = ("d176f887b86c4a8bb4b287d1d316ca3e"
                "3744d9e20f20510b417c863df5840094")
SCHEMA = "wp07_b_paviau_frozen_range_pair_v1"
SEEDS = (64, 512, 2048)
PHASES = ("prefix", "resume", "completed")


def pin(path):
    return dev._pin(Path(path).resolve(strict=True))


def save(path, value):
    with BytesIO() as data:
        torch.save(value, data)
        _atomic_new(path, data.getvalue())
    return pin(path)


def qualify(root, out, gpu, code):
    require_frozen_code(code)
    graph.require(root in out.parents and not out.exists(), "Use a new isolated qualification path")
    out.mkdir()
    provider = training.FixtureProvider()
    entries = training.fixture_cache(provider, code["commit"])
    cache = save(out / "fixture-cache.pt", entries)
    proof = []
    with observation_only_resources(), worker.arithmetic(root, gpu) as execution:
        for arm in graph.ARMS:
            spec = training.contract(provider, 64, arm, code["commit"], execution.numeric,
                                     cache, {"synthetic": True, "out": str(out), "gpu": gpu})
            folder = out / arm
            folder.mkdir()
            values = {}
            stages = []
            for phase in ("full", "prefix", "resume", "completed"):
                prior = values.get({"resume": "prefix", "completed": "resume"}.get(phase))
                path = folder / (phase + ".pt")
                stage = training.execute(spec, phase, path, provider, entries, code["commit"], execution, prior)
                values[phase] = packet(pin(path), root)
                stages.append(stage)
            graph.require(_exact(values["full"], values["resume"]) and _exact(values["full"], values["completed"])
                          and [s["updates"] for s in stages] == [6, 3, 3, 0], "Range full/split/completed qualification differs")
            if arm == "content":
                graph.require(any(e["changed"] and e["gradient_l1"] > 0
                    for row in values["full"]["history"] for e in row["fresh"].values()), "Existing graph selector did not update")
            proof.append({"arm": arm, "updates": 12, "full_split_complete_exact": True})
    return dev._json_new(out / "qualification.json", {"schema": SCHEMA, "status": "passed", "synthetic": True,
        "producer": code["commit"], "source_hash": code["source_hash"], "gpu": gpu, "arms": proof,
        "optimizer_updates": 48, "real_optimizer_updates": 0, "completed_updates": 0,
        "query_accessed": False, "raw_gt_loaded": False, "python_optimization_refused": True})


def prepare(root, out, geometry, qualifications, code):
    require_frozen_code(code)
    graph.require(geometry["sha256"] == GEOMETRY_SHA, "Expected the frozen pre-score geometry decision")
    decision = read(geometry, root)
    graph.require(decision["selected_protocol"] == "paviaU-factor1-scale50"
                  and decision["labels_scored"] == decision["classifier_heads_evaluated"] == decision["optimizer_updates"] == 0,
                  "Geometry decision or no-score boundary differs")
    admitted_pin = decision["protocols"][decision["selected_protocol"]]
    admitted = read(admitted_pin, root)
    graph.require(admitted["all_views_distinct"] and admitted["eligible"]
                  and all(v["comparison"]["identifiable"] for v in admitted["views"]), "Every chosen view must distinguish graph support")
    bindings = []
    for item in qualifications:
        q = read(item, root)
        graph.require(q["schema"] == SCHEMA and q["status"] == "passed" and q["synthetic"] is True
                      and q["producer"] == code["commit"] and q["source_hash"] == code["source_hash"]
                      and q["real_optimizer_updates"] == q["completed_updates"] == 0
                      and resource.UUID.fullmatch(q["gpu"]) and all(a["full_split_complete_exact"] for a in q["arms"]),
                      "Missing same-source physical-device recovery qualification")
        bindings.append(q["gpu"])
    graph.require(len(bindings) in (1, 2) and len(set(bindings)) == len(bindings), "One or two independently qualified devices required")
    graph.require(root in out.parents and not out.exists(), "Use a fresh external pair directory")
    preflight = admitted["preflight"]
    receipt = read(preflight, root)
    cache = packet(admitted["cache"], root)
    graph.require(cache["source"] == preflight and cache["bank"] == receipt["bank"] and cache["factor"] == 1 and cache["scale"] == 50,
                  "Geometry cache source differs")
    cases = []
    shared_backbone = None
    for seed in SEEDS:
        provider = training.BankProvider(root, receipt, seed)
        graph.require(state_hash(provider.baseline(64)) == cache["baseline_state"], "Geometry source initialization differs")
        backbone = state_hash(provider.baseline(seed).backbone)
        shared_backbone = backbone if shared_backbone is None else shared_backbone
        graph.require(backbone == shared_backbone, "Model-seed initialization changed the frozen geometry encoder")
        train, val, events = provider.events()
        graph.require(len(train) == 60 and len(val) == 82 and events == [0, 30, 60], "Original PaviaU budget or validation roster differs")
        for role, rows in (("devtrain", train), ("devval", val)):
            for event in rows:
                entry = cache["entries"][training.key(role, event)]
                graph.compare_supports(entry["near"], entry["wide"], require_distinct=True)
        for arm in graph.ARMS:
            gpu = bindings[1] if len(bindings) == 2 and seed == 512 else bindings[0]
            cases.append({"id": f"paviaU/{seed}/{arm}", "seed": seed, "arm": arm, "gpu": gpu, "updates": 60,
                          "destination": str(out / "execution" / f"paviaU-{seed}-{arm}")})
    out.mkdir()
    (out / "execution").mkdir()
    return dev._json_new(out / "plan.json", {"schema": SCHEMA, "producer": code["commit"], "source_hash": code["source_hash"],
        "geometry": geometry, "admission": admitted_pin, "cache": admitted["cache"], "preflight": preflight,
        "qualifications": qualifications, "cases": cases, "optimizer_updates": 720,
        "authority": "2026-11-26 coordinator-approved B graph-range package after accepted geometry enumeration",
        "primary_comparison": "content-minus-mean", "recall_delta_floor": -0.02,
        "effect_requires": "all 3 seeds and both bands lower fused NLL plus all-head per-class protection versus mean",
        "selection": "first_min_fused_NLL_0_30_60", "cnn": "fixed_dilation1_all_arms",
        "capacity": "same existing graph-content model and both message paths; fixed coefficients in controls",
        "Q_and_both_adjacency_values": "frozen_per_input_at_pre_score_initial_encoder",
        "estimated_wall_minutes": [20, 40], "independent_spatial_confirmation": False,
        "query_accessed": False, "raw_gt_loaded": False})


def load(root, plan_pin, code):
    require_frozen_code(code)
    plan = read(plan_pin, root)
    graph.require(plan["schema"] == SCHEMA and plan["producer"] == code["commit"]
                  and plan["source_hash"] == code["source_hash"] and plan["optimizer_updates"] == 720
                  and plan["geometry"]["sha256"] == GEOMETRY_SHA
                  and plan["query_accessed"] is plan["raw_gt_loaded"] is False,
                  "Range plan source, geometry or authority differs")
    graph.require(len(plan["cases"]) == 12 and {c["id"] for c in plan["cases"]} ==
                  {f"paviaU/{s}/{a}" for s in SEEDS for a in graph.ARMS}, "Range case roster differs")
    cache = packet(plan["cache"], root)
    receipt = read(plan["preflight"], root)
    return plan, cache["entries"], receipt


def stage(root, plan_pin, plan, entries, receipt, case, phase, code):
    out = Path(case["destination"])
    graph.require(root in out.parents and out.parent == Path(plan_pin["path"]).parent / "execution"
                  and out.name == case["id"].replace("/", "-"), "Range case path differs")
    out.mkdir(exist_ok=True)
    with _run_lock(out), observation_only_resources(), worker.arithmetic(root, case["gpu"]) as execution:
        provider = training.BankProvider(root, receipt, case["seed"])
        spec = training.contract(provider, case["seed"], case["arm"], code["commit"], execution.numeric, plan["cache"], plan_pin)
        registration = out / "registration.json"
        if registration.exists():
            spec_pin = pin(registration)
            graph.require(read(spec_pin, root) == spec, "Committed range registration differs")
        else:
            spec_pin = dev._json_new(registration, spec)
        prior_phase = worker.check_real_route(out, phase)
        prior = None
        if prior_phase:
            previous = read(pin(out / (prior_phase + ".json")), root)
            graph.require(previous["registration"] == spec_pin, "Range predecessor registration differs")
            prior = packet(previous["checkpoint"], root)
        graph.require(not any((out / (phase + suffix)).exists() for suffix in (".pt", ".json", ".intent.json")),
                      "Retain existing phase or failed intent for review")
        dev._json_new(out / (phase + ".intent.json"), {"registration": spec_pin, "phase": phase})
        result = training.execute(spec, phase, out / (phase + ".pt"), provider, entries, code["commit"], execution, prior)
        return dev._json_new(out / (phase + ".json"), {**result, "registration": spec_pin, "checkpoint": pin(out / (phase + ".pt"))})


def audit_case(root, plan_pin, plan, entries, receipt, case, code, *, independent=False):
    out = Path(case["destination"])
    spec = read(pin(out / "registration.json"), root)
    phases = [read(pin(out / (phase + ".json")), root) for phase in PHASES]
    graph.require([p["updates"] for p in phases] == [30, 30, 0], "Range budget differs")
    state = packet(phases[-1]["checkpoint"], root)
    graph.require(_exact(state, packet(phases[-2]["checkpoint"], root)), "Completed state differs")
    with observation_only_resources(), worker.arithmetic(root, case["gpu"]) as execution:
        provider = training.BankProvider(root, receipt, case["seed"])
        result = training.audit(spec, state, provider, entries, code["commit"], execution, cost=independent)
    graph.require(result["training_exposures_per_class"] == [10]*9, "Released training exposure differs")
    return {**result, "case": case["id"], "parameters": spec["parameters"], "source_state": spec["source_state"],
            "numeric": spec["numeric"], "independent_process_replay": independent}


def lane(root, plan_pin, gpu, code, *, independent=False):
    plan, entries, receipt = load(root, plan_pin, code)
    cases = [c for c in plan["cases"] if c["gpu"] == gpu]
    graph.require(cases and os.environ.get("CUDA_VISIBLE_DEVICES") == gpu, "Assigned visible UUID required")
    out = Path(plan_pin["path"]).parent
    with resource.lease(out / (gpu + (".audit.lock" if independent else ".lane.lock"))):
        for case in cases:
            dest = Path(case["destination"])
            if not independent:
                for phase in PHASES:
                    if (dest / (phase + ".json")).exists():
                        previous = read(pin(dest / (phase + ".json")), root)
                        packet(previous["checkpoint"], root)
                        continue
                    started = perf_counter()
                    sample = resource.snapshot(gpu)
                    result = stage(root, plan_pin, plan, entries, receipt, case, phase, code)
                    print(json.dumps({"case": case["id"], "phase": phase, "receipt": result,
                                      "wall_seconds": perf_counter()-started, "resource_sample": sample,
                                      "resource_policy": "observation_only_no_threshold"}), flush=True)
            name = "independent.json" if independent else "audit.json"
            if not (dest / name).exists():
                started = perf_counter()
                result = audit_case(root, plan_pin, plan, entries, receipt, case, code, independent=independent)
                dev._json_new(dest / name, {**result, "audit_wall_seconds": perf_counter()-started})
            print(json.dumps({"case": case["id"], "status": "independent" if independent else "audited",
                              "receipt": pin(dest / name)}), flush=True)


def summarize(root, plan_pin, out, code):
    plan, entries, receipt = load(root, plan_pin, code)
    graph.require(root in out.parents and not out.exists(), "Use a fresh external summary directory")
    out.mkdir()
    endpoints, classes, preferences, costs = [], [], [], []
    chosen, initialization, parameters = {}, {}, set()
    for case in plan["cases"]:
        dest = Path(case["destination"])
        original = read(pin(dest / "audit.json"), root)
        independent = read(pin(dest / "independent.json"), root)
        graph.require(independent["independent_process_replay"] and independent["audit_optimizer_updates"] == 0
                      and independent["case"] == case["id"], "Independent replay missing")
        parameters.add(independent["parameters"])
        initialization.setdefault(case["seed"], set()).add(independent["source_state"])
        chosen[case["id"]] = independent["selected_step"]
        for endpoint, values in independent["endpoints"].items():
            graph.require(values["metrics"] == original["endpoints"][endpoint]["metrics"], "Independent metrics differ")
            for head, bands in values["metrics"].items():
                for band, score in bands.items():
                    graph.require(score["count"] == 45 and [c["support"] for c in score["per_class"]] == [5]*9,
                                  "Released validation support differs")
                    common = {"case": case["id"], "seed": case["seed"], "arm": case["arm"], "endpoint": endpoint,
                              "step": chosen[case["id"]] if endpoint == "selected" else 60, "head": head, "band": band}
                    endpoints.append({**common, **{k: score[k] for k in ("oa", "aa", "nll", "kappa", "macro_f1")}})
                    for i, c in enumerate(score["per_class"]):
                        classes.append({**common, "class_id": c["class_id"], "correct": score["confusion_matrix"][i][i],
                                        "support": c["support"], "recall": c["recall"]})
        state = packet(read(pin(dest / "completed.json"), root)["checkpoint"], root)
        collections = [("train", state["history"]),
                       ("selected_devval", independent["endpoints"]["selected"]["validation"]["rows"])]
        for role, rows in collections:
            w = [r["graph"]["weights"]["graph"]["weights"] for r in rows]
            preferences.append({"case": case["id"], "role": role, "views": len(w),
                                "mean_near_weight": sum(v[0] for v in w)/len(w),
                                "min_near_weight": min(v[0] for v in w), "max_near_weight": max(v[0] for v in w),
                                "near_preference": sum(v[0] > v[1] for v in w),
                                "wide_preference": sum(v[1] > v[0] for v in w), "ties": sum(v[0] == v[1] for v in w),
                                "interpretation": "continuous coefficients; both graph paths always execute"})
        costs.extend({"case": case["id"], "parameters": independent["parameters"], **v} for v in independent["cost"])
    graph.require(len(parameters) == 1 and all(len(v) == 1 for v in initialization.values())
                  and len({next(iter(v)) for v in initialization.values()}) == 3,
                  "Capacity or paired/independent initializations differ")
    idx = {(r["seed"], r["arm"], r["endpoint"], r["head"], r["band"]): r for r in endpoints}
    ci = {(r["seed"], r["arm"], r["endpoint"], r["head"], r["band"], r["class_id"]): r for r in classes}
    pairs, class_pairs = [], []
    for seed in SEEDS:
        for candidate, baseline in (("wide", "near"), ("mean", "near"), ("content", "near"), ("content", "mean")):
            for endpoint in ("selected", "last"):
                for head in training.HEADS:
                    for band in ("full", "even_retained_positions"):
                        a, b = (idx[seed, arm, endpoint, head, band] for arm in (candidate, baseline))
                        row = {"seed": seed, "candidate": candidate, "baseline": baseline,
                               "endpoint": endpoint, "head": head, "band": band}
                        protected = []
                        for cls in range(1, 10):
                            x, y = (ci[seed, arm, endpoint, head, band, cls] for arm in (candidate, baseline))
                            change = x["recall"]-y["recall"]
                            protected.append(change >= plan["recall_delta_floor"])
                            class_pairs.append({**row, "class_id": cls, "candidate_correct": x["correct"],
                                "baseline_correct": y["correct"], "support": 5, "recall_delta": change,
                                "protected": protected[-1]})
                        pairs.append({**row, **{k+"_delta": a[k]-b[k] for k in ("oa", "aa", "nll")},
                                      "all_classes_protected": all(protected)})
    for name, rows in (("endpoints.csv", endpoints), ("per-class-correct.csv", classes), ("paired-endpoints.csv", pairs),
                       ("paired-per-class.csv", class_pairs), ("range-preferences.csv", preferences), ("resident-cost.csv", costs)):
        with (out / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    primary = [r for r in pairs if r["candidate"] == "content" and r["baseline"] == "mean" and r["endpoint"] == "selected"]
    passed = all(r["all_classes_protected"] for r in primary) and all(r["nll_delta"] < 0 for r in primary if r["head"] == "fused")
    return dev._json_new(out / "summary.json", {"schema": SCHEMA, "status": "passed", "producer": code["commit"],
        "plan": plan_pin, "optimizer_updates": 720, "independent_optimizer_updates": 0,
        "primary_effect_passed": passed, "primary_pairs": primary, "selected_steps": chosen,
        "parameters_all_arms": next(iter(parameters)), "query_accessed": False, "raw_gt_loaded": False,
        "independent_spatial_confirmation": False, "scope": "frozen-Q released K5/V5 development pairing only"})


def main():
    graph.require_checks()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--root", type=Path, required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    q = actions.add_parser("qualify")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--gpu")
    q = actions.add_parser("prepare")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--geometry", type=Path, required=True)
    q.add_argument("--qualification", type=Path, action="append", required=True)
    for name in ("lane", "audit"):
        q = actions.add_parser(name)
        q.add_argument("--plan", type=Path, required=True)
        q.add_argument("--gpu", required=True)
    q = actions.add_parser("summarize")
    q.add_argument("--plan", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    graph.require(code["commit"] == args.expected_commit, "Unexpected graph-range source")
    require_frozen_code(code)
    root = args.root.resolve(strict=True)
    torch.set_num_threads(1)
    if args.action == "qualify":
        print(json.dumps(qualify(root, args.out.resolve(), args.gpu, code)), flush=True)
    elif args.action == "prepare":
        print(json.dumps(prepare(root, args.out.resolve(), pin(args.geometry), [pin(p) for p in args.qualification], code)), flush=True)
    elif args.action in ("lane", "audit"):
        lane(root, pin(args.plan), args.gpu, code, independent=args.action == "audit")
    else:
        print(json.dumps(summarize(root, pin(args.plan), args.out.resolve(), code)), flush=True)


if __name__ == "__main__":
    main()
