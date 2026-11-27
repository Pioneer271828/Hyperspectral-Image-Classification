"""Verify frozen controls, shared tensors and the paired WP04 joint factorial."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from evaluation.wp12_edge_weight_evidence import _comparison, _qualified_phases, _recompute
from evaluation.wp12_residual_interaction_evidence import _factorial, contract, published_roles
from joint_training import _tree_fingerprint
from research_runtime import code_snapshot, external_directory, require_frozen_code, tensor_hash, write_json
from scripts.audit_wp04_message_correspondence import scores
from scripts.run_wp04_factorial import (
    HEADS, MATCHED_GROUP, MATCHED_PARAMS, MATCHED_SCOPE, SEEDS, STEPS, attest, audit_case, identity,
    pinned, read, require, roster, terminal, verify,
)
from transfer import p4_runtime_device as p4
from transfer.shared_gpu_lease import physical_lease

HISTORY_SHA = "561af2b2a1d9f81a205c926c89631971fca0dbcc0b788babd6d2076ca8dd9d9e"
PREFIXES = {"M00": (), "M10": ("backbone.interaction.",), "M01": ("backbone.edge_updater.",),
            "M11": ("backbone.interaction.", "backbone.edge_updater.")}
DESIGNS = {
    "message": {"M10": .5, "M00": -.5, "M11": .5, "M01": -.5},
    "joint_edge": {"M01": .5, "M00": -.5, "M11": .5, "M10": -.5},
    "interaction": {"M11": 1, "M10": -1, "M01": -1, "M00": 1},
}
CONDITIONAL = (("M10", "M00"), ("M11", "M01"), ("M01", "M00"), ("M11", "M10"))


def common_tensors(reg):
    common = read(verify(reg["common_initialization"]))
    require(common["schema"] == "wp04_common_message_edge_tensors_v1"
            and tuple(common["seeds"]) == tuple(map(str, SEEDS))
            and common["inactive_role"] == "registered_tensor_artifact_no_forward_or_optimizer_presence",
            "Different common initialization metadata")
    for item in common["seeds"].values():
        tensors = torch.load(verify(item["tensors"]), map_location="cpu", weights_only=True)
        require({n: tensor_hash(t) for n, t in tensors.items()} == item["initialization"]["additional_tensors"],
                "Common tensor material differs from its fingerprint")
    return common["seeds"]


def match_initial(init, common, arm):
    bound = common["initialization"]
    require(init["shared"] == bound["shared"] and init["parameters"] == MATCHED_PARAMS[arm]
            and init["additional_tensors"] == {n: h for n, h in bound["additional_tensors"].items()
                                               if n.startswith(PREFIXES[arm])},
            "Shared, active or inactive initialization differed from the seed's registration")


def historical(reg, pin, common):
    require(pin["sha256"] == HISTORY_SHA, "Use the independently qualified previous 21-case receipt")
    history = read(verify(pin))
    require(history["release"] == reg["published_catalog"] and history["updates"] == 0
            and history["query_access"] is False, "Historical role or scope differs")
    rows = []
    aliases = {"I0": "M00", "I2": "M10", "I6": "M01"}
    for row in history["cases"]:
        seed, origin = row["seed"], row["arm"]
        if origin not in aliases or (origin == "I6" and seed != 2048):
            continue
        arm = aliases[origin]
        original = read(verify(row["origin"]))
        _qualified_phases(original, reg["devices"][str(seed)])
        require(row["device"] == reg["devices"][str(seed)]
                and row["initial"] == original["init"]
                and all(row["input"][k] == v for k, v in history["supervision"].items()),
                "Historical device or released supervision differs")
        match_initial(row["initial"], common[str(seed)], arm)
        rows.append({**row, "arm": arm, "reused": True, "steps": original["steps"],
                     "initialization_tensors": common[str(seed)]["tensors"]})
    require(len(rows) == 7, "Incorrect predeclared historical roster")
    return sorted(rows, key=lambda r: (r["seed"], r["arm"]))


def preflight(args, reg, reg_pin, common, code):
    seed = args.seed
    gpu = reg["devices"][str(seed)]
    counts = Counter()
    rows = []
    with physical_lease(gpu), p4.arithmetic(reg["arithmetic"], gpu, kernel_calls=counts,
                                           enforce_resource_thresholds=False):
        card = attest(gpu)
        for case in reg["cases"]:
            if case["seed"] != seed:
                continue
            row = audit_case(reg, case, args.out.parent / "new-preflight" / str(seed) / case["arm"], gpu, real=True)
            match_initial(row["init"], common[str(seed)], case["arm"])
            rows.append(row)
    require(counts["sparse_mm"] > 0 and all(r["optimizer_updates"] == 0 for r in rows),
            "The preregistered GPU did not exercise the real sparse forward")
    write_json(args.out, {"producer": identity(code), "registration": reg_pin, "device": card, "seed": seed,
                         "rows": rows, "kernel_calls": dict(counts), "updates": 0, "query_access": False})


def independent_replay(tree, result, row, labels):
    route_paths = (tree / "whole", tree / "split")
    fields = ("model", "optimizer", "scheduler", "history", "trace", "rng", "best_model", "best_probabilities")
    refs = {}
    for tag, paths in (("step_one", tuple(p / "state-000001.pt" for p in route_paths)),
                       ("terminal", tuple(p / "state-000012.pt" for p in route_paths))):
        pins = [pinned(path) for path in paths]
        saved = [terminal(path.parent) if tag == "terminal" else torch.load(verify(pin), map_location="cpu", weights_only=True)
                 for path, pin in zip(paths, pins, strict=True)]
        before, after = ([_tree_fingerprint(val[k]) for k in fields] for val in saved)
        require(before == after, f"Independent exact {tag} restoration failed")
        if tag == "terminal":
            require(min(saved[1]["history"], key=lambda x: x["val_loss"])["step"] == saved[1]["best_step"]
                    == row["steps"]["selected"], "Independent first-min selection differs")
        refs[tag] = {"files": pins, "same_state_by_field": dict(zip(fields, before, strict=True))}
    path = tree / "split" / "selected-probabilities.pt"
    pin = {"path": str(path), "sha256": result["artifacts"][path.name]}
    saved = torch.load(verify(pin), map_location="cpu", weights_only=True)["xuzhou"]["val"]
    require(tuple(saved) == tuple(HEADS), "Unexpected stored history heads")
    for head, probabilities in saved.items():
        numeric = scores(probabilities, labels)
        phases = ("selected", "last") if row["steps"]["selected"] == row["steps"]["last"] else ("selected",)
        require(all(numeric["probability_sha256"] == row["endpoints"][phase][head]["probability_sha256"]
                    and numeric["points"] == row["endpoints"][phase][head]["points"] for phase in phases),
                "Stored probability point scores differ from WP04 validation")
    return {**refs, "selected_probabilities": pin}


def assemble(args, reg, reg_pin, previous, common, code):
    data = read(verify(args.evidence))
    expected = roster(MATCHED_GROUP)
    require(data["registration"] == reg_pin and data["scope"] == reg["scope"] == MATCHED_SCOPE
            and reg["cases"] == expected and data["code"] == reg["code"] and data["devices"] == reg["devices"]
            and data["real_updates"] == reg["real_updates_authorized"] == 120
            and data["research_updates"] == 60 and data["query_access"] is False
            and not any(data["lane_exit_codes"].values()), "Invalid trained matrix receipt")
    for pin in data["tables"]:
        verify(pin)
    roles, input_hashes, labels = published_roles(reg, Path(reg_pin["path"]).parent.parent)
    new = []
    for planned, pin in zip(expected, data["case_pins"], strict=True):
        row = read(verify(pin))
        seed, arm = planned["seed"], planned["arm"]
        gpu = reg["devices"][str(seed)]
        require(row["case"] == planned, "Unexpected new case")
        _qualified_phases(row, gpu)
        match_initial(row["init"], common[str(seed)], arm)
        tree = Path(pin["path"]).parent
        result_pin = pinned(tree / "split" / "result.json")
        result = read(verify(result_pin))
        require(identity(result["start_code"]) == data["code"]
                and result["selected_step"] == row["steps"]["selected"]
                and all(row["inputs"][k] == v for k, v in input_hashes.items()), "Selected checkpoint lineage differs")
        proof = independent_replay(tree, result, row, labels)
        new.append({"seed": seed, "arm": arm, "origin": pin, "reused": False, "device": gpu,
                    "input": row["inputs"], "shared": row["init"]["shared"], "initial": row["init"],
                    "result": result_pin, "independent_recovery": proof,
                    "initialization_tensors": common[str(seed)]["tensors"], "phases": row["phases"],
                    "activity": row["learned_components"], "steps": row["steps"], "contract": contract(result, seed),
                    "endpoints": {e: {h: _recompute(row["endpoints"][e][h]) for h in HEADS}
                                  for e in ("selected", "last")}})
    rows = previous + new
    cases = {(r["seed"], r["arm"]): r for r in rows}
    require(len(cases) == len(rows) == 12, "Factorial must have four cells for each of three seeds")
    graph_identities = {}
    for seed in SEEDS:
        cohort = [cases[seed, a] for a in MATCHED_PARAMS]
        for key in ("input", "shared", "contract", "device", "initialization_tensors"):
            require(all(r[key] == cohort[0][key] for r in cohort), f"Unmatched factorial field: {key}")
        active = [read(verify(cases[seed, a]["origin"]))["diagnostics"] for a in ("M01", "M11")]
        graph_identities[str(seed)] = {}
        for e in ("initial", "selected", "last"):
            before, after = (d[e] for d in active)
            for key in ("support_sha256", "prior_sha256", "candidate_edges"):
                require(before[key] == after[key], "Matched edge cells changed positive support or row prior")
            graph_identities[str(seed)][e] = {k: before[k] for k in ("support_sha256", "prior_sha256", "candidate_edges")}
        for arm in ("M01", "M11"):
            item = cases[seed, arm]
            if not item["reused"]:
                for name, value in item["activity"].items():
                    require((value is not None) == (name == "edges" or arm == "M11"),
                            "An inactive component entered the optimizer or an active one has no trace")
    effects = {}
    for name, weights in DESIGNS.items():
        endpoints = _factorial(cases, weights)
        values = list(endpoints["selected"]["fused"].values())
        means = {k: mean(v[k] for v in values) for k in ("delta_oa", "delta_aa", "delta_nll")}
        recalls = [mean(v["correct_delta_by_class"][i] / 5 for v in values) for i in range(9)]
        failed = [i for i, diff in enumerate(recalls, 1) if diff < -.02]
        positive = sum(v["delta_oa"] > 0 for v in values)
        effects[name] = {"coefficients": weights, "endpoints": endpoints, "mean": means,
                         "per_class_mean_recall_delta": recalls, "failed_classes": failed,
                         "positive_seeds": positive, "primary": name != "interaction",
                         "passes": name != "interaction" and means["delta_oa"] > 0 and positive >= 2
                         and means["delta_aa"] >= -.005 and not failed}
    endpoints = {k: r["endpoints"] for k, r in cases.items()}
    conditional = {f"{a}-{b}": _comparison(endpoints, a, b, primary=False) for a, b in CONDITIONAL}
    args.out.mkdir(mode=0o700)
    write_json(args.out / "supplement.json", {"schema": "wp12_wp04_common_initialization_joint_factorial_v1",
               "producer": data["code"], "adapter": identity(code), "registration": reg_pin, "history": args.history,
               "primary": args.evidence, "common_initialization": reg["common_initialization"], "devices": reg["devices"],
               "released_role_overlay": roles,
               "new_cases": 5, "reused_cases": 7, "research_updates": 60, "restore_replay_updates": 60,
               "query_access": False, "production_accuracy_qualified": False, "released_points_per_class": 5,
               "edges": graph_identities, "effects": effects, "conditional": conditional, "cases": rows})
    for filename, by_class in (("paper-effects.csv", False), ("paper-classes.csv", True)):
        fields = ["effect", "endpoint", "head", "seed", "denominator",
                  *(("class_id", "delta_correct") if by_class else ("delta_oa", "delta_aa", "delta_nll"))]
        with (args.out / filename).open("x", encoding="utf8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for name, effect in effects.items():
                for end, heads in effect["endpoints"].items():
                    for h, seeds in heads.items():
                        for seed, value in seeds.items():
                            if by_class:
                                writer.writerows((name, end, h, seed, 5, i, c)
                                                 for i, c in enumerate(value["correct_delta_by_class"], 1))
                            else:
                                writer.writerow((name, end, h, seed, 45, value["delta_oa"],
                                                 value["delta_aa"], value["delta_nll"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--registration", nargs=2, required=True)
    parser.add_argument("--history", nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int, choices=SEEDS)
    mode.add_argument("--evidence", nargs=2)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    require(code["commit"] == args.expected_commit, "Wrong audited source")
    args.out = external_directory(args.out)
    require(not args.out.exists(), "Refusing to overwrite an audit")
    for key in ("registration", "history", "evidence"):
        value = getattr(args, key)
        if value:
            setattr(args, key, dict(zip(("path", "sha256"), value, strict=True)))
    reg = read(verify(args.registration))
    require(reg["scope"] == MATCHED_SCOPE and reg["group"] == MATCHED_GROUP and reg["phase_updates"] == STEPS,
            "Wrong bounded factorial registration")
    torch.set_num_threads(1)
    common = common_tensors(reg)
    prior = historical(reg, args.history, common)
    if args.seed is not None:
        require(reg["code"] == identity(code), "The preflight must bind the training source")
        preflight(args, reg, args.registration, common, code)
    else:
        assemble(args, reg, args.registration, prior, common, code)


if __name__ == "__main__":
    main()
