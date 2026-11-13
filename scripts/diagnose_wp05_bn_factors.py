"""Factor frozen HongHu DomainBN buffer identity without training or calibration."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation import p4_bn_factors as bn
from research_runtime import code_snapshot, require_frozen_code, write_json
from scripts.diagnose_wp05_p4_fit import bounded_scene
from transfer import p4_coverage_batch as batch, p4_coverage_review as prior
from transfer import p4_frozen_diagnostics as diagnostic, p4_full_budget_audit as audit
from transfer import p4_full_budget_campaign as campaign, p4_full_budget_registration as registration
from transfer import p4_independent_registration as scope, p4_runtime_device as resource
from transfer.p4_three_arm_contract import PROTECTION


SCHEMA = "wp05_p4_bounded_domainbn_factor_v1"
CASES = ("s2048-k1-shared_transfer-m512-none", "s2048-k1-shared_transfer-m512-continuous")


def protection(by_head, baseline, train_support):
    result = {}
    for head, row in by_head.items():
        new, old = row["metrics"], baseline[head]["metrics"]
        delta = {field: 100*(new[field]-old[field]) for field in ("oa", "aa")}
        recall = [100*(a["recall"]-b["recall"]) for a, b in zip(new["per_class"], old["per_class"], strict=True)]
        failures = [i for i, change in enumerate(recall, 1) if train_support <= PROTECTION["max_small_train_support"]
                    and change < -PROTECTION["max_small_recall_drop_pp"]-1e-10]
        result[head] = {"delta_pp": delta, "recall_delta_pp": recall, "small_class_failure": failures,
                        "passes": all(x >= -PROTECTION["max_oa_aa_drop_pp"]-1e-10 for x in delta.values()) and not failures}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for field in ("scope", "release", "baseline", "out"):
        parser.add_argument("--"+field, type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    bn.require(code["commit"] == args.expected_commit and not args.out.exists(), "Expected frozen diagnostic source")
    torch.set_num_threads(1)
    p, auth, baseline = map(scope.pin, (args.scope, args.release, args.baseline))
    source = scope.read(baseline)
    matrix = scope.read(p)
    sc = scope.validate(matrix, live_worker=True)
    scope.validate_release(p, matrix, auth)
    bn.require(source["schema"] == "wp05_p4_fit_sufficiency_v1"
               and source["producer"] == "b02104f716b6e1495772b8376c94dc2d631e0c7c"
               and source["resource"] == p and source["release"] == auth
               and source["new_optimizer_updates"] == 0, "Wrong prior bounded fitting diagnostic")
    remembered = {r["case"]: r for r in source["cases"]}
    out = Path(matrix["execution"])
    sha = matrix["science"]["sha256"]
    reg = scope.read({"path": sc["registration"], "sha256": sc["protected_files"][sc["registration"]]})
    pin_for = lambda path: {"path": reg[path], "sha256": reg["pins"][reg[path]]}
    exposure_file = reg["exposure"]["plan"]
    exposure = scope.read({"path": exposure_file, "sha256": reg["pins"][exposure_file]})
    plan = scope.read(pin_for("plan"))
    started = perf_counter()
    records = []
    with resource.lease(out / ".coordinator.lock"):
        owner = campaign.read(out / "resource-owner.json")
        bn.require(owner["resource"] == p and owner["release"] == auth, "Original run owner changed")
        geo, allowed, scene, frozen, image = bounded_scene(reg, plan, exposure, out.parent)
        for case in sc["cases"]:
            if case["id"] not in CASES:
                continue
            case_tick = perf_counter()
            folder = campaign.bound_folder(out, case)
            accepted = campaign.accepted(folder, sc, case, sha)
            state, *_ = audit.train_state(folder / "split", sc, reg, case, sha, accepted["numeric"], path_name="split")
            checkpoint = scope.pin(folder / "split" / "state.pt")
            original = remembered[case["id"]]
            bn.require(original["bounded_observation"] == image, "Observation pin differs from prior diagnostic")
            model, init = registration.initialize_case(reg, case)
            bn.require(init == case["initialization"], "Invalid source initialization")
            model.eval()
            initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
            release = prior.released_roles(reg, plan, geo, allowed, case)
            data = {}
            for role, quota in (("train", case["k"]), ("val", 5)):
                layout, labels, population = registration.fixed_population(release[role], allowed, role, plan["shape"][1], quota)
                bn.require(population == original["roles"][role]["population"], "Released pixel roster changed")
                condition = {"sampling": frozen["records"][role], "metadata": "nominal"}
                data[role] = diagnostic.fixed_views(scene, allowed, layout, condition, role), labels
            observed = {}
            for tag, weights in (("initial", initial), ("selected", state["best"]["model"]), ("last", state["model"])):
                model.load_state_dict(weights, strict=True)
                measured = bn.factor_state(model, initial, scene, data)
                base = measured["arms"]["original"]["probability_metrics"]
                for role in data:
                    for head in bn.HEADS:
                        old = original["frozen_cpu_readout"][tag][role]["metrics"][head]
                        new = base[role]["heads"][head]
                        bn.require(abs(old["nll"]-new["nll"]) < 1e-10
                                   and old["confusion"] == new["metrics"]["confusion_matrix"],
                                   "Frozen baseline differs from the independent bounded forward")
                for item in measured["arms"].values():
                    item["devval_protection"] = protection(item["probability_metrics"]["val"]["heads"],
                                                           base["val"]["heads"], case["k"])
                observed[tag] = measured
            bn.require(scope.pin(folder / "split" / "state.pt") == checkpoint, "Frozen checkpoint bytes changed")
            records.append({"case": case["id"], "source": checkpoint, "initialization": init,
                            "selected_step": state["best"]["step"], "updates_in_original_path": state["cursor"],
                            "states": observed, "wall_seconds": perf_counter()-case_tick})
        bn.require([row["case"] for row in records] == list(CASES), "Expected the preregistered bounded contrasting arms")
        batch.check_pins(sc["protected_files"])
        bn.require(scope.pin(args.baseline) == baseline, "Prior accepted diagnostic changed")
    args.out.mkdir()
    output = args.out / "diagnostic.json"
    write_json(output, {"schema": SCHEMA, "scope": "frozen_real_devtrain_devval_diagnostic", "status": "measured",
               "producer": {"commit": code["commit"], "source_hash": code["source_hash"]},
               "resource": p, "release": auth, "prior_bounded_diagnostic": baseline, "observation": image,
               "cases": records, "training_updates": 0, "persistent_state_updates": 0,
               "statistics_calibration_updates": 0, "raw_gt_loaded": False, "query_accessed": False,
               "validation_statistics_fit": False, "confirmed_metrics": None, "real_gain": None,
               "arms": bn.ARMS, "protection": PROTECTION, "wall_seconds": perf_counter()-started})
    print(scope.pin(output), flush=True)


if __name__ == "__main__":
    main()
