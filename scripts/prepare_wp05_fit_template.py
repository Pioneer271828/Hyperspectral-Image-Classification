"""Project authorized HongHu fitting inputs into a zero-update evidence template."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.research_config import digest
from research_runtime import code_snapshot, external_directory, write_json
from transfer import p4_coverage_batch as batch, p4_independent_registration as source
from transfer import p4_three_arm_contract as fit, p4_three_arm_review as review


FIT_COMMIT = "65dd7864ad2d54e658b1efb91e408bb99e687c04"
FIT_SCOPE_SHA = "0676d13736c0c6be4106171205e873fde9d81e9124245a9d0fe5dbb6244b2ca8"
PREFLIGHT_SHA = "7e41acc2d9b0359a2be03aebcb0f297d11239418e6285c7fa3148739513721ba"


def template(fit_scope, preflight):
    source.require(fit_scope["sha256"] == FIT_SCOPE_SHA and preflight["sha256"] == PREFLIGHT_SHA,
                   "Expected the original scope and independent HongHu preflight")
    scope, proof = source.read(fit_scope), source.read(preflight)
    pins = scope["protected_files"]
    batch.check_pins(pins)
    source.require(scope["commit"] == proof["tested"] == FIT_COMMIT and proof["scope_sha256"] == FIT_SCOPE_SHA
                   and proof["status"] == "passed" and proof["tiles"] == 137
                   and scope["preflight"]["optimizer_updates"] == proof["optimizer_updates"] == 0
                   and proof["raw_gt_loaded"] is proof["query_accessed"] is False
                   and scope["fit"]["release"] is None and scope["synthetic"] is False
                   and scope["budget"]["planned_optimizer_updates"] == 13500
                   and set(proof["conditions"]) == {"train/training_stride", *(f"val/{n}" for n in fit.CONDITIONS)},
                   "Unreleased zero-update 137-tile preflight required")
    path = scope["registration"]
    registration = source.read({"path": path, "sha256": pins[path]})
    plan_path = registration["exposure"]["plan"]
    plan = source.read({"path": plan_path, "sha256": pins[plan_path]})
    release_paths = {registration["cases"][case["index"]]["release"] for case in scope["cases"]}
    source.require(len(release_paths) == 1, "Fitting cases must share one support and devval release")
    release_path, = release_paths
    release = source.read({"path": release_path, "sha256": pins[release_path]})
    exposure = scope["exposure"]["s2048-k5"]
    cohort, = (p for p in plan["cohorts"] if (p["support_seed"], p["k"]) == (2048, 5))
    labels = release["support"]["labels"]
    devval = release["validation"]["labels"]
    steps = scope["budget"]["validation_steps"]
    source.require(pins[release_path] == fit.EXPECTED_RELEASE_SHA == exposure["release_sha256"]
                   and digest(release["support"]) == exposure["support_order_sha256"]
                   and digest(cohort["tile_sequence"]) == exposure["sequence_sha256"] == fit.EXPECTED_SEQUENCE_SHA
                   and len(labels) == len(devval) == 110 and steps == list(range(30, 1501, 30)),
                   "Original released roles, exposure order or selection events differ")
    per_class = []
    for label in range(1, 23):
        members = [i for i, value in enumerate(labels) if value == label]
        support = devval.count(label)
        source.require(len(members) == support == 5, "Original per-class train or devval quotas changed")
        counts = []
        for step in steps:
            samples = exposure["per_label_counts_at_events"][str(step)]
            source.require(len(samples) == 110, "Per-label exposure serial differs")
            counts.append(sum(samples[i] for i in members))
        per_class.append({"class_id": label, "train_support": len(members), "devval_support": support,
                          "planned_train_exposures_in_validation_order": counts,
                          "small_class_protection": len(members) <= fit.PROTECTION["max_small_train_support"]})
    cases = []
    for case in scope["cases"]:
        cases.append({"id": case["id"], "arm": case["arm"], "model_seed": case["model_seed"],
                      "physical_uuid": case["physical_uuid"], "planned_updates": 1500, "actual_updates": 0,
                      "selected_step": None, "train_loss": None, "tail_train_loss_slope_per_100_steps": None,
                      "tail_devval_fused_nll_slope_per_100_steps": None, "scheduler_history": None,
                      "actual_training_exposure": None, "status": "NOT_RUN",
                      "devval": {condition: {head: None for head in review.HEADS} for condition in fit.CONDITIONS}})
    source.require(len(cases) == 9 and sum(c["planned_updates"] for c in cases) == 13500,
                   "Unexpected fitting matrix")
    return {"schema": "wp05_honghu_fit_input_template_v1", "status": "NOT_RUN",
            "fit_scope": fit_scope, "independent_preflight": preflight, "preflight_tiles": proof["tiles"],
            "protected_input_count": len(pins), "role": {"support_seed": 2048, "k": 5, "method": "scratch"},
            "release_sha256": pins[release_path], "sequence_sha256": exposure["sequence_sha256"],
            "validation_steps": steps, "selection": scope["budget"]["selection"],
            "selection_condition": scope["fit"]["selection_condition"], "small_class_protection": fit.PROTECTION,
            "tail_definition": "Least-squares slope per 100 updates over the last fifth (rounded up)",
            "shared_class_inputs": per_class, "cases": cases, "raw_gt_loaded": False, "query_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for key in ("fit-scope", "preflight", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    source.require(not code["dirty"] and code["commit"] == args.expected_commit, "Unexpected template source")
    out = external_directory(args.out)
    source.require(not out.exists(), "Refusing to overwrite a previous fitting template")
    row = template(source.pin(args.fit_scope), source.pin(args.preflight))
    write_json(out, row)
    print(source.pin(out), flush=True)


if __name__ == "__main__":
    main()
