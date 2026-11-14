"""Read an idle P4 execution and independently recompute its admitted case results."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer import p4_coverage_batch as batch, p4_coverage_review as prior
from transfer import p4_full_budget_audit as audit, p4_full_budget_campaign as campaign
from transfer import p4_full_budget_registration as registration, p4_guarded
from transfer import p4_frozen_diagnostic_metrics as metrics
from transfer import p4_independent_acceptance as acceptance
from transfer import p4_independent_registration as scope
from transfer import p4_runtime_device as device, p4_three_arm_contract as fit


SCHEMA = "wp05_p4_real_selected_recompute_v1"
ARTIFACT_METRICS = ("oa", "aa", "kappa", "macro_f1", "nll", "recall", "confusion")
TIMING_KEYS = ("preparation_seconds", "train_with_hooks_seconds", "trace_evidence_seconds",
               "validation_seconds", "checkpoint_seconds", "end_to_end_step_seconds")


def case_labels(reg, case, budget):
    plan, geometry = (scope.read({"path": reg[k], "sha256": reg["pins"][reg[k]]})
                      for k in ("plan", "development"))
    _, allowed = p4_guarded.bind_inputs(reg, plan, geometry)
    released = prior.released_roles(reg, plan, geometry, allowed, case)
    _, labels, population = registration.fixed_population(released["val"], allowed, "val", plan["shape"][1], 5)
    scope.require(population == budget["populations"]["val"], "Original devval support changed")
    return labels


def recompute_case(folder, science, reg, case, sha, row):
    labels = case_labels(reg, case, science["exposure"][f's{case["support_seed"]}-k{case["k"]}'])
    paths, budgets = {}, {}
    total = 0
    for name in science["paths"]:
        state, result, ledger, proof = audit.train_state(folder / name, science, reg, case, sha, row["numeric"], path_name=name)
        selected = prior.chosen_event(state, result, science["budget"]["validation_steps"])
        heads = {key: metrics.recompute(value.numpy(), labels) for key, value in state["best"]["scores"].items()}
        scope.require(selected["step"] == row["selected_step"]
                      and all(metrics.equivalent(head, row["fit"]["selected"]["val"]["heads"][key])
                              for key, head in heads.items()), "Independent selected probabilities or event differ")
        scope.require([event["step"] for event in state["history"]] == [30, 60]
                      and all(torch.isfinite(value).all() for value in state["best"]["scores"].values()),
                      "Two frozen validation events and finite selected scores required")
        calls = [entry["timing"] for entry in ledger]
        stages = [entry for call in calls for entry in call["steps"]]
        lower = datetime.fromisoformat(ledger[0]["started_utc"])
        upper = datetime.fromisoformat(ledger[-1]["finished_utc"])
        paths[name] = {"state": state["fingerprints"], "selected_step": selected["step"], "heads": heads,
                       "first_loss": state["trace"][0]["loss"], "last_loss": state["trace"][-1]["loss"],
                       "validation_nll": [event["metrics"]["fused"]["nll"] for event in state["history"]]}
        budgets[name] = {"optimizer_updates": sum(call["new_optimizer_updates"] for call in ledger),
                         "completed_resume_updates": proof["new_optimizer_updates"],
                         "recorded_training_call_seconds": sum(call["wall_seconds"] for call in ledger),
                         "training_start": lower.isoformat(), "training_end": upper.isoformat(),
                         "elapsed_between_first_and_last_training_calls": (upper-lower).total_seconds(),
                         "startup_seconds": sum(call["startup_seconds"] for call in calls),
                         "checkpoint_recovery_seconds": sum(call["reload_seconds"] for call in calls),
                         "completed_resume_seconds": proof["timing"]["call_seconds"],
                         "sum_synchronized_step_phases": {k: sum(entry[k] for entry in stages) for k in TIMING_KEYS}}
        total += budgets[name]["optimizer_updates"]
    scope.require(paths["full"] == paths["split"] and total == 120, "Paired scientific output or update budget differs")
    return {"case": case, "optimizer_updates": total, "paired": paths["split"], "paths": budgets,
            "audit_wall_seconds": row["audit_wall_seconds"], "audit_forwards": row["actual_forward_calls"],
            "audit_updates": 0, "parameters": row["parameter_audit"], "coverage": row["coverage_participation"],
            "exposure": row["exposure"], "populations": row["populations"], "audit": scope.pin(folder / "audit.json")}


def review(item, release, output, code, *, published=None):
    start = perf_counter()
    resource = scope.read(item)
    science = scope.validate(resource, live_worker=True)
    scope.require(resource["mode"] == scope.REAL, "Only the approved real P4 scope enters this review")
    scope.validate_release(item, resource, release)
    out = Path(resource["execution"])
    sha = resource["science"]["sha256"]
    source = science["registration"]
    reg = scope.read({"path": source, "sha256": science["protected_files"][source]})
    with device.lease(out / ".coordinator.lock"):
        scope.require(campaign.read(out / "resource-owner.json") == {
            "resource": item, "worker_owner": campaign.owner(out, resource["science"]["path"], sha),
            "release": release}, "Review owner differs")
        batch.check_pins(science["protected_files"])
        cases, slot_stats = [], {}
        for name, slot in resource["slots"].items():
            committed, audited = 0, 0
            for case in (row for row in science["cases"] if row["id"] in slot["cases"]):
                folder = campaign.bound_folder(out, case)
                committed += campaign.case_progress(science, out, case, sha)
                if not (folder / "audit.json").is_file():
                    continue
                row = campaign.accepted(folder, science, case, sha)
                cases.append(recompute_case(folder, science, reg, case, sha, row))
                audited += 1
            slot_stats[name] = {"planned_updates": slot["updates"], "committed_updates": committed,
                                "planned_cases": len(slot["cases"]), "audited_cases": audited}
        total = sum(row["committed_updates"] for row in slot_stats.values())
        complete = len(cases) == 20 and total == 2400
        if published is not None:
            scope.require(complete, "Partial run cannot have the full acceptance")
            reconstructed = acceptance.reconstruct(item, resource, science, release=release)
            scope.require(scope.read(published) == reconstructed, "Independent acceptance differs from publication")
        first = min((p["training_start"] for c in cases for p in c["paths"].values()), default=None)
        last = max((p["training_end"] for c in cases for p in c["paths"].values()), default=None)
        timing = {"first_completed_case_training_start": first, "last_completed_case_training_end": last,
                  "completed_case_elapsed_span_seconds": None if not first else (
                      datetime.fromisoformat(last)-datetime.fromisoformat(first)).total_seconds(),
                  "completed_case_worker_call_seconds": sum(p["recorded_training_call_seconds"]
                      for c in cases for p in c["paths"].values()),
                  "completed_case_resume_seconds": sum(p["completed_resume_seconds"] for c in cases for p in c["paths"].values()),
                  "completed_case_audit_seconds": sum(c["audit_wall_seconds"] for c in cases),
                  "resource_waiting_seconds": None}
        result = {"schema": SCHEMA, "status": "accepted" if published else "partial",
                  "producer": code["commit"], "resource": item, "release": release, "acceptance": published,
                  "scientific_scope": resource["science"], "worker": scope.identity(resource["worker"]),
                  "observed_cases": len(cases), "planned_cases": 20, "committed_optimizer_updates": total,
                  "audit_optimizer_updates": 0, "slots": slot_stats, "cases": cases, "timing": timing,
                  "input_pins": science["protected_files"], "review_seconds": perf_counter()-start,
                  "raw_gt_loaded": False, "query_accessed": False}
        batch.check_pins(science["protected_files"])
        require_frozen_code(code)
        output.mkdir()
        write_json(output / "review.json", result)
        with (output / "per-head.csv").open("x", encoding="utf-8", newline="") as stream:
            table = csv.writer(stream)
            table.writerow(["coverage", "k", "method", "arm", "head", "selected_step",
                            *ARTIFACT_METRICS[:5], *(f"recall_{i}" for i in range(1, 23))])
            for row in cases:
                c, paired = row["case"], row["paired"]
                for name, values in paired["heads"].items():
                    table.writerow([result["status"], c["k"], c["method"], c["arm"], name, paired["selected_step"],
                                    *(values[k] for k in ARTIFACT_METRICS[:5]), *values["recall"]])
    return scope.pin(output / "review.json")


def fit_draft(item, proof, template_pin, destination, code, *, review_pin):
    resource = scope.read(item)
    template, independent = scope.read(template_pin), scope.read(review_pin)
    scope.require(template["schema"] == "wp05_honghu_fit_input_template_v1" and template["status"] == "NOT_RUN"
                  and len(template["cases"]) == 9 and sum(v["planned_updates"] for v in template["cases"]) == 13500,
                  "Original nine-case fitting template required")
    scope.require(independent["status"] == "accepted" and independent["acceptance"] == proof
                  and independent["resource"] == item, "Accepted independent P4 review required")
    pins = acceptance.predecessor(item, proof, resource["controller"])
    # Approval and fresh source-bound qualifications are inserted only by the
    # later release owner after reviewing this draft.
    return {"schema": "wp05_three_arm_execution_release_v2", "decision": "pending_review",
            "research_issue": 470, "source_commit": code["commit"], "source_hash": code["source_hash"],
            "gpus": list(resource["slots"]), "execution": str(destination),
            "qualification_sha256": None, "admission_scope": item, "admission_acceptance": proof,
            "admission_controller": resource["controller"], "planned_optimizer_updates": 13500,
            "prerequisites": {"fresh_fitting_arithmetic_qualifications": list(resource["slots"]),
                              "verified_predecessor_inputs": len(pins), "independent_review": review_pin},
            "fit_template": template_pin, "validation_steps": template["validation_steps"],
            "primary_conditions": list(fit.UNSEEN), "protection": fit.PROTECTION,
            "estimated_optimizer_seconds_per_gpu_from_p4": {
                gpu: sum(c["paths"]["full"]["sum_synchronized_step_phases"]["train_with_hooks_seconds"]
                         for c in independent["cases"] if c["case"]["physical_uuid"] == gpu) /
                     (60 * len(resource["slots"][gpu]["cases"])) *
                     (9000 if i == 0 else 4500)
                for i, gpu in enumerate(resource["slots"])},
            "cost_boundary": "Historical P4 per-update train-with-hooks proxy; real fitting end-to-end unmeasured"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for k in ("scope", "release", "out"):
        parser.add_argument("--" + k, type=Path, required=True)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--fit-template", type=Path)
    parser.add_argument("--fit-execution", type=Path)
    args = parser.parse_args()
    code = code_snapshot()
    scope.require(code["commit"] == args.expected_commit, "Independent review source differs")
    require_frozen_code(code)
    item, release = scope.pin(args.scope), scope.pin(args.release)
    published = scope.pin(args.acceptance) if args.acceptance else None
    dest = external_directory(args.out)
    result = review(item, release, dest, code, published=published)
    if args.fit_template:
        scope.require(published is not None and args.fit_execution is not None and not args.fit_execution.exists(),
                      "Fitting draft requires accepted P4 and a fresh authorized directory")
        draft = fit_draft(item, published, scope.pin(args.fit_template), args.fit_execution, code, review_pin=result)
        write_json(dest / "fit-release-review-draft.json", draft)
    print(result, flush=True)


if __name__ == "__main__":
    main()
