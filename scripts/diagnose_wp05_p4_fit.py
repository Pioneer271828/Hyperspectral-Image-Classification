"""Explain the bounded P4 fit from saved states and released development labels."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluation.classification_metrics import metrics_from_confusion
from research_runtime import code_snapshot, require_frozen_code, write_json
from train_spatial import SpatialConfig
from transfer import p4_coverage_batch as batch, p4_coverage_review as prior
from transfer import p4_full_budget_audit as audit, p4_full_budget_campaign as campaign
from transfer import p4_full_budget_registration as registration, p4_guarded
from transfer import p4_independent_registration as scope, p4_runtime_device as device
from transfer.p4_frozen_diagnostic_metrics import recompute
from transfer.p4_spatial import initialize
from scripts.review_wp05_p4_real import case_labels


def scores(values, labels):
    result = recompute(values.numpy(), labels)
    values = values.numpy()
    picked = values.argmax(axis=1)
    result.update(predicted_classes=sorted({int(i)+1 for i in picked}),
                  mean_max_probability=float(values.max(axis=1).mean()),
                  max_probability=float(values.max()),
                  independent_class_metrics=metrics_from_confusion(result["confusion"], range(1, 23)))
    return result


def case_diagnostic(folder, case, science, reg, sha, accepted):
    state, result, calls, _ = audit.train_state(folder / "split", science, reg, case, sha,
                                               accepted["numeric"], path_name="split")
    labels = case_labels(reg, case, science["exposure"][f's{case["support_seed"]}-k{case["k"]}'])
    plan, geometry = (scope.read({"path": reg[k], "sha256": reg["pins"][reg[k]]})
                      for k in ("plan", "development"))
    _, allowed = p4_guarded.bind_inputs(reg, plan, geometry)
    release = prior.released_roles(reg, plan, geometry, allowed, case)
    train = dict((row["id"], counts) for row, counts in release["train"])
    tile_exposure = Counter(entry["tile"] for entry in state["trace"])
    visits = np.zeros((len(state["trace"]), 22), dtype=np.int64)
    for i, step in enumerate(state["trace"]):
        visits[i] = np.bincount(train[step["tile"]].ravel(), minlength=23)[1:]
    exposed = {str(stop): visits[:stop].sum(axis=0).tolist() for stop in (30, 60)}
    covered = {str(stop): int(np.count_nonzero(visits[:stop].sum(axis=0))) for stop in (30, 60)}
    source = reg["source"] if case["method"] == "shared_transfer" else None
    fresh, initial = initialize(SpatialConfig(reg["config"]), source["weight"] if source else None,
                                source["result"] if source else None, coverage_arm=case["arm"],
                                model_seed=case["model_seed"])
    scope.require(initial == case["initialization"], "Fresh model or loaded source differs from its registration")
    head_layout = {name: list(value.shape) for name, value in fresh.state_dict().items()
                   if name.startswith(("heads.", "hgnn_heads.", "cnn_heads."))}
    changed = [name for name in head_layout if not torch.equal(fresh.state_dict()[name], state["model"][name])]
    domain = {name: {"shape": list(tensor.shape), "changed": not torch.equal(tensor, state["model"][name])}
              for name, tensor in fresh.state_dict().items() if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    _, _, class_pop = registration.fixed_population(release["val"], allowed, "val", plan["shape"][1], 5)
    chosen = state["best"]
    classes = list(range(1, 23))
    events = [{"step": row["step"], "metrics": row["metrics"],
               "validation_payload": "selected_probabilities_available" if row["step"] == chosen["step"]
                                     else "committed_metrics_only"} for row in state["history"]]
    roles = {}
    for role in ("train", "val"):
        tiles = release[role]
        layout, local_labels, population = registration.fixed_population(
            tiles, allowed, role, plan["shape"][1], case["k"] if role == "train" else 5)
        roles[role] = {"tiles": [tile["id"] for tile, _ in layout], "population": population,
                       "class_support": np.bincount(local_labels, minlength=23)[1:].tolist()}
    return {"case": case["id"], "class_to_column": {str(label): label-1 for label in classes},
            "selection_step": chosen["step"], "roles": roles, "fixed_validation": class_pop,
            "seed": case["model_seed"], "actual_optimizer_updates_on_one_path": len(state["trace"]),
            "events": events, "selected": {head: scores(p, labels) for head, p in chosen["scores"].items()},
            "first_loss": state["trace"][0]["loss"], "last_loss": state["trace"][-1]["loss"],
            "source_initialization": initial, "head_layout": head_layout, "target_head_parameters_changed": changed,
            "branch_norm": reg["config"]["model"].get("norm_type"), "normalization": reg["statistics"],
            "domain_buffer_changes": domain, "class_exposure": exposed, "class_coverage": covered,
            "tile_visits": {str(k): v for k, v in sorted(tile_exposure.items())},
            "parameter_update": accepted["parameter_audit"], "optimizer_updates": 0,
            "observed_probability_source": "recorded_selected_checkpoint"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for arg in ("scope", "release", "out"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    scope.require(code["commit"] == args.expected_commit and not args.out.exists(), "Unexpected diagnostic source or output")
    item, release = scope.pin(args.scope), scope.pin(args.release)
    resource = scope.read(item)
    science = scope.validate(resource, live_worker=True)
    scope.validate_release(item, resource, release)
    out = Path(resource["execution"])
    reg = scope.read({"path": science["registration"], "sha256": science["protected_files"][science["registration"]]})
    sha = resource["science"]["sha256"]
    with device.lease(out / ".coordinator.lock"):
        owner = campaign.read(out / "resource-owner.json")
        scope.require(owner["release"] == release and owner["resource"] == item, "Run ownership differs")
        rows = []
        total = 0
        for case in science["cases"]:
            folder = campaign.bound_folder(out, case)
            total += campaign.case_progress(science, out, case, sha)
            if (folder / "audit.json").is_file():
                accepted = campaign.accepted(folder, science, case, sha)
                rows.append(case_diagnostic(folder, case, science, reg, sha, accepted))
        batch.check_pins(science["protected_files"])
        args.out.mkdir()
        write_json(args.out / "diagnostic.json", {"schema": "wp05_p4_fit_sufficiency_v1", "producer": code["commit"],
                   "resource": item, "release": release, "cases": rows, "actual_scope_updates": total,
                   "complete_cases": len(rows), "planned_cases": 20, "validation_steps": [30, 60],
                   "new_optimizer_updates": 0, "raw_gt_loaded": False, "query_accessed": False})
    print(scope.pin(args.out / "diagnostic.json"), flush=True)


if __name__ == "__main__":
    main()
