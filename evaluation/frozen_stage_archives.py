"""Reconcile historical stage predictions using released train/val labels only."""

import json
from pathlib import Path
import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, json_pin, pinned, require
from evaluation.g4_target import independent_trainval
from g4_matched_target_audit import AUDIT_POLICY
from g4_matched_target_study import grid
from loadData.target_label_protocol import phase_view
from model.research_config import digest


def admission(root, pins):
    require(set(pins) == {"launch", "independent", "freeze"}, "Expected the full54 launch, audit and freeze")
    launch, audit, freeze = (json_pin(root, pins[key]) for key in ("launch", "independent", "freeze"))
    require(audit.get("policy") == AUDIT_POLICY and audit.get("status") == "passed"
            and freeze.get("status") == "all54_frozen_before_query"
            and audit["launch_sha256"] == freeze["launch_sha256"] == pins["launch"]["sha256"]
            and freeze["independent_audit_sha256"] == pins["independent"]["sha256"]
            and audit["acceptance_sha256"] == freeze["acceptance_sha256"]
            and audit.get("new_optimizer_updates") == freeze.get("new_optimizer_updates") == 0
            and audit.get("query_labels_accessed") is freeze.get("query_labels_accessed") is False,
            "Independent all54 development acceptance required")
    rows = {row["name"]: row for row in freeze["models"]}
    accepted = {row["name"]: row for row in audit["cases"]}
    require(len(rows) == len(accepted) == len(freeze["models"]) == len(audit["cases"]) == 54
            and rows.keys() == accepted.keys()
            and {(row["target"], row["seed"], row["method"], row["mode"]) for row in rows.values()}
            == set(grid()), "Fixed complete model roster required")
    cases = {row["name"]: row for row in launch["cases"]}
    require(rows.keys() == cases.keys() and len(launch["cases"]) == len(cases), "Foreign launch roster")
    targets = {}
    for name, row in rows.items():
        original, independent = cases[name], accepted[name]
        context = original["target_context"]
        require(independent["status"] == "passed" and independent["updates"] == 1500
                and independent["query_labels_accessed"] is False
                and all(row[key] == independent[key] for key in
                        ("result_sha256", "primary_sha256", "sidecar_sha256", "selected_step"))
                and (row["target"], row["seed"], row["method"], row["mode"])
                == (original["target"], original["seed"], original["method"], original["mode"])
                and row["family_hash"] == context["family_hash"]
                and context["support_seed"] == 64 and context["full_target_scene_transductive"] is True
                and original["settings"]["support_per_class"] == 10
                and original["settings"]["validation_per_class"] == 5
                and original["settings"]["config"]["model"]["graph_mode"] == "static",
                "Different model/seed/support/graph family in historical freeze")
        targets[name] = dict(scene=row["target"], seed=row["seed"], method=row["method"], mode=row["mode"],
                             family=row["family_hash"], init=original["initial_state_sha256"],
                             graph=original["graph_binding"], phase_hashes=context["phase_hashes"],
                             observation=context["observation_sha256"], result=row["result_sha256"],
                             selected_checkpoint=row["primary_sha256"], selected_step=row["selected_step"])
    return dict(pins=pins, scope="original_full_band_transductive_development_only", support_seed=64,
                train_per_class=10, validation_per_class=5, budget=1500, cases=targets)


def read_stage_probabilities(path, scene):
    packet = torch.load(path, map_location="cpu", weights_only=True)
    require(set(packet) == {scene} and set(packet[scene]) == {"train", "val"}
            and all(set(row) == set(HEADS) for row in packet[scene].values()),
            "Only original train/val three-head shards may be read")
    return packet[scene]


def aggregate_metrics(metrics, classes):
    require(set(metrics) == {"train", "val"}, "Unexpected historical metric role")
    for phase, per_class in (("train", 10), ("val", 5)):
        require(set(metrics[phase]) == set(HEADS), "Unexpected historical metric head")
        for head in HEADS:
            row = metrics[phase][head]
            matrix = np.asarray(row["confusion"])
            require(matrix.shape == (classes, classes) and matrix.dtype.kind in "iu" and (matrix >= 0).all()
                    and (matrix.sum(1) == per_class).all() and np.isfinite(row["nll"]) and row["nll"] >= 0,
                    "Historical confusion or released class denominator differs")
            recall = matrix.diagonal()/per_class
            np.testing.assert_allclose(row["recall"], recall, rtol=0, atol=1e-12)
            np.testing.assert_allclose([row["oa"], row["aa"]], [matrix.trace()/matrix.sum(), recall.mean()],
                                       rtol=0, atol=1e-12)


def archive(root, plan):
    require(plan == admission(root, plan["pins"]), "Historical stage admission differs")
    launch, audit = (json_pin(root, plan["pins"][name]) for name in ("launch", "independent"))
    launch_root = Path(plan["pins"]["launch"]["path"]).parent
    archived = {row["name"]: row for row in audit["cases"]}
    label_cache = {}
    output = {}
    for case in launch["cases"]:
        name, scene = case["name"], case["target"]
        identity = plan["cases"][name]
        folder = launch_root / "cases" / name
        require(Path(case["out"]) == folder and Path(name).name == name, "Foreign archived case directory")
        result = json_pin(root, {"path": str(folder/"result.json"), "sha256": identity["result"]})

        def artifact(filename):
            return pinned(root, {"path": str(folder/filename), "sha256": result["artifacts"][filename]})

        require(result["status"] == "completed" and result["test_scoring"] is False
                and result["optimizer_steps"] == plan["budget"] and result["selected_step"] == identity["selected_step"]
                and result["artifacts"]["primary.pt"] == identity["selected_checkpoint"],
                "Archive result no longer represents the accepted planned trajectory")
        history = json.loads(artifact("history.json").read_bytes())
        require([row["step"] for row in history] == [*range(1, 1472, 30), 1500]
                and min(history, key=lambda row: row["val_loss"])["step"] == identity["selected_step"]
                and result["selected_validation"] == archived[name]["selected_validation"]
                and history[-1] == archived[name]["last_validation"], "Historical validation clock/selection differs")
        inputs = case["inputs"]
        family_pin = dict(path=inputs["family"], sha256=inputs["family_sha256"])
        if (scene, family_pin["sha256"]) not in label_cache:
            family = json_pin(root, family_pin)
            view = {role: phase_view(family, 10, role, expected_hash=identity["family"])
                    for role in ("adapt", "validation")}
            require({role: digest(value) for role, value in view.items()} == identity["phase_hashes"],
                    "Historical development family differs")
            label_cache[(scene, family_pin["sha256"])] = view
        view = label_cache[(scene, family_pin["sha256"])]
        require({role: digest(value) for role, value in view.items()} == identity["phase_hashes"],
                "Historical case binds different development points")
        truth = {key: torch.tensor(view[role]["labels"], dtype=torch.long)-1
                 for key, role in (("train", "adapt"), ("val", "validation"))}
        classes = len(view["validation"]["labels"]) // 5
        chosen = next(row for row in history if row["step"] == identity["selected_step"])
        stages = {}
        for stage, event in (("first_update", history[0]), ("mid_751", history[25]),
                             ("selected", chosen), ("last", history[-1])):
            step = event["step"]
            path = artifact(f"raw-validation-{step:06d}.pt")
            metrics = event["metrics"][scene]
            aggregate_metrics(metrics, classes)
            if stage == "selected":
                require(metrics == archived[name]["selected_validation"]["metrics"][scene],
                        "Selected event differs from accepted three-head summary")
            stages[stage] = dict(step=step, metrics=metrics, raw_state_sha256=result["artifacts"][path.name])
        stored = read_stage_probabilities(artifact("selected-probabilities.pt"), scene)
        independent_trainval(stored, dict(class_num=classes, train_labels=truth["train"],
                                         val_labels=truth["val"]), chosen["metrics"][scene])
        output[name] = dict(identity=identity, stages=stages, retained_untouched_initial_head=False,
                            masked_replay_blocked="fixed graph fit on original full spectral scene")
    return dict(scope=plan["scope"], plan_sha256=digest(plan), checked_cases=len(output), cases=output,
                checked_confusion_rows=len(output)*4*2*3, independently_rescored_probability_rows=len(output)*2*3,
                optimizer_updates=0, model_forwards=0,
                raw_gt_loaded=False, query_accessed=False)


def honghu(root, release_pin, review_pin, status_pin, alternate_pin=None):
    release = json_pin(root, release_pin)
    review = json_pin(root, review_pin)
    status = json_pin(root, status_pin)
    execution = Path(release["execution"])
    require(root in execution.parents and execution.resolve(strict=True) == execution
            and Path(status_pin["path"]) == execution/"status.json"
            and release["decision"] == "approved" and release["raw_gt_authorized"] is False
            and release["query_authorized"] is False and review["release"] == release_pin
            and review["observed_cases"] == 10 and review["planned_cases"] == 20
            and review["audit_optimizer_updates"] == 0
            and review["query_accessed"] is review["raw_gt_loaded"] is False,
            "Original Honghu scope/partial independent audit differs")
    json_pin(root, release["science_scope"])
    json_pin(root, release["resource_scope"])
    require(review["resource"] == release["resource_scope"] and review["scientific_scope"] == release["science_scope"],
            "Partial audit binds a different worker or scientific scope")
    planned = {name: device for device, value in release["slots"].items() for name in value["cases"]}
    require(len(planned) == 20 and all(Path(name).name == name for name in planned), "Expected exact twenty cases")
    accepted = {row["case"]["id"]: row for row in review["cases"]}
    for name, row in accepted.items():
        require(row["audit_updates"] == 0 and row["optimizer_updates"] == 120
                and row["case"]["physical_uuid"] == planned.get(name), "Invalid historical accepted case")
        json_pin(root, row["audit"])
    require(set(accepted) <= set(planned) and len(accepted) == 10, "Historical accepted Honghu roster differs")
    missing = {}
    for name, device in planned.items():
        if name in accepted:
            continue
        parts = {kind: sorted(p.relative_to(execution).as_posix() for p in (execution/kind/name).rglob("*") if p.is_file())
                 for kind in ("cases", "stages")}
        worker = status["slots"][device]
        require(not any(parts.values()) and worker["optimizer_updates"] == worker["accepted_cases"] == 0
                and worker["state"] == "waiting_for_resources",
                "New or incomplete historical case requires a separate authenticated original-protocol audit")
        missing[name] = dict(status="not_started", registered_physical_uuid=device, stage_files=0, case_files=0,
                             reason="original worker waited for resources; no checkpoint or completed stage exists")
    alternate = None
    if alternate_pin is not None:
        migration = json_pin(root, alternate_pin)
        destination = Path(migration["execution"])
        require(migration["decision"] == "approved" and migration["parent_resource"] == release["resource_scope"]
                and migration["raw_gt_authorized"] is migration["query_authorized"] is False
                and {name for slot in migration["slots"].values() for name in slot["cases"]} == set(missing)
                and destination.is_absolute() and destination == destination.resolve() and root in destination.parents,
                "Alternative migration binds a different roster or science release")
        require(not destination.exists(), "Migrated output exists and requires original-protocol audit")
        alternate = dict(release_sha256=alternate_pin["sha256"], execution_directory_present=False,
                         physical_uuids=sorted(migration["slots"]))
    return dict(scope_sha256=release["science_scope"]["sha256"], resource_sha256=release["resource_scope"]["sha256"],
                release=release_pin["sha256"], audit=review_pin["sha256"], status_sha256=status_pin["sha256"],
                accepted=sorted(accepted), missing=missing, alternate=alternate, added=0, optimizer_updates=0,
                query_accessed=False, raw_gt_loaded=False)
