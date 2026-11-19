"""Read a published P4 migration as evidence for two distinct executions."""

import json
from pathlib import Path

from evaluation.classification_metrics import metrics_from_confusion
from evaluation.wp12_p4_progress import case_stages, observe_lock
from research_runtime import hash_file
from transfer.development_blocks import inside, read_pin


HEADS = ("fused", "graph", "cnn")
REFERENCE = {"cases": 20, "optimizer_updates": 2400, "predecessor_compatible": False}
RELEASE = "wp05_p4_unstarted_cohort_migration_release_v1"
OWNER = "wp05_p4_resource_owner_taken_over_v1"
COHORTS = ((1, "scratch"), (5, "shared_transfer"))
ARMS = {"none", "anchor_only", "binary", "continuous", "shuffled"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(pin, root):
    return json.loads(read_pin(pin, root))


def check_files(pins, root):
    for location, expected in pins.items():
        path = inside(location, root)
        require(hash_file(path) == expected, f"Accepted P4 artifact changed: {path}")


def directory(value, root):
    path = Path(value)
    require(path == path.resolve(strict=True) and path.is_relative_to(root) and path.is_dir(),
            "Execution directory escapes the authorized root or is aliased")
    return path


def combine_cases(parent, new, kept, moved, *, synthetic):
    """The old roster governs case identity, regardless of resource scheduling."""
    case_fields = {"physical_uuid", "pair_sha256"}
    original = {case["id"]: case for case in parent["cases"]}
    incoming = {case["id"]: case for case in new["cases"]}
    original_names = {row["case"]["id"] for row in kept}
    new_names = {row["case"]["id"] for row in moved}
    require(len(original) == len(parent["cases"]) == 20 and len(incoming) == len(new["cases"]) == 10
            and len(original_names) == len(kept) == len(new_names) == len(moved) == 10
            and not original_names & new_names and original_names | new_names == original.keys()
            and new_names == incoming.keys(), "Expected ten disjoint retained and ten migrated cases")
    require(parent["budget"]["planned_optimizer_updates"] == 2400
            and new["budget"]["planned_optimizer_updates"] == 1200
            and parent["synthetic"] is new["synthetic"] is synthetic
            and all(row["optimizer_updates"] == 120 for row in (*kept, *moved)),
            "P4 combined budget or synthetic identity differs")
    for row, roster in [(v, original) for v in kept] + [(v, incoming) for v in moved]:
        case = row["case"]
        require(case == roster[case["id"]], "Accepted case differs from its own scientific scope")
        require({k: v for k, v in case.items() if k not in case_fields} == {
                    k: v for k, v in original[case["id"]].items() if k not in case_fields},
                "Migrated case or original initialization differs")
    return {"reference": dict(REFERENCE), "synthetic": synthetic, "query_authorized": False,
            "retained": kept, "migrated": moved}


def unchanged_science(parent, parent_pin, new, plan, root):
    allowed = {"cases", "budget", "gpus", "qualifications", "migration_recoveries", "parent_science", "scope_hash",
               "recovery_receipt", "commit", "source_hash", "registration", "protected_files"}
    require({k: v for k, v in parent.items() if k not in allowed} ==
            {k: v for k, v in new.items() if k not in allowed}
            and new["parent_science"] == parent_pin
            and {k: v for k, v in parent["budget"].items() if k != "planned_optimizer_updates"} ==
                {k: v for k, v in new["budget"].items() if k != "planned_optimizer_updates"}
            and all(new["protected_files"].get(path) == sha for path, sha in parent["protected_files"].items()),
            "Original scientific configuration or protected input differs")
    old_reg = read({"path": parent["registration"], "sha256": parent["protected_files"][parent["registration"]]}, root)
    new_reg = read(plan["registration"], root)
    require(plan["registration"] == {"path": new["registration"], "sha256": new["protected_files"][new["registration"]]}
            and new_reg == {**old_reg, "commit": plan["code"]["commit"]}, "Original registration or source weights differ")


def custody(plan_pin, plan, released, retained, protected, root):
    control = directory(plan["control"], root)
    origin = directory(plan["parent_execution"], root)
    dest = directory(plan["execution"], root)
    require(len({control, origin, dest}) == 3 and control.parent == origin.parent == dest.parent,
            "Migration execution and control must be distinct siblings")
    authority = read(released, root)
    owner = dict(schema=OWNER, control=str(control), parent_execution=str(origin),
                 prior_owner_sha256=plan["prior_owner_sha256"], cases=plan["cases"],
                 migration=plan_pin, authority=released)
    expected_release = dict(schema=RELEASE, decision="approved", migration=plan_pin,
                             parent_resource=plan["parent_resource"], old_resource_owner_sha256=plan["prior_owner_sha256"],
                             control=str(control), execution=str(dest), source=plan["code"], science=plan["migrated_science"],
                             slots=plan["slots"], optimizer_updates=1200, raw_gt_authorized=False, query_authorized=False,
                             synthetic=False)
    if plan.get("dispatch_mode") == "serial":
        expected_release.update(dispatch_mode="serial", destinations=plan["destinations"])
    require(authority == expected_release, "Migration lacks its separately approved 1,200-step release")
    live = origin / "resource-owner.json"
    intent = control / "takeover-intent.json"
    previous = control / "original-resource-owner.bytes"
    archived_review = control / "original-partial-review.bytes"
    def protected_document(path):
        require(str(path) in protected, "Combined acceptance is missing a custody pin")
        return read({"path": str(path), "sha256": protected[str(path)]}, root)
    # The original owner path now carries a different, uniquely bound record.
    require(protected_document(live) == owner and json.loads(intent.read_bytes()) == owner
            and intent.resolve() == intent and hash_file(intent) == hash_file(live)
            and protected.get(str(previous)) == plan["prior_owner_sha256"]
            and protected.get(str(archived_review)) == retained["review"]["sha256"],
            "Original custody marker or archived evidence differs")
    previous_owner = protected_document(previous)
    require(previous_owner["resource"] == plan["parent_resource"] and previous_owner["release"] == plan["parent_release"]
            and previous_owner["worker_owner"] == protected_document(origin / "owner.json"),
            "Archived original owner differs from the old execution")
    require(not (origin / "failure.json").exists()
            and all(not (origin / folder / case).exists() for folder in ("cases", "stages") for case in plan["cases"]),
            "An original migrated case already has execution evidence")
    return origin, dest


def published_case(pin, case, scope, resource, slot, pins, root):
    require(pin["sha256"] == pins.get(pin["path"]), "Combined acceptance is missing a case digest")
    audit = read(pin, root)
    require(audit["case"] == case and audit["scope_sha256"] == scope["sha256"]
            and audit["status"] == "passed" and audit["optimizer_updates"] == 120
            and audit["completed_resume_updates"] == 0
            and audit["raw_gt_loaded"] is audit["query_accessed"] is False
            and all(pins.get(k) == v for k, v in audit["artifact_pins"].items()),
            "Case, scientific scope, completed recovery or artifact pins differ")
    execution = Path(pin["path"]).parents[2]
    steps, complete, receipts = case_stages(execution, case["id"], resource["sha256"], scope["sha256"], slot)
    require(steps == 120 and complete == 1 and all(pins.get(v["path"]) == v["sha256"] for v in receipts),
            "Combined acceptance is missing a committed stage")
    return audit


def case_row(audit, scope, pin, source, costs):
    heads = {}
    for head in HEADS:
        score = audit["fit"]["selected"]["val"]["heads"][head]
        metrics = metrics_from_confusion(score["confusion"], range(1, 23))
        require(all(item["support"] == 5 for item in metrics["per_class"]), "P4 selected class support differs")
        heads[head] = {**score, "per_class": metrics["per_class"]}
    return {"case": audit["case"], "scope": scope, "audit": pin, "source": source, "optimizer_updates": 120,
            "selected_step": audit["selected_step"], "heads": heads, "paths": costs,
            "audit_seconds": audit["audit_wall_seconds"]}


def migrated_cost(folder, paths, pins, root):
    result = {}
    for name, stops in paths.items():
        material = folder / name
        def document(path):
            key = str(path)
            require(key in pins, "Combined acceptance is missing a migrated worker commit")
            return read({"path": key, "sha256": pins[key]}, root)
        ledger = [document(material / "calls" / f"{start:06d}-{stop:06d}.json")
                  for start, stop in zip([0, *stops[:-1]], stops, strict=True)]
        completed = document(material / "completed-resume.json")
        require(sum(c["new_optimizer_updates"] for c in ledger) == 60 and completed["new_optimizer_updates"] == 0,
                "Migrated timing does not cover 60 committed and zero re-entry steps")
        result[name] = dict(optimizer_updates=60, recorded_training_call_seconds=sum(c["wall_seconds"] for c in ledger),
                            startup_seconds=sum(c["timing"]["startup_seconds"] for c in ledger),
                            checkpoint_recovery_seconds=sum(c["timing"]["reload_seconds"] for c in ledger),
                            completed_resume_seconds=completed["timing"]["call_seconds"], completed_resume_updates=0)
    return result


def allocation(plan, new, acceptance):
    """Bind each full scientific cohort to an explicitly registered queue."""
    cases = new["cases"]
    names = [case["id"] for case in cases]
    require(len(cases) == len(set(names)) == 10 and names == plan["cases"],
            "Expected ten unique migrated cases in the registered sequence")
    destinations = []
    expected = {}
    for index, cohort in enumerate(COHORTS):
        members = cases[5 * index:5 * (index + 1)]
        gpu = members[0]["physical_uuid"]
        require({(c["k"], c["method"], c["physical_uuid"]) for c in members} == {(*cohort, gpu)}
                and {c["arm"] for c in members} == ARMS, "A migrated five-arm cohort is incomplete or split")
        destinations.append(gpu)
        slot = expected.setdefault(gpu, {"physical_uuid": gpu, "cases": [], "updates": 0})
        slot["cases"].extend(c["id"] for c in members)
        slot["updates"] += 600
    mode = plan.get("dispatch_mode", "parallel")
    require((mode, len(expected)) in (("parallel", 2), ("serial", 1))
            and plan.get("destinations", destinations if mode == "parallel" else None) == destinations
            and plan["slots"] == expected and new["gpus"] == list(expected)
            and acceptance["slots"] == {gpu: {"cases": len(slot["cases"]), "optimizer_updates": slot["updates"]}
                                       for gpu, slot in expected.items()},
            "Migrated physical allocation, dispatch mode or budget differs")
    return {name: gpu for gpu, slot in expected.items() for name in slot["cases"]}


def intake(item, parent_pin, parent_resource, parent, review_pin, release_pin, root):
    combined = read(item, root)
    require(combined.get("schema") == "wp05_p4_independent_slot_acceptance_v1"
            and combined.get("status") == "passed" and combined.get("synthetic") is False
            and combined.get("combined_reference") == REFERENCE, "Published real combined P4 acceptance is required")
    plan_pin = combined["resource_scope"]
    plan = read(plan_pin, root)
    proposal = read(plan["proposal"], root)
    original_evidence = proposal["parent_evidence"]
    retained_pin = proposal["retained_evidence"]
    retained = read(retained_pin, root)
    protected = combined["protected_files"]
    require(plan["synthetic"] is False
            and combined["raw_gt_loaded"] is combined["query_accessed"] is False
            and combined["audit_optimizer_updates"] == combined["completed_resume_updates"] == 0
            and combined["cases"] == 10 and combined["optimizer_updates"] == 1200
            and combined["retained_original"] == retained
            and plan["parent_resource"] == parent_pin and plan["parent_science"] == parent_resource["science"]
            and plan["parent_execution"] == parent_resource["execution"] and plan["parent_release"] == release_pin
            and proposal["prior_resource"] == parent_pin and proposal["prior_release"] == release_pin
            and proposal["science"] == plan["parent_science"] and proposal["execution"] == plan["execution"]
            and proposal["slots"] == plan["slots"] and original_evidence["review"] == retained["review"] == review_pin
            and proposal.get("destinations") == plan.get("destinations")
            and proposal.get("dispatch_mode", "parallel") == plan.get("dispatch_mode", "parallel")
            and original_evidence["resource_owner"]["sha256"] == plan["prior_owner_sha256"]
            and combined["science_scope"] == plan["migrated_science"] and combined["worker"] == combined["controller"] == plan["code"],
            "Combined result does not bind the original and migrated P4 executions")
    for pin in (plan_pin, combined["release"], plan["migrated_science"], retained_pin,
                original_evidence["status"], original_evidence["owner"]):
        require(protected.get(pin["path"]) == pin["sha256"], "Combined result is missing a frozen artifact")
    require(original_evidence["resource_owner"]["path"] not in retained["protected_files"]
            and all(protected.get(k) == v for k, v in retained["protected_files"].items()),
            "Combined result omitted the original protected artifacts")
    origin, dest = (directory(plan[key], root) for key in ("parent_execution", "execution"))
    with observe_lock(origin / ".coordinator.lock") as busy_old, observe_lock(dest / ".coordinator.lock") as busy_new:
        require(busy_old is busy_new is False, "Both P4 coordinators must be idle")
        require(item["path"] == str(dest / "migration-acceptance.json"), "Expected the published migration receipt")
        check_files(protected, root.parent)
        custody(plan_pin, plan, combined["release"], retained, protected, root)
        independent = read(review_pin, root)
        require(independent["schema"] == "wp05_p4_real_selected_recompute_v1" and independent["status"] == "partial"
                and independent["observed_cases"] == 10 and independent["planned_cases"] == 20
                and independent["committed_optimizer_updates"] == retained["optimizer_updates"] == 1200
                and independent["acceptance"] is None and independent["resource"] == parent_pin
                and independent["scientific_scope"] == parent_resource["science"] and independent["release"] == release_pin
                and independent["input_pins"] == parent["protected_files"] and independent["audit_optimizer_updates"] == 0
                and independent["raw_gt_loaded"] is independent["query_accessed"] is False
                and retained["case_audits"] == {r["case"]["id"]: r["audit"] for r in independent["cases"]}
                and set(retained["cases"]) == set(retained["case_audits"]), "Retained independent review is incomplete or unrelated")
        new = read(plan["migrated_science"], root)
        unchanged_science(parent, parent_resource["science"], new, plan, root)
        by_case = allocation(plan, new, combined)
        old_rows = []
        for row in independent["cases"]:
            pin, case = row["audit"], row["case"]
            require(pin["path"] == str(origin / "cases" / case["id"] / "audit.json"), "Original audit path differs")
            audit = published_case(pin, case, parent_resource["science"], parent_pin,
                                   case["physical_uuid"], retained["protected_files"], root)
            old_rows.append(case_row(audit, parent_resource["science"], pin, "original", row["paths"]))
        new_rows = []
        require(len(combined["case_audits"]) == 10, "Ten migrated audits are required")
        for case in new["cases"]:
            name = case["id"]
            path = dest / "cases" / name / "audit.json"
            require(str(path) in combined["case_audits"] and case["physical_uuid"] == by_case.get(name),
                    "Migrated case audit or physical destination is missing")
            pin = {"path": str(path), "sha256": combined["case_audits"][str(path)]}
            audit = published_case(pin, case, plan["migrated_science"], plan_pin,
                                   case["physical_uuid"], protected, root)
            costs = migrated_cost(path.parent, new["paths"], protected, root)
            new_rows.append(case_row(audit, plan["migrated_science"], pin, "migrated", costs))
    result = combine_cases(parent, new, old_rows, new_rows, synthetic=False)
    result.update(receipt=item, plan=plan_pin, original_status=read(original_evidence["status"], root),
                  protected_original_paths=len(retained["protected_files"]))
    return result
