"""Prepare, admit and audit an explicitly released P4 resource takeover."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from model.research_config import digest
from research_runtime import ROOT, code_snapshot, external_directory, require_frozen_code
from train_spatial import SpatialConfig
from transfer import p4_coverage_batch as batch, p4_cuda_qualification as qualification
from transfer import p4_full_budget_campaign as campaign, p4_full_budget_registration as registration
from transfer import p4_independent_acceptance as acceptance, p4_independent_queue as queue
from transfer import p4_independent_registration as original, p4_resource_takeover as custody
from transfer import p4_runtime_device as device, p4_update_evidence as updates

from scripts.prepare_wp05_resource_migration import (
    COHORTS, PARALLEL, PARENT_RELEASE, PARENT_RESOURCE, SERIAL, physical_destinations, slots, verify_retained,
)


RELEASE = "wp05_p4_unstarted_cohort_migration_release_v1"
LEGACY_WORKER = "3a6381cb4b0c089a10123292bc831a3a613e6106"
RUNTIME_PATHS = ("model", "loadData", "configs", "transfer/p4_runtime_device.py", "transfer/p4_exposure.py",
                 "transfer/p4_cuda_qualification.py", "transfer/p4_coverage.py", "transfer/p4_update_evidence.py",
                 "transfer/g3_training.py", "train_spatial.py")
FIXTURE_GROUPS = ("cpu-k1-scratch", "cpu-k5-shared-transfer")


def require(ok, message):
    custody.require(ok, message)


def seal(path, data):
    custody.create(path, custody.payload(data))
    return original.pin(path)


def reference(plan):
    resource = original.read(plan["parent_resource"])
    science = original.validate(resource)
    fixture = science["synthetic"]
    require(fixture == plan["synthetic"] and (not fixture or resource["mode"] == original.CPU),
            "The migration requires the original real scope or an isolated CPU fixture")
    original.validate_release(plan["parent_resource"], resource, plan["parent_release"])
    if not fixture:
        require(plan["parent_resource"]["sha256"] == PARENT_RESOURCE
                and plan["parent_release"]["sha256"] == PARENT_RELEASE, "Original P4 parent differs")
    require(resource["science"] == plan["parent_science"] and resource["execution"] == plan["parent_execution"]
            and science["paths"] == original.PAIRED and science["budget"]["steps"] == 60
            and science["budget"]["validation_steps"] == [30, 60], "Original scientific scope or budget differs")
    cases = {case["id"]: case for case in science["cases"]}
    if fixture:
        require(plan.get("dispatch_mode", PARALLEL) == PARALLEL, "The existing CPU fixture uses separate lanes")
        chosen = {lane: dict(physical_uuid=None, cases=[case["id"] for case in science["cases"]
                    if (case["k"], case["method"]) == (k, method)], updates=600)
                  for lane, (k, method) in zip(FIXTURE_GROUPS, COHORTS, strict=True)}
        identifiers = [case for spec in chosen.values() for case in spec["cases"]]
    else:
        chosen, identifiers = slots(resource, cases, plan["destinations"], plan.get("dispatch_mode", PARALLEL))
    require(len(set(identifiers)) == len(identifiers) == 10
            and chosen == plan["slots"] and identifiers == plan["cases"], "Only the two unstarted complete five-arm groups may move")
    return resource, science, cases


def qualified(plan, old):
    if plan["synthetic"]:
        require(not plan["proofs"], "CPU fixture cannot claim the real qualification")
        return {}
    gpus = set(plan["slots"])
    proofs = plan["proofs"]
    require(set(proofs) == gpus, "Every physical destination requires its own v4 operator and recovery proof")
    pins = {}
    frozen_worker = dict(commit=old["commit"], source_hash=old["source_hash"])
    for gpu, pair in proofs.items():
        op, recovery = pair["operator"], pair["recovery"]
        op_row, rec_row = original.read(op), original.read(recovery)
        require(qualification.check_receipt(op_row, old["protocol"], frozen_worker, old["operator_geometry"]) == gpu,
                "Operator qualification belongs to another UUID")
        require(rec_row["physical_uuid"] == gpu, "Recovery qualification belongs to another UUID")
        for role in rec_row["cases"]:
            proof = original.read(dict(path=role["scope"], sha256=role["scope_sha256"]))
            require(proof["qualifications"] == {gpu: op["path"]} and proof["protected_files"][op["path"]] == op["sha256"],
                    "Recovery did not consume the same accepted physical qualification")
        pins.update(registration.bind_recovery(Path(recovery["path"]), registration.ADMISSION, old["protocol"], False,
                                               frozen_worker))
        pins[op["path"]] = op["sha256"]
    return pins


def expected_science(plan, old, old_cases, operator_pins, reg, reg_path):
    new = deepcopy(old)
    selected = []
    lanes = FIXTURE_GROUPS if plan["synthetic"] else physical_destinations(
        plan["destinations"], plan.get("dispatch_mode", PARALLEL))
    for lane in lanes:
        slot = plan["slots"][lane]
        for case_id in slot["cases"]:
            case = deepcopy(old_cases[case_id])
            case["physical_uuid"] = slot["physical_uuid"]
            case["pair_sha256"] = registration.pair_identity(reg, case, case["initialization"])
            selected.append(case)
    new.update(**plan["code"], registration=str(reg_path), cases=selected,
               gpus=[] if plan["synthetic"] else lanes,
               qualifications={gpu: row["operator"]["path"] for gpu, row in plan["proofs"].items()},
               protected_files={**old["protected_files"], **operator_pins, str(reg_path): original.pin(reg_path)["sha256"]})
    new["budget"]["planned_optimizer_updates"] = 1200
    new["parent_science"] = plan["parent_science"]
    new["migration_recoveries"] = {gpu: row["recovery"] for gpu, row in plan["proofs"].items()}
    new.pop("recovery_receipt", None)
    new["scope_hash"] = digest({k: v for k, v in new.items() if k != "scope_hash"})
    return new


def locations(request, out):
    parent = Path(request["parent_execution"])
    working = Path(request["execution"])
    require(all(path.parent == parent.parent and path == path.resolve() for path in (out, parent, working))
            and len({parent, working, out}) == 3, "Original, new execution and control must be distinct sibling directories")
    return parent, working


def retained_parent(request, *, transferred):
    proposal = original.read(request["proposal"])
    evidence = proposal["parent_evidence"]
    names = ("status", "owner", "review") if transferred else ("status", "owner", "review", "resource_owner")
    for name in names:
        original.read(evidence[name])
    resource, science, _ = reference(request)
    retained = verify_retained(request["parent_resource"], request["parent_release"], resource, science,
                               evidence["review"], original.GPU6, request["cases"])
    require(retained == original.read(proposal["retained_evidence"]), "Retained stage or artifact digest changed")
    return retained


def prepare(request, code, out):
    require_frozen_code(code)
    request = deepcopy(request)
    request["code"] = original.identity(code)
    _, old, old_cases = reference(request)
    parent, working = locations(request, out)
    require(not out.exists() and not working.exists(), "Migration outputs must be fresh")
    if not request["synthetic"]:
        subprocess.run(["git", "diff", "--exit-code", LEGACY_WORKER, code["commit"], "--", *RUNTIME_PATHS],
                       cwd=ROOT, check=True)
    if not request["synthetic"]:
        draft = original.read(request["proposal"])
        require(draft["prior_resource"] == request["parent_resource"] and draft["prior_release"] == request["parent_release"]
                and draft["science"] == request["parent_science"] and draft["slots"] == request["slots"]
                and draft["destinations"] == request["destinations"]
                and draft.get("dispatch_mode", PARALLEL) == request.get("dispatch_mode", PARALLEL)
                and draft["execution"] == request["execution"] and draft["protected_origin_path"] == str(parent)
                and draft["parent_evidence"]["resource_owner"] == original.pin(parent / "resource-owner.json"),
                "Original identity differs from the reviewed resource proposal")
        with device.lease(parent / ".coordinator.lock"):
            custody.untouched(parent, request["cases"])
            retained_parent(request, transferred=False)
    proofs = qualified(request, old)
    batch.check_pins(old["protected_files"])
    reg = original.read(dict(path=old["registration"], sha256=old["protected_files"][old["registration"]]))
    reg["commit"] = code["commit"]
    for case_id in request["cases"]:
        case = old_cases[case_id]
        model, initial = registration.initialize_case(reg, case)
        optimizer = registration._optimizer(model, SpatialConfig(reg["config"]))
        require(initial == case["initialization"]
                and digest(updates.layout(model, optimizer)) == case["parameter_layout_sha256"],
                "Migrated initialization tensor or optimizer layout differs from the parent")
    out.mkdir()
    reg_pin = seal(out / "registration.json", reg)
    science = expected_science(request, old, old_cases, proofs, reg, Path(reg_pin["path"]))
    science_pin = seal(out / "scope.json", science)
    request.update(registration=reg_pin, migrated_science=science_pin, control=str(out))
    parent_owner = parent / "resource-owner.json"
    request["prior_owner_sha256"] = original.pin(parent_owner)["sha256"]
    request_pin = seal(out / "migration.json", request)
    seal(out / "release-draft.json", release_for(request_pin, request, decision="review_required"))
    parent_cases = {case["id"]: case for case in old["cases"]}
    comparable = ("physical_uuid", "pair_sha256")
    require(all({k: v for k, v in case.items() if k not in comparable}
                == {k: v for k, v in parent_cases[case["id"]].items() if k not in comparable}
                for case in science["cases"]), "Non-resource case fields differ")
    seal(out / "scientific-diff.json", dict(
        changed_top_level=sorted(key for key in set(old) | set(science) if old.get(key) != science.get(key)),
        changed_case_fields=list(comparable), input_registration_commit_only=True, case_initialization_exact=True,
        retained=[case["id"] for case in old["cases"] if case["id"] not in request["cases"]],
        parent= request["parent_science"], scope=science_pin, planned_new_steps=1200))
    return request_pin


def release_for(item, plan, *, decision="approved"):
    release = dict(schema=RELEASE, decision=decision, migration=item, parent_resource=plan["parent_resource"],
                old_resource_owner_sha256=plan["prior_owner_sha256"], control=plan["control"],
                execution=plan["execution"], source=plan["code"], science=plan["migrated_science"], slots=plan["slots"],
                optimizer_updates=1200, raw_gt_authorized=False, query_authorized=False, synthetic=plan["synthetic"])
    if plan.get("dispatch_mode") == SERIAL:
        release.update(dispatch_mode=SERIAL, destinations=plan["destinations"])
    return release


def admitted(item, authority, code):
    require_frozen_code(code)
    plan = original.read(item)
    require(plan["code"] == original.identity(code), "Migration code differs")
    require(original.read(authority) == release_for(item, plan), "Explicit approved migration release required")
    _, old, old_cases = reference(plan)
    parent, execution = locations(plan, Path(plan["control"]))
    reg = original.read(plan["registration"])
    old_reg = original.read(dict(path=old["registration"], sha256=old["protected_files"][old["registration"]]))
    old_reg["commit"] = code["commit"]
    require(reg == old_reg, "Only the execution commit may differ in the scientific input registration")
    proofs = qualified(plan, old)
    science = original.read(plan["migrated_science"])
    require(science == expected_science(plan, old, old_cases, proofs, reg, Path(plan["registration"]["path"])),
            "Scientific parent projection or migrated resource scope differs")
    resource = dict(science=plan["migrated_science"], worker=dict(root=str(ROOT), **original.identity(code)),
                    controller=original.identity(code), execution=str(execution), slots=plan["slots"],
                    mode=original.CPU if plan["synthetic"] else original.REAL,
                    classes=22, physical_recoveries={gpu: value["recovery"] for gpu, value in plan["proofs"].items()})
    record = dict(schema=custody.SCHEMA, control=plan["control"], parent_execution=str(parent),
                  prior_owner_sha256=plan["prior_owner_sha256"], cases=plan["cases"],
                  migration=item, authority=authority)
    return plan, record, resource, science, reg


def start(item, authority, code, *, transition=False, recover=True, fixture_wait=None):
    plan, record, resource, science, _ = admitted(item, authority, code)
    parent, destination = Path(plan["parent_execution"]), Path(resource["execution"])
    require(fixture_wait is None or plan["synthetic"], "Fixture limits are CPU-only")
    with device.lease(parent / ".coordinator.lock") as old_lease:
        archived_review = Path(plan["control"]) / "original-partial-review.bytes"

        def validate_parent():
            resource_old = original.read(plan["parent_resource"])
            expected_owner = dict(resource=plan["parent_resource"],
                worker_owner=campaign.owner(parent, resource_old["science"]["path"], resource_old["science"]["sha256"]))
            if plan["parent_release"] is not None:
                expected_owner["release"] = plan["parent_release"]
            require(campaign.read(parent / "resource-owner.json") == expected_owner
                    and campaign.read(parent / "owner.json") == expected_owner["worker_owner"], "Original owner differs")
            if not plan["synthetic"]:
                retained = retained_parent(plan, transferred=False)
                custody.create(archived_review, Path(retained["review"]["path"]).read_bytes())

        if transition:
            require(not destination.exists(), "Migration execution already exists before takeover")
            custody.takeover(record, old_lease, validate_parent)
            return dict(status="taken_over", updates=0, owner=original.pin(parent/"resource-owner.json"))
        custody.verify(record, old_lease)
        if not plan["synthetic"]:
            proof = original.read(plan["proposal"])["parent_evidence"]["review"]
            require(custody.digest(archived_review.read_bytes()) == proof["sha256"], "Retained independent review changed")
        if not recover:
            return dict(status="verified_custody", updates=0)
        custody.untouched(parent, plan["cases"])

        def launch(scope, case, stage, locks, log):
            path, operation, _, until = stage
            command = [sys.executable, "-m", "transfer.p4_resource_migration", "worker", "--expected-commit", code["commit"],
                       "--plan", item["path"], "--plan-sha", item["sha256"], "--authority", authority["path"],
                       "--authority-sha", authority["sha256"], "--case", case["id"], "--operation", operation,
                       "--parent-fd", str(old_lease.fileno()), "--queue-fd", str(locks[0].fileno()),
                       "--slot-fd", str(locks[1].fileno())]
            if path is not None:
                command.extend(("--path", path))
            if until is not None:
                command.extend(("--until", str(until)))
            with log.open("xb") as stream:
                return subprocess.Popen(command, cwd=ROOT, env=original.environment(ROOT, case["physical_uuid"]),
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        pass_fds=tuple(lease.fileno() for lease in (old_lease, *locks)))

        return queue.dispatch_registered(item, code, resource, science, resume=destination.exists(), release=authority,
                                         launch=launch, fixture_wait=fixture_wait, require_confirmed_intent=True)


def publish(item, authority, code):
    plan, record, resource, science, _ = admitted(item, authority, code)
    parent = Path(plan["parent_execution"])
    dest = Path(resource["execution"])
    with device.lease(parent / ".coordinator.lock") as lock, device.lease(dest / ".coordinator.lock"):
        custody.verify(record, lock)
        custody.untouched(parent, plan["cases"])
        result = acceptance.reconstruct_registered(item, resource, science, release=authority)
        require(result["cases"] == 10 and result["optimizer_updates"] == 1200, "Only the exact migrated budget may be accepted")
        if not plan["synthetic"]:
            retained = retained_parent(plan, transferred=True)
            archive = Path(plan["control"]) / "original-partial-review.bytes"
            require(custody.digest(archive.read_bytes()) == retained["review"]["sha256"],
                    "Archived original independent review changed")
            prior_owner = Path(plan["control"]) / "original-resource-owner.bytes"
            proposal = original.read(plan["proposal"])
            result["retained_original"] = retained
            result["combined_reference"] = dict(
                cases=result["cases"] + len(retained["cases"]),
                optimizer_updates=result["optimizer_updates"] + retained["optimizer_updates"],
                predecessor_compatible=False)
            controls = [proposal["parent_evidence"][name] for name in ("owner", "status")]
            controls.append(original.pin(parent / "resource-owner.json"))
            for path, sha in {
                **retained["protected_files"], **{pin["path"]: pin["sha256"] for pin in controls},
                str(archive): retained["review"]["sha256"], str(prior_owner): plan["prior_owner_sha256"],
                proposal["retained_evidence"]["path"]: proposal["retained_evidence"]["sha256"],
            }.items():
                require(result["protected_files"].setdefault(path, sha) == sha, "Merged artifacts claim conflicting digests")
            batch.check_pins(result["protected_files"])
        path = dest / "migration-acceptance.json"
        if path.exists():
            require(campaign.read(path) == result, "Published migration acceptance differs")
            return original.pin(path)
        return seal(path, result)


def run_worker(args, item, authority, code):
    plan, record, resource, science, reg = admitted(item, authority, code)
    device.check_inherited_lease(args.parent_fd, Path(plan["parent_execution"]) / ".coordinator.lock")
    # Duplicate descriptor also lets the verifier authenticate the exact inode.
    import os
    with os.fdopen(os.dup(args.parent_fd), "r") as lock:
        custody.verify(record, lock)
    return campaign.execute_registered(resource["science"]["path"], resource["science"]["sha256"],
        Path(resource["execution"]), args.case, args.operation, args.until, code, science, reg,
        lease_fds=(args.queue_fd, args.slot_fd), path_name=args.path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "takeover", "run", "worker", "accept"))
    parser.add_argument("--expected-commit", required=True)
    for name in ("plan", "plan-sha", "authority", "authority-sha", "out", "case", "operation", "path"):
        parser.add_argument(f"--{name}")
    for name in ("until", "parent-fd", "queue-fd", "slot-fd"):
        parser.add_argument(f"--{name}", type=int)
    args = parser.parse_args()
    code = code_snapshot()
    require(code["commit"] == args.expected_commit, "Expected full migration code SHA")
    item = dict(path=args.plan, sha256=args.plan_sha)
    release = dict(path=args.authority, sha256=args.authority_sha)
    try:
        if args.action == "prepare":
            result = prepare(original.read(item), code, external_directory(args.out))
        elif args.action == "worker":
            result = run_worker(args, item, release, code)
        elif args.action == "accept":
            result = publish(item, release, code)
        else:
            result = start(item, release, code, transition=args.action == "takeover")
    except device.ResourceUnavailable as error:
        if args.action != "worker":
            raise
        result = dict(status="waiting_for_resources", reason=str(error), case=args.case, updates=0)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
