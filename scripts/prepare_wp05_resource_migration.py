"""Prepare a review-only migration of the two untouched HongHu P4 cohorts."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import p4_coverage_batch as batch, p4_full_budget_campaign as campaign
from transfer import p4_independent_acceptance as acceptance, p4_independent_registration as original
from transfer import p4_runtime_device as device
from transfer.container_runtime import training_container_kind


PARENT_CONTROLLER = "72e22231b4889e03ea37c4f3629cfbc9494c74b9"
PARENT_RESOURCE = "eb9cacef34c0cb5b10d654a52d8a8a6eccb645e4d4bd7908160ab2942cce9e43"
PARENT_RELEASE = "bec9247d9200234b5c88634f5be944c15c9e847a49f135f511f9ae6971358da7"
ARMS = ("none", "anchor_only", "binary", "continuous", "shuffled")
COHORTS = ((1, "scratch"), (5, "shared_transfer"))
PARALLEL, SERIAL = "parallel", "serial"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def physical_destinations(destinations, dispatch_mode):
    require(isinstance(destinations, list) and len(destinations) == len(COHORTS)
            and all(isinstance(gpu, str) and device.UUID.fullmatch(gpu) for gpu in destinations),
            "Two physical destination UUID assignments are required")
    unique = list(dict.fromkeys(destinations))
    require((dispatch_mode, len(unique)) in ((PARALLEL, 2), (SERIAL, 1)),
            "Dispatch mode requires two distinct UUIDs in parallel or one shared UUID in serial")
    return unique


def slots(parent, cases, destinations, dispatch_mode=PARALLEL):
    physical_destinations(destinations, dispatch_mode)
    records = {}
    selected = []
    for (k, method), gpu in zip(COHORTS, destinations, strict=True):
        ids = [f"s2048-k{k}-{method}-m512-{arm}" for arm in ARMS]
        require(all(case_id in cases for case_id in ids), "A comparison cohort is incomplete")
        members = [cases[case_id] for case_id in ids]
        require(all((c["physical_uuid"], c["support_seed"], c["model_seed"], c["sampler_seed"], c["k"], c["method"])
                    == (original.GPU1, 2048, 512, 64, k, method) and c["protocol"] == device.CUDA_FP64_WEIGHT
                    for c in members), "Only the original GPU 1 unstarted role is eligible")
        slot = records.setdefault(gpu, dict(physical_uuid=gpu, cases=[], updates=0))
        slot["cases"].extend(ids)
        slot["updates"] += 600
        selected.extend(ids)
    require(len(set(selected)) == 10 and set(selected) == set(parent["slots"][original.GPU1]["cases"]),
            "Expected both complete five-arm cohorts and no other cases")
    return records, selected


def untouched(execution, identifiers):
    require(not (execution / "failure.json").exists(), "Original failure or unconfirmed operations require review")
    for case_id in identifiers:
        for folder in ("cases", "stages"):
            path = execution / folder / case_id
            require(not path.exists() and not path.is_symlink(), f"Original case has state or a dispatch: {case_id}")
        require(not list((execution / "logs").glob(f"{case_id}-*")), f"Original case has a worker launch: {case_id}")


def no_live_dispatch(out):
    active = []
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = entry.read_bytes().split(b"\0")
        except OSError:
            continue
        if (b"-m" in argv and any(x in argv for x in (b"transfer.p4_independent_queue", b"transfer.p4_full_budget_campaign"))
                and any(str(out).encode() in arg or b"wp05-independent-queue-real-prepared03/resource-scope.json" in arg
                        for arg in argv)):
            active.append(int(entry.parent.name))
    require(not active, "Original P4 coordinator or worker remains active")
    return active


def verify_retained(resource_pin, release_pin, resource, science, review_pin, slot, departed):
    """Reconstruct and pin completed cases without reading their images or running any forward."""
    review = original.read(review_pin)
    kept = resource["slots"][slot]["cases"]
    require(len(kept) == len(set(kept)) == 10 and not set(kept).intersection(departed),
            "Original and migrated cases must be disjoint")
    expected = {case["id"]: case for case in science["cases"]}
    audited = {row["case"]["id"]: row for row in review["cases"]}
    require(review["resource"] == resource_pin and review["release"] == release_pin
            and review["scientific_scope"] == resource["science"] and review["status"] == "partial"
            and review["committed_optimizer_updates"] == 1200 and review["audit_optimizer_updates"] == 0
            and review["observed_cases"] == len(review["cases"]) == 10 and set(audited) == set(kept)
            and review["input_pins"] == science["protected_files"]
            and review["raw_gt_loaded"] is review["query_accessed"] is False, "Original independent review differs")
    pins = {**science["protected_files"], review_pin["path"]: review_pin["sha256"],
            resource_pin["path"]: resource_pin["sha256"], resource["science"]["path"]: resource["science"]["sha256"]}
    if release_pin is not None:
        pins[release_pin["path"]] = release_pin["sha256"]
    audits, total = {}, 0
    for name in kept:
        reference, case = audited[name], expected[name]
        require(reference["case"] == case and reference["optimizer_updates"] == 120,
                "Retained reviewed case differs from the frozen roster")
        pinned_audit = reference["audit"]
        original.read(pinned_audit)
        row, audit_pin, case_pins = acceptance.reconstruct_case(resource_pin, resource, science, case, slot)
        require(audit_pin == pinned_audit, "Current case audit differs from its review")
        audits[name] = audit_pin
        for path, sha in {audit_pin["path"]: audit_pin["sha256"], **case_pins}.items():
            require(pins.setdefault(path, sha) == sha, "Retained artifacts claim conflicting digests")
        total += row["optimizer_updates"]
    require(total == 1200, "Original audited budget must be 1200 steps")
    batch.check_pins(pins)
    return dict(review=review_pin, cases=sorted(kept), optimizer_updates=total, case_audits=audits, protected_files=pins)


def parent_evidence(resource_pin, release_pin, review_pin, parent_controller):
    resource = original.read(resource_pin)
    code = code_snapshot(parent_controller)
    require(not code["dirty"] and code["commit"] == PARENT_CONTROLLER
            and original.identity(code) == resource["controller"], "Original clean controller identity required")
    science = original.validate(resource, live_worker=True)
    original.validate_release(resource_pin, resource, release_pin)
    out = Path(resource["execution"])
    status = original.pin(out / "status.json")
    owner = original.pin(out / "resource-owner.json")
    worker_owner = campaign.owner(out, resource["science"]["path"], resource["science"]["sha256"])
    require(original.read(owner) == dict(resource=resource_pin, release=release_pin, worker_owner=worker_owner)
            and original.read(original.pin(out / "owner.json")) == worker_owner, "Original owner differs")
    states = original.read(status)
    require(states["resource_sha256"] == resource_pin["sha256"] and states["status"] == "waiting_for_resources"
            and states["slots"][original.GPU1]["optimizer_updates"] == states["slots"][original.GPU1]["accepted_cases"] == 0
            and states["slots"][original.GPU6]["optimizer_updates"] == 1200
            and states["slots"][original.GPU6]["accepted_cases"] == 10, "Expected original 0 + 1200 optimized steps")
    pins = dict(status=status, resource_owner=owner, owner=original.pin(out / "owner.json"), review=review_pin)
    return resource, science, pins


def field_diff(old, new, names):
    return {name: dict(previous=old.get(name), proposed=new.get(name), unchanged=old.get(name)==new.get(name))
            for name in names}


def write(path, row):
    with path.open("x", encoding="utf-8") as file:
        json.dump(row, file, ensure_ascii=False, sort_keys=True, indent=2)
        file.write("\n")
    return original.pin(path)


def prepare(args):
    code = code_snapshot()
    require_frozen_code(code)
    require(code["commit"] == args.expected_commit and training_container_kind() == "lxc",
            "Expected clean review code in the authorized LXC")
    rp, lp, review = (original.pin(path) for path in (args.resource, args.release, args.review))
    require(rp["sha256"] == PARENT_RESOURCE and lp["sha256"] == PARENT_RELEASE, "Original resource and release pins required")
    resource = original.read(rp)
    out = Path(resource["execution"])
    draft = external_directory(args.out)
    execution = external_directory(args.execution)
    require(draft.parent == out.parent == execution.parent and len({out, draft, execution}) == 3
            and not draft.exists() and not execution.exists(), "Use separate fresh sibling locations")
    gpus = physical_destinations(args.gpus, args.dispatch_mode)
    inventory_gpus = tuple(dict.fromkeys((original.GPU1, original.GPU6, *gpus)))
    locks = [out / ".coordinator.lock", *(out.parent / ".p4_gpu_leases" / f"{gpu}.lock" for gpu in inventory_gpus)]
    require(all(path.is_file() and not path.is_symlink() for path in locks), "Existing shared physical and coordinator locks required")
    with ExitStack() as stack:
        for lock in locks:
            stack.enter_context(device.lease(lock))
        active = no_live_dispatch(out)
        resource, science, evidence = parent_evidence(rp, lp, review, args.parent_controller)
        cases = {case["id"]: case for case in science["cases"]}
        proposed, case_ids = slots(resource, cases, args.gpus, args.dispatch_mode)
        untouched(out, case_ids)
        retained = verify_retained(rp, lp, resource, science, review, original.GPU6, case_ids)
        inventory = device.simultaneous_inventory(inventory_gpus)
        required = {gpu: dict(qualification=None, recovery=None, status="no_approved_proof",
                              geometry=science["operator_geometry"]["geometry_sha256"],
                              required_recovery_classes=[9,22], expected_synthetic_recovery_updates=240)
                    for gpu in gpus}
        draft.mkdir()
        retained_pin = write(draft / "retained-artifacts.json", retained)
        stamp = datetime.now(timezone.utc).isoformat()
        separate = dict(
            schema="wp05_p4_limited_resource_migration_draft_v1", mode=resource["mode"], classes=resource["classes"],
            science=resource["science"], science_fields=resource["science_fields"], thresholds=resource["thresholds"],
            controller=resource["controller"], worker=resource["worker"], slots=proposed, destinations=args.gpus,
            dispatch_mode=args.dispatch_mode,
            physical_recoveries=dict.fromkeys(gpus), execution=str(execution), prior_resource=rp,
            prior_release=lp, parent_evidence=evidence, protected_origin_path=str(out),
            process_evidence=dict(original_live_pids=active, acquired_shared_leases=[str(p) for p in locks], utc=stamp),
            inventory=inventory, original_unstarted_case_ids=case_ids, retained_original_case_ids=retained["cases"],
            retained_evidence=retained_pin,
            original_coordinator_reentry_status="original_authority_not_retired",
            required_eligibility_and_qualifications=required,
            physical_protocol=science["protocol"], per_case_paths=science["paths"],
            validation_steps=science["budget"]["validation_steps"], optimizer_updates=1200,
            release_compatible_with_existing_controller=False)
        sb = write(draft / "resource-scope-draft.json", separate)
        release = dict(schema="wp05_p4_limited_resource_migration_release_draft_v1", decision="review_required",
                       research_issue=470, resource_scope=sb, science_scope=resource["science"],
                       controller=resource["controller"], worker=original.identity(resource["worker"]),
                       execution=str(execution), slots=proposed, destinations=args.gpus, dispatch_mode=args.dispatch_mode,
                       physical_recoveries=dict.fromkeys(gpus),
                       optimizer_updates=1200, raw_gt_authorized=False, query_authorized=False,
                       preconditions=["qualify_each_destination_with_frozen_arithmetic_and_same_device_recovery",
                                      "approve_new_resource_admission_and_exclusivity_implementation",
                                      "retire_original_gpu1_dispatch_under_original_coordinator_lease",
                                      "verify_original_remapped_cases_are_untouched_immediately_before_execution"],
                       real_dispatch_authorized=False)
        lb = write(draft / "release-draft.json", release)
        previous_release = original.read(lp)
        diff = write(draft / "scope-diff.json", dict(
            resource_fields=field_diff(resource, separate, resource),
            release_fields=field_diff(previous_release, release, previous_release),
            added_resource_fields=sorted(set(separate)-set(resource)),
            added_release_fields=sorted(set(release)-set(previous_release)),
            scientific_parent_equals_original=True, historical_loader_never_accepts_draft=True,
            compatible_worker_changes_required=["new per-case UUIDs enter numeric binding and pair identity",
                                               "original roster ties both unstarted groups to one index",
                                               "legacy controller enforces original slots, fixed full release and exact scope digest"],
            safe_retirement_constraint="Archived frozen controller does not read a migration marker. "
                                       "A durable authorized admission entry and explicit original release supersession are required."))
        require_frozen_code(code)
        print(json.dumps(dict(resource=sb, release=lb, diff=diff, updated_optimizer_steps=0), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--gpus", nargs=2, required=True, metavar=("K1_SCRATCH_UUID", "K5_SHARED_UUID"))
    parser.add_argument("--dispatch-mode", choices=(PARALLEL, SERIAL), default=PARALLEL)
    for arg in ("resource", "release", "review", "parent-controller", "out", "execution"):
        parser.add_argument("--" + arg, type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
