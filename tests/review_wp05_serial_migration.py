"""Verify serial migration against the frozen roster and existing CPU receipts."""

import argparse
import json
from pathlib import Path
from unittest.mock import Mock, patch

from research_runtime import code_snapshot, require_frozen_code
from tests.test_p4_independent_queue import reject
from transfer import p4_resource_migration as m


def check_mapping(runs, gpu, alternate):
    old = m.original
    resource_pin = old.pin(runs / "wp05-independent-queue-real-prepared03/resource-scope.json")
    release = old.pin(runs / "wp05-independent-queue-real-authority01/release.json")
    parent = old.read(resource_pin)
    science = old.read(parent["science"])
    cases = {row["id"]: row for row in science["cases"]}
    slots, names = m.slots(parent, cases, [gpu, gpu], m.SERIAL)
    assert slots == {gpu: dict(physical_uuid=gpu, cases=names, updates=1200)}
    groups = [(cases[name]["k"], cases[name]["method"]) for name in names]
    assert groups == [group for group in m.COHORTS for _ in range(5)]
    request = dict(parent_resource=resource_pin, parent_release=release, parent_science=parent["science"],
                   parent_execution=parent["execution"], synthetic=False, destinations=[gpu, gpu],
                   dispatch_mode=m.SERIAL, slots=slots, cases=names)
    m.reference(request)
    dual_slots, dual_names = m.slots(parent, cases, [gpu, alternate])
    legacy = {name: value for name, value in request.items() if name != "dispatch_mode"}
    m.reference({**legacy, "destinations": [gpu, alternate], "slots": dual_slots, "cases": dual_names})
    failures = {}
    for name, change, error in (
        ("implicit_duplicate", legacy, "Dispatch mode"),
        ("serial_mixed_uuid", {**request, "destinations": [gpu, alternate]}, "Dispatch mode"),
        ("unknown_mode", {**request, "dispatch_mode": "unknown"}, "Dispatch mode"),
        ("wrong_case_order", {**request, "cases": names[5:] + names[:5]}, "complete five-arm"),
        ("missing_case", {**request, "cases": names[:-1]}, "complete five-arm"),
        ("extra_budget", {**request, "slots": {gpu: {**slots[gpu], "updates": 1201}}}, "complete five-arm"),
    ):
        failures[name] = reject(lambda: m.reference(change), error)
    for name, proofs in (("missing_proof", {}), ("extraneous_proof", {gpu: {}, alternate: {}})):
        failures[name] = reject(lambda: m.qualified({**request, "proofs": proofs}, science), "own v4 operator")
    return names, failures


def check_existing_queue(runs, out, names, code):
    source = runs / "wp05-resource-migration-fixture02"
    item = m.original.pin(source / "migration-control/migration.json")
    migration = m.original.read(item)
    source_out = Path(migration["execution"])
    science_pin = migration["migrated_science"]
    science = m.original.read(science_pin)
    cases = {row["id"]: row for row in science["cases"]}
    assert list(cases) == names
    slot = "cpu-serial"
    resource = dict(execution=str(out), science=science_pin, mode=m.original.CPU,
                    slots={slot: dict(physical_uuid=None, cases=names, updates=1200)})
    for name in ("cases", "logs", "stages"):
        (out / name).mkdir(parents=True)
    worker_owner = m.campaign.owner(out, science_pin["path"], science_pin["sha256"])
    m.seal(out / "owner.json", worker_owner)
    m.seal(out / "resource-owner.json", dict(resource=item, worker_owner=worker_owner))
    # The fixture redirects case evidence to the already completed CPU run. The
    # local intent/ack test files exercise the genuine serial coordinator without training.
    before = {}
    missing = (names[4], 6)
    steps = m.queue.stages(science)
    for case_id in names:
        local = out / "stages" / case_id
        local.mkdir()
        for index, (route, operation, start, until) in enumerate(steps):
            remote_receipt = source_out / "stages" / case_id / f"{index:02d}.receipt.json"
            proof = m.original.read(m.original.pin(remote_receipt))["evidence"]
            before[proof["path"]] = proof["sha256"]
            values = dict(resource_sha256=item["sha256"], case=case_id, slot=slot, path=route,
                          operation=operation, start=start, until=until)
            intent = m.seal(local / f"{index:02d}.intent.json", values)
            if (case_id, index) != missing:
                m.seal(local / f"{index:02d}.receipt.json", dict(attempt=intent, evidence=proof))
    blocked_launch = Mock(side_effect=AssertionError("A committed stage attempted to dispatch"))
    folder = m.campaign.bound_folder
    evidence = m.queue.evidence
    observed = []

    def capture(location, case, scope, sha, stage):
        observed.append((case["id"], stage))
        return evidence(location, case, scope, sha, stage)

    with patch.object(m.campaign, "bound_folder", side_effect=lambda _, case, route=None: folder(source_out, case, route)):
        with patch.object(m.queue, "evidence", side_effect=capture):
            first = m.queue.dispatch_registered(item, code, resource, science, resume=True, launch=blocked_launch,
                                                require_confirmed_intent=True)
            second = m.queue.dispatch_registered(item, code, resource, science, resume=True, launch=blocked_launch,
                                                 require_confirmed_intent=True)
    blocked_launch.assert_not_called()
    assert observed == [(case, stage) for _ in range(2) for case in names for stage in steps]
    assert first == second and first["status"] == "completed" and first["updates_this_invocation"] == 0
    assert first["slots"][slot]["optimizer_updates"] == 1200 and first["slots"][slot]["accepted_cases"] == 10
    assert (out / "stages" / missing[0] / f"{missing[1]:02d}.receipt.json").is_file()
    m.batch.check_pins(before)
    return dict(cases=names, stages=len(steps) * len(names), restored_acknowledgment=list(missing),
                new_updates=0, launches=0, observed_slots=list(first["slots"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--alternate", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    assert code["commit"] == args.expected_commit
    names, failures = check_mapping(args.runs, args.gpu, args.alternate)
    args.out.mkdir()
    counts = check_existing_queue(args.runs, args.out / "serial-fixture", names, code)
    print(json.dumps(m.seal(args.out / "result.json", dict(synthetic=counts, rejected=failures, legacy_dual=True))))
