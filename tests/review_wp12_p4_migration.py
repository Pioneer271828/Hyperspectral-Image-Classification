"""Read the two existing CPU fixtures through the combined evidence checks."""

import argparse
import csv
from copy import deepcopy
from pathlib import Path

from evaluation import wp12_p4_migration as evidence
from evaluation.wp12_p4_released_evidence import write_migrated_case_metrics
from research_runtime import code_snapshot, hash_file, require_frozen_code, write_json


def pin(path):
    return {"path": str(path), "sha256": hash_file(path)}


def rejected(call, expected):
    try:
        call()
    except ValueError as error:
        if expected not in str(error):
            raise
        return str(error)
    raise AssertionError("Invalid combined evidence was accepted")


def verify_serial_control(root, control, alternate):
    plan = evidence.read(pin(control / "migration.json"), root)
    scope = evidence.read(plan["migrated_science"], root)
    parent = evidence.read(plan["parent_science"], root)
    evidence.unchanged_science(parent, plan["parent_science"], scope, plan, root)
    gpu, = scope["gpus"]
    assert gpu != alternate and plan["dispatch_mode"] == "serial"
    counts = {"slots": {gpu: {"cases": 10, "optimizer_updates": 1200}}}
    serial = evidence.allocation(plan, scope, counts)
    assert len(serial) == 10 and set(serial.values()) == {gpu}
    names = plan["cases"]
    dual_cases = [*scope["cases"][:5], *({**r, "physical_uuid": alternate} for r in scope["cases"][5:])]
    dual_scope = {**scope, "cases": dual_cases, "gpus": [gpu, alternate]}
    dual_plan = {key: value for key, value in plan.items() if key != "dispatch_mode"}
    dual_plan.update(destinations=[gpu, alternate], slots={
        gpu: dict(physical_uuid=gpu, cases=names[:5], updates=600),
        alternate: dict(physical_uuid=alternate, cases=names[5:], updates=600)})
    dual_counts = {"slots": {physical: {"cases": 5, "optimizer_updates": 600} for physical in (gpu, alternate)}}
    assert len(evidence.allocation(dual_plan, dual_scope, dual_counts)) == 10
    rejected_rows = {
        "parallel_shared": rejected(lambda: evidence.allocation({**plan, "dispatch_mode": "parallel"}, scope, counts), "mode"),
        "serial_distinct": rejected(lambda: evidence.allocation({**dual_plan, "dispatch_mode": "serial"}, dual_scope, dual_counts), "mode"),
        "short_roster": rejected(lambda: evidence.allocation(plan, {**scope, "cases": scope["cases"][:-1]}, counts), "ten unique"),
        "altered_budget": rejected(lambda: evidence.allocation(plan, scope, {"slots": {gpu: {"cases": 10, "optimizer_updates": 1201}}}), "budget"),
        "cohort_split": rejected(lambda: evidence.allocation(
            plan, {**scope, "cases": [{**scope["cases"][0], "physical_uuid": alternate}, *scope["cases"][1:]]},
            counts), "cohort is incomplete or split"),
        "duplicate_gpu_scope": rejected(lambda: evidence.allocation(plan, {**scope, "gpus": [gpu, gpu]}, counts), "mode"),
    }
    return dict(serial=serial, parallel_legacy=True, rejected=rejected_rows, published_real_acceptance=False)


def fixture(root, out, serial_control=None, alternate=None):
    first = root / "wp05-resource-migration-validation03/legacy-fixture"
    second = root / "wp05-resource-migration-fixture02"
    old_accept = evidence.read(pin(first / "matrix-execution/independent-acceptance.json"), root)
    new_accept = evidence.read(pin(second / "migrated-execution/migration-acceptance.json"), root)
    old_scope_pin, new_scope_pin = (r["science_scope"] for r in (old_accept, new_accept))
    old, new = (evidence.read(p, root) for p in (old_scope_pin, new_scope_pin))
    plan = evidence.read(new_accept["resource_scope"], root)
    bound_parent = evidence.read(plan["parent_science"], root)
    assert old_accept["synthetic"] is new_accept["synthetic"] is plan["synthetic"] is True
    evidence.unchanged_science(bound_parent, plan["parent_science"], new, plan, root)
    cases = {case["id"]: case for case in old["cases"]}
    moved = {case["id"]: case for case in new["cases"]}
    departing = {case["id"]: case for case in bound_parent["cases"] if case["id"] in moved}
    locators = {"observation_sha256", "pair_sha256"}
    assert len(departing) == 10
    assert all({k: v for k, v in cases[name].items() if k not in locators} ==
               {k: v for k, v in row.items() if k not in locators} for name, row in departing.items())
    # In the two isolated CPU runs the observation records include absolute input
    # paths. Preserve both actual scope IDs and use each original case for its part
    # of this synthetic-only combined roster. Real intake has a single frozen parent.
    roster = {**old, "cases": [departing.get(c["id"], c) for c in old["cases"]]}
    evidence.check_files(old_accept["protected_files"], root.parent)
    evidence.check_files(new_accept["protected_files"], root.parent)
    rows = {"original": [], "migrated": []}
    for source, accepted, scope_pin, execution, roster_by_id, slots in (
            ("original", old_accept, old_scope_pin, first / "matrix-execution", cases,
             evidence.read(old_accept["resource_scope"], root)["slots"]),
            ("migrated", new_accept, new_scope_pin, second / "migrated-execution", moved, plan["slots"])):
        for slot, slot_rows in slots.items():
            for name in slot_rows["cases"]:
                if source == "original" and name in moved:
                    continue
                p = pin(execution / "cases" / name / "audit.json")
                pins = accepted["protected_files"]
                assert accepted["case_audits"][p["path"]] == p["sha256"]
                a = evidence.published_case(p, roster_by_id[name], scope_pin, accepted["resource_scope"], slot, pins, root)
                source_scope = old if source == "original" else new
                costs = evidence.migrated_cost(Path(p["path"]).parent, source_scope["paths"], pins, root)
                rows[source].append(evidence.case_row(a, scope_pin, p, source, costs))
    kept, changed = rows["original"], rows["migrated"]
    combined = evidence.combine_cases(roster, new, kept, changed, synthetic=True)
    negatives = {
        "missing_original": rejected(lambda: evidence.combine_cases(roster, new, kept[1:], changed, synthetic=True), "disjoint"),
        "missing_migrated": rejected(lambda: evidence.combine_cases(roster, new, kept, changed[1:], synthetic=True), "disjoint"),
        "overlap": rejected(lambda: evidence.combine_cases(roster, new, kept, [kept[0], *changed[1:]], synthetic=True), "disjoint"),
        "wrong_parent": rejected(lambda: evidence.unchanged_science(bound_parent, old_scope_pin, new, plan, root), "configuration"),
        "changed_initial_weights": "",
        "wrong_budget": rejected(lambda: evidence.combine_cases(roster, new, kept,
                           [{**changed[0], "optimizer_updates": 119}, *changed[1:]], synthetic=True), "budget"),
    }
    broken = deepcopy(new)
    broken["cases"][0]["initialization"]["initial_state_sha256"] = "0" * 64
    bad_rows = [{**r, "case": next(c for c in broken["cases"] if c["id"] == r["case"]["id"])} for r in changed]
    negatives["changed_initial_weights"] = rejected(
        lambda: evidence.combine_cases(roster, broken, kept, bad_rows, synthetic=True), "initialization")
    refused_unbound = rejected(lambda: evidence.combine_cases(old, new, kept, changed, synthetic=True), "initialization")
    refused_real_mode = rejected(lambda: evidence.intake(pin(second / "migrated-execution/migration-acceptance.json"),
                                 None, None, None, None, None, root), "Published real combined")
    out.mkdir()
    output = out / "synthetic-case-metrics.csv"
    write_migrated_case_metrics(output, combined)
    with output.open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 60 and len({r["case"] for r in records}) == 20
    assert {r["scope_sha256"] for r in records} == {old_scope_pin["sha256"], new_scope_pin["sha256"]}
    serial = verify_serial_control(root, serial_control, alternate) if serial_control is not None else None
    write_json(out / "result.json", dict(combined_reference=combined["reference"], synthetic=True,
               new_optimizer_updates=0, new_forwards=0, negatives=negatives, unbound_roster=refused_unbound,
               synthetic_rejected_in_real_mode=refused_real_mode,
               serial_registration=serial,
               source_acceptances=[pin(first / "matrix-execution/independent-acceptance.json"),
                                   pin(second / "migrated-execution/migration-acceptance.json")],
               case_metrics=pin(output)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--serial-control", type=Path)
    parser.add_argument("--alternate")
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    assert code["commit"] == args.expected_commit
    fixture(args.runs.resolve(strict=True), args.out, args.serial_control, args.alternate)
