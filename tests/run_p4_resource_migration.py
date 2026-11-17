"""Exercise the P4 custody change against the archived controller and live CPU workers."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from research_runtime import code_snapshot, hash_file, require_frozen_code
from transfer import p4_independent_queue as queue, p4_independent_registration as old, p4_resource_migration as migration
from transfer import p4_resource_takeover as custody, p4_runtime_device as device
from tests.test_p4_independent_queue import reject


def origin(root, controller, worker):
    driver = """
from pathlib import Path
import sys
from research_runtime import code_snapshot
from tests.test_p4_independent_queue import bind, cpu_science
from transfer import p4_independent_queue as q, p4_runtime_device as dev
from transfer import p4_independent_registration as s

root, worker = map(Path, sys.argv[1:])
science = cpu_science(worker, root, 22)
code = code_snapshot()
reg = bind(worker, root, science, s.CPU, 22, code, 'prior')
def unavailable(*args):
    raise dev.ResourceUnavailable('Fixture resource intentionally unavailable')
assert q.run(reg, code, fixture_wait=unavailable)['updates_this_invocation'] == 0
"""
    subprocess.run([sys.executable, "-c", driver, str(root), str(worker)], cwd=controller,
                   env=old.environment(controller), check=True)
    return old.pin(root / "prior-resource/resource-scope.json")


def interrupted(root):
    here, parent = root / "isolated-control", root / "isolated-parent"
    parent.mkdir()
    lockfile = parent / ".coordinator.lock"
    lockfile.touch()
    original = parent / "resource-owner.json"
    original.write_bytes(b'{"original":1}\n')
    marker = dict(schema=custody.SCHEMA, control=str(here), parent_execution=str(parent),
                  prior_owner_sha256=hash_file(original), cases=["sample"])
    with device.lease(lockfile) as lock, patch("transfer.p4_resource_takeover.os.replace", side_effect=OSError("crash")):
        try:
            custody.takeover(marker, lock, lambda: None)
        except OSError as error:
            assert str(error) == "crash"
        else:
            raise AssertionError("Missing modeled interruption")
    assert hash_file(original) == marker["prior_owner_sha256"]
    with device.lease(lockfile) as lock:
        reject(lambda: custody.takeover(marker, lock, lambda: None), "isolated intent")
        reject(lambda: custody.verify(marker, lock), "not been transferred")
    return True


def run(root, controller, worker, code):
    root.mkdir()
    item = origin(root, controller, worker)
    resource = old.read(item)
    parent_scope = old.read(resource["science"])
    parent = Path(resource["execution"])
    destination = root / "migrated-execution"
    groups = {}
    for lane, (k, method) in zip(migration.FIXTURE_GROUPS, migration.COHORTS, strict=True):
        cases = [x["id"] for x in parent_scope["cases"] if (x["k"], x["method"]) == (k, method)]
        groups[lane] = dict(physical_uuid=None, cases=cases, updates=600)
    cases = [case for group in groups.values() for case in group["cases"]]
    request = dict(parent_resource=item, parent_release=None, parent_science=resource["science"],
                   parent_execution=str(parent), synthetic=True, execution=str(destination),
                   slots=groups, cases=cases, proofs={})
    broken = []
    for change, message in (
        ({"cases": cases[:-1]}, "complete five-arm"),
        ({"cases": cases[:-1] + [next(x["id"] for x in parent_scope["cases"] if x["id"] not in cases)]}, "complete five-arm"),
        ({"parent_science": dict(path=resource["science"]["path"], sha256="0"*64)}, "scientific scope"),
        ({"slots": {**groups, next(iter(groups)): dict(groups[next(iter(groups))], updates=601)}}, "complete five-arm"),
        ({"slots": {**groups, next(iter(groups)): dict(groups[next(iter(groups))], physical_uuid="GPU-fake")}}, "complete five-arm"),
    ):
        broken.append(reject(lambda: migration.reference({**request, **change}), message))
    assert interrupted(root)
    log_dir = root / "isolated-wait/logs"
    stages = root / "isolated-wait/stages"
    log_dir.mkdir(parents=True)
    stages.mkdir()
    log = log_dir / "missing-worker.log"
    prefix = stages / "00"
    dispatch = dict(log=str(log), case=cases[0])
    (stages / "00.0001.attempt.json").write_text(json.dumps(dispatch))
    reject(lambda: queue.confirmed_migration_wait(root / "isolated-wait", prefix, dispatch, dispatch), "unconfirmed")
    log.write_text(json.dumps(dict(status="waiting_for_resources", case=cases[0], reason="resource", updates=60)))
    reject(lambda: queue.confirmed_migration_wait(root / "isolated-wait", prefix, dispatch, dispatch), "unconfirmed")
    log.write_text(json.dumps(dict(status="waiting_for_resources", case=cases[0], reason="resource", updates=0)))
    queue.confirmed_migration_wait(root / "isolated-wait", prefix, dispatch, dispatch)
    planned = migration.prepare(request, code, root / "migration-control")
    plan = old.read(planned)
    review_only = old.pin(root / "migration-control/release-draft.json")
    reject(lambda: migration.start(planned, review_only, code, transition=True), "Explicit approved")
    approved = migration.seal(root / "fixture-explicit-authority.json", migration.release_for(planned, plan))
    assert hash_file(parent / "resource-owner.json") == plan["prior_owner_sha256"]

    before = root / "parent-status.json"
    before.write_bytes((parent / "status.json").read_bytes())
    stale = cases[0]
    for folder, name in (("stages", "00.intent.json"), ("cases", "state.pt")):
        path = parent / folder / stale
        path.mkdir()
        log = path / name
        log.touch()
        reject(lambda: migration.start(planned, approved, code, transition=True), "has state or a dispatch")
        log.unlink()
        path.rmdir()

    acquired = migration.start(planned, approved, code, transition=True)
    assert acquired["updates"] == 0 and acquired["status"] == "taken_over"
    reject(lambda: migration.start(planned, approved, code, transition=True), "Duplicate takeover")
    parent_code = resource["controller"]["commit"]
    legacy = [sys.executable, "-m", "transfer.p4_independent_queue", "--expected-commit", parent_code,
              "run", "--scope", item["path"], "--scope-sha256", item["sha256"], "--resume"]
    denial = subprocess.run(legacy, cwd=controller, env=old.environment(controller), capture_output=True, text=True)
    (root / "legacy-after-takeover.log").write_text(denial.stdout+denial.stderr)
    assert denial.returncode and "Execution ownership differs" in denial.stderr
    assert not list((parent/"cases").iterdir()) and not list((parent/"stages").iterdir())
    assert (parent/"status.json").read_bytes() == before.read_bytes()

    def pause(lane, case, stage):
        if lane == migration.FIXTURE_GROUPS[1] or case["id"] != stale or stage == ("split", "advance", 2, 30):
            raise device.ResourceUnavailable("Fixture interrupts between registered stops")

    first = migration.start(planned, approved, code, fixture_wait=pause)
    assert first["updates_this_invocation"] == 62 and first["status"] == "waiting_for_resources"
    first_artifact = destination / "cases" / stale / "split/calls/000000-000002.json"
    prior = hash_file(first_artifact)
    resumed = migration.start(planned, approved, code)
    assert resumed["updates_this_invocation"] == 1138 and resumed["status"] == "completed"
    assert hash_file(first_artifact) == prior
    accepted = migration.publish(planned, approved, code)
    receipt = old.read(accepted)
    assert receipt["cases"] == 10 and receipt["optimizer_updates"] == 1200 and receipt["completed_resume_updates"] == 0
    replay = migration.start(planned, approved, code)
    assert replay["updates_this_invocation"] == 0 and migration.publish(planned, approved, code) == accepted
    assert (parent/"status.json").read_bytes() == before.read_bytes()
    unconfirmed = destination / "stages" / stale / "02.receipt.json"
    original_ack = unconfirmed.read_bytes()
    # An accepted checkpoint can restore a lost receipt. Verify before leaving the fixture.
    unconfirmed.unlink()
    assert migration.start(planned, approved, code)["updates_this_invocation"] == 0
    assert unconfirmed.read_bytes() == original_ack
    assert migration.publish(planned, approved, code) == accepted
    return dict(status="passed", prior=planned, legacy_rejection=denial.returncode,
                negative_cases=len(broken)+7, first=first["updates_this_invocation"], resume=resumed["updates_this_invocation"],
                completed_reentry=0, treatment=accepted, synthetic=True, real_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "controller", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    assert code["commit"] == args.expected_commit and args.out.parent != Path(__file__).resolve().parents[1]
    result = run(args.out, args.controller.resolve(), args.worker.resolve(), code)
    print(json.dumps(migration.seal(args.out / "result.json", result)))
