"""Controller faults with frozen workers, real checkpoints and external fixtures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from unittest.mock import Mock

from research_runtime import code_snapshot, hash_file, write_json
from transfer import p4_full_budget_campaign as campaign
from transfer import p4_independent_acceptance as acceptance
from transfer import p4_independent_queue as queue
from transfer import p4_independent_registration as scope, p4_runtime_device as device
from transfer import p4_three_arm_contract as fit


def cpu_science(worker, root, classes):
    folder = root / f"inputs-{classes}"
    folder.mkdir()
    script = """
from pathlib import Path
import sys

from research_runtime import code_snapshot, hash_file, write_json
from tests.test_p4_coverage import fixture
from contextlib import nullcontext
from tests.test_p4_stable_cuda import synthetic_schema
from transfer import p4_full_budget_registration as r, p4_runtime_device as d
from transfer import p4_full_budget_campaign as c

root, classes = Path(sys.argv[1]), int(sys.argv[2])
dest = root.parent / f"cpu{classes}-science"
with synthetic_schema(classes) if classes == 9 else nullcontext():
    parent, _, _ = fixture(root, classes=classes)
    prior = root / "prior.json"
    write_json(prior, {"status": "passed", "query_accessed": False, "protected_files": {str(parent): hash_file(parent)}})
    options = dict(purpose=r.RECOVERY, recovery_classes=9) if classes == 9 else dict(purpose=r.ADMISSION)
    scope = r.prepare(parent, hash_file(parent), prior, hash_file(prior), dest,
                      code_snapshot(), protocol=d.CPU, gpus=(), synthetic=True, **options)
    if classes == 9:
        def launch(command, logs, case, operation, locks):
            wrapped = [*command[:2], "tests.test_p4_stable_cuda", "--worker-classes", "9", *command[3:]]
            return c.invoke(wrapped, logs, case, operation, locks)
        result = c.run(scope, hash_file(scope), root.parent / "cpu9-original", code_snapshot(), launcher=launch)
        assert result["optimizer_updates"] == 120
"""
    subprocess.run([sys.executable, "-c", script, str(folder), str(classes)], cwd=worker,
                   env=scope.environment(worker), check=True)
    return scope.pin(root / f"cpu{classes}-science/scope.json")


def bind(worker, root, science, mode, classes, code, name):
    return scope.prepare(science, root / (name + "-execution"), root / (name + "-resource"),
                         worker, code, mode=mode, classes=classes, recoveries={})


def reject(call, expected):
    try:
        call()
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as error:
        if expected not in str(error):
            raise
        return type(error).__name__
    raise AssertionError(f"Expected rejection: {expected}")


def cpu(worker, root, code):
    frozen = cpu_science(worker, root, 9)
    nine = scope.read(frozen)
    one = bind(worker, root, frozen, scope.CPU, 9, code, "cpu9")
    nine_status = queue.run(one, code)
    nine_receipt = acceptance.publish(one, code)
    assert nine_status["updates_this_invocation"] == 120
    resource = scope.read(one)
    old_folder = root / "cpu9-original/cases" / nine["cases"][0]["id"]
    new_folder = Path(resource["execution"]) / "cases" / nine["cases"][0]["id"]
    old, new = [campaign.read(path / "audit.json") for path in (old_folder, new_folder)]
    fields = ("exact_path_fingerprints", "selected_step", "fit", "exposure", "coverage_participation")
    assert all(old[key] == new[key] for key in fields)

    science = cpu_science(worker, root, 22)
    item = bind(worker, root, science, scope.CPU, 22, code, "matrix")
    resource = scope.read(item)
    old_science = scope.read(science)
    out = Path(resource["execution"])
    blocked = []
    a = resource["slots"]["cpu-a"]["cases"]
    b = resource["slots"]["cpu-b"]["cases"]

    def limited(slot, case, stage):
        if slot == "cpu-b" or case["id"] != a[0] or stage == ("split", "advance", 2, 30):
            blocked.append((slot, case["id"], stage))
            raise device.ResourceUnavailable("Synthetic queue became busy or stayed busy")

    first = queue.run(item, code, fixture_wait=limited)
    assert first["status"] == "waiting_for_resources" and first["updates_this_invocation"] == 62
    assert first["slots"]["cpu-a"]["optimizer_updates"] == 62 and first["slots"]["cpu-b"]["optimizer_updates"] == 0
    assert [len(x) for x in (a, b)] == [10, 10]
    with device.lease(out / ".cpu-b.lock"):
        second = queue.run(item, code, resume=True,
                           fixture_wait=lambda *args: (_ for _ in ()).throw(device.ResourceUnavailable("busy")))
    assert second["updates_this_invocation"] == 0 and "Active lease" in second["waiting"]["cpu-b"]["reason"]
    incomplete = reject(lambda: acceptance.publish(item, code), "audit.json")
    nclobber = reject(lambda: queue.run(item, code), out.name)

    # Remove only the controller acknowledgment, leaving the genuine worker
    # commit and dispatch intent to model an abrupt process exit after commit.
    saved = out / "stages" / a[0] / "02.receipt.json"
    prior_digest = hash_file(saved)
    saved.rename(root / "interrupted-before-ack.saved.json")
    first_call = out / "cases" / a[0] / "split/calls/000000-000002.json"
    first_sha = hash_file(first_call)
    busy_b = Event()

    def b_first(slot, case, stage):
        if slot == "cpu-a":
            raise device.ResourceUnavailable("Synthetic A waits")
        if case["id"] == b[1]:
            busy_b.set()
            raise device.ResourceUnavailable("Synthetic B became busy")

    third = queue.run(item, code, resume=True, fixture_wait=b_first)
    assert third["updates_this_invocation"] == 120 and busy_b.is_set()
    final = queue.run(item, code, resume=True)
    assert final["status"] == "completed" and final["updates_this_invocation"] == 2218
    assert hash_file(saved) == prior_digest and hash_file(first_call) == first_sha
    receipt = acceptance.publish(item, code)
    accepted = scope.read(receipt)
    assert accepted["cases"] == 20 and accepted["optimizer_updates"] == 2400
    assert len(accepted["case_audits"]) == 20 and all(row["cases"] == 10 for row in accepted["slots"].values())
    never = Mock(side_effect=AssertionError("Replayed an accepted stage"))
    done = queue.run(item, code, resume=True, launch=never)
    never.assert_not_called()
    assert done["updates_this_invocation"] == 0 and acceptance.publish(item, code) == receipt

    negatives = {"incomplete": incomplete, "no_clobber": nclobber}
    for name, mutation in {
        "unknown_resource_field": {"extra": True},
        "mapping": {"slots": dict(resource["slots"], bogus={})},
        "directory": {"execution": frozen["path"]},
        "worker": {"worker": {**resource["worker"], "commit": "0"*40}},
        "qualification": {"physical_recoveries": {scope.GPU1: frozen}},
    }.items():
        negatives[name] = reject(lambda: scope.validate({**resource, **mutation}), {
            "unknown_resource_field": "schema", "mapping": "mapping", "directory": "sibling",
            "worker": "identity", "qualification": "recovery proofs"}[name])
    for name, mutation in {
        "budget": {"budget": {**old_science["budget"], "planned_optimizer_updates": 2399}},
        "arithmetic": {"protocol": device.CUDA_FP64_WEIGHT},
    }.items():
        forged = {**old_science, **mutation}
        expected = "budget" if name == "budget" else "CPU engineering"
        negatives[name] = reject(lambda: (
            scope.projection(science, forged, real=False), scope.slots(forged, scope.CPU, 22)), expected)
    negatives["synthetic_predecessor"] = reject(
        lambda: acceptance.predecessor(item, receipt, resource["controller"]), "approved real")

    failures = bind(worker, root, science, scope.CPU, 22, code, "failure")
    other = Event()
    go = Event()
    failure_resource = scope.read(failures)

    class Failing:
        def wait(self):
            other.wait(40)
            return 91

    class Drain:
        def __init__(self, process):
            self.process = process

        def wait(self):
            other.set()
            go.wait(40)
            return self.process.wait()

    def fail_after_dispatch(resource, case, stage, locks, log):
        if case["id"] == a[0]:
            with log.open("x") as stream:
                stream.write("Injected worker exit 91\n")
            return Failing()
        assert case["id"] == b[0] and stage == ("full", "advance", 0, 60)
        return Drain(queue.start_worker(resource, case, stage, locks, log))

    # Release the other worker only after the failure boundary is recorded.
    from concurrent.futures import ThreadPoolExecutor
    from time import monotonic, sleep

    failure_out = Path(failure_resource["execution"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(queue.run, failures, code, launch=fail_after_dispatch)
        until = monotonic() + 50
        while not (failure_out / "failure.json").exists() and monotonic() < until and not job.done():
            sleep(.03)
        go.set()
        negatives["failed_worker"] = reject(job.result, "Stopped dispatch")
    assert (failure_out / "cases" / b[0] / "full" / "calls/000000-000060.json").is_file()
    assert len(list((failure_out / "logs").iterdir())) == 2
    assert campaign.read(failure_out / "status.json")["slots"]["cpu-b"]["optimizer_updates"] == 60
    negatives["failed_resume"] = reject(lambda: queue.run(failures, code, resume=True), "retained")
    negatives["failed_acceptance"] = reject(lambda: acceptance.publish(failures, code), "nonfailed")
    write_json(root / "cpu-evidence.json", {"worker": resource["worker"], "nine": nine_status,
        "nine_receipt": nine_receipt, "nine_original_same": list(fields), "matrix": final, "matrix_receipt": receipt,
        "partial": first, "contended": second, "opposite_queue": third, "complete_resume": done,
        "negative": negatives, "failed_directory": str(failure_out), "drained_worker_updates": 60})


def gpu(worker, root, runs, code):
    holder = runs / "wp05-meta-p4-second-recovery01"
    rows = []
    for classes in (9, 22):
        science = scope.pin(holder / f"classes-{classes}-registration/scope.json")
        name = root.name + f"-cuda{classes}"
        previous = holder / (name + "-resource") / "resource-scope.json"
        item = scope.pin(previous) if previous.exists() else bind(worker, holder, science, scope.CUDA, classes, code, name)
        execution = Path(scope.read(item)["execution"])
        run = queue.run(item, code, resume=execution.exists())
        if run["status"] == "waiting_for_resources":
            rows.append({"classes": classes, "resource": item, "run": run})
            write_json(root / "gpu6-evidence.json", rows)
            return
        receipt = acceptance.publish(item, code)
        resource = scope.read(item)
        case = scope.read(science)["cases"][0]
        original = holder / f"classes-{classes}-execution/cases" / case["id"] / "audit.json"
        audit = Path(resource["execution"]) / "cases" / case["id"] / "audit.json"
        old, new = campaign.read(original), campaign.read(audit)
        assert old["numeric"] == new["numeric"] and new["numeric"]["physical_uuid"] == scope.GPU6
        assert old["exact_path_fingerprints"] == new["exact_path_fingerprints"]
        assert old["selected_step"] == new["selected_step"] and old["fit"] == new["fit"]
        completed = queue.run(item, code, resume=True)
        assert completed["updates_this_invocation"] == 0
        rows.append({"classes": classes, "resource": item, "run": run, "completed": completed,
                     "acceptance": receipt, "audit": scope.pin(audit), "original": scope.pin(original),
                     "numeric": new["numeric"], "original_output_identical": True})
        write_json(root / "gpu6-evidence.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("cpu", "gpu"), required=True)
    options = parser.parse_args()
    code = code_snapshot()
    assert code["commit"] == options.expected_commit and not code["dirty"]
    if options.kind == "cpu":
        cpu(options.worker, options.out, code)
    else:
        gpu(options.worker, options.out, options.runs, code)
