"""Dispatch frozen paired P4 workers in independent resource queues."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock

from research_runtime import code_snapshot, hash_file, now, require_frozen_code, write_json
from transfer import p4_full_budget_audit as audit, p4_full_budget_campaign as campaign
from transfer import p4_independent_registration as scope, p4_runtime_device as device


def stages(science):
    result = []
    for name, stops in science["paths"].items():
        start = 0
        for until in stops:
            result.append((name, "advance", start, until))
            start = until
        result.append((name, "completed", start, None))
    return [*result, (None, "audit", science["budget"]["steps"], None)]


def evidence(out, case, science, sha, stage):
    name, operation, start, until = stage
    folder = campaign.bound_folder(out, case, name)
    if operation == "audit":
        path = folder / "audit.json"
        if path.exists():
            campaign.accepted(folder, science, case, sha)
            return scope.pin(path)
        return None
    cursor, calls = audit.ledger(folder, case, sha, path_name=name, stops=science["paths"][name])
    if operation == "completed":
        path = folder / "completed-resume.json"
        if not path.exists():
            return None
        data = campaign.read(path)
        scope.require(cursor == start == data["cursor"] and data["scope_sha256"] == sha
                      and data["case"] == case["id"] and data["path"] == name
                      and data["new_optimizer_updates"] == 0 and data["numeric"] == calls[-1]["numeric"]
                      and data["state_sha256"] == hash_file(folder / "state.pt")
                      and data["result_sha256"] == hash_file(folder / "result.json"),
                      "Invalid completed recovery evidence")
        return scope.pin(path)
    if cursor < until:
        scope.require(cursor == start, "Unexpected committed advance cursor")
        return None
    return scope.pin(folder / "calls" / f"{start:06d}-{until:06d}.json")


def start_worker(resource, case, stage, locks, log):
    name, operation, _, until = stage
    root = Path(resource["worker"]["root"])
    entry = ["-m", "transfer.p4_full_budget_campaign"]
    if resource["classes"] == 9:
        entry = ["-m", "tests.test_p4_stable_cuda", "--worker-classes", "9"]
    command = [sys.executable, *entry, "worker", "--expected-commit", resource["worker"]["commit"],
               "--scope", resource["science"]["path"], "--scope-sha256", resource["science"]["sha256"],
               "--out", resource["execution"], "--case", case["id"], "--operation", operation,
               "--coordinator-fd", str(locks[0].fileno()), "--slot-fd", str(locks[1].fileno())]
    if name is not None:
        command.extend(["--path", name])
    if until is not None:
        command.extend(["--until", str(until)])
    with log.open("xb") as stream:
        return subprocess.Popen(command, cwd=root, env=scope.environment(root, case["physical_uuid"]),
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                pass_fds=tuple(lock.fileno() for lock in locks))


def summary(resource, out, science):
    sha = resource["science"]["sha256"]
    cases = {case["id"]: case for case in science["cases"]}
    result = {}
    for slot, spec in resource["slots"].items():
        members = [cases[i] for i in spec["cases"]]
        audits = [case for case in members if (campaign.bound_folder(out, case) / "audit.json").exists()]
        for case in audits:
            campaign.accepted(campaign.bound_folder(out, case), science, case, sha)
        result[slot] = {"planned_cases": len(members), "accepted_cases": len(audits),
                        "planned_updates": spec["updates"],
                        "optimizer_updates": sum(campaign.case_progress(science, out, c, sha) for c in members)}
    return result


def run(item, code, *, resume=False, probe=device.gate, launch=start_worker, fixture_wait=None):
    resource = scope.read(item)
    science = scope.validate(resource, code=code, live_worker=True)
    scope.require(resource["mode"] != scope.REAL, "Real independent scope is preparation-only")
    scope.require(fixture_wait is None or resource["mode"] == scope.CPU, "Fixture gates are CPU-only")
    out, sha = Path(resource["execution"]), resource["science"]["sha256"]
    owner = {"resource": item, "worker_owner": campaign.owner(out, resource["science"]["path"], sha)}
    if not resume:
        out.mkdir()
    else:
        scope.require(out.is_dir(), "Resume requires the original execution directory")
    markers = out / "stages"
    cases = {case["id"]: case for case in science["cases"]}
    with ExitStack() as leases:
        coordinator = leases.enter_context(device.lease(out / ".coordinator.lock"))
        if resume:
            scope.require(campaign.read(out / "resource-owner.json") == owner
                          and campaign.read(out / "owner.json") == owner["worker_owner"],
                          "Execution ownership differs")
        else:
            for directory in ("cases", "logs", "stages"):
                (out / directory).mkdir()
            write_json(out / "owner.json", owner["worker_owner"])
            write_json(out / "resource-owner.json", owner)
        scope.require(not (out / "failure.json").exists(), "Failed execution is retained for diagnosis")
        cpu_lock = None
        if resource["mode"] == scope.CPU:
            cpu_lock = leases.enter_context(device.lease(campaign.slot_lock(out, None)))
        else:
            root = campaign.slot_lock(out, next(iter(resource["slots"]))).parent
            root.mkdir(exist_ok=True)
            scope.require(root.resolve() == root, "Shared lease root cannot traverse symlinks")
        stop, mutex = Event(), Lock()
        waits, errors = {}, {}
        updates = 0

        def step(slot, case, stage, index, lock):
            nonlocal updates
            name, operation, start, until = stage
            prefix = markers / case["id"] / f"{index:02d}"
            intent_path, receipt_path = prefix.with_suffix(".intent.json"), prefix.with_suffix(".receipt.json")
            prior = evidence(out, case, science, sha, stage)
            binding = {"resource_sha256": item["sha256"], "case": case["id"], "slot": slot,
                       "path": name, "operation": operation, "start": start, "until": until}
            if receipt_path.exists():
                receipt = campaign.read(receipt_path)
                scope.require(prior is not None and receipt["evidence"] == prior
                              and receipt["attempt"] == scope.pin(intent_path), "Committed dispatch evidence differs")
                return
            if intent_path.exists():
                saved = campaign.read(intent_path)
                scope.require(all(saved[key] == value for key, value in binding.items()),
                              "Unfinished dispatch identity differs")
                if prior is not None:
                    write_json(receipt_path, {"attempt": scope.pin(intent_path), "evidence": prior})
                    return
            else:
                scope.require(prior is None, "Worker evidence without a recorded intent")
            with mutex:
                if stop.is_set():
                    return
                gpu = case["physical_uuid"]
                if fixture_wait is not None:
                    fixture_wait(slot, case, stage)
                inventory = probe(gpu) if gpu is not None else None
                if gpu is not None:
                    scope.require(inventory["physical_uuid"] == gpu and device.eligible(inventory),
                                  "Physical admission does not satisfy the resource scope")
                prefix.parent.mkdir(exist_ok=True)
                attempts = sorted(prefix.parent.glob(prefix.name + ".*.attempt.json"))
                serial = len(attempts) + 1
                attempt = prefix.parent / f"{prefix.name}.{serial:04d}.attempt.json"
                log = out / "logs" / f"{case['id']}-{index:02d}-{serial:04d}.log"
                dispatch = {**binding, "utc": now(), "inventory": inventory,
                            "cuda_visible_devices": gpu or "", "lease": lock.name, "log": str(log)}
                write_json(attempt, dispatch)
                write_json(intent_path, dispatch)
                process = launch(resource, case, stage, (coordinator, cpu_lock or lock), log)
            exit_code = process.wait()
            if exit_code != 0:
                raise RuntimeError(f"Frozen worker exited {exit_code}. See {log}")
            output = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
            if output["status"] == "waiting_for_resources":
                scope.require(evidence(out, case, science, sha, stage) is None and output["updates"] == 0
                              and output["case"] == case["id"], "Waiting worker changed committed state")
                raise device.ResourceUnavailable(output["reason"])
            expected = {"advance": "advanced", "completed": "verified_completed", "audit": "accepted"}
            delta = until - start if operation == "advance" else 0
            scope.require(output == {"status": expected[operation], "case": case["id"], "updates": delta},
                          "Frozen worker result differs from the requested stage")
            result = evidence(out, case, science, sha, stage)
            scope.require(result is not None, "Worker did not commit the requested stage")
            write_json(receipt_path, {"attempt": scope.pin(intent_path), "evidence": result})
            with mutex:
                updates += delta

        def queue(slot, spec):
            active = None
            try:
                lock_path = (out / f".{slot}.lock" if cpu_lock else
                             campaign.slot_lock(out, spec["physical_uuid"]))
                scope.require(lock_path.resolve() == lock_path, "Slot lease cannot traverse symlinks")
                with device.lease(lock_path) as lock:
                    for case_id in spec["cases"]:
                        active = case_id
                        for index, stage in enumerate(stages(science)):
                            if stop.is_set():
                                return
                            step(slot, cases[case_id], stage, index, lock)
            except device.ResourceUnavailable as error:
                waits[slot] = {"case": active, "reason": str(error)}
            except Exception as error:
                with mutex:
                    stop.set()
                    errors[slot] = {"case": active, "reason": repr(error)}
                    write_json(out / "failure.json", errors)

        with ThreadPoolExecutor(max_workers=len(resource["slots"])) as pool:
            jobs = [pool.submit(queue, slot, spec) for slot, spec in resource["slots"].items()]
            for job in jobs:
                job.result()
        if errors:
            raise RuntimeError(f"Stopped dispatch and drained in-flight workers. See {out / 'failure.json'}")
        report = summary(resource, out, science)
        for name, value in report.items():
            value["state"] = "waiting_for_resources" if name in waits else "completed"
            scope.require(name in waits or value["accepted_cases"] == value["planned_cases"],
                          "Finished queue is missing registered cases")
        require_frozen_code(code)
        status = {"status": "waiting_for_resources" if waits else "completed", "resource_sha256": item["sha256"],
                  "slots": report, "waiting": waits, "updates_this_invocation": updates}
        write_json(out / "status.json", status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    modes = parser.add_subparsers(dest="command", required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("--science", type=Path, required=True)
    prepare.add_argument("--science-sha256", required=True)
    prepare.add_argument("--worker-root", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--execution", type=Path, required=True)
    prepare.add_argument("--mode", choices=(scope.REAL, scope.CPU, scope.CUDA), required=True)
    prepare.add_argument("--classes", type=int, choices=(9, 22), required=True)
    prepare.add_argument("--recovery", action="append", nargs=3, metavar=("UUID", "PATH", "SHA256"), default=[])
    for name in ("run", "verify", "accept"):
        sub = modes.add_parser(name)
        sub.add_argument("--scope", type=Path, required=True)
        sub.add_argument("--scope-sha256", required=True)
        if name == "run":
            sub.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    code = code_snapshot()
    scope.require(code["commit"] == args.expected_commit, "Controller commit differs")
    if args.command == "prepare":
        recoveries = {gpu: {"path": path, "sha256": sha} for gpu, path, sha in args.recovery}
        scope.require(len(recoveries) == len(args.recovery), "Duplicate recovery binding")
        result = scope.prepare({"path": str(args.science.resolve()), "sha256": args.science_sha256},
                               args.execution, args.out, args.worker_root, code,
                               mode=args.mode, classes=args.classes, recoveries=recoveries)
    else:
        item = {"path": str(args.scope.resolve()), "sha256": args.scope_sha256}
        if args.command == "run":
            result = run(item, code, resume=args.resume)
        elif args.command == "accept":
            from transfer.p4_independent_acceptance import publish

            result = publish(item, code)
        else:
            resource = scope.read(item)
            science = scope.validate(resource, code=code, live_worker=True)
            result = {"status": "verified", "scene": resource["mode"], "science": resource["science"],
                      "same_science_fields": len(resource["science_fields"]), "slots": resource["slots"],
                      "protected_inputs": len(science["protected_files"]), "optimizer_updates": 0}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
