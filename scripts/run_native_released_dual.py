"""Dispatch frozen per-case WP07 stages through two independent worker processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
from threading import Event, Lock
from time import perf_counter


SCHEMA = "wp07_native_released_dual_slots_v2"
OUTCOME = "wp07_native_released_dual_attempt_v2"
WAITING = 75


def arguments():
    def pin(value):
        return dict(zip(("path", "sha256"), value, strict=True))

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("runs", "worker", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("worker-commit", "controller-commit"):
        parser.add_argument("--" + key, required=True)
    for key in ("matrix", "release", "routing"):
        parser.add_argument("--" + key, nargs=2, required=True)
    parser.add_argument("--operation", nargs=5, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("matrix", "release", "routing"):
        setattr(args, key, pin(getattr(args, key)))
    return args


def controller_identity(code_snapshot):
    root = Path(__file__).resolve().parents[1]
    code = code_snapshot(root)
    if code["dirty"]:
        raise ValueError("Freeze the independent controller before dispatch")
    return {key: code[key] for key in ("commit", "source_hash")}


def admitted(args, code):
    from transfer import native_released_campaign as campaign
    from transfer import native_released_real as real
    from transfer import p4_runtime_device as resource
    from transfer.native_scale_campaign import read

    root = args.runs
    scope = read(args.routing, root)
    campaign.require(set(scope) == {"schema", "matrix", "release", "worker", "slots", "synthetic",
                                    "cases", "execution", "controller"},
                     "Unknown dual-slot routing fields")
    synthetic = scope["synthetic"]
    spec, released = campaign.load(root, args.matrix, code, args.release, engineering=synthetic)
    worker = {key: code[key] for key in ("commit", "source_hash")}
    routes = scope["cases"]
    cases = {case["id"]: case for case in spec["cases"]}
    slots = scope["slots"]
    cpu_fixture = synthetic is True and slots in [["a", "b"], ["a"]]
    physical = 1 <= len(slots) <= 2 and all(resource.UUID.fullmatch(slot) for slot in slots)
    valid_slots = cpu_fixture or physical
    campaign.require(type(synthetic) is bool and scope["schema"] == SCHEMA
                     and scope["matrix"] == args.matrix and scope["release"] == args.release
                     and scope["worker"] == worker and args.out == Path(scope["execution"])
                     and args.out == args.out.resolve() and root in args.out.parents
                     and args.out != Path(spec["execution"]) and valid_slots and len(set(slots)) == len(slots)
                     and routes and set(routes.values()) <= set(slots) and set(routes) <= set(cases)
                     and (synthetic or set(routes) == set(cases)),
                     "Routing does not bind the frozen matrix, worker, release, slots and execution")
    selected = {}
    for case_id, slot in routes.items():
        case, release = cases[case_id], released[case_id]
        approval = read(release, root)
        uuid = None if cpu_fixture else slot
        protocol = resource.CPU if uuid is None else resource.CUDA_FP64_WEIGHT
        campaign.require(approval["schema"] == (real.ENGINEERING_RELEASE if synthetic else real.RELEASE)
                         and approval["source_commit"] == worker["commit"]
                         and approval["source_hash"] == worker["source_hash"]
                         and approval["preflight"] == case["preflight"]
                         and approval["case_sha256"] == case["sha256"]
                         and approval["destination"] == case["destination"]
                         and Path(case["destination"]).parent == Path(spec["execution"])
                         and Path(case["destination"]).resolve() == Path(case["destination"])
                         and approval["optimizer_updates"] == case["updates"]
                         and approval["decision"] == "approved" and approval["route"] in campaign.ROUTES
                         and approval["physical_uuid"] == uuid and approval["protocol"] == protocol
                         and approval["raw_gt_authorized"] is approval["query_authorized"] is False,
                         "Case moved across worker, route, source, protocol or device")
        selected[case_id] = (case, release, uuid, approval["route"])
    return scope, selected


def fixture_intake(source):
    from dataclasses import replace
    from types import SimpleNamespace

    from transfer import development_bank as bank
    from transfer import development_bank_training as generator
    from transfer import registered_development as transport
    from transfer.native_scale_campaign import read

    def intake(root, scene):
        classes = {"xuzhou": 9, "whu-honghu": 22}[scene]
        release, _ = generator.fixture(classes)
        rows = [dict(row, samples=[list(point) for point in row["samples"]]) for row in release.rows]
        for row in rows:
            ids = {point[0] for point in row["samples"]}
            first = row["box"][0]*release.shape[1]+row["box"][2]
            free = next((pixel for pixel in (first+1, first+release.shape[1]) if pixel not in ids), None)
            if free is not None:
                row["samples"].append([free, 1])
                row["samples"].sort()
        release = replace(release, rows=tuple(rows))
        prior = transport._pin(source / f"source-{classes}.json")
        authority = read(prior, root)
        return (bank.Bank(authority["bank"], source, release), prior,
                SimpleNamespace(baseline=generator._baseline(release, 81271)))

    return intake


def committed_stage(root, dest, phase, guard, code):
    from model.research_config import digest
    from transfer import native_released_campaign as campaign, registered_development as transport
    from transfer.native_scale_campaign import read

    spec_pin = transport._pin(dest / "registration.json")
    stage_pin, row, packet = campaign.receipt(root, dest, phase, spec_pin)
    spec = read(spec_pin, root)
    case = guard.case
    expected = guard.contract(guard.classes, case["model_seed"], case["arm"], case["factor"],
                              code["commit"], numeric=spec["numeric"])
    tail = spec["steps"]
    prefix = spec["prefix"]
    campaign.require(spec == expected and packet["registration"] == digest(spec)
                     and packet["cursor"] == (prefix if phase == "prefix" else tail)
                     and row["updates"] == {"full": tail, "prefix": prefix, "resume": tail-prefix,
                                            "completed": 0}[phase],
                     "Stage cannot be reused under a different frozen case")
    return stage_pin, row["updates"]


def operate(args, scope, selected, code):
    from scripts import run_native_released_points as worker
    from transfer import native_released_campaign as campaign, p4_runtime_device as resource
    from transfer import registered_development as transport
    from transfer.native_scale_campaign import read

    case_id, phase, receipt_path, matrix_fd, journal_fd = args.operation
    matrix_out = Path(read(args.matrix, args.runs)["execution"])
    resource.check_inherited_lease(int(matrix_fd), matrix_out / ".matrix.lock")
    resource.check_inherited_lease(int(journal_fd), args.out / ".matrix.lock")
    receipt_path = Path(receipt_path)
    case, release, gpu, route = selected[case_id]
    slot = scope["cases"][case_id]
    campaign.require(receipt_path.parent == args.out and receipt_path == receipt_path.resolve()
                     and not receipt_path.exists() and phase in (*campaign.ROUTES[route], "audit")
                     and os.environ.get("CUDA_VISIBLE_DEVICES") == (gpu or ""),
                     "Worker operation has another phase, device or receipt location")
    intake = fixture_intake(Path(case["preflight"]["path"]).parent) if scope["synthetic"] else None
    guard, dest = campaign.authorized_case(args.runs, case, release, code, intake=intake)
    if phase == "audit":
        existing = dest / "dual-audit.json"
        if existing.exists():
            result = transport._pin(existing)
            value = read(result, args.runs)
            pins = [committed_stage(args.runs, dest, name, guard, code)[0] for name in campaign.ROUTES[route]]
            campaign.require(value["case"] == case_id and value["stages"] == pins
                             and value["optimizer_updates"] == case["updates"]
                             and value["completed_resume_updates"] == 0, "Committed audit differs from its phases")
        else:
            if gpu:
                resource.gate(gpu)
            result = transport._json_new(existing, campaign.audit_case(
                args.runs, case, release, code, intake=intake, cli=worker))
        new = 0
    elif (dest / f"{phase}.json").is_file():
        result, _ = committed_stage(args.runs, dest, phase, guard, code)
        new = 0
    else:
        campaign.require(not (dest / f"{phase}.intent.json").exists(),
                         "Interrupted stage retains its original intent")
        if gpu:
            resource.gate(gpu)
        result = worker.real_stage(args.runs, dest, case["preflight"], release, case_id, phase,
                                   code, gpu, intake=intake)
        _, new = committed_stage(args.runs, dest, phase, guard, code)
    transport._json_new(receipt_path, {"case": case_id, "phase": phase, "slot": slot,
                         "physical_uuid": gpu, "protocol": guard._protocol,
                         "pid": os.getpid(), "worker": scope["worker"], "result": result, "new_updates": new})


def dispatch(args, scope, selected, code):
    from transfer import native_released_campaign as campaign, p4_runtime_device as resource
    from transfer import registered_development as transport
    from transfer.native_scale_campaign import read
    from wp07_dual_journal import Journal, Operation, seal

    args.out.mkdir(exist_ok=True)
    owner = args.out / "owner.json"
    matrix_root = Path(read(args.matrix, args.runs)["execution"])
    campaign.require(matrix_root == matrix_root.resolve() and args.runs in matrix_root.parents,
                     "Matrix output escapes the authorized runs root")
    matrix_root.mkdir(exist_ok=True)
    with (resource.lease(args.out / ".matrix.lock") as journal_lease,
          resource.lease(matrix_root / ".matrix.lock") as matrix_lease):
        identity = {"routing": args.routing, "controller": scope["controller"]}
        matrix_owner = matrix_root / "owner.json"
        authority = {"matrix": args.matrix, "release": args.release}
        if matrix_owner.exists():
            campaign.require(read(transport._pin(matrix_owner), args.runs) == authority, "Matrix has another owner")
        else:
            campaign.require({p.name for p in matrix_root.iterdir()} == {".matrix.lock"},
                             "Unowned matrix execution directory")
            transport._json_new(matrix_owner, authority)
        if owner.exists():
            campaign.require(read(transport._pin(owner), args.runs) == identity, "Dual-slot journal has another owner")
        else:
            campaign.require({p.name for p in args.out.iterdir()} == {".matrix.lock"}, "Unowned dual-slot journal")
            transport._json_new(owner, identity)
        expected = {(case, phase): {"case": case, "phase": phase, "slot": scope["cases"][case],
                    "physical_uuid": gpu, "protocol": resource.CPU if gpu is None else resource.CUDA_FP64_WEIGHT,
                    "worker": scope["worker"]}
                    for case, (_, _, gpu, route) in selected.items() for phase in (*campaign.ROUTES[route], "audit")}
        journal = Journal(args.runs, args.out, identity, expected)
        session, attempt = journal.start()
        began = perf_counter()
        stopping, lock = Event(), Lock()
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        dispatch_count = 0
        acknowledged = {}

        def failure(reason, operation=None, returncode=None):
            # The first observation and next dispatch obey the same mutex.
            with lock:
                journal.failure(attempt, dispatch=operation, reason=reason, returncode=returncode)
                stopping.set()

        def lane(slot):
            nonlocal dispatch_count
            rows, pending = [], None
            try:
                with resource.lease(args.out / f".lane-{slot}.lock"):
                    for case_id, (_, _, gpu, route) in selected.items():
                        if scope["cases"][case_id] != slot:
                            continue
                        for phase in (*campaign.ROUTES[route], "audit"):
                            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu or "", "PYTHONDONTWRITEBYTECODE": "1",
                                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                            started = perf_counter()
                            with lock:
                                if stopping.is_set():
                                    return {"status": "drained", "stages": rows}
                                order = dispatch_count
                                op = Operation(args.out, session, order)
                                pending = seal(op.path(".dispatch.json"), {
                                    **expected[case_id, phase], "attempt": attempt, "order": order})
                                dispatch_count += 1
                                fds = (matrix_lease.fileno(), journal_lease.fileno())
                                path = op.path(".receipt.json")
                                child = [*command, "--operation", case_id, phase, str(path), *map(str, fds)]
                                with op.path(".log").open("xb") as log:
                                    process = subprocess.Popen(child, env=env, cwd=args.worker,
                                                               stdout=log, stderr=subprocess.STDOUT, pass_fds=fds)
                                lp = seal(op.path(".launch.json"), {"dispatch": pending, "pid": process.pid})
                            returncode = process.wait()
                            if returncode not in (0, WAITING):
                                failure("worker_nonzero_exit", pending, returncode)
                                return {"status": "failed", "case": case_id, "phase": phase,
                                        "returncode": returncode, "stages": rows}
                            pin = None
                            if returncode == 0:
                                pin = transport._pin(path)
                                committed = read(pin, args.runs)
                                journal.check_receipt(committed, expected[case_id, phase], process.pid)
                                rows.append({**committed, "started_seconds": started-began,
                                             "finished_seconds": perf_counter()-began})
                            elif path.exists():
                                raise ValueError("Worker waiting after writing a receipt")
                            with lock:
                                acknowledged[order] = seal(op.path(".ack.json"),
                                    {"launch": lp, "returncode": returncode, "receipt": pin})
                            pending = None
                            if returncode == WAITING:
                                return {"status": "waiting", "case": case_id, "phase": phase, "stages": rows}
            except resource.ResourceUnavailable as error:
                return {"status": "waiting", "reason": str(error), "stages": rows}
            except Exception as error:
                failure(str(error), pending)
                return {"status": "failed", "reason": str(error), "stages": rows}
            return {"status": "completed", "stages": rows}

        with ThreadPoolExecutor(max_workers=len(scope["slots"])) as pool:
            outcomes = dict(zip(scope["slots"], pool.map(lane, scope["slots"]), strict=True))
        failed = any(row["status"] == "failed" for row in outcomes.values())
        status = "failed" if failed else ("waiting" if any(row["status"] == "waiting" for row in outcomes.values()) else "completed")
        report = {"schema": OUTCOME, "status": status, "worker": scope["worker"], "controller": scope["controller"],
                  "attempt": attempt, "operation_acks": [acknowledged[k] for k in sorted(acknowledged)],
                  "first_failure": transport._pin(journal.first) if failed else None,
                  "routing": args.routing, "synthetic": scope["synthetic"], "slots": outcomes,
                  "physical_uuids": sorted({gpu for _, _, gpu, _ in selected.values() if gpu}),
                  "query_accessed": False, "raw_gt_loaded": False,
                  "new_updates": sum(stage["new_updates"] for lane in outcomes.values() for stage in lane["stages"]),
                  "elapsed_seconds": perf_counter()-began}
        print(seal(args.out / f"summary-{session:04d}.json", report), flush=True)
        return 1 if failed else 0


def main():
    args = arguments()
    args.worker = args.worker.resolve(strict=True)
    args.runs = args.runs.resolve(strict=True)
    sys.path.insert(0, str(args.worker))
    import torch
    from research_runtime import code_snapshot, require_frozen_code
    from transfer import p4_runtime_device as resource

    code = code_snapshot()
    require_frozen_code(code)
    identity = controller_identity(code_snapshot)
    if (code["commit"], identity["commit"]) != (args.worker_commit, args.controller_commit):
        raise ValueError("External controller or science worker SHA differs")
    scope, selected = admitted(args, code)
    if identity != scope["controller"]:
        raise ValueError("Controller differs from the approved routing")
    torch.set_num_threads(1)
    try:
        if args.operation:
            operate(args, scope, selected, code)
            return 0
        return dispatch(args, scope, selected, code)
    except resource.ResourceUnavailable as error:
        print(str(error), flush=True)
        return WAITING


if __name__ == "__main__":
    sys.exit(main())
