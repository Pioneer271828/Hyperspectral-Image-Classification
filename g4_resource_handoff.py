"""Resource-only handoff of an existing full54 launch; frozen model code is loaded separately.

The original launch/registration/model provenance is never rewritten. A pinned
resource appendix and per-stage events record this external scheduling layer.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

POLICY = "g4_registered_resource_handoff_v1"
HERE = Path(__file__).resolve().parent
STEPS = {"first2": ("interrupted", 2), "continue": ("completed", 1498), "readonly": ("verified_completed", 0)}
NEXT = {"first2": "continue", "continue": "readonly", "readonly": "audit", "audit": None}


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()) + "-" + str(threading.get_ident()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tree(root): return {str(p): sha(p) for p in Path(root).rglob("*") if p.is_file() and p.name != ".lock"}


def check(pins):
    if any(sha(p) != h for p, h in pins.items()): raise ValueError("Protected handoff file changed")


def bootstrap(frozen):
    frozen = Path(frozen).resolve(strict=True)
    sys.path.insert(0, str(frozen)); os.chdir(frozen)
    import research_runtime as rr
    from g4_matched_target_execution import read_registration, read_full_case, case_data, unchanged_producer
    from g4_matched_target_study import EXPECTED, runtime
    from g4_matched_target_audit import audit_case, ACCEPTANCE_POLICY
    from joint_training import run_training
    if Path(rr.__file__).resolve().parent != frozen: raise ValueError("Model imports are not from pinned frozen checkout")
    return locals()


def process(pid):
    p = Path("/proc") / str(pid)
    if not p.exists(): return None
    try:
        tail = (p / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "state": tail[0], "ppid": int(tail[1]), "start_ticks": tail[19],
                "command_sha256": hashlib.sha256((p / "cmdline").read_bytes()).hexdigest(),
                "command": (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
                "cwd": str((p / "cwd").resolve()), "uid": p.stat().st_uid}
    except (FileNotFoundError, ProcessLookupError): return None


def same_process(pin):
    now = process(pin["pid"])
    return now and all(now[k] == pin[k] for k in ("start_ticks", "command_sha256", "cwd", "uid"))


def live_children(pid):
    rows = [process(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()]
    return [p for p in rows if p and p["ppid"] == pid and p["state"] != "Z"]


def phase(case):
    folder = Path(case["out"])
    if not folder.exists(): return "first2"
    if (folder / ".lock").exists(): raise ValueError("A case still owns its training lock")
    value = read(folder / "progress.json")
    if value["status"] == "interrupted" and value["cursor"] == 2 and not (folder / "result.json").exists():
        if sha(folder / value["state_file"]) != value["state_sha256"]: raise ValueError("Partial state changed")
        return "continue"
    if value["status"] == "completed" and value["cursor"] == 1500 and (folder / "result.json").exists(): return "readonly"
    raise ValueError("Only unstarted, exact2 checkpoint or completed1500 handoff allowed")


class Ledger:
    """Single ownership of each case stage, with one active subprocess per GPU."""
    def __init__(self, jobs, rows, order):
        if len(set(order)) != len(order) or any(s not in STEPS for _, s in jobs): raise ValueError("Unknown/duplicate stage contract")
        if len({n for n, _ in jobs}) != len(jobs) or len({r["name"] for r in rows}) != len(rows):
            raise ValueError("Duplicate handoff case")
        if {n for n, _ in jobs} & {r["name"] for r in rows} or {n for n, _ in jobs} | {r["name"] for r in rows} != set(order):
            raise ValueError("Exact completed/remaining partition required")
        self.pending = deque(jobs); self.active = {}; self.rows = {r["name"]: r for r in rows}
        self.events = []; self.error = None; self.order = order
        self.lock = threading.RLock(); self.started = {}; self.stages = set()

    def claim(self, gpu):
        with self.lock:
            if self.error or not self.pending: return None
            if gpu in self.active: raise ValueError("GPU already owns a task")
            job = self.pending.popleft()
            if job in self.stages or any(j[0] == job[0] for j in self.active.values()): raise ValueError("Duplicate active case stage")
            self.stages.add(job); self.active[gpu] = job; self.started.setdefault(job[0], time.monotonic())
            return job

    def finish(self, gpu, event, row=None):
        with self.lock:
            name, stage = self.active.pop(gpu)
            if (event["case"], event["stage"], event["gpu"]) != (name, stage, gpu): raise ValueError("Stage owner changed")
            self.events.append(event)
            if row is not None:
                if stage != "audit" or row["name"] != name or name in self.rows: raise ValueError("Unexpected acceptance row")
                self.rows[name] = row
            elif NEXT[stage] is not None and not self.error:
                self.pending.appendleft((name, NEXT[stage]))


def gate(gpu, slot, aborted=lambda: False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, str(gpu)): raise ValueError("Explicit physical GPU mismatch")
    while not aborted():
        free, util = map(int, subprocess.check_output(["nvidia-smi", "-i", str(gpu),
            "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().split(","))
        write(slot, {"status": "ready" if free >= 12288 and util <= 60 else "waiting_for_resources",
                     "gpu": gpu, "free_mib": free, "utilization": util})
        if free >= 12288 and util <= 60: return True
        time.sleep(5)
    return False


def load_amendment(path, expected):
    path = Path(path).resolve(strict=True)
    if sha(path) != expected: raise ValueError("Pinned resource appendix required")
    a = read(path)
    if (a["policy"] != POLICY or len(a["gpus"]) not in (2, 3) or len(set(a["gpus"])) != len(a["gpus"])
            or any(type(g) is not int or g < 0 for g in a["gpus"]) or a["optimizer_updates_total"] != 81000
            or a["query_labels_accessed"] is not False): raise ValueError("Resource/budget/label contract differs")
    b = bootstrap(a["frozen_checkout"]); rr = b["rr"]
    producer = rr.code_snapshot()
    if producer != a["producer"]: raise ValueError("Frozen producer identity changed")
    controller = rr.code_snapshot(a["controller_checkout"])
    if controller != a["controller_code"] or controller["dirty"]: raise ValueError("Scheduling code changed")
    check(a["pins"])
    launch = read(a["launch"]["path"])
    if launch["code"] != producer: raise ValueError("Launch/producer differ")
    Ledger(list(a["jobs"].items()), a["completed_rows"], [c["name"] for c in launch["cases"]])
    remaining = sum(1500 if s == "first2" else 1498 if s == "continue" else 0 for s in a["jobs"].values())
    spent_partial = sum(2 if s == "continue" else 1500 if s == "readonly" else 0 for s in a["jobs"].values())
    if remaining != a["remaining_updates"] or 1500 * len(a["completed_rows"]) + spent_partial + remaining != 81000:
        raise ValueError("Exact81000 continuation budget required")
    b["read_registration"](a["registration"]["path"], a["registration"]["sha256"], producer)
    return a, b, launch


def provenance(case, a, b):
    # Identical to the original frozen worker: resource changes live in the appendix.
    items, value, guard = b["case_data"](case)
    value.update(source_spec=case["source_spec"], m4_mode=case["mode"], matched_full_target_launch={
        "protocol": b["EXPECTED"]["protocol"], "launch_sha256": a["launch"]["sha256"], "case": case["name"],
        "registration_sha256": a["registration"]["sha256"], "initial_state_sha256": case["initial_state_sha256"]})
    pins = value["protected_files"]
    pins.update({a[k]["path"]: a[k]["sha256"] for k in ("launch", "registration")})
    reg = read(a["registration"]["path"])
    for k in ("engineering_acceptance", "analysis_plan"): pins[reg[k]["path"]] = reg[k]["sha256"]
    spec = case["source_spec"]
    for field, path in spec.items():
        if not field.endswith("sha256"): pins[path] = spec["source_weight_sha256" if field == "source_path" else field + "_sha256"]
    if spec: pins[spec["source_path"] + ".json"] = spec["source_sidecar_sha256"]
    def finish():
        guard(); b["rr"].require_frozen_code(a["producer"]); check(pins); check(a["pins"])
        if b["rr"].code_snapshot(a["controller_checkout"]) != a["controller_code"]: raise ValueError("Resource code changed")
    return items, value, finish


def worker(args):
    a, b, launch = load_amendment(args.amendment, args.amendment_sha256)
    if args.gpu not in a["gpus"] or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu): raise ValueError("Unregistered physical GPU")
    if args.case not in a["jobs"]: raise ValueError("Completed historical case cannot be retrained")
    case = next(c for c in launch["cases"] if c["name"] == args.case)
    case, _ = b["read_full_case"](a["launch"]["path"], a["launch"]["sha256"], a["registration"]["path"],
        a["registration"]["sha256"], a["producer"], (case["target"], case["seed"], case["method"], case["mode"]))
    gate(args.gpu, Path(args.amendment).parent / ("gpu-" + str(args.gpu) + ".json"))
    if args.stage == "audit":
        row = b["audit_case"](case, a["producer"], a["launch"]["sha256"], a["registration"]["sha256"])
        out = Path(args.amendment).parent / "events" / (args.case + "-audit.json")
        if out.exists(): raise ValueError("Audit output already exists")
        write(out, row); print(json.dumps({"audit": str(out), "sha256": sha(out)})); return
    current = phase(case)
    if (args.stage == "first2" and current != "first2") or (args.stage == "continue" and current != "continue") or (args.stage == "readonly" and current != "readonly"):
        raise ValueError("Case cursor does not match assigned operation")
    items, value, finish = provenance(case, a, b); finish()
    before = tree(case["out"]) if args.stage == "readonly" else None
    result = b["run_training"](case["settings"], items, case["out"], value, a["producer"], device="cuda",
        resume=args.stage != "first2", stop_after=2 if args.stage == "first2" else None,
        finish_guard=finish, runtime_adapter=b["runtime"](case, items))
    if (result["operation"], result["updates_this_call"]) != STEPS[args.stage]: raise ValueError("Stage update budget differs")
    if before is not None and before != tree(case["out"]): raise ValueError("Readonly stage mutated files")
    print(json.dumps({k: result[k] for k in ("operation", "updates_this_call")}))


def controller(args):
    a, b, launch = load_amendment(args.amendment, args.amendment_sha256)
    old = process(a["legacy_process"]["pid"])
    if old and old["state"] != "Z": raise ValueError("Legacy controller still alive; no double ownership")
    import fcntl
    root = Path(a["run"]); folder = Path(args.amendment).parent
    lock = (folder / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(folder / "controller-start.json", {"pid": os.getpid(), "amendment_sha256": args.amendment_sha256})
    order = [c["name"] for c in launch["cases"]]
    ledger = Ledger(list(a["jobs"].items()), a["completed_rows"], order)
    begun = time.monotonic()
    def publish():
        with ledger.lock:
            rows = [ledger.rows[n] for n in order if n in ledger.rows]
            write(root / "progress.json", rows)
            write(root / "status.json", {"status": "failed" if ledger.error else "running", "case": "parallel_resource_handoff",
                "operation": "resource_amended_queue", "completed_cases": len(rows), "planned_optimizer_updates": 81000,
                "controller_pid": os.getpid(), "resource_amendment": str(Path(args.amendment).resolve()),
                "resource_amendment_sha256": args.amendment_sha256, "gpus": a["gpus"],
                "active_cases": [{"gpu": g, "case": n, "stage": s} for g, (n, s) in ledger.active.items()], "error": ledger.error})
            write(folder / "events.json", ledger.events)
    def queue(gpu):
        while True:
            with ledger.lock:
                if ledger.error or not ledger.pending: return
            if not gate(gpu, folder / ("gpu-" + str(gpu) + ".json"), lambda: bool(ledger.error) or not ledger.pending): return
            job = ledger.claim(gpu)
            if job is None: return
            name, stage = job; publish()
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--amendment", args.amendment,
                "--amendment-sha256", args.amendment_sha256, "--case", name, "--stage", stage, "--gpu", str(gpu)]
            path = folder / "logs" / (name + "-" + stage + ".log")
            started = time.monotonic()
            try:
                with path.open("x") as log:
                    proc = subprocess.Popen(command, cwd=a["frozen_checkout"], env=dict(os.environ,
                        CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1"), stdout=log, stderr=subprocess.STDOUT)
                    pid = proc.pid; rc = proc.wait()
                if rc: raise RuntimeError("Resource stage failed: " + str(path))
                result = json.loads(path.read_text().splitlines()[-1])
                row = None
                if stage == "audit":
                    if sha(result["audit"]) != result["sha256"]: raise ValueError("Audit receipt changed")
                    row = read(result["audit"])
                    row.update(completed_resume_updates=0, elapsed_seconds=time.monotonic() - ledger.started[name],
                        elapsed_scope="resource_handoff_case_segment_excludes_prior_legacy_wait_and_first2" if a["jobs"][name] != "first2" else "resource_case_wall_with_waits",
                        resource_amendment_sha256=args.amendment_sha256)
                elif (result["operation"], result["updates_this_call"]) != STEPS[stage]: raise ValueError("Worker operation differs")
                ledger.finish(gpu, {"case": name, "stage": stage, "gpu": gpu, "pid": pid, "exit_code": rc,
                    "log": str(path), "log_sha256": sha(path), "seconds": time.monotonic() - started, "result": result}, row)
            except BaseException as error:
                with ledger.lock: ledger.error = type(error).__name__ + ": " + str(error)
                publish(); return
            publish()
    publish()
    with ThreadPoolExecutor(max_workers=len(a["gpus"])) as pool:
        futures = [pool.submit(queue, gpu) for gpu in a["gpus"]]
        for future in futures: future.result()
    if ledger.error: raise RuntimeError(ledger.error)
    rows = [ledger.rows[n] for n in order]
    if (len(rows) != 54 or sum(r["updates"] for r in rows) != 81000 or sum(r["head_metrics"] for r in rows) != 324
            or sum(r["class_recalls"] for r in rows) != 4320 or sum(r["exact_reload_repeats"] for r in rows) != 162):
        raise ValueError("Full54 acceptance coverage differs")
    check(a["pins"]); check(read(root / "protected-files.json"))
    receipt = {"status": "passed", "policy": POLICY, "amendment_sha256": args.amendment_sha256,
        "old_completed_files_verified_unchanged": len(a["preserved_completed_files"]), "events": ledger.events,
        "remaining_updates": a["remaining_updates"], "query_labels_accessed": False,
        "elapsed_seconds": time.monotonic() - begun, "independent_resource_audit_required": True}
    check(a["preserved_completed_files"])
    for prior in a["preserved_partial_files"].values(): check(prior)
    write(folder / "receipt.json", receipt)
    write(root / "training-acceptance.json", {"policy": b["ACCEPTANCE_POLICY"], "status": "passed", "code": a["producer"],
        "launch_sha256": a["launch"]["sha256"], "registration_sha256": a["registration"]["sha256"], "optimizer_updates": 81000,
        "models": rows, "protected_manifest_sha256": sha(root / "protected-files.json"),
        "protected_files_and_inputs": len(read(root / "protected-files.json")), "query_labels_accessed": False, "test_scoring": False,
        "elapsed_seconds": time.monotonic() - begun, "elapsed_scope": "resource_controller_segment_only",
        "resource_execution": {"amendment": str(Path(args.amendment).resolve()), "sha256": args.amendment_sha256,
            "receipt": str(folder / "receipt.json"), "receipt_sha256": sha(folder / "receipt.json")}, "independent_audit_required": True})
    write(root / "status.json", {"status": "completed", "completed_cases": 54, "optimizer_updates": 81000,
        "resource_amendment": str(Path(args.amendment).resolve()), "resource_amendment_sha256": args.amendment_sha256,
        "independent_audit_required": True})


def register(args):
    b = bootstrap(args.frozen_checkout); rr = b["rr"]
    code = rr.code_snapshot(HERE)
    if code["dirty"] or code["commit"] != args.expected_controller_commit: raise ValueError("Clean tested scheduling commit required")
    if sha(args.engineering_receipt) != args.engineering_sha256: raise ValueError("Engineering receipt hash differs")
    evidence = read(args.engineering_receipt)
    if (evidence["status"] != "engineering_passed" or evidence["tested_commit"] != code["commit"]
            or any(evidence.get(k) is not True for k in ("cpu_passed", "gpu_passed", "full_passed"))):
        raise ValueError("Completed resource engineering acceptance required")
    launch = read(args.launch); producer = rr.code_snapshot()
    if sha(args.launch) != args.launch_sha256 or launch["code"] != producer: raise ValueError("Frozen launch/producer changed")
    b["unchanged_producer"](producer, code)
    reg = b["read_registration"](args.registration, args.registration_sha256, producer)
    root = Path(reg["out_dir"])
    if Path(args.launch).resolve() != root / "launch.json": raise ValueError("Original run mismatch")
    if len(args.gpu) not in (2, 3) or len(set(args.gpu)) != len(args.gpu): raise ValueError("Two or three distinct cards required")
    folder = Path(args.out).resolve()
    for protected in (root, HERE, Path(args.frozen_checkout).resolve()):
        if folder == protected or folder in protected.parents or protected in folder.parents: raise ValueError("Separate new resource directory required")
    if folder.exists(): raise ValueError("New resource appendix directory required")
    pin = process(args.legacy_pid)
    if not pin or pin["uid"] != os.getuid() or pin["cwd"] != str(Path(args.frozen_checkout).resolve()) or "matched_target_campaign.py" not in pin["command"]:
        raise ValueError("Owned original campaign process required")
    # Wait for a genuine checkpoint boundary. Active model locks are never removed.
    while True:
        if not same_process(pin): raise ValueError("Legacy process changed before handoff")
        rows = read(root / "progress.json"); cases = launch["cases"][len(rows):]
        if len(rows) == 54: raise ValueError("Batch already complete; no resource migration needed")
        try: jobs = {c["name"]: phase(c) for c in cases}
        except (ValueError, FileNotFoundError): time.sleep(2); continue
        os.kill(pin["pid"], signal.SIGSTOP)
        try:
            time.sleep(.1)
            children = live_children(pin["pid"])
            if children: raise BlockingIOError("Original child still executing")
            rows = read(root / "progress.json")
            if [r["name"] for r in rows] != [c["name"] for c in launch["cases"][:len(rows)]]: raise ValueError("Legacy prefix identity differs")
            cases = launch["cases"][len(rows):]; jobs = {c["name"]: phase(c) for c in cases}
            old_files = {}
            for c in launch["cases"][:len(rows)]: old_files.update(tree(c["out"]))
            check(read(root / "protected-files.json"))
            pins = {str(Path(args.launch).resolve()): args.launch_sha256,
                str(Path(args.registration).resolve()): args.registration_sha256,
                str(root / "protected-files.json"): sha(root / "protected-files.json"),
                str(Path(args.engineering_receipt).resolve()): args.engineering_sha256}
            partial = {c["name"]: {p: h for p, h in tree(c["out"]).items() if Path(p).name != "progress.json"}
                       for c in cases if jobs[c["name"]] != "first2"}
            folder.mkdir(); (folder / "logs").mkdir(); (folder / "events").mkdir()
            a = {"policy": POLICY, "producer": producer, "controller_code": code, "controller_checkout": str(HERE),
                "frozen_checkout": str(Path(args.frozen_checkout).resolve()), "run": str(root), "gpus": args.gpu,
                "registration": {"path": str(Path(args.registration).resolve()), "sha256": args.registration_sha256},
                "launch": {"path": str(Path(args.launch).resolve()), "sha256": args.launch_sha256},
                "legacy_process": pin, "completed_rows": rows, "jobs": jobs,
                "preserved_completed_files": old_files, "preserved_partial_files": partial, "pins": pins,
                "optimizer_updates_total": 81000, "remaining_updates": sum(1500 if s == "first2" else 1498 if s == "continue" else 0 for s in jobs.values()),
                "query_labels_accessed": False, "estimated_remaining_seconds": args.estimated_seconds,
                "user_authorization": "side chat user explicitly requested current remaining2-3 GPU handoff and main notification",
                "resource_override": "Only original registration.gpu is superseded for listed jobs/stages; all numeric/data/model contracts unchanged"}
            check(old_files)
            for files in partial.values(): check(files)
            write(folder / "amendment.json", a)
            print(json.dumps({"amendment": str(folder / "amendment.json"), "sha256": sha(folder / "amendment.json"),
                "completed": len(rows), "jobs": jobs, "remaining_updates": a["remaining_updates"],
                "legacy_controller_paused": pin["pid"]}), flush=True)
            return  # Intentionally leave only the verified coordinator stopped for takeover.
        except BlockingIOError:
            if same_process(pin): os.kill(pin["pid"], signal.SIGCONT)
            time.sleep(2)
        except BaseException:
            if same_process(pin): os.kill(pin["pid"], signal.SIGCONT)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("register")
    for n in ("frozen-checkout", "expected-controller-commit", "engineering-receipt", "engineering-sha256", "launch", "launch-sha256", "registration", "registration-sha256", "out"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--legacy-pid", type=int, required=True); p.add_argument("--gpu", type=int, action="append", required=True)
    p.add_argument("--estimated-seconds", type=float, required=True)
    for name in ("worker", "controller"):
        p = sub.add_parser(name); p.add_argument("--amendment", required=True); p.add_argument("--amendment-sha256", required=True)
        if name == "worker":
            p.add_argument("--case", required=True); p.add_argument("--stage", choices=tuple(NEXT), required=True); p.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    try:
        {"register": register, "worker": worker, "controller": controller}[args.action](args)
    except BaseException as error:
        if args.action == "controller":
            folder = Path(args.amendment).resolve().parent
            marker = folder / "controller-start.json"
            if marker.exists() and read(marker).get("pid") == os.getpid():
                a = read(args.amendment)
                write(Path(a["run"]) / "status.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                    "resource_amendment": args.amendment, "retry_policy": "preserve_and_diagnose"})
        raise


if __name__ == "__main__": main()
