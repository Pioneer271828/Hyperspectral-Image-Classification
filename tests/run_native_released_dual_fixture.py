"""Run the frozen worker through real CPU subprocesses with two logical slots."""
import argparse
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
from time import monotonic, sleep

import torch


def pin_args(pin):
    return [pin["path"], pin["sha256"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    args = parser.parse_args()
    worker = args.worker.resolve(strict=True)
    controller = args.controller.resolve(strict=True)
    sys.path.insert(0, str(worker))
    from research_runtime import code_snapshot
    from scripts import run_native_released_points as cli
    from tests.native_released_bank_fixture import fixture
    from tests.test_native_released_production import release_for
    from transfer import native_released_campaign as campaign, native_released_real as real
    from transfer import p4_runtime_device as resource, registered_development as transport
    from transfer.native_scale_campaign import packet, read
    from transfer.xuzhou_multiscale_audit import _exact

    root = args.out.resolve()
    root.mkdir()
    torch.set_num_threads(1)
    code = code_snapshot(worker)
    owner = code_snapshot(controller)
    identity = lambda snapshot: {key: snapshot[key] for key in ("commit", "source_hash")}
    assert not code["dirty"] and not owner["dirty"]
    pins, intake = fixture(root, code, design=real.DUTY)
    manifest = transport._json_new(root / "fixture-preflights.json", {
        "schema": real.ENGINEERING, "status": "preflight_only", "scenes": pins,
        "optimization_updates": 0, "raw_gt_loaded": False, "query_accessed": False})
    cases = ("xuzhou/64/2/joint", "whu-honghu/64/4/constant_joint")

    def prepare(stem, routes):
        execution = root / f"{stem}-execution"
        matrix = campaign.prepare(root, root / f"{stem}-matrix", execution, manifest, code,
                                  engineering=True, design=real.DUTY)
        value = read(matrix, root)
        proofs = {}
        release_dir = root / f"{stem}-case-releases"
        release_dir.mkdir()
        for entry in value["cases"]:
            case = next(row for row in read(entry["preflight"], root)["draft"]["cases"] if row["id"] == entry["id"])
            proofs[case["id"]] = release_for(release_dir, entry["preflight"], case, Path(entry["destination"]),
                                            code, routes.get(case["id"], "full"))
        release = transport._json_new(root / f"{stem}-release.json", {
            "schema": campaign.RELEASE, "decision": "approved", "matrix": matrix, "execution": str(execution),
            "synthetic": True, "optimizer_updates": value["optimizer_updates"], "cases": proofs, "query_authorized": False})
        journal = root / f"{stem}-journal"
        plan = transport._json_new(root / f"{stem}-routing.json", {
            "schema": "wp07_native_released_dual_slots_v2", "matrix": matrix, "release": release,
            "worker": identity(code), "controller": identity(owner), "slots": ["a", "b"], "synthetic": True,
            "execution": str(journal), "cases": dict(zip(cases, ("a", "b"), strict=True))})
        command = [sys.executable, str(controller / "scripts/run_native_released_dual.py"),
                   "--runs", str(root), "--worker", str(worker), "--out", str(journal),
                   "--controller-commit", owner["commit"], "--worker-commit", code["commit"],
                   "--matrix", *pin_args(matrix), "--release", *pin_args(release), "--routing", *pin_args(plan)]
        return value, proofs, execution, journal, command

    def dispatch(command, journal, label, expected=0):
        run = subprocess.run(command, cwd=worker, text=True, capture_output=True, check=False,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        (root / f"{label}.log").write_text(run.stdout + run.stderr)
        if run.returncode != expected:
            raise ValueError(run.stdout + run.stderr)
        found = sorted(journal.glob("summary-*.json"))
        return read(transport._pin(found[-1]), root) if found else None

    value, proofs, execution, journal, command = prepare("recovery", {cases[0]: "split"})
    journal.mkdir()
    with resource.lease(journal / ".matrix.lock"):
        transport._json_new(journal / "owner.json", {"routing": dict(zip(("path", "sha256"), command[-2:])),
                                                    "controller": identity(owner)})
    with resource.lease(journal / ".lane-a.lock"):
        first = dispatch(command, journal, "slot-a-held")
    assert first["status"] == "waiting" and first["slots"]["a"]["stages"] == []
    assert first["new_updates"] == 12 and first["slots"]["b"]["status"] == "completed"
    resumed = dispatch(command, journal, "recovered")
    assert resumed["status"] == "completed" and resumed["new_updates"] == 12
    assert all(row["new_updates"] == 0 for row in resumed["slots"]["b"]["stages"])
    again = dispatch(command, journal, "completed")
    assert again["new_updates"] == 0
    children = [stage["pid"] for lane in resumed["slots"].values() for stage in lane["stages"]]
    assert len(children) == len(set(children))

    ref_spec, direct, reference, _, _ = prepare("direct", {cases[1]: "split"})
    reference.mkdir()
    compared = {}
    for case in (row for row in ref_spec["cases"] if row["id"] in cases):
        case_id = case["id"]
        dest = Path(case["destination"])
        route = "full" if case_id == cases[0] else "split"
        for phase in campaign.ROUTES[route]:
            cli.real_stage(root, dest, case["preflight"], direct[case_id], case_id, phase, code, None,
                           intake=intake)
        base = packet(transport._pin(dest / "completed.pt"), root)
        other = packet(transport._pin(execution / dest.name / "completed.pt"), root)
        assert set(base) == set(other) and _exact({k: v for k, v in base.items() if k != "registration"},
                                                {k: v for k, v in other.items() if k != "registration"})
        compared[case_id] = {"model_adam_rng_history_selected_exact": True, "direct_optimizer_updates": 12,
                             "subprocess_completed": transport._pin(execution / dest.name / "completed.json"),
                             "direct_completed": transport._pin(dest / "completed.json")}

    _, _, broken, failure_journal, failure_command = prepare("interrupt", {})
    target = broken / cases[0].replace("/", "-")
    marker = target / "full.intent.json"
    def launched(folder, case_id):
        for entry in sorted(folder.glob("attempt0000-*.launch.json")):
            row = json.loads(entry.read_text())
            dispatch = read(row["dispatch"], root)
            if dispatch["case"] == case_id and dispatch["phase"] == "full":
                return row["pid"], entry
        return None
    with (root / "interrupt.log").open("x") as log:
        proc = subprocess.Popen(failure_command, cwd=worker, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        deadline = monotonic() + 120
        killed = None
        try:
            while proc.poll() is None and monotonic() < deadline:
                if marker.exists() and (child := launched(failure_journal, cases[0])):
                    killed, _ = child
                    os.kill(killed, signal.SIGKILL)
                    break
                sleep(.01)
        finally:
            returncode = proc.wait(timeout=120)
    assert killed and returncode == 1 and marker.exists()
    gate = read(transport._pin(failure_journal / "failure.json"), root)
    failed = read(transport._pin(failure_journal / "summary-0000.json"), root)
    assert failed["first_failure"] == transport._pin(failure_journal / "failure.json")
    assert gate["returncode"] == -signal.SIGKILL
    assert failed["status"] == "failed" and failed["slots"]["a"]["phase"] == "full"
    assert failed["slots"]["a"]["returncode"] == -signal.SIGKILL
    assert failed["slots"]["a"]["stages"] == [] and not (target / "full.json").exists()
    assert failed["slots"]["b"]["status"] == "drained"
    assert failed["slots"]["b"]["stages"] and len(failed["slots"]["b"]["stages"]) < 3
    dispatch(failure_command, failure_journal, "failure-retry-refused", expected=1)
    assert not (target / "full.json").exists()
    def snapshot(folder):
        return {str(p.relative_to(folder)): transport._pin(p)["sha256"]
                for p in folder.rglob("*") if p.is_file() and p.suffix in (".json", ".pt")}

    def pause_when(command, label, test):
        with (root / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=worker, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        deadline = monotonic() + 120
        try:
            while proc.poll() is None and monotonic() < deadline:
                outcome = test(proc)
                if outcome:
                    return outcome
                sleep(.005)
            raise AssertionError(f"Missed controlled coordinator interruption: {label}")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=120)

    def dead_parent(label, fail_child):
        _, _, work, folder, cmd = prepare(label, {})
        def kill_after_dispatch(proc):
            pair = [launched(folder, case) for case in cases]
            if not all(pair):
                return None
            pids = [p[0] for p in pair]
            os.kill(proc.pid, signal.SIGSTOP)
            if fail_child:
                os.kill(pids[0], signal.SIGKILL)
            peer_case = cases[1].replace("/", "-")
            limit = monotonic() + 90
            while monotonic() < limit:
                if (work / peer_case / "full.json").is_file() and list(folder.glob("*.receipt.json")):
                    break
                sleep(.01)
            else:
                raise AssertionError("In-flight worker did not commit its real stage")
            assert not list(folder.glob("*.ack.json"))
            return pids
        pids = pause_when(cmd, label, kill_after_dispatch)
        limit = monotonic() + 30
        while monotonic() < limit:
            try:
                with resource.lease(folder / ".matrix.lock"):
                    break
            except resource.ResourceUnavailable:
                sleep(.01)
        else:
            raise AssertionError("Worker leases failed to drain after coordinator death")
        original = snapshot(work)
        prior_journal = snapshot(folder)
        run = subprocess.run(cmd, cwd=worker, capture_output=True, text=True, check=False)
        (root / f"{label}-reentry.log").write_text(run.stdout + run.stderr)
        assert run.returncode == 1 and "Dispatch reconciliation refused" in run.stderr
        assert snapshot(work) == original
        assert all(snapshot(folder).get(key) == value for key, value in prior_journal.items())
        assert len(list(folder.glob("attempt-*.json"))) == 1
        assert transport._pin(folder / "failure.json")
        committed_updates = sum(read(transport._pin(p), root)["updates"] for p in work.rglob("full.json"))
        assert committed_updates >= 12
        return {"pids": pids, "committed_updates": committed_updates, "new_updates_on_reentry": 0,
                "unchanged_science": True, "gate": transport._pin(folder / "failure.json"),
                "original_journal_entries": len(prior_journal)}

    interrupted = dead_parent("coordinator-killed-after-worker-failure", True)
    unconfirmed = dead_parent("coordinator-killed-with-success-receipt", False)
    with resource.lease(journal / ".matrix.lock"):
        conflict = subprocess.run(command, cwd=worker, capture_output=True, check=False)
        assert conflict.returncode == 75
    result = {"schema": "wp07_native_dual_slot_crash_qualification_v2", "status": "passed",
              "worker": identity(code), "controller": identity(owner), "physical_gpus": 0,
              "actual_optimizer_updates": first["new_updates"]+resumed["new_updates"]+24+failed["new_updates"],
              "completed_resume_updates": 0, "synthetic": True, "query_accessed": False, "raw_gt_loaded": False,
              "resource_wait": first, "continuation": resumed, "completed": again, "failure": failed,
              "paired_states": compared, "interrupted_pid": killed, "real_scientific_cost": None,
              "coordinator_failure_window": interrupted, "unacknowledged_success": unconfirmed,
              "orphan_committed_optimizer_updates": interrupted["committed_updates"]+unconfirmed["committed_updates"],
              "concurrent_exit": conflict.returncode}
    print(transport._json_new(root / "cpu-qualification.json", result), flush=True)


if __name__ == "__main__":
    main()
