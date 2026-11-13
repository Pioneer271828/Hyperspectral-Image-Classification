"""Run the frozen worker through real CPU subprocesses with two logical slots."""
import argparse
from pathlib import Path
import os
import subprocess
import sys

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
            "schema": "wp07_native_released_dual_slots_v1", "matrix": matrix, "release": release,
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
    broken.mkdir()
    target = broken / cases[0].replace("/", "-")
    target.mkdir()
    transport._json_new(target / "full.intent.json", {"phase": "full", "fixture_interrupted_after_intent": True})
    failed = dispatch(failure_command, failure_journal, "interrupt", expected=1)
    assert failed["status"] == "failed" and failed["slots"]["a"]["phase"] == "full"
    assert failed["slots"]["a"]["stages"] == [] and not (target / "full.pt").exists()
    assert len(failed["slots"]["b"]["stages"]) <= 1 and failed["slots"]["b"]["status"] == "drained"
    dispatch(failure_command, failure_journal, "failure-retry-refused", expected=1)
    assert not (target / "full.pt").exists()
    result = {"schema": "wp07_native_dual_slot_cpu_qualification_v1", "status": "passed",
              "worker": identity(code), "controller": identity(owner), "physical_gpus": 0,
              "actual_optimizer_updates": first["new_updates"]+resumed["new_updates"]+24+failed["new_updates"],
              "completed_resume_updates": 0, "synthetic": True, "query_accessed": False, "raw_gt_loaded": False,
              "resource_wait": first, "continuation": resumed, "completed": again, "failure": failed,
              "paired_states": compared, "real_scientific_cost": None}
    print(transport._json_new(root / "cpu-qualification.json", result), flush=True)


if __name__ == "__main__":
    main()
