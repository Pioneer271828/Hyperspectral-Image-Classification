"""Prepare a scoped engineering release for the two-slot physical CLI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


SCHEMA = "wp07_dual_physical_qualification_plan_v2"
ASSIGNED = ("GPU-2fd29d4e-c4c7-618b-22d8-3c2fa73e48e5", "GPU-1c123d1c-2962-6a75-ca3a-b2f13a09e964")
CASES = ("xuzhou/64/2/joint", "whu-honghu/64/4/constant_joint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("out", "worker", "controller"):
        parser.add_argument("--"+key, type=Path, required=True)
    for key in ("worker-commit", "controller-commit"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--gpu", nargs="+", required=True, choices=ASSIGNED)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root, worker, controller = (getattr(args, key).resolve() for key in ("out", "worker", "controller"))
    sys.path.insert(0, str(worker))
    import torch
    from research_runtime import code_snapshot
    from tests.native_released_bank_fixture import fixture
    from tests.test_native_released_production import release_for
    from transfer import native_released_campaign as campaign, native_released_real as real
    from transfer import p4_runtime_device as resource, registered_development as transport
    from transfer.native_scale_campaign import packet, read
    from transfer.xuzhou_multiscale_audit import _exact

    if len(set(args.gpu)) != len(args.gpu):
        raise ValueError("Use one or two distinct registered physical UUIDs")
    worker_id, control_id = code_snapshot(worker), code_snapshot(controller)
    for actual, expected in ((worker_id, args.worker_commit), (control_id, args.controller_commit)):
        if actual["dirty"] or actual["commit"] != expected:
            raise ValueError("Freeze the worker and controller before physical qualification")
    torch.set_num_threads(1)
    source = lambda code: {k: code[k] for k in ("commit", "source_hash")}
    root.mkdir()
    pinned = lambda item: [item["path"], item["sha256"]]
    pins, _ = fixture(root, worker_id, design=real.DUTY)
    manifest = transport._json_new(root / "fixture-preflights.json", {
        "schema": real.ENGINEERING, "status": "preflight_only", "scenes": pins,
        "optimization_updates": 0, "raw_gt_loaded": False, "query_accessed": False})
    mapped = {case: args.gpu[i % len(args.gpu)] for i, case in enumerate(CASES)}
    plans = {}
    for path, route in (("continuous", "full"), ("segmented", "split")):
        execution = root / (path+"-execution")
        matrix = campaign.prepare(root, root / (path+"-matrix"), execution, manifest, worker_id,
                                  engineering=True, design=real.DUTY)
        science = read(matrix, root)
        release_dir = root / (path+"-case-releases")
        release_dir.mkdir()
        releases = {}
        for case in science["cases"]:
            key = case["id"]
            preflight = case["preflight"]
            selected = next(row for row in read(preflight, root)["draft"]["cases"] if row["id"] == key)
            releases[key] = release_for(release_dir, preflight, selected, Path(case["destination"]), worker_id, route,
                                        mapped.get(key))
        release = transport._json_new(root / (path+"-authorization.json"), {
            "schema": campaign.RELEASE, "decision": "approved", "matrix": matrix, "execution": str(execution),
            "synthetic": True, "cases": releases, "optimizer_updates": science["optimizer_updates"],
            "query_authorized": False})
        journal = root / (path+"-journal")
        routing = transport._json_new(root / (path+"-routing.json"), {"schema": "wp07_native_released_dual_slots_v2",
            "matrix": matrix, "release": release, "worker": source(worker_id), "controller": source(control_id),
            "slots": args.gpu, "synthetic": True, "cases": mapped, "execution": str(journal)})
        command = [sys.executable, str(controller / "scripts/run_native_released_dual.py"), "--runs", str(root),
                   "--out", str(journal), "--worker", str(worker), "--worker-commit", args.worker_commit,
                   "--controller-commit", args.controller_commit, "--matrix", *pinned(matrix),
                   "--release", *pinned(release), "--routing", *pinned(routing)]
        plans[path] = {"journal": str(journal), "output": str(execution), "command": command,
                       "release": release, "routing": routing}
    snapshot = resource.simultaneous_inventory(args.gpu)
    plan = transport._json_new(root / "qualification-plan.json", {"schema": SCHEMA, "synthetic": True,
         "matrix": "two synthetic cases, full and split, two completion proofs and zero reread",
         "worker": source(worker_id), "controller": source(control_id), "prospective_optimizer_updates": 48,
         "actual_optimizer_updates": 0, "real_optimizer_updates": 0, "mapping": mapped,
         "min_free_mib": resource.MIN_FREE_MIB, "max_utilization": resource.MAX_UTILIZATION,
         "inventory": snapshot, "plans": plans, "physical_qualification": "not_executed", "query_accessed": False})
    print(json.dumps(plan), flush=True)
    if not args.run:
        return
    if not all(resource.eligible(row) for row in snapshot.values()):
        raise resource.ResourceUnavailable("Registered physical qualification threshold was not met")
    # Keep the original P4 batch as the first claimant of the designated cards.
    if not all(resource.eligible(row) for row in resource.simultaneous_inventory(args.gpu).values()):
        raise resource.ResourceUnavailable("Physical resources are no longer eligible")
    started = perf_counter()
    for name, plan_dir in plans.items():
        for iteration, expected in ((0, 24), (1, 0)):
            with (root / f"{name}-{iteration}.log").open("x") as log:
                subprocess.run(plan_dir["command"], cwd=worker, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
            report = read(transport._pin(Path(plan_dir["journal"]) / f"summary-{iteration:04d}.json"), root)
            if report["status"] != "completed" or report["new_updates"] != expected:
                raise ValueError("Controller did not complete its bounded physical assignment")
    proof = {}
    for case, gpu in mapped.items():
        records = [Path(path["output"]) / case.replace("/", "-") / "completed.pt" for path in plans.values()]
        a, b = (packet(transport._pin(rec), root) for rec in records)
        if not _exact({k: v for k, v in a.items() if k != "registration"},
                      {k: v for k, v in b.items() if k != "registration"}):
            raise ValueError("Physical continuous and segmented science differs")
        proof[case] = {"physical_uuid": gpu, "completed": [transport._pin(rec) for rec in records]}
    print(transport._json_new(root / "qualification.json", {"schema": SCHEMA, "scope": "synthetic_physical",
          "plan": plan, "physical_gpus": len(args.gpu), "paired_exact": proof, "actual_optimizer_updates": 48,
          "real_optimizer_updates": 0, "completed_resume_updates": 0, "query_accessed": False,
          "seconds": perf_counter()-started}))


if __name__ == "__main__":
    main()
