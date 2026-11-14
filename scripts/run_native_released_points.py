"""Execute bounded sparse-point recovery and authorized zero-update preflights."""
import argparse
from contextlib import contextmanager, nullcontext
from itertools import product
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import native_released_preflight as preflight, native_released_training as released
from transfer import native_scale_training as training, native_scale_device as device
from transfer import p4_runtime_device as p4, registered_development as transport
from transfer.native_scale_campaign import packet
from transfer.xuzhou_multiscale_audit import _exact


@contextmanager
def arithmetic(root, gpu):
    if gpu is None:
        with device.arithmetic(p4.CPU) as execution:
            yield execution
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or p4.UUID.fullmatch(gpu) is None:
        raise ValueError("Require the one physical GPU UUID in CUDA_VISIBLE_DEVICES")
    leases = root / ".p4_gpu_leases"
    leases.mkdir(exist_ok=True)
    with p4.lease(leases / f"{gpu}.lock"), device.arithmetic(p4.CUDA_FP64_WEIGHT, gpu) as execution:
        yield execution


def synthetic(root, out, producer, gpu):
    results = {}
    coverage = tuple(product((9, 22), training.SEEDS, released.COMPARISONS))
    if gpu:
        coverage = tuple((classes, 64, (4, "joint")) for classes in (9, 22))
    with arithmetic(root, gpu) as exec:
        for classes, seed, (factor, arm) in coverage:
            if gpu:
                p4.gate(gpu)
            name = f"{classes}-{seed}-f{factor}-{arm}"
            folder = out / name
            folder.mkdir()
            spec = released.contract(classes, seed, arm, factor, producer, numeric=exec.numeric)
            spec_pin = transport._json_new(folder / "registration.json", spec)
            saved, stages = {}, []
            source = {"resume": "prefix", "completed": "resume"}
            for phase in ("full", "prefix", "resume", "completed"):
                if gpu:
                    p4.gate(gpu)
                path = folder / f"{phase}.pt"
                prior = packet(saved[source[phase]], root) if phase in source else None
                actual = training.execute(spec, phase, path, producer, prior, execution=exec, released=released)
                saved[phase] = transport._pin(path)
                stages.append({**actual, "checkpoint": saved[phase], "registration": spec_pin})
            full, prefix, split, done = (packet(saved[phase], root) for phase in ("full", "prefix", "resume", "completed"))
            if (not _exact(full, split) or not _exact(full, done)
                    or not _exact(full["history"][:spec["prefix"]], prefix["history"])
                    or [row["updates"] for row in stages] != [6, 2, 4, 0]):
                raise ValueError("Sparse native continuation differs")
            best = min(full["validation"], key=lambda row: row["fused_nll"])
            if full["selected"]["step"] != best["step"]:
                raise ValueError("Sparse native first-min selection differs")
            replay = training.Run(spec, producer, execution=exec, released=released)
            replay.model.load_state_dict(full["selected"]["state"])
            endpoint = replay.evaluate(best["step"], detailed=True)
            if endpoint["fused_nll"] != best["fused_nll"]:
                raise ValueError("Sparse endpoint replay differs")
            results[name] = {"stages": stages, "endpoint": endpoint, "numeric": exec.numeric,
                             "exposure": [row["released"] for row in full["history"]],
                             "validation_clock": [row["step"] for row in full["validation"]],
                             "input_roster_hash": spec["labels_root"], "exact_full_split_complete": True}
            transport._json_new(folder / "audit.json", results[name])
    result = {"schema": "wp07_native_released_synthetic_acceptance_v1", "producer": producer,
              "status": "passed", "scope": "synthetic_engineering", "cases": results,
              "optimizer_updates": len(results)*12, "completed_resume_updates": 0,
              "audit_optimizer_updates": 0, "raw_gt_loaded": False, "query_accessed": False}
    return transport._json_new(out / "synthetic-acceptance.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--mode", choices=("synthetic", "preflight"), required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    if args.expected_commit != code["commit"] or (args.gpu is not None and args.mode != "synthetic"):
        raise ValueError("Scope, worker or real optimization permission differs")
    root = args.runs.resolve(strict=True)
    out = external_directory(args.out)
    if root not in out.parents:
        raise ValueError("Output must stay in the authorized runs root")
    out.mkdir()
    torch.set_num_threads(1)
    if args.mode == "synthetic":
        result = synthetic(root, out, code["commit"], args.gpu)
    else:
        rows = {scene: transport._json_new(out / (scene+".json"), preflight.scan(root, scene, code["commit"]))
                for scene in preflight.SCENES}
        result = transport._json_new(out / "preflight.json", {"schema": preflight.SCHEMA,
            "producer": code["commit"], "scenes": rows, "status": "preflight_only",
            "blocked": {"paviaU": "DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED"},
            "optimization_updates": 0, "raw_gt_loaded": False, "query_accessed": False})
    require_frozen_code(code)
    print(result, flush=True)


if __name__ == "__main__":
    main()
