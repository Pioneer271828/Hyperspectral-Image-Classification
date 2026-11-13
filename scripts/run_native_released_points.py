"""Execute bounded sparse-point recovery and authorized zero-update preflights."""
import argparse
from contextlib import contextmanager
from itertools import product
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import native_released_preflight as preflight_api, native_released_training as released
from transfer import native_scale_training as training, native_scale_device as device
from transfer import p4_runtime_device as p4, registered_development as transport
from transfer.native_scale_campaign import packet, read
from transfer.native_released_real import RealReleasedRun
from joint_training import _run_lock
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


def check_real_route(out, phase):
    # A released budget owns one training path. An interrupted phase retains
    # its intent for diagnosis instead of allowing another update attempt.
    phases = ("full", "prefix", "resume", "completed")
    seen = {name for name in phases if (out / f"{name}.intent.json").exists()}
    expected = {"full": set(), "prefix": set(), "resume": {"prefix"}}
    admissible = (seen in ({"full"}, {"prefix", "resume"}) if phase == "completed" else seen == expected[phase])
    if not admissible:
        raise ValueError("A released case allows one training path and no repeated stage")
    return "full" if seen == {"full"} else {"resume": "prefix", "completed": "resume"}.get(phase)


def real_stage(root, out, preflight_pin, release, case, phase, code, gpu, *, intake=None):
    guarded = RealReleasedRun(root, preflight_pin, case, release, out, code, intake=intake)
    if gpu != guarded.physical_uuid:
        raise ValueError("Real sparse execution GPU must match the approved release")
    selected = guarded.case
    if (guarded.route == "full" and phase in ("prefix", "resume")) or (guarded.route == "split" and phase == "full"):
        raise ValueError("Phase differs from the released route")
    out.mkdir(exist_ok=True)
    with _run_lock(out), arithmetic(root, gpu) as execution:
        spec = guarded.contract(guarded.classes, selected["model_seed"], selected["arm"],
                                selected["factor"], code["commit"], numeric=execution.numeric)
        identity = out / "registration.json"
        if identity.is_file():
            spec_pin = transport._pin(identity)
            if read(spec_pin, root) != spec:
                raise ValueError("Approved checkpoint registration differs from prior execution")
        else:
            spec_pin = transport._json_new(identity, spec)
        previous = check_real_route(out, phase)
        state = None
        if previous:
            prior = read(transport._pin(out / f"{previous}.json"), root)
            if (prior["registration"] != spec_pin or prior["phase"] != previous
                    or Path(prior["checkpoint"]["path"]) != out / f"{previous}.pt"
                    or prior.get("updates") != {"full": spec["steps"], "prefix": spec["prefix"],
                                               "resume": spec["steps"] - spec["prefix"]}[previous]):
                raise ValueError("Real predecessor belongs to another phase or registration")
            state = packet(prior["checkpoint"], root)
        path = out / f"{phase}.pt"
        artifacts = [out / f"{phase}{part}" for part in (".pt", ".json", ".intent.json")]
        if any(file.exists() for file in artifacts):
            raise ValueError("A prior real phase artifact is already present")
        if gpu:
            p4.gate(gpu)
        transport._json_new(artifacts[2], {"registration": spec_pin, "phase": phase})
        actual = training.execute(spec, phase, path, code["commit"], state, execution=execution, released=guarded)
        return transport._json_new(out / f"{phase}.json",
                                   {**actual, "registration": spec_pin, "checkpoint": transport._pin(path)})


def main(argv=None, *, intake=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--mode", choices=("synthetic", "preflight", "real-stage"), required=True)
    parser.add_argument("--preflight", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--release", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--case")
    parser.add_argument("--phase", choices=("full", "prefix", "resume", "completed"))
    args = parser.parse_args(argv)
    code = code_snapshot()
    require_frozen_code(code)
    if args.expected_commit != code["commit"] or (args.gpu is not None and args.mode == "preflight"):
        raise ValueError("Scope, worker or real optimization permission differs")
    root = args.runs.resolve(strict=True)
    out = external_directory(args.out)
    if root not in out.parents:
        raise ValueError("Output must stay in the authorized runs root")
    real = args.mode == "real-stage"
    if real != bool(args.preflight and args.case and args.phase) or (not real and args.release is not None):
        raise ValueError("A real stage requires its own explicit preflight, case and phase")
    torch.set_num_threads(1)
    if real:
        pinned = {"path": args.preflight[0], "sha256": args.preflight[1]}
        release = {"path": args.release[0], "sha256": args.release[1]} if args.release else None
        result = real_stage(root, out, pinned, release, args.case, args.phase, code, args.gpu, intake=intake)
    elif args.mode == "synthetic":
        out.mkdir()
        result = synthetic(root, out, code["commit"], args.gpu)
    else:
        out.mkdir()
        rows = {scene: transport._json_new(out / (scene+".json"), preflight_api.scan(root, scene, code["commit"]))
                for scene in preflight_api.SCENES}
        result = transport._json_new(out / "preflight.json", {"schema": preflight_api.SCHEMA,
            "producer": code["commit"], "scenes": rows, "status": "preflight_only",
            "blocked": {"paviaU": "DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED"},
            "optimization_updates": 0, "raw_gt_loaded": False, "query_accessed": False})
    require_frozen_code(code)
    print(result, flush=True)


if __name__ == "__main__":
    main()
