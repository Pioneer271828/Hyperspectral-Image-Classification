"""Execute bounded sparse-point recovery and authorized zero-update preflights."""
import argparse
import json
from contextlib import contextmanager
from itertools import product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import native_released_preflight as preflight_api, native_released_training as released
from transfer import native_scale_training as training, native_scale_device as device
from transfer import p4_runtime_device as p4, registered_development as transport
from transfer.native_scale_campaign import packet, read
from transfer.native_released_real import RealReleasedRun
from transfer.shared_gpu_lease import physical_lease
from joint_training import _run_lock
from transfer.xuzhou_multiscale_audit import _exact


@contextmanager
def arithmetic(root, gpu):
    if gpu is None:
        with device.arithmetic(p4.CPU) as execution:
            yield execution
        return
    with physical_lease(gpu), device.arithmetic(p4.CUDA_FP64_WEIGHT, gpu) as execution:
        yield execution


def synthetic(root, out, producer, gpu, *, paviau=False):
    results = {}
    coverage = tuple(product((9, 22), training.SEEDS, released.COMPARISONS))
    if paviau:
        coverage = tuple((9, 64, row) for row in ([(4, "joint")] if gpu else released.PAVIA_COMPARISONS))
    elif gpu:
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
            initialization_update = {}
            if paviau:
                final = full["model"]
                initial = replay.model.state_dict()
                changes = [key for key in final if not torch.equal(final[key], initial[key].detach().cpu())]
                trained_fresh = {key for event in full["history"] for key, evidence in event["fresh"].items()
                                 if evidence["gradient_l1"] > 0 and evidence["changed"]}
                if not changes or (arm in ("content", "metadata", "joint") and not trained_fresh):
                    raise ValueError("PaviaU engineering arm did not update its expected parameters")
                initialization_update = {"changed_state_tensors": len(changes), "updated_selectors": sorted(trained_fresh)}
            replay.model.load_state_dict(full["selected"]["state"])
            endpoint = replay.evaluate(best["step"], detailed=True)
            if endpoint["fused_nll"] != best["fused_nll"]:
                raise ValueError("Sparse endpoint replay differs")
            results[name] = {"stages": stages, "endpoint": endpoint, "numeric": exec.numeric,
                             "exposure": [row["released"] for row in full["history"]],
                             "validation_clock": [row["step"] for row in full["validation"]],
                             "input_roster_hash": spec["labels_root"], "exact_full_split_complete": True,
                             **initialization_update}
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


def coverage(args, root, out, code):
    from transfer import native_coverage_queue as queue

    if sys.platform != "linux" or not __debug__:
        raise ValueError("Coverage commands require the authorized Linux environment and assertions")
    queue_modes = ("coverage-run", "coverage-reload")
    if args.mode not in queue_modes and (args.gpu or args.case or args.workers != 1):
        raise ValueError("Device, case and workers apply only to coverage execution")
    if (args.mode == "coverage-reload") != bool(args.from_device):
        raise ValueError("Cross-device reload requires its explicit source device")
    if args.mode != "coverage-replay" and args.archive:
        raise ValueError("Historical pins apply only to frozen replay")
    release_modes = ("coverage-release-review","coverage-release-history","coverage-release-demo")
    if args.mode in release_modes:
        return independent_release(args, root, out, code)
    if args.manifest or args.authority:
        raise ValueError("Release and authority pins apply only to independent intake")
    if args.mode in ("coverage-register", "coverage-stage-register"):
        out.mkdir(exist_ok=True)
        with queue.case_lock(out):
            return queue.prepare(out, code, frozen_stages=args.mode == "coverage-stage-register")
    if args.mode == "coverage-run":
        plan = queue.load_queue(out, code)
        names = list(plan["cases"]) if args.case in (None, "all") else [args.case]
        if any(name not in plan["cases"] for name in names):
            raise ValueError("Unknown registered coverage case")
        if args.gpu is None:
            return json.dumps(queue.run_queue(out, code, names, args.workers))
        with arithmetic(root, args.gpu) as execution:
            return json.dumps(queue.run_queue(out, code, names, args.workers, execution=execution))
    if args.mode == "coverage-reload":
        if args.workers != 1 or args.case not in queue.design():
            raise ValueError("Cross-device reload requires a single registered case")
        if args.gpu is None:
            return queue.reload_selected(out, code, args.case, args.from_device)
        with arithmetic(root, args.gpu) as execution:
            return queue.reload_selected(out, code, args.case, args.from_device, execution=execution)
    if args.mode == "coverage-replay":
        from transfer.native_coverage_replay import replay
        return replay(root, out, code, args.archive)
    from evaluation.wp12_native_coverage import publish

    return publish(out, code)


def independent_release(args, root, out, code):
    from transfer import native_independent_release as independent

    if args.mode == "coverage-release-review":
        if not (args.manifest and args.authority):
            raise ValueError("Independent release review requires manifest and authority pins")
        manifest, authority = (dict(zip(("path", "sha256"), item, strict=True))
                               for item in (args.manifest,args.authority))
        _, result, _ = independent.open_independent(root,manifest,authority)
        require_frozen_code(code)
        out.mkdir()
        return independent.archive._write(out/"admission.json",dict(producer=code["commit"],report=result,
                                         manifest=manifest["sha256"],authority=authority["sha256"]))
    if args.manifest or args.authority:
        raise ValueError("Independent manifest pins apply only to external review")
    if args.mode == "coverage-release-history":
        return independent.audit_existing(root,out,code)
    return independent.synthetic_demo(root,out,code)


def long_coverage(root, out, code, args):
    from transfer import native_long_run as run
    from transfer.native_long_campaign import STEPS

    if sys.platform != "linux" or not __debug__:
        raise ValueError("Long native coverage requires the authorized Linux environment")
    if args.archive or args.from_device or args.preflight or args.release or args.phase or args.workers != 1:
        raise ValueError("Long coverage has an explicit campaign and case device")
    if args.mode == "coverage-long-register":
        if args.gpu or args.case or args.drain_after is not None:
            raise ValueError("Registration does not create updates or device contexts")
        request, approval = (dict(zip(("path", "sha256"), row, strict=True)) if row else None
                             for row in (args.manifest,args.authority))
        result = run.register(root, out, code, request=request, approval=approval, real=args.long_real)
    elif args.long_real or args.manifest or args.authority:
        raise ValueError("Execution authority is taken only from the registered campaign")
    elif args.mode == "coverage-long-freeze":
        if args.gpu or args.case or args.drain_after is not None:
            raise ValueError("Freeze requires the entire completed campaign")
        result = run.freeze(root, out, code)
    else:
        if not args.case:
            raise ValueError("One registered case is required per worker")
        with arithmetic(root, args.gpu) as execution:
            result = run.execute(root, out, code, args.case, execution,
                                 until=STEPS if args.drain_after is None else args.drain_after,
                                 reference=args.mode == "coverage-long-reference")
    print(json.dumps(result), flush=True)


def positive_control(root,out,code,args):
    from transfer import native_positive_training as controls

    if (sys.platform != "linux" or not __debug__ or args.long_real or args.manifest or args.authority
            or args.release or args.preflight or args.phase or args.frozen_plan or args.archive or args.from_device
            or args.drain_after is not None or args.workers != 1 or args.paviau_publication or args.paviau_engineering):
        raise ValueError("Positive controls require the isolated synthetic scope only")
    torch.set_num_threads(1)
    if args.mode == "positive-control-run":
        if not args.case:
            raise ValueError("An exact registered case is required")
        with arithmetic(root,args.gpu) as execution:
            result = controls.run_case(root,out,code,args.case,execution)
    else:
        if args.case or args.gpu:
            raise ValueError("Registration/freeze are CPU-only roster operations")
        operation = controls.register if args.mode == "positive-control-register" else controls.freeze
        result = operation(root,out,code)
    print(json.dumps(result),flush=True)


def production(root, out, code, args):
    from transfer import native_production_campaign as campaign

    if (sys.platform != "linux" or not __debug__ or args.long_real or args.manifest or args.authority
            or args.release or args.preflight or args.phase or args.frozen_plan or args.archive or args.from_device
            or args.drain_after is not None or args.workers != 1 or args.paviau_publication or args.paviau_engineering):
        raise ValueError("Production verification accepts only the exact registered synthetic program")
    if args.mode == "production-register":
        if args.case or args.gpu:
            raise ValueError("Registration takes no model case or GPU")
        result = campaign.register(root, out, code)
    else:
        plan, _ = campaign.plan_for(root,out,code)
        if args.case in plan["design"]:
            cases = [args.case]
        else:
            cases = [case for case in plan["design"] if args.case and case.startswith(args.case+"-")]
        if not cases:
            raise ValueError("An exact production case or paired case group is required")
        with arithmetic(root,args.gpu) as execution:
            result = [campaign.execute_case(root,out,code,case,execution) for case in cases]
    print(json.dumps(result),flush=True)


def paired_heads(root,out,code,args):
    from transfer import native_paired_head_campaign as campaign

    if (sys.platform!="linux" or not __debug__ or args.long_real or args.manifest or args.authority
            or args.preflight or args.release or args.phase or args.drain_after is not None or args.frozen_plan
            or args.archive or args.from_device or args.workers!=1 or args.paviau_publication or args.paviau_engineering):
        raise ValueError("Paired optimizer admission requires the exact synthetic registration")
    if args.mode=="paired-head-register":
        if args.case or args.gpu:
            raise ValueError("Paired registration has no model case or GPU")
        result=campaign.register(root,out,code)
    else:
        if not args.case:
            raise ValueError("Choose a registered paired model/scene/support case")
        with arithmetic(root,args.gpu) as execution:
            result=campaign.fit(root,out,code,args.case,execution)
    print(json.dumps(result),flush=True)


def main(argv=None, *, intake=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--mode", choices=("synthetic", "preflight", "real-stage", "coverage-register",
                                            "coverage-run", "coverage-replay", "coverage-reload",
                                            "coverage-audit", "coverage-stage-register", "coverage-release-review","coverage-release-history",
                                            "coverage-release-demo", "frozen-stage-register",
                                            "frozen-stage-run", "coverage-long-register", "coverage-long-run",
                                            "coverage-long-reference", "coverage-long-freeze",
                                            "positive-control-register", "positive-control-run",
                                            "positive-control-freeze", "production-register", "production-run",
                                            "paired-head-register", "paired-head-run"), required=True)
    parser.add_argument("--manifest", nargs=2, metavar=("PATH","SHA256"))
    parser.add_argument("--authority", nargs=2, metavar=("PATH","SHA256"))
    parser.add_argument("--frozen-plan", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--long-real", action="store_true")
    parser.add_argument("--drain-after", type=int)
    parser.add_argument("--from-device")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--archive", nargs=2, action="append", default=[], metavar=("PATH", "SHA256"))
    parser.add_argument("--preflight", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--release", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--paviau-publication", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--paviau-engineering", action="store_true")
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
    if args.mode.startswith("paired-head-"):
        return paired_heads(root,out,code,args)
    if args.mode.startswith("production-"):
        return production(root, out, code, args)
    if args.mode.startswith("positive-control-"):
        return positive_control(root,out,code,args)
    if args.mode.startswith("coverage-long-"):
        return long_coverage(root, out, code, args)
    if args.long_real or args.drain_after is not None:
        parser.error("Long authority or drain cursor applies only to the long campaign")
    if args.mode.startswith("frozen-stage-"):
        from evaluation.frozen_stage_cli import replay_cli

        if (args.case or args.phase or args.preflight or args.release or args.archive or args.authority
                or args.from_device or args.workers != 1 or args.paviau_publication or args.paviau_engineering
                or (args.mode == "frozen-stage-register") != bool(args.manifest)):
            parser.error("Frozen stages accept a development release to register or a frozen plan to replay")
        torch.set_num_threads(1)
        return replay_cli(code, root, out, args.gpu, args.manifest, plan_pin=args.frozen_plan)
    if args.frozen_plan:
        parser.error("A frozen plan applies only to frozen-stage replay")
    real = args.mode == "real-stage"
    if real != bool(args.preflight and args.case and args.phase) or (not real and (args.release or args.preflight or args.phase)):
        raise ValueError("A real stage requires its own explicit preflight, case and phase")
    if (args.paviau_publication and args.mode != "preflight") or (args.paviau_engineering and args.mode != "synthetic"):
        parser.error("PaviaU publication is for preflight, and PaviaU engineering is synthetic only")
    torch.set_num_threads(1)
    if args.mode.startswith("coverage-"):
        result = coverage(args, root, out, code)
    elif real:
        pinned = {"path": args.preflight[0], "sha256": args.preflight[1]}
        release = {"path": args.release[0], "sha256": args.release[1]} if args.release else None
        result = real_stage(root, out, pinned, release, args.case, args.phase, code, args.gpu, intake=intake)
    elif args.mode == "synthetic":
        out.mkdir()
        result = synthetic(root, out, code["commit"], args.gpu, paviau=args.paviau_engineering)
    elif args.paviau_publication:
        publication = dict(zip(("path", "sha256"), args.paviau_publication, strict=True))
        out.mkdir()
        scene_pin = transport._json_new(out / "paviaU.json",
            preflight_api.scan(root, "paviaU", code["commit"], publication=publication))
        result = transport._json_new(out / "preflight.json", {
            "schema": preflight_api.SCHEMA, "producer": code["commit"], "design": released.PAVIA_DESIGN,
            "scenes": {"paviaU": scene_pin}, "status": "preflight_only", "blocked": {},
            "optimization_updates": 0, "raw_gt_loaded": False, "query_accessed": False})
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
