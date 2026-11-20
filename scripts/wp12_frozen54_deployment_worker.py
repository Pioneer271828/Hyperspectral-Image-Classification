"""Time sealed full-scene requests using only the preserved model runtime."""

import argparse
import fcntl
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
from time import perf_counter_ns

from wp12_frozen54_deployment_contract import (
    PLAN, WARM, COUNT, WARMUPS, PHASES, SETUP, TIMING, approval, digest_text, read, worker_binding,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("--out", type=Path, required=True)
    execute = commands.add_parser("measure")
    for field in ("plan", "plan-sha", "case", "scope", "device", "attempt"):
        execute.add_argument("--" + field, required=True)
    execute.add_argument("--invocation", type=int, required=True)
    execute.add_argument("--approval", nargs=2)
    execute.add_argument("--lease", type=int)
    execute.add_argument("--lease-path", type=Path)
    args = parser.parse_args()
    runtime = Path.cwd().resolve()
    sys.path.insert(0, str(runtime))
    import torch
    from research_runtime import code_snapshot, require_frozen_code

    code = code_snapshot()
    root = args.runs.resolve(strict=True)
    if code["commit"] != args.worker or code["dirty"]:
        raise ValueError("Expected the clean preserved benchmark worker")
    torch.set_num_threads(1)
    if args.action == "fixture":
        result = create_fixture(root, args.out)
    else:
        result = measure(root, args, code)
    require_frozen_code(code)
    print(json.dumps(result, allow_nan=False), flush=True)


def create_fixture(root, out):
    import torch
    from evaluation.wp12_resident_cost import OBSERVATION
    from joint_baseline import load_config
    from joint_training import _tree_fingerprint
    from model.checkpoint_io import save_model_checkpoint
    from research_runtime import hash_file
    sys.path.insert(0, str(Path.cwd()/"tests"))
    from test_joint_baseline import tiny_items

    if root not in out.parents or out.exists() or out.resolve() != out:
        raise ValueError("Fresh engineering fixture directory required")
    out.mkdir()
    checkpoint = out / "fixture.pt"
    observation = out / "observation.pt"
    torch.manual_seed(64)
    model = load_config("configs/experiments/base97_joint.json").build_model({"paviaU": 9}).eval()
    save_model_checkpoint(model, checkpoint)
    item = {k: tiny_items()["paviaU"][k] for k in OBSERVATION.split()}
    torch.save(item, observation)

    def pin(path):
        return dict(path=str(path), sha256=hash_file(path))

    return dict(scene="paviaU", seed=64, arm="engineering_fixture", evaluation_sha256="e"*64,
                checkpoint=pin(checkpoint), sidecar=pin(out/"fixture.pt.json"),
                workload=dict(file=pin(observation), tensor_sha256=_tree_fingerprint(item),
                              work_items=math.prod(item["net_input"].shape[:2])),
                legacy_output=None, legacy_cost=None)


def measure(root, args, code):
    import torch
    from evaluation.wp12_resident_cost import OBSERVATION, environment, pinned_path
    from joint_baseline import forward
    from joint_training import _tree_fingerprint
    from m4_ordered_source import OrderedValidation
    from model.checkpoint_io import load_model_checkpoint
    from scripts.measure_wp12_historical_cost import preserved_runtime

    pin = {"path": args.plan, "sha256": args.plan_sha}
    plan = json.loads(read(root, pin))
    permit = dict(zip(("path", "sha256"), args.approval, strict=True)) if args.approval else None
    approval(root, plan, pin, permit, args.device)
    if (plan["schema"] != PLAN or args.scope not in COUNT or plan["timing"] != TIMING
            or plan["adapter_sha256"] != digest_text(Path(__file__).read_bytes())):
        raise ValueError("Worker, plan or timing boundary changed")
    if not plan["engineering"]:
        if preserved_runtime(json.loads(read(root, plan["freeze"])), code) != plan["runtime"]:
            raise ValueError("Preserved frozen model runtime differs")
    job = plan["cases"][args.case]
    cuda = args.device != "cpu"
    device = "cuda:0" if cuda else "cpu"
    if cuda:
        path = args.lease_path
        if (args.lease is None or path is None or path != path.resolve(strict=True)
                or path.name != args.device + ".lock" or not os.path.samestat(os.fstat(args.lease), path.stat())):
            raise ValueError("Expected inherited physical-device lock")
        fcntl.flock(args.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif args.lease is not None or args.lease_path is not None:
        raise ValueError("CPU requests cannot inherit a GPU lock")
    env = environment("cuda" if cuda else "cpu", "ordered")
    # The original helper describes the resident benchmark. Keep its device and software identity.
    env.pop("scope")
    env.pop("excluded")
    if cuda and (os.environ.get("CUDA_VISIBLE_DEVICES") != args.device or env["gpu"]["uuid"] != args.device):
        raise ValueError("Physical CUDA device differs from the registered lease")
    sync = (lambda: torch.cuda.synchronize(0)) if cuda else (lambda: None)
    setup = {}

    def operation(table, name, fn):
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(0)
        start = perf_counter_ns()
        value = fn()
        sync()
        stop = perf_counter_ns()
        table[name] = dict(start_ns=start, finish_ns=stop, wall_ms=(stop-start)/1e6,
                           cuda_peak_bytes=torch.cuda.max_memory_allocated(0) if cuda else None)
        return value

    def load():
        path, sidecar = (pinned_path(job[key]) for key in ("checkpoint", "sidecar"))
        if sidecar != Path(str(path) + ".json"):
            raise ValueError("Sidecar does not belong to the pinned checkpoint")
        loaded = load_model_checkpoint(path, device="cpu")
        if loaded.backbone.graph_mode != "static" or job["scene"] not in loaded.heads:
            raise ValueError("Frozen54 requires the original static-graph architecture")
        return loaded

    model = operation(setup, SETUP[0], load)
    state = _tree_fingerprint(model.state_dict())
    model = operation(setup, SETUP[1], lambda: model.to(device))
    workload = job["workload"]
    classes = model.heads[job["scene"]].out_features

    def decode(raw):
        item = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(item, dict) or set(item) != set(OBSERVATION.split()):
            raise ValueError("Only the sealed label-free observation fields are admitted")
        if (item["net_input"].ndim != 3 or math.prod(item["net_input"].shape[:2]) != workload["work_items"]
                or _tree_fingerprint(item) != workload["tensor_sha256"]):
            raise ValueError("Frozen workload or graph differs")
        return item

    def trial(sequence, role):
        phases = {}
        sync()
        start = perf_counter_ns()
        raw = operation(phases, PHASES[0], lambda: read(root, workload["file"]))
        item = operation(phases, PHASES[1], lambda: decode(raw))
        del raw
        request = operation(phases, PHASES[2], lambda: {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()})
        scores = operation(phases, PHASES[3], lambda: forward(model, job["scene"], request)[:3])
        predictions = operation(phases, PHASES[4], lambda: [v.argmax(1).cpu() for v in scores])
        stop = phases[PHASES[-1]]["finish_ns"]
        if (len(scores) != 3 or any(tuple(v.shape) != (workload["work_items"], classes) or
                                    not torch.isfinite(v).all() for v in scores)):
            raise ValueError("The three frozen score heads must be finite and complete")
        # Provenance validation is after the prediction deadline.
        identity = _tree_fingerprint(scores)
        if job["legacy_output"] is not None and identity != job["legacy_output"]:
            raise ValueError("Ordered probabilities differ from the accepted resident baseline")
        return dict(sequence=sequence, role=role, phases=phases, request_started_ns=start,
                    first_prediction_ns=stop,
                    prediction=_tree_fingerprint(predictions), probability_state=identity,
                    checkpoint=job["checkpoint"]["sha256"], observation=workload["file"]["sha256"],
                    workload=workload["tensor_sha256"], pixels=workload["work_items"], batch=1,
                    observation_reads=1, graph_builds=0, skip_count=None,
                    rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    with torch.inference_mode(), OrderedValidation():
        warmups = WARMUPS[args.scope]
        measured = COUNT[WARM] if args.scope == WARM else 1
        requests = [trial(i, "warmup" if i < warmups else "measured") for i in range(warmups + measured)]
    if _tree_fingerprint(model.state_dict()) != state:
        raise ValueError("Frozen model state changed during measurement")
    if json.loads(read(root, pin)) != plan:
        raise ValueError("Plan changed during the worker run")
    return dict(binding=worker_binding(args.plan_sha, args.case, args.scope, args.device, args.attempt, args.invocation),
                timing=TIMING[args.scope], requests=requests, setup=setup, pid=os.getpid(), environment=env,
                optimizer_updates=0, query_accessed=False)


if __name__ == "__main__":
    main()
