"""Pair verified ZIP loaders on package-local, label-free synthetic observations only."""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import fcntl
import io
import json
import math
import os
from pathlib import Path
import resource
import selectors
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.wp12_frozen54_deployment_contract import PHASES, SETUP, read
from scripts.wp12_decoded_observation_cache import DecodedObservationCache
from scripts.wp12_observation_snapshot import load_snapshot, verified_snapshot

WORKER = "3ba683439ed6c5ea3ae219bea0840e41b738ffce"
SCENES = ("paviaU", "xuzhou", "whu-honghu")
BACKENDS = ("buffered", "sealed-mmap")
CACHE_BACKEND = "decoded-cache"
COLD_PAIRS, WARM_PAIRS, WARMUPS = 8, 12, 2


def require_clean(repo, expected):
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(repo), *argv], text=True).strip()

    if git("rev-parse", "HEAD") != expected or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Clean expected commit required at {repo}")


def manifest(root, filename, digest):
    return json.loads(read(root, {"path": str(root / filename), "sha256": digest}))


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def worker(args):
    sys.path.insert(0, str(args.runtime))
    import torch
    from evaluation.wp12_resident_cost import OBSERVATION
    from joint_baseline import forward
    from joint_training import _tree_fingerprint
    from m4_ordered_source import OrderedValidation
    from model.checkpoint_io import load_model_checkpoint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.fixtures
    case = manifest(root, "fixtures.json", args.fixture_sha)[args.scene]
    model_info = manifest(root, "models.json", args.model_sha)[args.scene]
    if (case["pin"]["path"] != str(root / (args.scene + ".pt"))
            or model_info["checkpoint"]["path"] != str(root / (args.scene + "-synthetic-model.pt"))
            or model_info["sidecar"]["path"] != model_info["checkpoint"]["path"] + ".json"):
        raise ValueError("Only named engineering observation and weight files are admitted")

    gpu = args.gpu_uuid is not None
    device = "cuda:0" if gpu else "cpu"
    if gpu:
        if (args.lease_fd is None or args.lease_path.name != args.gpu_uuid + ".lock"
                or not os.path.samestat(os.fstat(args.lease_fd), args.lease_path.stat())):
            raise ValueError("Expected inherited physical-device lock")
        fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
        if (os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid or torch.cuda.device_count() != 1
                or "GPU-" + uuid != args.gpu_uuid):
            raise ValueError("Unexpected GPU identity")
    sync = (lambda: torch.cuda.synchronize(0)) if gpu else (lambda: None)

    def stage(record, key, action):
        sync()
        if gpu:
            torch.cuda.reset_peak_memory_stats(0)
        start = perf_counter_ns()
        result = action()
        sync()
        stop = perf_counter_ns()
        record[key] = dict(ms=(stop-start)/1e6,
                           cuda_peak_bytes=torch.cuda.max_memory_allocated(0) if gpu else None)
        return result

    def load_model():
        for key in ("checkpoint", "sidecar"):
            read(root, model_info[key])
        model = load_model_checkpoint(model_info["checkpoint"]["path"], device="cpu").eval()
        if model.backbone.graph_mode != "static" or model.heads[args.scene].out_features != case["classes"]:
            raise ValueError("Engineering model shape or graph differs")
        return model

    setup = {}
    model = stage(setup, SETUP[0], load_model)
    original = _tree_fingerprint(model.state_dict())
    model = stage(setup, SETUP[1], lambda: model.to(device))

    def validate(item):
        if (not isinstance(item, dict) or set(item) != set(OBSERVATION.split())
                or item["net_input"].ndim != 3 or math.prod(item["net_input"].shape[:2]) != case["pixels"]
                or _tree_fingerprint(item) != case["tree_sha256"]):
            raise ValueError("Frozen label-free observation or decoded tensor identity differs")
        return item

    cache = (DecodedObservationCache(root, case["pin"], validate)
             if args.backend == CACHE_BACKEND else None)

    def request():
        phases = {}
        sync()
        started = perf_counter_ns()
        if args.backend == "buffered":
            raw = stage(phases, PHASES[0], lambda: read(root, case["pin"]))
            item = stage(phases, PHASES[1], lambda: validate(torch.load(
                io.BytesIO(raw), map_location="cpu", weights_only=True)))
            del raw
        elif cache is None:
            with ExitStack() as snapshot:
                path = stage(phases, PHASES[0], lambda: snapshot.enter_context(verified_snapshot(root, case["pin"])))
                item = stage(phases, PHASES[1], lambda: validate(load_snapshot(path)))
        else:
            hit = cache.retained_bytes != 0
            stage(phases, PHASES[0], cache.check_source)
            item = stage(phases, PHASES[1], cache.checkout)
        features = stage(phases, PHASES[2], lambda: {k:v.to(device) if torch.is_tensor(v) else v
                                                     for k,v in item.items()})
        scores = stage(phases, PHASES[3], lambda: forward(model, args.scene, features)[:3])
        decisions = stage(phases, PHASES[4], lambda: [score.argmax(1).cpu() for score in scores])
        stop = perf_counter_ns()
        if len(scores) != 3 or any(score.shape != (case["pixels"], case["classes"])
                                   or not torch.isfinite(score).all() for score in scores):
            raise ValueError("Expected three complete finite score heads")
        return dict(request_started_ns=started, first_prediction_ns=stop, phases=phases, setup=setup,
                    observation_sha256=case["pin"]["sha256"], decoded_sha256=case["tree_sha256"],
                    score_sha256=[_tree_fingerprint(tensor) for tensor in scores],
                    decision_sha256=[_tree_fingerprint(tensor) for tensor in decisions],
                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    snapshot_bytes=case["size"] if args.backend == "sealed-mmap" else 0,
                    **({"decoded_cache_hit": hit, "decoded_cache_retained_bytes": cache.retained_bytes}
                       if cache is not None else {}),
                    pid=os.getpid())

    with torch.inference_mode(), OrderedValidation():
        if args.worker == "cold":
            emit(request())
        else:
            emit({"ready": True})
            for line in sys.stdin:
                if line.strip() != "request":
                    raise ValueError("Unexpected resident worker command")
                emit(request())
    if _tree_fingerprint(model.state_dict()) != original:
        raise ValueError("Engineering model changed")


def receive(proc):
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        if not selector.select(timeout=240):
            raise TimeoutError("Engineering subprocess produced no result")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Engineering subprocess exited before its result. See stderr.log.")
    return json.loads(line)


@contextmanager
def launch(args, scope, backend, lease, log):
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--expected-commit", args.expected_commit,
           "--runtime", str(args.runtime), "--fixtures", str(args.fixtures), "--out", str(args.out),
           "--fixture-sha", args.fixture_sha, "--model-sha", args.model_sha, "--scene", args.scene,
           "--worker", scope, "--backend", backend]
    descriptors = ()
    if lease is not None:
        from transfer.shared_gpu_lease import lease_path

        descriptors = (lease.fileno(),)
        cmd.extend(["--gpu-uuid", args.gpu_uuid, "--lease-fd", str(lease.fileno()),
                    "--lease-path", str(lease_path(args.gpu_uuid))])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid or "", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    start = perf_counter_ns()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                            text=True, pass_fds=descriptors, env=env)
    try:
        yield proc, start
        proc.stdin.close()
        if proc.wait(timeout=60):
            raise RuntimeError("Engineering subprocess failed. See stderr.log.")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdin.close()
        proc.stdout.close()


def campaign(args, lease):
    args.out.mkdir()
    samples = []
    backends = ("buffered", CACHE_BACKEND) if args.compare_decoded_cache else BACKENDS
    cold_pairs = 0 if args.compare_decoded_cache else COLD_PAIRS
    with (args.out/"stderr.log").open("x") as log:
        for index in range(cold_pairs):
            order = backends if index%2 == 0 else backends[::-1]
            for position, backend in enumerate(order):
                with launch(args, "cold", backend, lease, log) as (proc, started):
                    row = receive(proc)
                    row.update(scope="cold", pair=index, position=position, backend=backend,
                               wall_ms=(row["first_prediction_ns"]-started)/1e6)
                    samples.append(row)
        with ExitStack() as workers:
            pool = {}
            for backend in backends:
                proc, _ = workers.enter_context(launch(args, "warm", backend, lease, log))
                if receive(proc) != {"ready":True}:
                    raise ValueError("Resident worker failed to acknowledge setup")
                pool[backend] = proc
            for index in range(-WARMUPS, WARM_PAIRS):
                order = backends if index%2 == 0 else backends[::-1]
                for position, backend in enumerate(order):
                    proc = pool[backend]
                    proc.stdin.write("request\n")
                    proc.stdin.flush()
                    row = receive(proc)
                    row.update(scope="warmup" if index < 0 else "warm", pair=index, position=position,
                               backend=backend, wall_ms=(row["first_prediction_ns"]-row["request_started_ns"])/1e6)
                    samples.append(row)

    identities = ("observation_sha256", "decoded_sha256", "score_sha256", "decision_sha256")
    if any(any(row[key] != samples[0][key] for key in identities) for row in samples):
        raise ValueError("Whole-input, decoded tensors, score heads or decisions changed")
    if args.compare_decoded_cache:
        cached = [row for row in samples if row["backend"] == CACHE_BACKEND]
        if ([row["decoded_cache_hit"] for row in cached] !=
                [False] + [True] * (WARM_PAIRS + WARMUPS - 1)):
            raise ValueError("The verified cache must prime exactly once before timed warm requests")
    require_clean(ROOT, args.expected_commit)
    require_clean(args.runtime, WORKER)
    with (args.out/"receipt.json").open("x") as stream:
        json.dump(dict(scene=args.scene, gpu=args.gpu_uuid, code=args.expected_commit, runtime=WORKER,
                       affinity=sorted(os.sched_getaffinity(0)), warmups_per_backend=WARMUPS,
                       cold_pairs=cold_pairs, warm_pairs=WARM_PAIRS, rows=samples,
                       **({"engineering_comparison": list(backends)} if args.compare_decoded_cache else {}),
                       prescored_dataset_reads=0), stream, allow_nan=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("runtime","fixtures","out"):
        parser.add_argument("--"+field, type=Path, required=True)
    for field in ("expected-commit","fixture-sha","model-sha"):
        parser.add_argument("--"+field, required=True)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--compare-decoded-cache", action="store_true",
                        help="Pair buffered reads with opt-in decoded reuse on warm engineering requests")
    parser.add_argument("--worker", choices=("cold","warm"), help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=(*BACKENDS, CACHE_BACKEND), help=argparse.SUPPRESS)
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--lease-path", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    require_clean(ROOT, args.expected_commit)
    require_clean(args.runtime, WORKER)
    if (any(not p.is_absolute() or p.resolve() != p for p in (args.fixtures,args.runtime,args.out))
            or args.out.exists() or args.out.parent != args.fixtures.parent
            or args.fixtures.name != "fixtures" or ROOT in args.fixtures.parents):
        raise ValueError("Use a fresh engineering output next to the isolated fixture directory")
    lock = nullcontext()
    if args.gpu_uuid:
        from evaluation.wp12_resident_cost import gpu_measurement_lease

        lock = gpu_measurement_lease(args.gpu_uuid)
    with lock as lease:
        campaign(args, lease)


if __name__ == "__main__":
    main()
