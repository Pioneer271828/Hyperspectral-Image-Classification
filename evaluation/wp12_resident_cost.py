"""Native frozen-checkpoint resident costs; no labels, calibration or training.

CPU peak is fresh-worker RSS through validation (including imports/load), CUDA
peak is allocated tensor memory over measured forwards (including resident
model/input). Neither is incremental model memory. Artifact bytes are the
checkpoint plus architecture sidecar, excluding runtime and input data.
"""

from contextlib import nullcontext
import os
import math
from pathlib import Path
import platform
import resource
import subprocess
import time

import torch

from evaluation.wp12_cost_primitives import fields as _fields, pin, timing_summary
from joint_baseline import forward
from joint_training import _tree_fingerprint
from m4_ordered_source import OrderedValidation
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import hash_file


JOB_POLICY = "wp12-native-resident-job-v1"
OBSERVATION = "net_input Q A wavelengths fwhm superpixel_scale"
MEMORY = {"cpu": "fresh_worker_peak_rss_through_validation_bytes",
          "cuda": "measured_forward_peak_allocated_bytes_including_resident_tensors"}


def environment(device, arithmetic):
    if device not in MEMORY or arithmetic not in ("native", "ordered"):
        raise ValueError("Explicit CPU/CUDA and arithmetic required")
    if torch.get_num_threads() != 1:
        raise ValueError("Registered resident worker requires one CPU thread")
    cpu = [line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
           if line.startswith("model name")]
    result = dict(host=platform.node(), kernel=platform.release(), cpu=sorted(set(cpu)),
                  affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
                  torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  device=device, arithmetic=arithmetic, scope="resident_forward",
                  memory=MEMORY[device], package="checkpoint_and_architecture_sidecar_bytes",
                  excluded="imports,load,preparation,transfer,calibration,scoring,output_serialization",
                  deterministic=torch.are_deterministic_algorithms_enabled(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one visible CUDA device required per worker")
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = dict(name=props.name, uuid=str(props.uuid), total_memory=props.total_memory,
                             driver=subprocess.check_output(
                                 ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 text=True).splitlines()[0])
    return result


def pinned_path(spec):
    _fields(spec, "path sha256")
    path = Path(spec["path"])
    if not path.is_absolute() or not pin(spec["sha256"]) or hash_file(path) != spec["sha256"]:
        raise ValueError("Pinned absolute input file required")
    return path.resolve(strict=True)


def validate_job(job):
    _fields(job, "policy case timing arithmetic checkpoint sidecar workload")
    _fields(job["case"], "scene seed arm evaluation_sha256 checkpoint_sha256")
    _fields(job["timing"], "environment_sha256 scope device warmups repeats")
    _fields(job["workload"], "file tensor_sha256 work_items")
    case, timing, work = job["case"], job["timing"], job["workload"]
    if (job["policy"] != JOB_POLICY or timing["scope"] != "resident_forward"
            or timing["device"] not in MEMORY or job["arithmetic"] not in ("native", "ordered")
            or not pin(timing["environment_sha256"]) or not pin(case["evaluation_sha256"])
            or not pin(case["checkpoint_sha256"]) or not pin(work["tensor_sha256"])
            or type(case["seed"]) is not int or case["seed"] < 0
            or any(not isinstance(case[k], str) or not case[k] for k in ("scene", "arm"))
            or type(work["work_items"]) is not int or not 0 < work["work_items"] < 2**63
            or any(type(timing[k]) is not int or timing[k] < minimum
                   for k, minimum in (("warmups", 1), ("repeats", 4)))):
        raise ValueError("Invalid resident job identity, scope or budget")
    if case["checkpoint_sha256"] != job["checkpoint"]["sha256"]:
        raise ValueError("Case checkpoint differs from measured checkpoint")


def admit_gpu(env):
    """Read-only admission; do not queue or interfere with other GPU processes."""
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    matches = [line.split(",") for line in rows if line.split(",")[0].strip() == env["gpu"]["uuid"]]
    if len(matches) != 1:
        raise ValueError("Visible GPU UUID cannot be matched to physical admission")
    _, free, utilization = matches[0]
    admission = dict(free_mib=int(free), utilization_percent=int(utilization))
    if admission["free_mib"] < 12288 or admission["utilization_percent"] > 60:
        raise ValueError("GPU resource gate not met; no measurement started")
    return admission


def measure(job):
    """Call once per fresh CLI worker; fingerprints/validation are outside timing."""
    validate_job(job)
    timing, case, work = job["timing"], job["case"], job["workload"]
    device = timing["device"]
    env = environment(device, job["arithmetic"])
    if digest(env) != timing["environment_sha256"]:
        raise ValueError("Measured environment differs from registered environment")
    admission = admit_gpu(env) if device == "cuda" else None
    checkpoint, sidecar, observation = [pinned_path(s) for s in
                                       (job["checkpoint"], job["sidecar"], work["file"])]
    if sidecar != Path(str(checkpoint) + ".json"):
        raise ValueError("Architecture sidecar must belong to the pinned checkpoint")
    if len({checkpoint, sidecar, observation}) != 3:
        raise ValueError("Distinct model and observation artifacts required")
    # This tensor-only payload deliberately cannot carry train/val/query labels.
    item = torch.load(observation, map_location="cpu", weights_only=True)
    _fields(item, OBSERVATION)
    if (any(not torch.is_tensor(item[k]) for k in ("net_input", "Q", "A"))
            or any(item[k] is not None and not torch.is_tensor(item[k]) for k in ("wavelengths", "fwhm"))
            or type(item["superpixel_scale"]) not in (int, float)
            or not math.isfinite(item["superpixel_scale"]) or item["superpixel_scale"] <= 0
            or item["net_input"].ndim != 3
            or item["net_input"].shape[0] * item["net_input"].shape[1] != work["work_items"]
            or _tree_fingerprint(item) != work["tensor_sha256"]):
        raise ValueError("Observation tensor identity or work-item count differs")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not raw or any(not torch.is_tensor(v) for v in raw.values()):
        raise ValueError("Native state-dict checkpoint required")
    state_sha = _tree_fingerprint(raw)
    del raw
    model = load_model_checkpoint(checkpoint, device=device)
    if _tree_fingerprint(model.state_dict()) != state_sha:
        raise ValueError("Loaded weights/buffers differ from the frozen state")
    if case["scene"] not in model.heads or any(m.training for m in model.modules()):
        raise ValueError("Frozen evaluation model with the declared scene required")
    classes = model.heads[case["scene"]].out_features
    item = {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
    synchronize = (lambda: torch.cuda.synchronize(0)) if device == "cuda" else (lambda: None)
    context = OrderedValidation if job["arithmetic"] == "ordered" else nullcontext

    def output_id():
        outputs = forward(model, case["scene"], item)
        if len(outputs) < 3 or any(not torch.is_tensor(v) or tuple(v.shape) != (work["work_items"], classes)
                                   or not torch.isfinite(v).all() for v in outputs[:3]):
            raise ValueError("Three finite full-scene heads required")
        return _tree_fingerprint(outputs[:3])

    with torch.inference_mode(), context():
        before = output_id()
        for _ in range(timing["warmups"]):
            outputs = forward(model, case["scene"], item)
            synchronize()
            del outputs
        synchronize()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(0)
        samples = []
        for _ in range(timing["repeats"]):
            synchronize()
            start = time.perf_counter_ns()
            outputs = forward(model, case["scene"], item)
            synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
            del outputs
        peak = (torch.cuda.max_memory_allocated(0) if device == "cuda"
                else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        after = output_id()
    timing_summary(samples, repeats=timing["repeats"], blocks=work["work_items"])
    if (before != after or _tree_fingerprint(model.state_dict()) != state_sha
            or _tree_fingerprint(item) != work["tensor_sha256"]
            or any(m.training for m in model.modules()) or environment(device, job["arithmetic"]) != env):
        raise ValueError("Frozen output, model, observation or environment changed")
    for spec in (job["checkpoint"], job["sidecar"], work["file"]):
        pinned_path(spec)
    if device == "cpu":
        # Include final tensor/file validation, not just the timing loop. This
        # ends before receipt/CLI serialization and is not incremental RSS.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    row = {k: case[k] for k in ("scene", "seed", "arm")}
    row["cost"] = dict(checkpoint_sha256=case["checkpoint_sha256"],
                       workload_sha256=work["tensor_sha256"], wall_ms=samples, peak_bytes=int(peak),
                       parameters=sum(p.numel() for p in model.parameters()),
                       package_bytes=checkpoint.stat().st_size + sidecar.stat().st_size)
    return dict(status="passed", job=job, environment=env, row=row, resource_admission=admission,
                model_state_sha256=state_sha, output_sha256=before,
                optimizer_updates=0, query_labels_accessed=False)


def assemble(plan, results, *, plan_sha256):
    """Join complete pinned worker records; no substitute or missing case accepted."""
    # Historical workers only measure; aggregation stays in the modern consumer.
    from evaluation.unified_cohort import _index
    from evaluation.wp12_evidence import COST_POLICY
    _fields(plan, "policy protocol_sha256 timing workloads cases")
    if plan["policy"] != COST_POLICY or not pin(plan["protocol_sha256"]) or not pin(plan_sha256):
        raise ValueError("Pinned hardware cost plan required")
    expected = {(r["scene"], r["seed"], r["arm"]) for r in plan["cases"]}
    registered = _index(plan["cases"], expected, "scene seed arm evaluation_sha256 checkpoint_sha256")
    rows = []
    for result in results:
        if (result["status"] != "passed" or result["optimizer_updates"] != 0
                or result["query_labels_accessed"] is not False):
            raise ValueError("Successful read-only worker required")
        job = result["job"]
        validate_job(job)
        key = tuple(job["case"][k] for k in ("scene", "seed", "arm"))
        work = plan["workloads"].get(key[0])
        if (job["case"] != registered.get(key) or job["timing"] != plan["timing"]
                or digest(result["environment"]) != plan["timing"]["environment_sha256"]
                or result["environment"]["arithmetic"] != job["arithmetic"]
                or result["environment"]["memory"] != MEMORY[job["timing"]["device"]]
                or work != {"workload_sha256": job["workload"]["tensor_sha256"],
                            "work_items": job["workload"]["work_items"]}):
            raise ValueError("Worker differs from cohort registration")
        row = result["row"]
        _fields(row, "scene seed arm cost")
        cost = row["cost"]
        _fields(cost, "checkpoint_sha256 workload_sha256 wall_ms peak_bytes parameters package_bytes")
        if (tuple(row[k] for k in ("scene", "seed", "arm")) != key
                or cost["checkpoint_sha256"] != job["case"]["checkpoint_sha256"]
                or cost["workload_sha256"] != work["workload_sha256"]
                or any(type(cost[k]) is not int or cost[k] <= 0
                       for k in ("peak_bytes", "parameters", "package_bytes"))):
            raise ValueError("Worker measurement identity or cost differs")
        timing_summary(cost["wall_ms"], repeats=plan["timing"]["repeats"], blocks=work["work_items"])
        rows.append(row)
    _index(rows, expected, "scene seed arm cost")
    return dict(policy=COST_POLICY, status="passed", plan_sha256=plan_sha256,
                timing=plan["timing"], cases=rows)
