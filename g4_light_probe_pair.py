"""Pair and independently review the frozen HongHu backward pass with zero updates."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from g4_light_engineering_registration import code_identity, pin, pinned
from g4_light_memory_probe import (FAILED_ROLE, ISOLATED_START, PRODUCER_COMMIT, PROBE_ROOT, REGISTERED_GPU_UUID,
                                   REGISTRATION_SHA256, backward_snapshot, frozen_probe)
from g4_light_target import ADAPTER_HEADS, allowed
from g4_support_context import check_files
from joint_training import _rng_state, _tree_fingerprint
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, tensor_hash, write_json
from transfer import p4_runtime_device as resources
from transfer.shared_gpu_lease import physical_lease

SAMPLES = ("offload", "offload-repeat", "recompute", "recompute-repeat")
DEVICE_REQUEST = "wp03_honghu_zero_update_device_request_v1"
DEVICE_APPROVAL = "wp03_honghu_zero_update_device_approval_v1"
PROBE_REVIEW = "wp03_honghu_zero_update_comparison_v1"
REAL = "frozen_honghu_real_zero_update"
FIXTURE = "honghu_shape_synthetic_zero_update"
ARITHMETIC = {
    "torch": "2.7.1+cu128", "dtype": "float32", "cudnn_tf32": True,
    "matmul_tf32": False, "deterministic": False,
}
NATIVE_FREE_MIB = 20 * 1024
ALTERNATE_FREE_MIB = 16 * 1024
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
ORIGINAL = {"slot": 5, "uuid": REGISTERED_GPU_UUID}
LIMITS = {"optimizer_updates": 0, "query_access": False, "real_training_authorized": False}

ISOLATED_PAIR = ISOLATED_START.rsplit('probe["isolated_recompute"]', 1)[0] + """
import transfer
for name, relative in (
    ("g4_light_engineering_registration", "g4_light_engineering_registration.py"),
    ("transfer.p4_runtime_device", "transfer/p4_runtime_device.py"),
    ("transfer.shared_gpu_lease", "transfer/shared_gpu_lease.py"),
):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    dependency = importlib.util.module_from_spec(spec)
    sys.modules[name] = dependency
    spec.loader.exec_module(dependency)
    if name.startswith("transfer."):
        setattr(transfer, name.split(".")[1], dependency)
name = "g4_light_probe_pair"
spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
pair = importlib.util.module_from_spec(spec)
sys.modules["g4_light_memory_probe"] = type(sys)("g4_light_memory_probe")
sys.modules["g4_light_memory_probe"].__dict__.update(probe)
spec.loader.exec_module(pair)
pair.isolated_sample(json.loads(sys.argv[2]), sys.argv[3])
"""


def check_device(row):
    if (not isinstance(row, dict) or set(row) != {"slot", "uuid"} or type(row["slot"]) is not int
            or row["slot"] < 0 or not resources.UUID.fullmatch(row["uuid"])):
        raise ValueError("An explicit physical index and UUID are required")
    return row


def arithmetic():
    return {"torch": str(torch.__version__), "dtype": "float32",
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def snapshot(device):
    row = resources.snapshot(device["uuid"])
    identity = subprocess.check_output(["nvidia-smi", "-i", str(device["slot"]), "--query-gpu=uuid,name",
                                        "--format=csv,noheader"], text=True).strip()
    uuid, name = (value.strip() for value in identity.split(",", 1))
    if uuid != device["uuid"] or "RTX 4090" not in name:
        raise ValueError("Physical UUID and RTX 4090 model must match the exact slot")
    return {**row, "name": name}


def hardware_gate(device, minimum):
    row = snapshot(device)
    if row["free_mib"] < minimum or row["utilization"] > resources.MAX_UTILIZATION:
        raise resources.ResourceUnavailable(f'{device["uuid"]} needs {minimum} MiB free and at most 60% utilization')
    return row


def device_request(registration, device, engineering, code):
    if registration["sha256"] != REGISTRATION_SHA256:
        raise ValueError("Exact original Issue249 registration pin required")
    pinned(registration)
    check_device(device)
    if device["slot"] == ORIGINAL["slot"] or device["uuid"] == ORIGINAL["uuid"]:
        raise ValueError("Alternate diagnostic request must name a different physical card")
    review = pinned(engineering)
    if (review["policy"] != PROBE_REVIEW or review["scope"] != FIXTURE or review["device"] != device
            or review["arithmetic"] != ARITHMETIC or review["shape"] != [940, 475, 272]):
        raise ValueError("Same-device full-shape arithmetic engineering review required")
    if review != review_pair(review["request"], review["samples"]):
        raise ValueError("Synthetic engineering comparison did not replay independently")
    if review["status"] != "within_existing_tolerance":
        raise ValueError("Synthetic engineering comparison failed the original tolerance")
    return {"policy": DEVICE_REQUEST, "producer": code_identity(code), "registration": registration,
            "original_producer": PRODUCER_COMMIT, "role": list(FAILED_ROLE), "device": device,
            "engineering": engineering, "arithmetic": ARITHMETIC, "minimum_free_mib": ALTERNATE_FREE_MIB,
            "maximum_utilization": resources.MAX_UTILIZATION, "allowed_scope": REAL, **LIMITS}


def approval_for(request, plan, *, decision):
    return {"policy": DEVICE_APPROVAL, "decision": decision, "request": request, "producer": plan["producer"],
            "registration": plan["registration"], "device": plan["device"], "role": plan["role"],
            "allowed_scope": REAL, **LIMITS}


def probe_intent(registration, device, code, *, migration=None, authority=None):
    check_device(device)
    if registration["sha256"] != REGISTRATION_SHA256:
        raise ValueError("Exact original registration required")
    original = pinned(registration)
    if original["experiment_issue"] != 249 or original["producer_commit"] != PRODUCER_COMMIT:
        raise ValueError("Original producer and experiment differ")
    minimum = NATIVE_FREE_MIB
    if device != ORIGINAL:
        if migration is None or authority is None:
            raise ValueError("Alternate physical card requires a separate approved zero-update release")
        request = pinned(migration)
        if request != device_request(registration, device, request["engineering"], code):
            raise ValueError("Alternate zero-update request no longer reproduces")
        if pinned(authority) != approval_for(migration, request, decision="approved"):
            raise ValueError("Independent approved zero-update release is required")
        minimum = request["minimum_free_mib"]
    elif migration is not None or authority is not None:
        raise ValueError("Original device must retain the original qualification")
    return {"scope": REAL, "producer": code_identity(code), "registration": registration,
            "original_producer": PRODUCER_COMMIT, "device": device, "role": list(FAILED_ROLE),
            "arithmetic": ARITHMETIC, "migration": migration, "authority": authority,
            "minimum_free_mib": minimum, **LIMITS}


def sample(runtime, model, item, config, intent, label, observed, output, *, finish=None):
    """Run one admitted process, saving only what the separate reviewer needs."""
    if label not in SAMPLES or arithmetic() != ARITHMETIC:
        raise ValueError("Original arithmetic is required")
    device = next(model.parameters()).device
    random.seed(64)
    np.random.seed(64)
    torch.manual_seed(64)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(64)
    raw = {"gradients": {}}
    before_rng = _tree_fingerprint(_rng_state(device))
    before_bn = digest({n: tensor_hash(x) for n, x in model.named_buffers()})
    recompute = label.startswith("recompute")
    record = backward_snapshot(runtime, model, item, config, offload=not recompute,
                               recompute=recompute, capture=raw)
    after_bn = digest({n: tensor_hash(x) for n, x in model.named_buffers()})
    if finish is not None:
        finish()
    raw_path = output / "tensors.pt"
    torch.save(raw, raw_path)
    write_json(output / "receipt.json", {
        "intent": intent, "sample": label, "admission": observed, "shape": list(item["net_input"].shape),
        "raw": pin(raw_path), "rng_before": before_rng, "rng_after": _tree_fingerprint(_rng_state(device)),
        "buffers_before": before_bn, "buffers_after": after_bn, **LIMITS, **record,
    })


def isolated_sample(arguments, expected_source_hash):
    args = argparse.Namespace(**arguments)
    frozen = Path(args.producer_checkout)
    code = code_snapshot(PROBE_ROOT)
    producer = code_snapshot(frozen)
    if (not sys.flags.isolated or Path.cwd() != frozen or code["dirty"]
            or code["source_hash"] != expected_source_hash or producer["dirty"] or producer["commit"] != PRODUCER_COMMIT):
        raise ValueError("Isolated original producer and clean current probe required")
    for name in ("g4_light_full_target", "g4_light_target", "target_training", "research_runtime"):
        file = Path(sys.modules[name].__file__).resolve()
        if not file.is_relative_to(frozen) or file.relative_to(frozen).as_posix() not in producer["files"]:
            raise ValueError("Frozen worker imported a foreign module")
    if getattr(args, "verify_source_only", False):
        print(json.dumps({"current_probe": code_identity(code), "original_worker": code_identity(producer), **LIMITS}))
        return
    request = probe_intent(args.registration, args.device, code, migration=args.migration, authority=args.authority)
    if request != args.intent:
        raise ValueError("Zero-update intent changed after dispatch")
    with physical_lease(args.device["uuid"], inherited_fd=args.lease_fd):
        admitted = hardware_gate(args.device, request["minimum_free_mib"])
        free, _ = torch.cuda.mem_get_info()
        if free < request["minimum_free_mib"] * 2**20:
            raise resources.ResourceUnavailable("CUDA context free memory no longer qualifies")
        torch.cuda.set_per_process_memory_fraction((free - 2 * 2**30) / torch.cuda.get_device_properties(0).total_memory, 0)
        parent_args = argparse.Namespace(producer_checkout=frozen, registration=args.registration["path"],
                                         gpu=ORIGINAL["slot"], out=args.out)
        runtime, item, _, out, pins, finish = frozen_probe(parent_args, producer)
        config = ResearchConfig.from_dict(runtime.settings["config"])
        model = runtime.build_model(config, {runtime.settings["target"]: item["class_num"]})
        moved = {k: x.to("cuda:0") if torch.is_tensor(x) else x for k, x in item.items()}
        output = out / args.label
        output.mkdir()
        sample(runtime, model, moved, config,
               {**request, "original_settings": digest(runtime.settings), "protected_files": pins},
               args.label, admitted, output, finish=finish)


def execute(intent, producer_root, out, code):
    """The sole real command is a zero-update diagnostic, under an inherited physical lock."""
    path = Path(producer_root).resolve(strict=True)
    producer = code_snapshot(path)
    if producer["dirty"] or producer["commit"] != PRODUCER_COMMIT:
        raise ValueError("Exact clean Issue249 producer checkout required")
    from g4_matched_target_evaluation import new_output

    registration = pinned(intent["registration"])
    launch = pinned(registration["reference_launch"])
    out = new_output(out, launch, (registration["out_dir"],))
    require_frozen_code(code)
    with physical_lease(intent["device"]["uuid"]) as lock:
        hardware_gate(intent["device"], intent["minimum_free_mib"])
        out.mkdir(exist_ok=False)
        for label in SAMPLES:
            arguments = {"producer_checkout": str(path), "out": str(out), "label": label, "intent": intent,
                         "device": intent["device"], "registration": intent["registration"],
                         "migration": intent["migration"], "authority": intent["authority"], "lease_fd": lock.fileno()}
            with (out / (label + ".log")).open("x") as log:
                subprocess.run([sys.executable, "-I", "-B", "-c", ISOLATED_PAIR, str(PROBE_ROOT),
                                json.dumps(arguments), code["source_hash"]], cwd=path, pass_fds=(lock.fileno(),),
                               stdout=log, stderr=subprocess.STDOUT, check=True)


def load_sample(sample_pin, label):
    report = pinned(sample_pin)
    if (report["sample"] != label or any(report[k] != v for k, v in LIMITS.items())
            or report["raw"]["path"] != str(Path(sample_pin["path"]).with_name("tensors.pt"))):
        raise ValueError("Sample label, scope or owned raw payload differs")
    raw = Path(report["raw"]["path"])
    check_files({str(raw): report["raw"]["sha256"]})
    tensors = torch.load(raw, map_location="cpu", weights_only=True)
    if ([tensor_hash(p) for p in tensors["probabilities"]] != report["output_sha256"]
            or {n: tensor_hash(g) for n, g in tensors["gradients"].items()} != report["gradient_sha256"]
            or len(tensors["probabilities"]) != 3 or any(tuple(p.shape) != (220, 22) for p in tensors["probabilities"])
            or len(tensors["gradients"]) != 19 or not all(allowed(name, ADAPTER_HEADS) for name in tensors["gradients"])):
        raise ValueError("Raw three-head or permitted gradient identity differs")
    return report, tensors


def diff(before, after):
    if (before.shape != after.shape or before.dtype != torch.float32 or after.dtype != torch.float32
            or not before.numel() or not torch.isfinite(before).all() or not torch.isfinite(after).all()):
        raise ValueError("Compared tensor shape, type or finiteness differs")
    return {"max_abs": float((after - before).abs().max()),
            "outside": int(torch.count_nonzero(~torch.isclose(after, before, **TOLERANCE))),
            "exact": torch.equal(before, after)}


def review_pair(request_pin, samples):
    request = pinned(request_pin)
    if (request.get("scope") not in (REAL, FIXTURE) or request.get("arithmetic") != ARITHMETIC
            or any(request.get(k) != v for k, v in LIMITS.items()) or set(samples) != set(SAMPLES)
            or len({v["path"] for v in samples.values()}) != len(SAMPLES)):
        raise ValueError("Four separate zero-update sample receipts are required")
    check_device(request["device"])
    if request["role"] != list(FAILED_ROLE):
        raise ValueError("Original exact failed role required")
    sample_data = {label: load_sample(samples[label], label) for label in SAMPLES}
    metadata, ref = sample_data["offload"]
    intent = metadata["intent"]
    if (any(intent.get(k) != v for k, v in request.items()) or metadata["shape"] != [940, 475, 272]
            or any(row["intent"] != intent or row["state_sha256"] != metadata["state_sha256"]
                   or row["rng_before"] != metadata["rng_before"] or row["rng_after"] != metadata["rng_after"]
                   or row["buffers_before"] != row["buffers_after"] or row["buffers_before"] != metadata["buffers_before"]
                   or row["shape"] != metadata["shape"] or row["admission"]["physical_uuid"] != request["device"]["uuid"]
                   or row["admission"]["driver"] != metadata["admission"]["driver"]
                   or row["saved_tensors_on_cpu"] != label.startswith("offload")
                   for label,(row,_) in sample_data.items())):
        raise ValueError("Frozen role, BN, RNG, device or saved-tensor treatment differs")
    if request["scope"] == REAL:
        if (not intent.get("protected_files") or not intent.get("original_settings")
                or pinned(request["registration"])["producer_commit"] != PRODUCER_COMMIT):
            raise ValueError("Missing frozen original settings and input facts")
        check_files(intent["protected_files"])
    comparisons = {}
    for label,(row, data) in sample_data.items():
        if label == "offload":
            continue
        if data["gradients"].keys() != ref["gradients"].keys() or row["losses"].keys() != metadata["losses"].keys():
            raise ValueError("Permitted parameter or objective terms changed")
        heads = {}
        for name, before, after in zip(("fused", "graph", "cnn"), ref["probabilities"], data["probabilities"], strict=True):
            heads[name] = {**diff(before, after), "changed": int((before.argmax(1) != after.argmax(1)).sum())}
        gradients = {k: diff(x, data["gradients"][k]) for k,x in ref["gradients"].items()}
        losses = {k: diff(torch.tensor(x), torch.tensor(row["losses"][k])) for k,x in metadata["losses"].items()}
        comparisons[label] = {"heads": heads, "gradients": gradients, "losses": losses}
    rows = [row for part in comparisons.values() for group in part.values() for row in group.values()]
    by_sample = {label: {k: row[k] for k in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                                            "peak_process_rss_kib", "backward_seconds_including_forward")}
                 for label,(row,_) in sample_data.items()}
    return {"policy": PROBE_REVIEW, "scope": request["scope"], "device": request["device"], "request": request_pin,
            "samples": samples, "original_role": request["role"], "arithmetic": ARITHMETIC, "shape": metadata["shape"],
            "tolerance": TOLERANCE, "status": "numeric_failure" if any(r["outside"] or r.get("changed") for r in rows)
            else "within_existing_tolerance", "bitwise_equal": all(r["exact"] for r in rows),
            "measurements": by_sample, "comparisons": comparisons, **LIMITS}
