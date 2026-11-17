"""Isolated, zero-update backward probe for the failed HongHu light role."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import resource
import sys
import time

import torch

from g4_light_full_campaign import case_directory
from g4_light_full_target import FullLightTargetRuntime, read_registration, require_gpu, roles
from g4_light_ordered_query import REGISTRATION_SHA256
from g4_light_target import ADAPTER_HEADS, allowed
from g4_matched_target_evaluation import new_output
from g4_matched_target_execution import case_data, unchanged_producer
from g4_matched_target_study import METHODS, select_case
from g4_support_context import check_files, read
from g4_support_target import source_pins
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig
from research_runtime import code_snapshot, require_frozen_code, tensor_hash, write_json
from target_initialization import state_id
from target_training import loss_terms, selected_outputs


PRODUCER_COMMIT = "2cfca12ae52308189c81c6d32dcea10f0ac1b50c"
FAILED_ROLE = ("whu-honghu", 64, ADAPTER_HEADS, "none")


def backward_snapshot(runtime, model, item, config, *, offload):
    if runtime.settings["method"] != ADAPTER_HEADS:
        raise ValueError("VCA plus heads is the only profiled optimizer phase")
    model.train()
    model.backbone.eval()
    model.zero_grad(set_to_none=True)
    target = runtime.settings["target"]
    before = state_id(model)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    storage = torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else nullcontext()
    with storage:
        outputs = selected_outputs(model, target, item, "train")
        terms = loss_terms(outputs, item["train_labels"].to(device), config)
    if not torch.isfinite(terms["total"]):
        raise ValueError("Nonfinite profiled training loss")
    terms["total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    gradients = {}
    for name, parameter in model.named_parameters():
        if allowed(name, ADAPTER_HEADS):
            if not parameter.requires_grad or parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError("Missing or invalid permitted gradient: " + name)
            gradients[name] = tensor_hash(parameter.grad)
        elif parameter.requires_grad or parameter.grad is not None:
            raise ValueError("Frozen parameter received a gradient: " + name)
    if state_id(model) != before or any(m.training for m in model.backbone.modules()):
        raise ValueError("Backward changed model state or frozen backbone mode")
    return {"saved_tensors_on_cpu": offload, "losses": {k: float(v.detach()) for k, v in terms.items()},
            "output_sha256": [tensor_hash(v) for v in outputs], "gradient_sha256": gradients,
            "state_sha256": before, "backward_seconds_including_forward": seconds,
            "peak_cuda_allocated_bytes": peak, "peak_cuda_reserved_bytes": reserved,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def profile(args, code):
    producer = code_snapshot(Path(args.producer_checkout).resolve(strict=True))
    if producer["commit"] != PRODUCER_COMMIT:
        raise ValueError("Exact Issue249 training producer required")
    unchanged_producer(producer, code)
    registration, pins = read_registration(args.registration, REGISTRATION_SHA256, producer)
    if registration["experiment_issue"] != 249:
        raise ValueError("Exact failed experiment required")
    root = Path(registration["out_dir"]).resolve(strict=True)
    status_path = root / "status.json"
    pins[str(status_path)] = file_hash(status_path)
    if read(status_path)["status"] != "failed":
        raise ValueError("Isolated probe requires terminal failed production state")
    gpu_ids = registration["gpu_ids"]
    assigned = gpu_ids[roles().index(FAILED_ROLE) % len(gpu_ids)]
    if args.gpu != assigned:
        raise ValueError("Use the physical card recorded for the failed role")
    target, seed, method, mode = FAILED_ROLE
    launch = read(registration["reference_launch"]["path"])
    reference = select_case(launch, launch["code"], target, seed, METHODS[1], mode, phase="full_campaign")
    identity_path = case_directory(root, FAILED_ROLE) / "identity.json"
    pins[str(identity_path)] = file_hash(identity_path)
    identity = read(identity_path)
    items, provenance, guard = case_data(reference)
    for path, sha in {**provenance["protected_files"], **source_pins(reference["source_spec"])}.items():
        if pins.setdefault(path, sha) != sha:
            raise ValueError("Conflicting protected inputs")
    if (identity["consumer_source_hash"] != producer["source_hash"]
            or identity["prepared"]["protected_files"] != {k: v for k, v in pins.items()
                                                              if k not in (str(status_path), str(identity_path))}):
        raise ValueError("Failed role evidence does not bind the original inputs")
    out = new_output(args.out, launch, (root,))
    if any(out == Path(path).resolve() or out in Path(path).resolve().parents for path in pins):
        raise ValueError("Probe output overlaps a protected file")
    runtime = FullLightTargetRuntime(reference, method, items, out, REGISTRATION_SHA256, assigned, device="cuda:0")
    if runtime.settings != identity["settings"]:
        raise ValueError("Probe must use the exact failed role settings")
    check_files(pins)
    require_gpu(args.gpu)
    initial_free, total = torch.cuda.mem_get_info()
    if initial_free < 20 * 1024 ** 3:
        raise RuntimeError("Diagnostic requires at least 20 GiB free on the registered card")
    out.mkdir(exist_ok=False)
    config = ResearchConfig.from_dict(runtime.settings["config"])
    model = runtime.build_model(config, {target: items[target]["class_num"]})
    moved = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in items[target].items()}
    evidence = backward_snapshot(runtime, model, moved, config, offload=True)
    guard()
    check_files(pins)
    require_frozen_code(code)
    receipt = {"status": "passed", "scope": "one_registered_role_zero_update_probe",
               "role": list(FAILED_ROLE), "physical_gpu": assigned, "code": code, "producer_commit": PRODUCER_COMMIT,
               "registration_sha256": REGISTRATION_SHA256, "initial_cuda_free_bytes": initial_free,
               "total_cuda_bytes": total, "query_labels_accessed": False, "optimizer_updates": 0, **evidence}
    write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "producer-checkout", "registration", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux probe commit required")
    profile(args, code)
    print(Path(args.out).resolve() / "receipt.json")


if __name__ == "__main__":
    main()
