"""Isolated, zero-update backward probe for the failed HongHu light role."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch

from g4_light_adapter_memory import RECOMPUTE, recompute_value_projection
from g4_light_full_campaign import case_directory
from g4_light_full_target import FullLightTargetRuntime, read_registration, require_gpu, roles
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
REGISTRATION_SHA256 = "b3cca8f1028004da30177704b107e959cd2fa47aa7892390522e4589fe39e735"
REGISTERED_GPU_UUID = "GPU-e8ab3c4d-e4f8-8dab-db06-fcd8d73040bc"
FAILED_ROLE = ("whu-honghu", 64, ADAPTER_HEADS, "none")
PROBE_ROOT = Path(__file__).resolve().parent

# Import the frozen worker from its own tree. Only these two files come from the probe tree.
ISOLATED_START = """
import importlib.util
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(Path.cwd()))
name = "g4_light_adapter_memory"
spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
policy = importlib.util.module_from_spec(spec)
sys.modules[name] = policy
spec.loader.exec_module(policy)
probe = runpy.run_path(str(root / "g4_light_memory_probe.py"))
probe["isolated_recompute"](json.loads(sys.argv[2]), sys.argv[3])
"""


def backward_snapshot(runtime, model, item, config, *, offload, recompute=False, capture_projection=False,
                      capture=None):
    if runtime.settings["method"] != ADAPTER_HEADS:
        raise ValueError("VCA plus heads is the only profiled optimizer phase")
    if offload and (recompute or capture_projection):
        raise ValueError("CPU offload is measured separately from recomputation and save tracing")
    model.train()
    model.backbone.eval()
    model.zero_grad(set_to_none=True)
    target = runtime.settings["target"]
    before = state_id(model)
    adapter = model.backbone.adapter
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    projection_storages = {}
    bands = item["bands"] if capture_projection else None
    hidden = adapter.value_proj[0].out_features if capture_projection else None

    def save(tensor):
        if (tensor.ndim == 3 and tensor.shape[-2:] == (bands, hidden)
                or tensor.ndim == 2 and tensor.shape[-1] == hidden and tensor.shape[0] >= bands
                and tensor.shape[0] % bands == 0):
            storage = tensor.untyped_storage()
            projection_storages[storage.data_ptr()] = storage.nbytes()
        return tensor.detach()

    saves = (torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor)
             if capture_projection else nullcontext())
    projection = recompute_value_projection(adapter) if recompute else nullcontext()
    with projection:
        storage = torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else saves
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
            if capture is not None:
                capture["gradients"][name] = parameter.grad.detach().cpu()
        elif parameter.requires_grad or parameter.grad is not None:
            raise ValueError("Frozen parameter received a gradient: " + name)
    if state_id(model) != before or any(m.training for m in model.backbone.modules()):
        raise ValueError("Backward changed model state or frozen backbone mode")
    if capture is not None:
        capture["probabilities"] = [value.detach().cpu() for value in outputs]
    return {**({"adapter_backward_policy": RECOMPUTE} if recompute else {}),
            **({"saved_value_hidden_storage_bytes": sum(projection_storages.values()),
                "saved_value_hidden_storages": len(projection_storages)} if capture_projection else {}),
            "saved_tensors_on_cpu": offload, "losses": {k: float(v.detach()) for k, v in terms.items()},
            "output_sha256": [tensor_hash(v) for v in outputs], "gradient_sha256": gradients,
            "state_sha256": before, "backward_seconds_including_forward": seconds,
            "peak_cuda_allocated_bytes": peak, "peak_cuda_reserved_bytes": reserved,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def frozen_probe(args, code):
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
    def finish():
        guard()
        check_files(pins)
        require_frozen_code(code)

    return runtime, items[target], assigned, out, pins, finish


def profile(args, code, *, probe_code=None):
    runtime, item, assigned, out, _, finish = frozen_probe(args, code)
    require_gpu(args.gpu)
    initial_free, total = torch.cuda.mem_get_info()
    if initial_free < 20 * 1024 ** 3:
        raise RuntimeError("Diagnostic requires at least 20 GiB free on the registered card")
    out.mkdir(exist_ok=False)
    config = ResearchConfig.from_dict(runtime.settings["config"])
    model = runtime.build_model(config, {runtime.settings["target"]: item["class_num"]})
    moved = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in item.items()}
    recompute = probe_code is not None
    evidence = backward_snapshot(runtime, model, moved, config, offload=not recompute, recompute=recompute)
    finish()
    if recompute and probe_code != checked_probe(args.expected_commit):
        raise RuntimeError("Probe source changed during diagnosis")
    receipt = {"status": "passed", "scope": "one_registered_role_zero_update_probe",
               "role": list(FAILED_ROLE), "physical_gpu": assigned, "code": code, "producer_commit": PRODUCER_COMMIT,
               "registration_sha256": REGISTRATION_SHA256, "initial_cuda_free_bytes": initial_free,
               "total_cuda_bytes": total, "query_labels_accessed": False, "optimizer_updates": 0, **evidence,
               **({"probe_code": probe_code, "physical_gpu_uuid": REGISTERED_GPU_UUID} if recompute else {})}
    write_json(out / "receipt.json", receipt)
    return receipt


def checked_probe(expected_commit):
    code = code_snapshot(PROBE_ROOT)
    if sys.platform != "linux" or code["dirty"] or code["commit"] != expected_commit:
        raise RuntimeError("Clean expected Linux probe commit required")
    return code


def isolated_recompute(arguments, expected_probe_hash):
    args = argparse.Namespace(**arguments)
    producer_root = Path(args.producer_checkout)
    if not sys.flags.isolated or Path.cwd() != producer_root or str(PROBE_ROOT) in sys.path:
        raise RuntimeError("The recomputation worker requires the isolated producer environment")
    probe = checked_probe(args.expected_commit)
    if probe["source_hash"] != expected_probe_hash:
        raise ValueError("External recomputation probe source differs")
    producer = code_snapshot(producer_root)
    if producer["commit"] != PRODUCER_COMMIT:
        raise ValueError("Exact Issue249 training producer required")
    require_frozen_code(producer)
    for name in ("g4_light_full_target", "g4_light_target", "research_runtime",
                 "target_training", "model.variable_channel_adapter"):
        module_file = Path(sys.modules[name].__file__).resolve()
        if not module_file.is_relative_to(producer_root) or module_file.relative_to(producer_root).as_posix() not in producer["files"]:
            raise RuntimeError("Module was not loaded from frozen production source: " + name)
    if args.verify_source_only:
        print(json.dumps({"strategy": RECOMPUTE, "producer_commit": producer["commit"],
                          "producer_source_hash": producer["source_hash"], "probe_commit": probe["commit"],
                          "probe_source_hash": probe["source_hash"], "optimizer_updates": 0}), flush=True)
        return
    profile(args, producer, probe_code=probe)
    print(Path(args.out) / "receipt.json")


def launch_recompute(args, code):
    # Paths are resolved before the isolated process moves to the producer checkout.
    for name in ("producer_checkout", "registration", "out"):
        setattr(args, name, str(Path(getattr(args, name)).resolve(strict=name == "producer_checkout")))
    if args.gpu != 5:
        raise ValueError("Use the physical card recorded for the failed role")
    producer = code_snapshot(args.producer_checkout)
    if producer["commit"] != PRODUCER_COMMIT or producer["dirty"]:
        raise ValueError("Exact clean Issue249 training producer required")
    command = [sys.executable, "-I", "-B", "-c", ISOLATED_START, str(PROBE_ROOT),
               json.dumps(vars(args)), code["source_hash"]]
    if args.verify_source_only:
        subprocess.run(command, check=True, cwd=args.producer_checkout)
        return

    from transfer.shared_gpu_lease import physical_lease

    uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip()
    if uuid != REGISTERED_GPU_UUID:
        raise ValueError("Registered HongHu GPU UUID differs")
    # The frozen worker binds CUDA visibility to the original integer index.
    with physical_lease(uuid) as lease:
        subprocess.run(command, check=True, cwd=args.producer_checkout, pass_fds=(lease.fileno(),),
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "producer-checkout", "registration", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--strategy", choices=("offload", "recompute"), default="offload")
    parser.add_argument("--verify-source-only", action="store_true")
    args = parser.parse_args(argv)
    code = checked_probe(args.expected_commit)
    if args.strategy == "recompute":
        launch_recompute(args, code)
        return
    if args.verify_source_only:
        parser.error("--verify-source-only requires --strategy recompute")
    profile(args, code)
    print(Path(args.out).resolve() / "receipt.json")


if __name__ == "__main__":
    main()
