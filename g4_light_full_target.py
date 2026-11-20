"""One registered, matched 1500-update light adaptation case, without query release."""
import argparse
import copy
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from g4_light_adapter_memory import RECOMPUTE
import g4_light_engineering_registration as engineering
from g4_light_target import ADAPTER_HEADS, CONTROLS, LightTargetRuntime, contract
from g4_matched_target_evaluation import new_output
from g4_matched_target_execution import case_data, unchanged_producer
from g4_matched_target_study import EXPECTED, METHODS, MODES, SEEDS, case_name, select_case
from g4_support_context import check_files, read
from g4_support_target import source_pins
from joint_baseline import NAMES, _item_fingerprint
from joint_training import run_training
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


POLICY = "g4_light_matched_full_target_v1"
REFERENCE_LAUNCH_SHA256 = "0c447087a9679538e0a792aa72cb033197e743bc959cd096ac2e831e46eafdb1"
UPDATES = EXPECTED["updates_per_case"]
MEMORY_FIELD = "adapter_backward_policy"


def roles():
    return [(target, seed, method, mode) for target in NAMES for seed in SEEDS
            for method in CONTROLS for mode in MODES]


def read_registration(path, sha256, code):
    path = Path(path).resolve(strict=True)
    pins = {str(path): sha256}
    check_files(pins)
    value = read(path)
    if value.get("policy") == engineering.REGISTRATION:
        return engineering.read_registration(value, path, sha256, code)
    fields = {"policy", "experiment_issue", "producer_commit", "planned_optimizer_updates",
              "estimated_seconds", "gpu_ids", "out_dir", "reference_launch", "engineering_acceptance",
              "analysis_plan", "query_labels_during_training", "test_scoring_during_training"}
    if (set(value) - {MEMORY_FIELD} != fields or value["policy"] != POLICY
            or type(value["experiment_issue"]) is not int or value["experiment_issue"] <= 245
            or value["producer_commit"] != code["commit"]
            or value["planned_optimizer_updates"] != len(roles()) * UPDATES
            or type(value["estimated_seconds"]) not in (int, float)
            or not math.isfinite(value["estimated_seconds"]) or value["estimated_seconds"] <= 0
            or value["query_labels_during_training"] is not False or value["test_scoring_during_training"] is not False):
        raise ValueError("Registered light adaptation budget, code, or label contract differs")
    memory = value.get(MEMORY_FIELD)
    if MEMORY_FIELD in value and (memory != RECOMPUTE or value["experiment_issue"] <= 249):
        raise ValueError("Adapter recomputation requires a new registered experiment")
    gpus = value["gpu_ids"]
    if (not isinstance(gpus, list) or len(gpus) not in (2, 3)
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus)):
        raise ValueError("Two or three distinct registered physical GPUs required")
    if value["reference_launch"]["sha256"] != REFERENCE_LAUNCH_SHA256:
        raise ValueError("Exact Issue184 reference launch required")
    for key in ("reference_launch", "engineering_acceptance", "analysis_plan"):
        ref = value[key]
        if set(ref) != {"path", "sha256"}:
            raise ValueError("Expected path and hash for " + key)
        resolved = str(Path(ref["path"]).resolve(strict=True))
        if pins.setdefault(resolved, ref["sha256"]) != ref["sha256"]:
            raise ValueError("Conflicting registration pin")
    check_files(pins)
    proof = read(value["engineering_acceptance"]["path"])
    if (proof["issue"] != 245 or proof["status"] != "engineering_passed"
            or proof["reference_launch_sha256"] != REFERENCE_LAUNCH_SHA256
            or any(proof[k] is not True for k in ("cpu_passed", "gpu_passed", "full_budget_recovery_passed"))
            or proof["real_training_updates"] != 0 or proof["query_labels_accessed"] is not False):
        raise ValueError("Accepted full-budget worker engineering required")
    if proof.get(MEMORY_FIELD) != memory:
        raise ValueError("Full-budget worker acceptance does not cover the registered adapter policy")
    if memory is not None:
        raise ValueError("Legacy worker proofs cannot authorize a new real adapter strategy")
    unchanged_producer(proof["code"], code)
    root = external_directory(value["out_dir"])
    if any(root == Path(p) or root in Path(p).parents for p in pins):
        raise ValueError("Registration inputs must remain outside the experiment output")
    return value, pins


def settings(reference, method, item, registration_sha256, gpu, *, device, adapter_backward_policy=None):
    value = copy.deepcopy(reference["settings"])
    value["config"]["device"] = str(device)
    value.update(protocol=POLICY, method=method, light_target_parameter_policy=contract(method, updates=UPDATES),
                 expected_initial_state_sha256=reference["initial_state_sha256"],
                 support_context_binding={"item_sha256": digest(_item_fingerprint(item))},
                 registration_sha256=registration_sha256, reference_case_sha256=digest(reference),
                 physical_gpu=gpu)
    if adapter_backward_policy == RECOMPUTE and method == ADAPTER_HEADS:
        value[MEMORY_FIELD] = RECOMPUTE
    return value


class FullLightTargetRuntime(LightTargetRuntime):
    def __init__(self, reference, method, items, directory, registration_sha256, gpu, *, device,
                 adapter_backward_policy=None):
        target = reference["target"]
        self.expected = settings(reference, method, items[target], registration_sha256, gpu, device=device,
                                 adapter_backward_policy=adapter_backward_policy)
        super().__init__(self.expected, items, directory, source_spec=reference["source_spec"])

    def validate(self, options):
        actual = {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")}
        if actual != self.expected or actual["optimizer_steps"] != UPDATES:
            raise ValueError("Registered full light target profile differs")
        return True


def prepare_case(registration_path, registration_sha256, code, key, *, device):
    registration, pins = read_registration(registration_path, registration_sha256, code)
    target, seed, method, mode = key
    gpu = registration["gpu_ids"][roles().index(key) % len(registration["gpu_ids"])]
    synthetic = registration["policy"] == engineering.REGISTRATION
    if synthetic:
        ledger = registration["roles"][roles().index(key)]
        if not ledger["scheduled"] or device != device_for(gpu):
            raise ValueError("Synthetic role is not registered for this device or execution profile")
    launch = reference_launch(registration)
    unchanged_producer(launch["code"], code)
    reference = select_case(launch, launch["code"], target, seed, METHODS[1], mode,
                            phase=engineering.PHASE if synthetic else "full_campaign")
    root = new_output(registration["out_dir"], launch)
    name = case_name(target, seed, METHODS[1], mode).removesuffix(METHODS[1]) + method
    directory = root / "cases" / name
    if directory.resolve() != directory:
        raise ValueError("Light target output cannot traverse a symlink")
    items, provenance, guard = case_data(reference)
    for group in (provenance["protected_files"], source_pins(reference["source_spec"])):
        for path, sha in group.items():
            if pins.setdefault(path, sha) != sha:
                raise ValueError("Conflicting light target input pin")
    if any(root == Path(p).resolve() or root in Path(p).resolve().parents for p in pins):
        raise ValueError("Light target output overlaps an input")
    runtime = FullLightTargetRuntime(reference, method, items, directory, registration_sha256, gpu, device=device,
                                     adapter_backward_policy=registration.get(MEMORY_FIELD))
    provenance.update(reference_case=reference["name"], registration_sha256=registration_sha256,
                      physical_gpu=gpu, protected_files=pins)
    if MEMORY_FIELD in runtime.settings:
        provenance[MEMORY_FIELD] = runtime.settings[MEMORY_FIELD]
    if synthetic:
        provenance.update(scope=engineering.SCOPE, scheduled_role=ledger)

    def finish():
        guard()
        require_frozen_code(code)
        check_files(pins)

    finish()
    return items, provenance, runtime, finish


def reference_launch(registration):
    if registration["policy"] == engineering.REGISTRATION:
        return engineering.pinned(engineering.pinned(registration["fixture"])["launch"])
    return read(registration["reference_launch"]["path"])


def device_for(slot):
    return "cpu" if isinstance(slot, str) and slot.startswith("cpu-") else "cuda:0"


def require_gpu(gpu):
    if device_for(gpu) == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU engineering slot requires CUDA hidden")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("Expose only the case's assigned physical GPU")
    free, util = map(int, subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip().split(","))
    if free < 12288 or util > 60:
        raise RuntimeError(f"GPU {gpu} is below the resource gate ({free} MiB free, {util}% utilized)")


def claim_output(root, registration_sha256):
    import fcntl

    root.mkdir(exist_ok=True)
    owner = root / "registration-owner.json"
    lock_path = root / ".registration.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected = {"policy": POLICY, "registration_sha256": registration_sha256}
        if owner.is_file():
            if read(owner) != expected:
                raise ValueError("Light campaign output is owned by another registration")
        else:
            if any(path != lock_path for path in root.iterdir()):
                raise ValueError("Light campaign needs a fresh owned output")
            write_json(owner, expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("expected-commit", "registration", "registration-sha256"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--target", choices=NAMES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--method", choices=CONTROLS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, choices=(2,))
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean registered Linux commit required")
    key = (args.target, args.seed, args.method, args.mode)
    registration, _ = read_registration(args.registration, args.registration_sha256, code)
    gpu = registration["gpu_ids"][roles().index(key) % len(registration["gpu_ids"])]
    device = device_for(gpu)
    items, provenance, runtime, finish = prepare_case(args.registration, args.registration_sha256, code, key, device=device)
    gpu = provenance["physical_gpu"]
    import fcntl
    root = runtime.directory.parent.parent
    claim_output(root, args.registration_sha256)
    with (root / f".gpu-{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if registration["policy"] == engineering.REGISTRATION:
            from g4_light_engineering_devices import check_child

            check_child(registration,gpu)
        require_gpu(gpu)
        runtime.directory.parent.mkdir(exist_ok=True)
        with redirect_stdout(sys.stderr):
            result = run_training(runtime.settings, items, runtime.directory, provenance, code,
                resume=args.resume, stop_after=args.stop_after, device=device,
                runtime_adapter=runtime, finish_guard=finish)
    print(json.dumps({k: result[k] for k in ("operation", "updates_this_call")}), flush=True)
