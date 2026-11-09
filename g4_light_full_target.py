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

from g4_light_target import CONTROLS, LightTargetRuntime, contract
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


def roles():
    return [(target, seed, method, mode) for target in NAMES for seed in SEEDS
            for method in CONTROLS for mode in MODES]


def read_registration(path, sha256, code):
    path = Path(path).resolve(strict=True)
    pins = {str(path): sha256}
    check_files(pins)
    value = read(path)
    fields = {"policy", "experiment_issue", "producer_commit", "planned_optimizer_updates",
              "estimated_seconds", "gpu_ids", "out_dir", "reference_launch", "engineering_acceptance",
              "analysis_plan", "query_labels_during_training", "test_scoring_during_training"}
    if (set(value) != fields or value["policy"] != POLICY
            or type(value["experiment_issue"]) is not int or value["experiment_issue"] <= 245
            or value["producer_commit"] != code["commit"]
            or value["planned_optimizer_updates"] != len(roles()) * UPDATES
            or type(value["estimated_seconds"]) not in (int, float)
            or not math.isfinite(value["estimated_seconds"]) or value["estimated_seconds"] <= 0
            or value["query_labels_during_training"] is not False or value["test_scoring_during_training"] is not False):
        raise ValueError("Registered light adaptation budget, code, or label contract differs")
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
    engineering = read(value["engineering_acceptance"]["path"])
    if (engineering["issue"] != 245 or engineering["status"] != "engineering_passed"
            or engineering["reference_launch_sha256"] != REFERENCE_LAUNCH_SHA256
            or any(engineering[k] is not True for k in ("cpu_passed", "gpu_passed", "full_budget_recovery_passed"))
            or engineering["real_training_updates"] != 0 or engineering["query_labels_accessed"] is not False):
        raise ValueError("Accepted full-budget worker engineering required")
    unchanged_producer(engineering["code"], code)
    root = external_directory(value["out_dir"])
    if any(root == Path(p) or root in Path(p).parents for p in pins):
        raise ValueError("Registration inputs must remain outside the experiment output")
    return value, pins


def settings(reference, method, item, registration_sha256, gpu, *, device):
    value = copy.deepcopy(reference["settings"])
    value["config"]["device"] = str(device)
    value.update(protocol=POLICY, method=method, light_target_parameter_policy=contract(method, updates=UPDATES),
                 expected_initial_state_sha256=reference["initial_state_sha256"],
                 support_context_binding={"item_sha256": digest(_item_fingerprint(item))},
                 registration_sha256=registration_sha256, reference_case_sha256=digest(reference),
                 physical_gpu=gpu)
    return value


class FullLightTargetRuntime(LightTargetRuntime):
    def __init__(self, reference, method, items, directory, registration_sha256, gpu, *, device):
        target = reference["target"]
        self.expected = settings(reference, method, items[target], registration_sha256, gpu, device=device)
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
    launch = read(registration["reference_launch"]["path"])
    unchanged_producer(launch["code"], code)
    reference = select_case(launch, launch["code"], target, seed, METHODS[1], mode, phase="full_campaign")
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
    runtime = FullLightTargetRuntime(reference, method, items, directory, registration_sha256, gpu, device=device)
    provenance.update(reference_case=reference["name"], registration_sha256=registration_sha256,
                      physical_gpu=gpu, protected_files=pins)

    def finish():
        guard()
        require_frozen_code(code)
        check_files(pins)

    finish()
    return items, provenance, runtime, finish


def require_gpu(gpu):
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
    items, provenance, runtime, finish = prepare_case(args.registration, args.registration_sha256, code,
        (args.target, args.seed, args.method, args.mode), device="cuda:0")
    gpu = provenance["physical_gpu"]
    import fcntl
    root = runtime.directory.parent.parent
    claim_output(root, args.registration_sha256)
    with (root / f".gpu-{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_gpu(gpu)
        runtime.directory.parent.mkdir(exist_ok=True)
        with redirect_stdout(sys.stderr):
            result = run_training(runtime.settings, items, runtime.directory, provenance, code,
                resume=args.resume, stop_after=args.stop_after, device="cuda:0",
                runtime_adapter=runtime, finish_guard=finish)
    print(json.dumps({k: result[k] for k in ("operation", "updates_this_call")}), flush=True)
