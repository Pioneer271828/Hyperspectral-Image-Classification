"""Explicit registered full1500 execution; engineering preflight is never executable."""
import argparse
import copy
import json
from pathlib import Path
import sys

from g4_matched_target_study import (EXPECTED, METHODS, MODES, SEEDS, STUDY_PATH, common_fresh_initials,
    grid, initialize, load_study, runtime, select_case, verify_fresh_target)
from joint_baseline import NAMES
from joint_training import run_training
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research.reproduction.m4g4b.none_source_campaign import wait_gpu
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.train_joint_m4 import graph_binding
from target_data_context import load_target_context

REGISTRATION_POLICY = "g4_matched_full_training_registration_v2"
REGISTRATION_FIELDS = {"policy", "experiment_issue", "study_file_sha256", "planned_optimizer_updates",
    "producer_commit", "gpu", "estimated_seconds", "engineering_acceptance", "preflight", "out_dir",
    "query_labels_during_training", "test_scoring_during_training", "analysis_plan"}


def unchanged_producer(producer, consumer):
    if (producer["dirty"] or consumer["dirty"] or producer["source_hash"] != digest(producer["files"])
            or consumer["source_hash"] != digest(consumer["files"])
            or any(consumer["files"].get(p) != h for p, h in producer["files"].items())):
        raise ValueError("Unchanged accepted engineering producer files required")


def read_registration(path, sha256, code):
    if file_hash(path) != sha256:
        raise ValueError("Pinned separate full experiment registration required")
    value = json.loads(Path(path).read_text())
    if (set(value) != REGISTRATION_FIELDS or value["policy"] != REGISTRATION_POLICY
            or type(value["experiment_issue"]) is not int or value["experiment_issue"] <= 182
            or value["producer_commit"] != code["commit"] or value["study_file_sha256"] != file_hash(STUDY_PATH)
            or value["planned_optimizer_updates"] != 81000 or type(value["gpu"]) is not int or value["gpu"] < 0
            or type(value["estimated_seconds"]) not in (int, float) or not 0 < value["estimated_seconds"] < float("inf")
            or value["query_labels_during_training"] is not False or value["test_scoring_during_training"] is not False):
        raise ValueError("Registered full54 budget/producer/resource/label contract differs")
    approval = value["engineering_acceptance"]
    if set(approval) != {"path", "sha256"} or file_hash(approval["path"]) != approval["sha256"]:
        raise ValueError("Pinned complete engineering acceptance required")
    accepted = json.loads(Path(approval["path"]).read_text())
    if (accepted["status"] != "engineering_passed" or accepted["issue"] != 182
            or any(accepted[k] is not True for k in ("cpu_passed", "gpu_passed", "full_regression_passed",
                "worker_campaign_audit_contracts_passed"))
            or accepted["preflight_receipt_sha256"] != value["preflight"]["receipt_sha256"]
            or accepted["query_labels_accessed"] is not False or accepted["full_training_completed"] is not False):
        raise ValueError("Complete worker/campaign/audit engineering acceptance required; preflight alone forbidden")
    unchanged_producer(accepted["code"], code)
    plan = value["analysis_plan"]
    if set(plan) != {"path", "sha256"} or file_hash(plan["path"]) != plan["sha256"]:
        raise ValueError("Pinned preregistered hypothesis/metrics/protection plan required")
    external_directory(value["out_dir"])
    return value


def read_preflight(registration, code):
    frozen = registration["preflight"]
    if set(frozen) != {"root", "receipt_sha256", "protected_sha256"}:
        raise ValueError("Explicit preflight receipt and input-manifest pins required")
    root = external_directory(frozen["root"])
    if file_hash(root / "receipt.json") != frozen["receipt_sha256"] or file_hash(root / "protected-files.json") != frozen["protected_sha256"]:
        raise ValueError("Pinned engineering preflight changed")
    receipt = json.loads((root / "receipt.json").read_text())
    launch = json.loads((root / "launch.json").read_text())
    if (receipt["status"] != "engineering_zero_update_passed" or receipt["optimizer_updates"] != 0
            or receipt["execution_enabled"] is not False or receipt["full_training_completed"] is not False
            or receipt["cases"] != 54 or receipt["common_initial_proofs"] != 9
            or receipt["launch_sha256"] != file_hash(root / "launch.json") or receipt["code"] != launch["code"]
            or receipt["query_labels_accessed"] is not False or receipt["test_scoring"] is not False):
        raise ValueError("Complete independent zero-update preflight required")
    unchanged_producer(receipt["code"], code)
    select_case(launch, receipt["code"], *grid()[0], phase="engineering_zero_update")
    pins = json.loads((root / "protected-files.json").read_text())
    if any(file_hash(p) != h for p, h in pins.items()):
        raise ValueError("Preflight source/short/input changed before full registration")
    pins.update({str(root / "receipt.json"): frozen["receipt_sha256"],
        str(root / "protected-files.json"): frozen["protected_sha256"],
        str(root / "launch.json"): receipt["launch_sha256"]})
    return launch, pins


def case_data(case):
    inputs = case["inputs"]
    items, provenance, guard = load_target_context(case["target"], inputs["observation"], inputs["record"],
        inputs["record_sha256"], inputs["family"], inputs["family_sha256"], per_class=10, support_seed=64)
    if provenance != case["target_context"] or graph_binding(items) != case["graph_binding"]:
        raise ValueError("Paired full target observation/labels/graph changed")
    return items, provenance, guard


def freeze_campaign(registration_path, registration_sha256, code):
    """Reprove all54 identities in a new experiment tree, without taking an update."""
    registration = read_registration(registration_path, registration_sha256, code)
    original, pins = read_preflight(registration, code)
    root, old = external_directory(registration["out_dir"]), external_directory(original["out_dir"])
    owners = {old, *[Path(v["root"]).resolve() for v in original["short_acceptances"].values()]}
    for case in original["cases"]:
        owners.update(Path(case["inputs"][name]).resolve().parent for name in ("observation", "family"))
        owners.update(Path(case["source_spec"][name]).resolve().parent
            for name in ("source_launch", "source_independent_audit") if name in case["source_spec"])
    if root.parent != old.parent or any(root == owner or root in owner.parents or owner in root.parents for owner in owners):
        raise ValueError("Fresh independent full experiment output required")
    launch = copy.deepcopy(original)
    launch.update(code=code, out_dir=str(root), phase="full_campaign", execution_enabled=True,
        optimizer_updates=81000, registration_path=str(Path(registration_path).resolve()), registration_sha256=registration_sha256)
    proofs = {(p["target"], p["seed"]): p for p in launch["common_initials"]}
    for t in NAMES:
        for s in SEEDS:
            if common_fresh_initials(t, s) != proofs[(t, s)]:
                raise ValueError("Previously accepted common fresh initialization changed")
    for case in launch["cases"]:
        case["out"] = str(root / "cases" / case["name"])
        _, _, guard = case_data(case)
        model, initial = initialize(case["target"], case["seed"], case["method"], case["mode"], source_spec=case["source_spec"])
        verify_fresh_target(model, proofs[(case["target"], case["seed"])])
        if initial != case["initialization"]:
            raise ValueError("Full source target initialization changed since preflight")
        guard()
        del model
    select_case(launch, code, *grid()[0], phase="full_campaign")
    require_frozen_code(code)
    if any(file_hash(p) != h for p, h in pins.items()) or file_hash(registration_path) != registration_sha256:
        raise ValueError("Registration/protected source/inputs changed while freezing full54")
    root.mkdir(exist_ok=False)
    (root / "logs").mkdir()
    pins.update({str(Path(registration_path).resolve()): registration_sha256,
        registration["engineering_acceptance"]["path"]: registration["engineering_acceptance"]["sha256"],
        registration["analysis_plan"]["path"]: registration["analysis_plan"]["sha256"]})
    write_json(root / "launch.json", launch)
    pins[str(root / "launch.json")] = file_hash(root / "launch.json")
    write_json(root / "protected-files.json", pins)
    write_json(root / "status.json", {"status": "frozen_before_first_update", "completed_cases": 0,
        "planned_optimizer_updates": 81000, "query_labels_accessed": False})
    return launch


def read_full_case(launch_path, launch_sha256, registration_path, registration_sha256, code, key):
    registration = read_registration(registration_path, registration_sha256, code)
    if file_hash(launch_path) != launch_sha256:
        raise ValueError("Pinned executable full54 launch required")
    launch = json.loads(Path(launch_path).read_text())
    if (Path(launch_path).resolve() != Path(registration["out_dir"]).resolve() / "launch.json"
            or launch.get("registration_path") != str(Path(registration_path).resolve())
            or launch.get("registration_sha256") != registration_sha256):
        raise ValueError("Full launch belongs to a different experiment registration")
    return select_case(launch, code, *key, phase="full_campaign"), registration


def worker_main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "registration", "registration-sha256", "launch-plan", "launch-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--target", choices=NAMES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--method", choices=METHODS[:2], required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, choices=(2,))
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean registered remote commit required")
    case, registration = read_full_case(args.launch_plan, args.launch_sha256, args.registration,
        args.registration_sha256, code, (args.target, args.seed, args.method, args.mode))
    items, provenance, guard = case_data(case)
    provenance.update(source_spec=case["source_spec"], m4_mode=case["mode"], matched_full_target_launch={
        "protocol": EXPECTED["protocol"], "launch_sha256": args.launch_sha256, "case": case["name"],
        "registration_sha256": args.registration_sha256, "initial_state_sha256": case["initial_state_sha256"]})
    pins = provenance["protected_files"]
    pins.update({args.launch_plan: args.launch_sha256, args.registration: args.registration_sha256})
    for field in ("engineering_acceptance", "analysis_plan"):
        pins[registration[field]["path"]] = registration[field]["sha256"]
    spec = case["source_spec"]
    for field, path in spec.items():
        if not field.endswith("sha256"):
            pins[path] = spec["source_weight_sha256" if field == "source_path" else field + "_sha256"]
    if spec:
        pins[spec["source_path"] + ".json"] = spec["source_sidecar_sha256"]
    def finish():
        guard()
        require_frozen_code(code)
        if any(file_hash(p) != h for p, h in pins.items()):
            raise ValueError("Protected full registration/source/target input changed")
    finish()
    wait_gpu(registration["gpu"], external_directory(registration["out_dir"]))
    result = run_training(case["settings"], items, case["out"], provenance, code, device="cuda",
        resume=args.resume, stop_after=args.stop_after, finish_guard=finish, runtime_adapter=runtime(case, items))
    print(json.dumps({k: result[k] for k in ("operation", "updates_this_call")}), flush=True)
