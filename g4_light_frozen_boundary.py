"""Freeze authenticated light-adaptation selections without opening query labels."""
import argparse
from pathlib import Path
import sys

from g4_light_full_campaign import POLICY as ACCEPTANCE_POLICY, case_directory
from g4_light_full_target import UPDATES, read_registration, roles
from g4_matched_target_evaluation import new_output
from g4_matched_target_execution import unchanged_producer
from g4_matched_target_study import METHODS, select_case
from g4_support_context import check_files, read
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


POLICY = "g4_light_all54_frozen_boundary_v1"
MODEL_FILES = {"result.json": "result_sha256", "primary.pt": "primary_sha256",
               "primary.pt.json": "sidecar_sha256"}


def model_binding(root, key, gpu, reference, receipt, protected, producer, registration_sha256):
    folder = case_directory(root, key)
    target, _, method, _ = key
    selected_step = receipt["selected_step"]
    if (receipt["status"] != "passed" or receipt["name"] != folder.name or receipt["role"] != list(key)
            or receipt["physical_gpu"] != gpu or receipt["updates"] != UPDATES or receipt["validation_events"] != 51
            or type(selected_step) is not int or not 1 <= selected_step <= UPDATES
            or receipt["head_metrics"] != 6 or receipt["class_recalls"] != 6 * DATASETS[target][2]
            or receipt["exact_reload_repeats"] != 3 or receipt["new_optimizer_updates"] != 0
            or receipt["query_labels_accessed"] is not False):
        raise ValueError(f"Independent acceptance differs for {folder.name}")
    for group in ("protected_inputs", "artifacts"):
        if not receipt[group] or any(protected.get(path) != sha for path, sha in receipt[group].items()):
            raise ValueError(f"Missing or conflicting accepted pins for {folder.name}")
    model_pins = {str(folder / filename): receipt[field] for filename, field in MODEL_FILES.items()}
    if any(receipt["artifacts"].get(path) != sha for path, sha in model_pins.items()):
        raise ValueError(f"Selected artifacts differ for {folder.name}")
    for field in ("record", "family"):
        path = reference["inputs"][field]
        if receipt["protected_inputs"].get(path) != reference["inputs"][field + "_sha256"]:
            raise ValueError(f"Matched reference labels differ for {folder.name}")

    result = read(folder / "result.json")
    settings = result["settings"]
    provenance = result["identity"]["prepared"]
    reference_sha256 = digest(reference)
    if (result["status"] != "completed" or result["start_code"] != producer
            or result["test_scoring"] is not False or result["optimizer_steps"] != UPDATES
            or result["per_dataset_updates"] != {target: UPDATES} or result["selected_step"] != selected_step
            or settings["target"] != target or settings["method"] != method
            or settings["physical_gpu"] != gpu or settings["registration_sha256"] != registration_sha256
            or settings["reference_case_sha256"] != reference_sha256
            or settings["expected_initial_state_sha256"] != reference["initial_state_sha256"]
            or provenance["reference_case"] != reference["name"]
            or provenance["protected_files"] != receipt["protected_inputs"]
            or any(result["artifacts"].get(name) != receipt[MODEL_FILES[name]]
                   for name in ("primary.pt", "primary.pt.json"))):
        raise ValueError(f"Training result differs from its registered reference for {folder.name}")
    return {"name": folder.name, "role": list(key), "physical_gpu": gpu,
            "selected_step": selected_step, "reference_case": reference["name"],
            "reference_case_sha256": reference_sha256, "model_files": model_pins}


def freeze(registration_path, registration_sha256, acceptance_sha256, out, code):
    registration_path = Path(registration_path).resolve(strict=True)
    check_files({str(registration_path): registration_sha256})
    root = external_directory(read(registration_path)["out_dir"])
    status_path, acceptance_path = root / "status.json", root / "training-acceptance.json"
    if status_path.resolve() != status_path or acceptance_path.resolve() != acceptance_path:
        raise ValueError("Campaign status and acceptance cannot traverse symlinks")
    status_sha256 = file_hash(status_path)
    status = read(status_path)
    keys = roles()
    updates = len(keys) * UPDATES
    if (status["status"] != "completed" or status["errors"] or status["completed_cases"] != len(keys)
            or status["planned_cases"] != len(keys) or status["planned_optimizer_updates"] != updates):
        raise ValueError("Completed training status for all 54 roles required")
    check_files({str(acceptance_path): acceptance_sha256})
    accepted = read(acceptance_path)
    producer = accepted["code"]
    unchanged_producer(producer, code)
    registration, registered_pins = read_registration(registration_path, registration_sha256, producer)
    launch = read(registration["reference_launch"]["path"])
    unchanged_producer(launch["code"], producer)
    target, seed, _, mode = keys[0]
    select_case(launch, launch["code"], target, seed, METHODS[1], mode, phase="full_campaign")
    root = new_output(root, launch)
    out = new_output(out, launch, (root,))
    if out.exists():
        raise FileExistsError(out)
    folders = [case_directory(root, key) for key in keys]
    receipts = [root / "audits" / (folder.name + ".json") for folder in folders]
    gpus = registration["gpu_ids"]
    header = {"policy": ACCEPTANCE_POLICY, "registration_sha256": registration_sha256, "gpu_ids": gpus,
              "campaign_engineering_sha256": status["campaign_engineering_sha256"]}
    protected = accepted["protected_files"]
    audit_pins = accepted["audit_receipts"]
    if (any(document[k] != value for document in (status, accepted) for k, value in header.items())
            or accepted["status"] != "passed" or accepted["optimizer_updates"] != updates
            or accepted["query_labels_accessed"] is not False or accepted["cases"] != [p.name for p in folders]
            or any(p.resolve() != p for p in receipts) or set(audit_pins) != {str(p) for p in receipts}
            or any(protected.get(path) != sha for path, sha in registered_pins.items())
            or header["campaign_engineering_sha256"] not in protected.values()):
        raise ValueError("Pinned campaign acceptance does not cover all registered roles and inputs")
    boundary_pins = {str(status_path): status_sha256, str(acceptance_path): acceptance_sha256, **audit_pins}
    if any(Path(path).resolve() == out or out in Path(path).resolve().parents for path in protected):
        raise ValueError("Freeze output overlaps an accepted input")
    check_files(boundary_pins)
    check_files(protected)

    reference_cases = {(row["target"], row["seed"], row["mode"]): row
                       for row in launch["cases"] if row["method"] == METHODS[1]}
    models = []
    for index, ((target, seed, method, mode), receipt_path) in enumerate(zip(keys, receipts)):
        reference = reference_cases[target, seed, mode]
        model = model_binding(root, (target, seed, method, mode), gpus[index % len(gpus)], reference,
                              read(receipt_path), protected, producer, registration_sha256)
        models.append({**model, "audit_path": str(receipt_path), "audit_sha256": audit_pins[str(receipt_path)]})
    check_files(boundary_pins)
    require_frozen_code(code)
    manifest = {"policy": POLICY, "status": "all54_frozen_before_query", "code": code, "producer": producer,
                "experiment_issue": registration["experiment_issue"], "training_root": str(root),
                "registration_path": str(registration_path), "registration_sha256": registration_sha256,
                "acceptance_sha256": acceptance_sha256, "reference_launch": registration["reference_launch"],
                "models": models, "protected_files": protected, "boundary_files": boundary_pins,
                "training_optimizer_updates": updates, "new_optimizer_updates": 0, "query_labels_accessed": False}
    out.mkdir(exist_ok=False)
    write_json(out / "freeze.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "registration", "registration-sha256", "acceptance-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean frozen-boundary Linux commit required")
    manifest = freeze(args.registration, args.registration_sha256, args.acceptance_sha256, args.out, code)
    print(f"Frozen {len(manifest['models'])} roles: {Path(args.out).resolve() / 'freeze.json'}")


if __name__ == "__main__":
    main()
