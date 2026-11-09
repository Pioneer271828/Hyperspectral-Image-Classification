"""Readonly physical-provenance sidecar for accepted27 contexts; no label decoding."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from g4_matched_target_execution import unchanged_producer
from g4_support_context import POLICY as CONTEXT_POLICY, check_files, context_name, grid, read
from joint_baseline import NAMES, _item_fingerprint
from loadData.data_read import read_band_metadata
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, tensor_hash, write_json
from target_bn_calibration import OBS_KEYS, observation_id

POLICY = "g4_observation_provenance_audit_v1"
INPUT_POLICY = "g4_observation_provenance_inputs_v1"
CATALOG = Path(__file__).resolve().parent / "docs/references/g4-observation-evidence-v1.json"
CLAIMS = {
    "few_shot_scene_transductive": (True, "Describes support/V5 and full-scene unlabeled context; no execution or effect acceptance"),
    "strict_spatial_isolation": (False, "Full-scene normalization/observation/Q/A; no independent partition context proof"),
    "unseen_sensor_DG": (False, "Target labels and unlabeled target context are used; no source-only semantic-head protocol"),
    "verified_sensor_family_holdout": (False, "Nominal descriptions do not audit local instrument identity or all source-family exclusions"),
    "measured_band_centers": (False, "Loader metadata approximate; no authenticated per-band calibration and retained sensor-band mapping"),
    "physical_FWHM": (False, "No authenticated physical FWHM/SRF in these frozen observations")}


def claim_table():
    return {k: {"allowed": v[0], "reason": v[1]} for k, v in CLAIMS.items()}


def require_claim(report, claim):
    if (report.get("policy") != POLICY or report.get("status") != "passed"
            or report.get("claims") != claim_table() or report.get("training_executable") is not False
            or report.get("observations") != 3 or report.get("contexts") != 27):
        raise ValueError("Authenticated conservative audit description required")
    if claim not in CLAIMS or not CLAIMS[claim][0]:
        raise ValueError("Unsupported physical/protocol claim: " + claim)
    return True


def load_catalog(path, sha256):
    if file_hash(path) != sha256:
        raise ValueError("Pinned reviewed evidence catalog required")
    value = read(path)
    if (value["policy"] != "g4_observation_evidence_catalog_v1" or set(value["scenes"]) != set(NAMES)
            or set(value["local_documents"]) != {"xuzhou_package_readme", "honghu_local_metadata"}
            or value["external_code_reused"] is not False or value["dataset_downloaded_by_this_module"] is not False):
        raise ValueError("Current readonly evidence catalog required")
    for name, row in value["scenes"].items():
        if (row["bands"] != {"paviaU": 103, "xuzhou": 436, "whu-honghu": 270}[name]
                or row["wavelength_quality"] != "approximate"
                or any(row[k] is not False for k in ("original_sensor_band_mapping_authenticated",
                    "per_band_centers_authenticated", "fwhm_authenticated", "srf_authenticated"))):
            raise ValueError("Unproved physical metadata upgrade requires another evidence/version review")
    return value


def inspect_observation(target, item, record, fact):
    if (set(item) != OBS_KEYS | {"name", "height", "width", "bands", "class_num"}
            or item["name"] != target or item["class_num"] != DATASETS[target][2]
            or item["bands"] != fact["bands"] or item["net_input"].shape != (item["height"], item["width"], item["bands"] + 2)
            or _item_fingerprint(item) != record["prepared_fields"]):
        raise ValueError("Exact label-free observation/geometry/record required")
    identity = observation_id({k: item[k] for k in OBS_KEYS})
    bands = read_band_metadata(target, item["bands"])
    expected = torch.tensor(bands.wavelengths, dtype=torch.float32)
    if (item["wavelengths"].dtype != torch.float32 or not torch.equal(item["wavelengths"], expected)
            or item["fwhm"] is not None or bands.fwhm is not None or bands.wavelength_unit != "nm"
            or bands.wavelength_quality != "approximate"
            or not bool((expected[1:] > expected[:-1]).all())):
        raise ValueError("Consumed wavelength/FWHM/unit/quality differs from frozen loader")
    return {"target": target, "shape": [item["height"], item["width"], item["bands"]],
        "legacy_registry_family": DATASETS[target][3], "reported_sensor": fact["reported_sensor"],
        "sensor_evidence_level": fact["evidence_level"], "wavelength_quality": bands.wavelength_quality,
        "wavelength_unit": bands.wavelength_unit, "wavelength_sha256": tensor_hash(item["wavelengths"]),
        "wavelength_range": [float(expected[0]), float(expected[-1])],
        "band_index_scope": "local_retained_array_positions_not_authenticated_original_sensor_indices",
        "original_sensor_band_mapping_authenticated": False, "per_band_centers_authenticated": False,
        "fwhm_present": False, "srf_authenticated": False, "observation_identity": identity,
        "graph_identity": digest(record["prepared_fields"]), "source_note": fact["note"]}


def audit(inputs_path, inputs_sha256, out, code, *, catalog_path=CATALOG, catalog_sha256=None):
    started = time.monotonic()
    require_frozen_code(code)
    inputs_path = Path(inputs_path).resolve(strict=True)
    if file_hash(inputs_path) != inputs_sha256:
        raise ValueError("Pinned audit inputs required")
    spec = read(inputs_path)
    if set(spec) != {"policy", "context_manifest", "context_audit", "evidence_files"} or spec["policy"] != INPUT_POLICY:
        raise ValueError("Explicit accepted contexts and evidence inputs required")
    catalog_path = Path(catalog_path).resolve(strict=True)
    catalog_sha256 = catalog_sha256 or file_hash(catalog_path)
    facts = load_catalog(catalog_path, catalog_sha256)
    if set(spec["evidence_files"]) != set(facts["local_documents"]):
        raise ValueError("Exactly the two reviewed local evidence documents required")
    pins = {str(inputs_path): inputs_sha256, str(catalog_path): catalog_sha256}
    for entry in (spec["context_manifest"], spec["context_audit"], *spec["evidence_files"].values()):
        if set(entry) != {"path", "sha256"}:
            raise ValueError("Explicit file path/hash pair required")
        pins[str(Path(entry["path"]).resolve(strict=True))] = entry["sha256"]
    check_files(pins)
    for key, entry in spec["evidence_files"].items():
        if entry["sha256"] != facts["local_documents"][key]["sha256"]:
            raise ValueError("Evidence bytes differ from reviewed catalog")
    text = Path(spec["evidence_files"]["xuzhou_package_readme"]["path"]).read_text(encoding="utf-8").upper()
    if not all(word in text for word in ("HYSPEX", "436", "415", "2508")):
        raise ValueError("Xuzhou package evidence does not support the recorded limited description")
    local = read(spec["evidence_files"]["honghu_local_metadata"]["path"])
    if (local["has_true_center_wavelengths"] is not False or local["bands"] != 270
            or local["spectral_range_nm"] != [400.0, 1000.0] or "Headwall Nano-Hyperspec" not in local["sensor"]):
        raise ValueError("HongHu local approximate metadata description differs")
    manifest_path = Path(spec["context_manifest"]["path"]).resolve(strict=True)
    manifest, accepted = read(manifest_path), read(spec["context_audit"]["path"])
    roles = {(r["target"], r["support_seed"], r["support_per_class"]) for r in manifest["contexts"]}
    if (manifest["policy"] != CONTEXT_POLICY or manifest["phase"] != "engineering_only"
            or manifest["execution_enabled"] is not False or manifest["optimizer_updates"] != 0
            or manifest["query_labels_accessed"] is not False or manifest["test_scoring"] is not False
            or manifest["full_target_scene_transductive"] is not True or manifest["spatial_isolation"] is not False
            or Path(manifest["out_dir"]).resolve() != manifest_path.parent or len(manifest["contexts"]) != 27
            or roles != set(grid()) or set(manifest["observations"]) != set(NAMES)):
        raise ValueError("Complete frozen engineering27 full-scene contexts required")
    producer_receipt = manifest_path.parent / "receipt.json"
    if (accepted["status"] != "passed" or accepted["issue"] != 188
            or accepted["manifest_sha256"] != spec["context_manifest"]["sha256"]
            or file_hash(producer_receipt) != accepted["producer_receipt_sha256"]
            or accepted["optimizer_updates"] != 0 or accepted["query_labels_accessed"] is not False
            or accepted["raw_gt_accessed"] is not False or accepted["training_executable"] is not False
            or (accepted["phase_checks"], accepted["label_tensor_checks"], accepted["graph_observation_checks"]) != (54, 108, 27)
            or len(accepted["contexts"]) != 27
            or {(r["target"], r["support_seed"], r["k"]) for r in accepted["contexts"]} != set(grid())):
        raise ValueError("Accepted independent188 context audit required")
    producer = read(producer_receipt)
    if producer["status"] != "passed" or producer["manifest_sha256"] != spec["context_manifest"]["sha256"]:
        raise ValueError("Accepted context producer receipt differs")
    unchanged_producer(manifest["code"], code)
    unchanged_producer(accepted["code"], code)
    pins[str(producer_receipt)] = accepted["producer_receipt_sha256"]
    pins.update(manifest["protected_files"])
    for row in manifest["contexts"]:
        if (row["name"] != context_name(row["target"], row["support_seed"], row["support_per_class"])
                or row["file"] != row["name"] + ".pt" or row["validation_per_class"] != 5
                or row["total_annotation_labels"] != DATASETS[row["target"]][2] * (row["support_per_class"] + 5)):
            raise ValueError("Context role/path/annotation accounting differs")
        pins[str(manifest_path.parent / row["file"])] = row["sha256"]
    check_files(pins)
    out = external_directory(out)
    input_roots = {Path(p).resolve().parent for p in pins}
    if any(out == root or out in root.parents or root in out.parents for root in input_roots):
        raise ValueError("Audit outputs must be separate from all input/source directories")
    if out.exists():
        raise ValueError("Fresh audit output required")
    rows = []
    for target in NAMES:
        ref = manifest["observations"][target]["input"]
        for key in ("observation", "record"):
            if pins.get(ref[key]) != ref[key + "_sha256"]:
                raise ValueError("Observation pair is outside accepted source pins")
        # Only these three .pt files are decoded. Label overlays/families are hashed.
        item = torch.load(ref["observation"], map_location="cpu", weights_only=True)
        record = read(ref["record"])
        if file_hash(ref["observation"]) != record["observation_sha256"]:
            raise ValueError("Observation record hash differs")
        row = inspect_observation(target, item, record, facts["scenes"][target])
        for context in (r for r in manifest["contexts"] if r["target"] == target):
            if context["graph_identity"] != row["graph_identity"] or context["shape"] != row["shape"][:2]:
                raise ValueError("Accepted context graph/geometry differs")
        rows.append(row)
    result = {"policy": POLICY, "status": "passed", "code": code, "catalog_sha256": catalog_sha256,
        "observations": 3, "contexts": 27, "rows": rows, "claims": claim_table(),
        "context_annotation_labels": {r["name"]: r["total_annotation_labels"] for r in manifest["contexts"]},
        "protected_files": pins, "files_verified_unchanged": len(pins), "optimizer_updates": 0,
        "model_forwards": 0, "training_executable": False, "raw_gt_accessed": False,
        "label_tensors_deserialized": 0, "query_labels_accessed": False,
        "full_target_scene_transductive": True, "spatial_isolation": False,
        "physical_metadata_upgraded": False, "elapsed_seconds": time.monotonic() - started}
    check_files(pins)
    require_frozen_code(code)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "inputs", "inputs-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--catalog", default=str(CATALOG))
    parser.add_argument("--catalog-sha256", required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        raise ValueError("Expected provenance audit commit differs")
    value = audit(args.inputs, args.inputs_sha256, args.out, code,
                  catalog_path=args.catalog, catalog_sha256=args.catalog_sha256)
    print(json.dumps({"status": value["status"], "observations": 3, "contexts": 27}))


if __name__ == "__main__":
    main()
