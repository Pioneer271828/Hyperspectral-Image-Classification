"""Freeze a registered WP11 gate using only the authorized calibration package."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp11_confirmation import build_requests, endpoint_records, load_held_aside_tiles, validate_admission, validate_gate_inputs
from evaluation.dynamic_curve import pin
from inference.gate_calibration import GateProtocol, SPARSE_PROTOCOL, calibrate, save_freeze
from inference.route_controls import _output_hash, request_hash
from evaluation.wp11_model_pair import MATCHED, forbid_aliases, load_pair
from inference.trained_light import trained_cascade
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.registered_development import _read, load_development


def run(registration, out, code):
    spec = json.loads(_read(registration))
    fields = {"development", "full_weights", "full_metadata", "full_history", "history_evidence", "light",
              "light_training_sha256", "light_contract_hash", "admission", "calibration_tiles", "gate_protocol",
              "calibration_receipt", "calibration_blocks", "calibration_labels"}
    if set(spec) not in (fields, fields | {"confirmation_design_sha256"}, fields | {"confirmation_design_sha256", MATCHED}):
        raise ValueError("Exact calibration-only registration required")
    if "confirmation_design_sha256" in spec and not pin(spec["confirmation_design_sha256"]):
        raise ValueError("Pinned confirmation design required")

    def read(key):
        return json.loads(_read(spec[key]))

    pins = [registration, *(spec[k] for k in fields - {"light_training_sha256", "light_contract_hash", "calibration_tiles"})]
    tiles = {row["tile_id"]: row["file"] for row in spec["calibration_tiles"]}
    pins.extend(tiles.values())
    label_path = Path(spec["calibration_labels"]["path"]).resolve()
    paths = [Path(p["path"]).resolve() for p in pins]
    if paths.count(label_path) != 1 or out.exists() or any(out == p or out in p.parents for p in paths):
        raise ValueError("Separate labels and new external output required")

    gate = read("gate_protocol")
    protocol = GateProtocol(gate, digest(gate))
    if gate["schema"] != SPARSE_PROTOCOL:
        raise ValueError("Calibration requires the sparse commitment protocol")
    admission = read("admission")
    if spec.get(MATCHED, {}).get("sha256") != admission.get("matched_release_sha256"):
        raise ValueError("Matched release differs from admission")
    if spec.get("confirmation_design_sha256") != admission.get("confirmation_design_sha256"):
        raise ValueError("Confirmation design must match admission before calibration")

    if admission["light_artifact_sha256"] != spec["light"]["sha256"]:
        raise ValueError("Light artifact differs from admission")
    forbidden = [spec["calibration_labels"]]
    if "confirmation_labels_file_sha256" in admission:
        forbidden.append({"sha256": admission["confirmation_labels_file_sha256"]})
    forbid_aliases([spec["development"]], forbidden)
    scene, record, development_files = load_development(spec["development"])
    if any(label_path == Path(p["path"]).resolve() for p in development_files):
        raise ValueError("Calibration labels cannot be part of the development package")
    pins.extend(development_files)
    full, light, history, pair_pins = load_pair(spec, scene, record, forbidden=forbidden)
    pins.extend(pair_pins)
    if any(out == Path(p["path"]).resolve() or out in Path(p["path"]).resolve().parents for p in pair_pins):
        raise ValueError("New external output outside model packages required")
    validate_admission(full, light, scene, record, history, admission, expected_registration_hash=gate["admission_hash"])
    blocks = validate_gate_inputs(gate, scene, history, admission)
    calibration = [b for b in blocks.values() if b["role"] == "calibration"]
    if len(tiles) != len(spec["calibration_tiles"]) or tiles.keys() != {b["tile_id"] for b in calibration}:
        raise ValueError("Exactly the calibration observations are required")
    load_held_aside_tiles(scene, tiles)
    requests = build_requests(scene, calibration, admission["scale"])
    if request_hash(requests) != gate["calibration_request_hash"]:
        raise ValueError("Calibrated input bytes differ")

    receipt = read("calibration_receipt")
    if (receipt["role"] != "calibration" or receipt["status"] != "packaged" or receipt["raw_labels_read"] is not False
            or receipt["query"] != 0 or receipt["labels_sha256"] != spec["calibration_labels"]["sha256"]
            or receipt["blocks_sha256"] != spec["calibration_blocks"]["sha256"]
            or receipt["labels_content_sha256"] != gate["data_hashes"]["calibration"]
            or read("calibration_blocks") != sorted(calibration, key=lambda b: b["tile_id"])):
        raise ValueError("Pinned calibration custody receipt required before opening labels")
    labels = read("calibration_labels")
    if (not isinstance(labels, dict) or labels.keys() != {b["id"] for b in calibration}
            or digest(labels) != gate["data_hashes"]["calibration"]):
        raise ValueError("Calibration labels differ from custody")
    for key, values in labels.items():
        if (not isinstance(values, list) or len(values) != len(blocks[key]["label_indices"])
                or any(type(y) is not int or not 1 <= y <= scene.classes for y in values)):
            raise ValueError("Calibration labels differ from the authorized subset")
    full_result = trained_cascade(full, light, requests)
    fast_result = trained_cascade(full, light, requests, mode="fast", min_coverage=gate["min_coverage"])
    records = endpoint_records(fast_result, full_result, requests, blocks, labels)
    frozen = calibrate(protocol, records)
    require_frozen_code(code)
    if any(hash_file(pin["path"]) != pin["sha256"] for pin in pins):
        raise ValueError("Calibration input changed")
    out.mkdir()
    file_hash = save_freeze(protocol, frozen, out / "gate-freeze.json")
    selected = frozen["candidates"][frozen["selected"]]
    result = dict(status="adaptive_frozen" if selected["kind"] == "adaptive" else "keep_full",
                  registration_sha256=registration["sha256"], protocol_hash=protocol.protocol_hash,
                  freeze_sha256=file_hash, freeze_hash=digest(frozen), selected=frozen["selected"],
                  accepted_tiles=selected["accepted_tiles"], support=selected["metrics"]["support"],
                  endpoints={"full": _output_hash(full_result), "fast": _output_hash(fast_result)},
                  calibration_tiles=len(requests), confirmation_observations=0, confirmation_labels_read=False,
                  optimizer_updates=0, query=0)
    if "confirmation_design_sha256" in spec:
        result["confirmation_design_sha256"] = spec["confirmation_design_sha256"]
    write_json(out / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument(f"--{key}", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux calibration checkout required")
    torch.set_num_threads(1)
    run(dict(path=args.registration, sha256=args.registration_sha256), external_directory(args.out), code)


if __name__ == "__main__":
    main()
