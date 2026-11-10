"""Run one preregistered WP11 confirmation using isolated tile and model packages."""

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.dynamic_cost_pairs import matched_cost_pairs
from evaluation.dynamic_curve import CurveProtocol
from evaluation.wp11_confirmation import confirmation_environment, layout, run_confirmation
from inference.gate_calibration import GateProtocol
from inference.trained_light import load_trained_light
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from transfer.registered_development import _load_source, _read, load_development


def run(registration, out, code):
    spec = json.loads(_read(registration))
    fields = {"development", "full_weights", "full_metadata", "full_history", "history_evidence", "light",
              "light_training_sha256", "light_contract_hash", "admission", "tiles", "gate_protocol",
              "gate_freeze", "fixed", "confirmation_labels", "curve_protocol", "comparison"}
    if set(spec) != fields:
        raise ValueError("Exact confirmation campaign artifacts required")

    def read(name):
        return json.loads(_read(spec[name]))

    pins = [registration, *[spec[k] for k in fields - {"tiles", "light_training_sha256", "light_contract_hash"}]]
    pins.extend(row["file"] for row in spec["tiles"])
    label_path = Path(spec["confirmation_labels"]["path"]).resolve()
    if sum(Path(p["path"]).resolve() == label_path for p in pins) != 1:
        raise ValueError("Confirmation labels require a separate deferred artifact")
    if out.exists() or any(out == Path(p["path"]).resolve() or out in Path(p["path"]).resolve().parents for p in pins):
        raise ValueError("New external output outside all input packages required")
    scene, record, development_files = load_development(spec["development"])
    if any(Path(p["path"]).resolve() == label_path for p in development_files):
        raise ValueError("Confirmation labels cannot be part of the training package")
    pins.extend(development_files)
    admission = read("admission")
    blocks = admission["blocks"]
    layout(scene.plan, record, blocks)
    tiles = {row["tile_id"]: row["file"] for row in spec["tiles"]}
    if len(tiles) != len(spec["tiles"]) or tiles.keys() != {b["tile_id"] for b in blocks}:
        raise ValueError("Exactly the registered held-aside tile files required")
    for tile_id, pin in tiles.items():
        sl = tile_slice(scene.plan["tiles"][tile_id])
        data = np.load(io.BytesIO(_read(pin)), allow_pickle=False)
        if not isinstance(data, np.ndarray) or data.shape != scene.cube[sl].shape or data.dtype != np.float32:
            raise ValueError("Held-aside file must be a raw FP32 spectral tile")
        scene.cube[sl] = data
    history = read("full_history")
    if history["history_evidence_sha256"] != spec["history_evidence"]["sha256"]:
        raise ValueError("Full history evidence differs")
    _read(spec["history_evidence"])
    if admission["light_artifact_sha256"] != spec["light"]["sha256"]:
        raise ValueError("Light artifact differs from admission")
    full = _load_source(spec["full_weights"], spec["full_metadata"])
    light = load_trained_light(full, spec["light"]["path"], expected_sha256=spec["light"]["sha256"],
                               expected_training_sha256=spec["light_training_sha256"],
                               expected_contract_hash=spec["light_contract_hash"])
    gate, frozen, fixed, curve_plan, pairs = (read(k) for k in
        ("gate_protocol", "gate_freeze", "fixed", "curve_protocol", "comparison"))
    protocol = GateProtocol(gate, digest(gate))
    curve = CurveProtocol(protocol, curve_plan, expected_hash=digest(curve_plan))
    if curve_plan["environment_hash"] != digest(confirmation_environment()):
        raise ValueError("Registered CPU timing environment differs")
    receipt, records, points = run_confirmation(
        full, light, scene, record, history, admission, expected_registration_hash=digest(admission),
        protocol=protocol, frozen=frozen, expected_freeze_hash=digest(frozen),
        read_labels=lambda: read("confirmation_labels"), fixed=fixed, expected_fixed_hash=digest(fixed))
    report = curve.report(records, points, expected_results_hash=digest(points))
    comparison = matched_cost_pairs(report, pairs, expected_spec_hash=digest(pairs),
                                    expected_curve_hash=digest(report))
    require_frozen_code(code)
    for pin in pins:
        _read(pin)
    out.mkdir()
    for name, value in (("runtime-points", points), ("curve-report", report),
                        ("matched-cost-report", comparison), ("receipt", receipt)):
        write_json(out / f"{name}.json", value)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux confirmation checkout required")
    run(dict(path=args.registration, sha256=args.registration_sha256), external_directory(args.out), code)


if __name__ == "__main__":
    main()
