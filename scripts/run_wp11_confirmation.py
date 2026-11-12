"""Run one preregistered WP11 confirmation using isolated tile and model packages."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.dynamic_cost_pairs import matched_cost_pairs
from evaluation.dynamic_curve import CurveProtocol
from evaluation.wp11_confirmation import confirmation_environment, layout, load_held_aside_tiles, run_confirmation
from inference.gate_calibration import GateProtocol
from evaluation.wp11_model_pair import MATCHED, forbid_aliases, load_pair
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.prepare_wp11_confirmation import resolve_confirmation
from transfer.registered_development import _read, load_development


def run(registration, out, code):
    spec = json.loads(_read(registration))
    fields = {"development", "full_weights", "full_metadata", "full_history", "history_evidence", "light",
              "light_training_sha256", "light_contract_hash", "admission", "tiles", "gate_protocol",
              "gate_freeze", "fixed", "confirmation_labels", "curve_protocol", "comparison"}
    if set(spec) not in (fields, fields | {"confirmation_binding"}, fields | {"confirmation_binding", MATCHED}):
        raise ValueError("Exact confirmation campaign artifacts required")

    def read(name):
        return json.loads(_read(spec[name]))

    pins = [registration, *[spec[k] for k in fields - {"tiles", "light_training_sha256", "light_contract_hash"}]]
    pins.extend(row["file"] for row in spec["tiles"])
    label_path = Path(spec["confirmation_labels"]["path"]).resolve()
    label_sha = spec["confirmation_labels"]["sha256"]
    if sum(Path(p["path"]).resolve() == label_path or p["sha256"] == label_sha for p in pins) != 1:
        raise ValueError("Confirmation labels require a separate deferred artifact")
    if out.exists() or any(out == Path(p["path"]).resolve() or out in Path(p["path"]).resolve().parents for p in pins):
        raise ValueError("New external output outside all input packages required")
    admission = read("admission")
    if spec.get(MATCHED, {}).get("sha256") != admission.get("matched_release_sha256"):
        raise ValueError("Matched release differs from admission")
    committed = "confirmation_design_sha256" in admission
    if committed != ("confirmation_binding" in spec):
        raise ValueError("Precommitted confirmation requires its calibration freeze binding")
    if committed:
        binding = spec["confirmation_binding"]
        if set(binding) != {"handoff", "freeze_receipt"}:
            raise ValueError("Exact confirmation binding required")
        resolved = resolve_confirmation(dict(binding, freeze=spec["gate_freeze"],
                                             confirmation_labels=spec["confirmation_labels"]), registration=registration)
        if (resolved["result"]["status"] != "registered" or
                any(spec[key] != value for key, value in resolved["inputs"].items()) or
                spec["fixed"] != resolved["fixed"] or read("curve_protocol") != resolved["curve"] or
                read("comparison") != resolved["comparison"]):
            raise ValueError("Confirmation inputs or controls differ from the precommitted design")
        pins.extend(resolved["pins"])
    forbid_aliases([spec["development"]], [spec["confirmation_labels"]])
    scene, record, development_files = load_development(spec["development"])
    if any(Path(p["path"]).resolve() == label_path for p in development_files):
        raise ValueError("Confirmation labels cannot be part of the training package")
    pins.extend(development_files)
    blocks = admission["blocks"]
    layout(scene.plan, record, blocks)
    tiles = {row["tile_id"]: row["file"] for row in spec["tiles"]}
    if len(tiles) != len(spec["tiles"]) or tiles.keys() != {b["tile_id"] for b in blocks}:
        raise ValueError("Exactly the registered held-aside tile files required")
    load_held_aside_tiles(scene, tiles)

    if admission["light_artifact_sha256"] != spec["light"]["sha256"]:
        raise ValueError("Light artifact differs from admission")
    full, light, history, pair_pins = load_pair(spec, scene, record, forbidden=[spec["confirmation_labels"]])
    pins.extend(pair_pins)
    if any(out == Path(p["path"]).resolve() or out in Path(p["path"]).resolve().parents for p in pair_pins):
        raise ValueError("New external output outside model packages required")
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
