"""Attach the accepted calibration freeze to preregistered confirmation controls."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.dynamic_cost_pairs import POLICY as PAIR_POLICY
from evaluation.dynamic_curve import CurveProtocol, number, pin
from evaluation.wp11_confirmation import TIMING_SCOPE, validate_benchmark
from inference.gate_calibration import GateProtocol, SPARSE_PROTOCOL, _validate_freeze
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, write_json
from scripts.prepare_wp11_label_custody import read, verify_inputs
from transfer.registered_development import _pin as file_pin


def confirmation_protocols(protocol, design, budget, fixed, freeze_hash):
    if set(design) != {"fixed", "environment_hash", "cost_interval_ms", "ece_bins", "nll_epsilon", "relative_tolerances"}:
        raise ValueError("Exact preregistered confirmation controls required")
    tolerances = design["relative_tolerances"]
    if set(tolerances) != {"p50_ms", "p95_ms"} or not all(number(v) and 0 <= v <= 1 for v in tolerances.values()):
        raise ValueError("Explicit p50 and p95 tolerances required")
    validate_benchmark(budget)
    gate = protocol.specification
    hashes = dict(full=digest({"mode": "full"}), fast=digest({"mode": "fast", "min_coverage": gate["min_coverage"]}),
                  adaptive=freeze_hash, fixed=digest(fixed))
    curve = dict(schema="dy-development-curve-v1", gate_protocol=protocol.protocol_hash, role="confirmation",
                 points=[dict(id=k, method=k, policy_hash=v) for k, v in hashes.items()],
                 environment_hash=design["environment_hash"], timing_scope=TIMING_SCOPE, warmups=budget["warmups"],
                 repeats=budget["pairs"], cost_interval_ms=design["cost_interval_ms"],
                 area_definition="all_points_cost_sorted_piecewise_linear_trapezoid_no_envelope",
                 ece_bins=design["ece_bins"], nll_epsilon=design["nll_epsilon"])
    CurveProtocol(protocol, curve, expected_hash=digest(curve))
    comparison = dict(policy=PAIR_POLICY, curve_protocol_hash=digest(curve), relative_tolerances=tolerances,
                      pairs=[dict(adaptive="adaptive", fixed="fixed")])
    return curve, comparison


def resolve_confirmation(spec, *, registration):
    """Rebuild the committed controls from metadata, with no label file access."""
    fields = {"handoff", "freeze", "freeze_receipt"}
    if set(spec) not in (fields, fields | {"confirmation_labels"}):
        raise ValueError("Pinned handoff and freeze artifacts required")
    labels = spec.get("confirmation_labels")
    if labels is not None and (set(labels) != {"path", "sha256"} or not pin(labels["sha256"])):
        raise ValueError("Confirmation release requires an explicit path and hash")
    label_path = external_directory(labels["path"]) if labels is not None else None
    committed_labels = None

    def metadata(items):
        if any(Path(item["path"]).resolve() == label_path or item["sha256"] == committed_labels for item in items):
            raise ValueError("Confirmation labels must remain a separate deferred artifact")

    inputs = [registration, *(spec[k] for k in fields)]
    metadata(inputs)
    handoff = read(spec["handoff"])
    if set(handoff) != {"inputs", "confirmation_labels_sha256", "required_after_freeze", "confirmation_design", "calibration"}:
        raise ValueError("Confirmation controls must be registered before calibration")
    committed_labels = handoff["confirmation_labels_sha256"]
    if not pin(committed_labels):
        raise ValueError("Exact confirmation custody commitment required")
    metadata(inputs)
    base = handoff["inputs"]
    sources = [handoff["calibration"], handoff["confirmation_design"], base["gate_protocol"], base["admission"]]
    metadata(sources)
    inputs.extend(sources)
    early, design, gate, admission = map(read, sources)
    design_sha = handoff["confirmation_design"]["sha256"]
    calibration_only = {"confirmation_design_sha256", "calibration_tiles", "calibration_labels",
                        "calibration_receipt", "calibration_blocks"}
    if (early.get("confirmation_design_sha256") != design_sha or admission.get("confirmation_design_sha256") != design_sha
            or admission.get("confirmation_labels_file_sha256") != committed_labels
            or {k: v for k, v in base.items() if k != "tiles"} != {k: v for k, v in early.items() if k not in calibration_only}
            or gate["schema"] != SPARSE_PROTOCOL or gate["admission_hash"] != digest(admission)):
        raise ValueError("Confirmation design or model admission differs from calibration registration")
    calibration_tiles = {row["tile_id"]: row["file"] for row in early["calibration_tiles"]}
    tiles = {row["tile_id"]: row["file"] for row in base["tiles"]}
    if len(tiles) != len(base["tiles"]) or any(tiles.get(key) != value for key, value in calibration_tiles.items()):
        raise ValueError("Confirmation inputs must include the frozen calibration tiles")
    metadata([base[key] for key in ("development", "full_weights", "full_metadata", "full_history", "history_evidence",
                                     "light", "admission", "gate_protocol")])
    if "matched_release" in base:
        metadata([base["matched_release"]])
        if base["matched_release"]["sha256"] != admission.get("matched_release_sha256"):
            raise ValueError("Matched release differs from the calibration admission")
    metadata(tiles.values())
    if label_path is not None and Path(base["development"]["path"]).resolve().parent in label_path.parents:
        raise ValueError("Confirmation release cannot be inside the development package")
    protocol = GateProtocol(gate, digest(gate))
    receipt = read(spec["freeze_receipt"])
    if (receipt.get("registration_sha256") != handoff["calibration"]["sha256"]
            or receipt.get("confirmation_design_sha256") != design_sha
            or receipt.get("protocol_hash") != protocol.protocol_hash or receipt.get("freeze_sha256") != spec["freeze"]["sha256"]
            or receipt.get("confirmation_labels_read") is not False or receipt.get("confirmation_observations") != 0
            or receipt.get("query") != 0):
        raise ValueError("Calibration receipt does not bind this registration and freeze")
    frozen = read(spec["freeze"])
    _validate_freeze(protocol, frozen, receipt["freeze_hash"])
    selected = frozen["candidates"][frozen["selected"]]
    adaptive = selected["kind"] == "adaptive"
    status = "adaptive_frozen" if adaptive else "keep_full"
    if receipt.get("status") != status or receipt.get("selected") != frozen["selected"]:
        raise ValueError("Calibration receipt selection differs")
    result = dict(status="registered" if adaptive else "keep_full", freeze_hash=receipt["freeze_hash"],
                  confirmation_design_sha256=design_sha, held_aside_labels_read=False, query=0)
    if not adaptive:
        return dict(result=result, pins=inputs)
    if labels is None or labels["sha256"] != committed_labels:
        raise ValueError("Confirmation release hash differs from custody")
    metadata([design["fixed"]])
    inputs.append(design["fixed"])
    fixed = read(design["fixed"])
    curve, comparison = confirmation_protocols(protocol, design, admission["benchmark"], fixed, receipt["freeze_hash"])
    return dict(result=result, pins=inputs, inputs=base, fixed=design["fixed"], curve=curve,
                comparison=comparison, label_path=label_path)


def run(registration, out, code):
    spec = read(registration)
    resolved = resolve_confirmation(spec, registration=registration)
    result = resolved["result"]
    label_path = resolved.get("label_path")
    if label_path is not None and (out == label_path or out in label_path.parents):
        raise ValueError("Output cannot contain the deferred label path")
    verify_inputs(resolved["pins"], code, out)
    out.mkdir()
    if result["status"] == "registered":
        write_json(out / "curve-protocol.json", resolved["curve"])
        write_json(out / "comparison.json", resolved["comparison"])
        path = out / "confirmation-registration.json"
        write_json(path, dict(resolved["inputs"], gate_freeze=spec["freeze"], fixed=resolved["fixed"],
                              confirmation_labels=spec["confirmation_labels"],
                              curve_protocol=file_pin(out / "curve-protocol.json"), comparison=file_pin(out / "comparison.json"),
                              confirmation_binding={k: spec[k] for k in ("handoff", "freeze_receipt")}))
        result["confirmation"] = file_pin(path)
    write_json(out / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux confirmation registration checkout required")
    run(dict(path=args.registration, sha256=args.registration_sha256), external_directory(args.out), code)


if __name__ == "__main__":
    main()
