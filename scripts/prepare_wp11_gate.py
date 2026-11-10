"""Assemble WP11 gate inputs from role commitments and authorized positions."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.dynamic_curve import pin
from evaluation.wp11_confirmation import (POLICY, build_requests, layout, load_held_aside_tiles, unused_tiles,
                                            validate_benchmark, validate_full_history, validate_gate_inputs)
from inference.gate_calibration import GateProtocol, SPARSE_PROTOCOL
from inference.route_controls import request_hash
from inference.trained_light import load_trained_light
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, write_json
from scripts.prepare_wp11_label_custody import COMMITMENT, PLAN, ROLES, read, verify_inputs
from transfer.registered_development import _load_source, _read, load_development


MODEL_FILES = ("development", "full_weights", "full_metadata", "full_history", "history_evidence", "light")
LIGHT_PINS = ("light_training_sha256", "light_contract_hash")
COMMITMENT_FIELDS = {"policy", "role", "plan_sha256", "authority_sha256", "labels_sha256",
                     "labels_content_sha256", "blocks_sha256"}


def file_pin(path):
    return dict(path=str(path), sha256=hash_file(path))


def run(registration, out, code):
    spec = read(registration)
    fields = {*MODEL_FILES, *LIGHT_PINS, "role_plan", "calibration", "confirmation", "settings", "scale", "benchmark"}
    if set(spec) != fields or set(spec["calibration"]) != {"commitment", "blocks", "labels", "receipt"} or set(
            spec["confirmation"]) != {"commitment", "blocks"}:
        raise ValueError("Only the registered calibration and confirmation metadata are allowed")
    if set(spec["settings"]) != {"name", "criterion", "thresholds", "min_coverage", "min_class_support", "max_drop"}:
        raise ValueError("Exact gate settings required")
    validate_benchmark(spec["benchmark"])
    deferred = [spec["calibration"][key] for key in ("labels", "receipt")]
    for item in deferred:
        if set(item) != {"path", "sha256"} or not pin(item["sha256"]):
            raise ValueError("Separate pinned calibration release required")
        external_directory(item["path"])
    protected = {Path(item["path"]).resolve() for item in deferred}
    if len(protected) != len(deferred) or out.exists() or any(out == p or out in p.parents for p in protected):
        raise ValueError("Separate pinned calibration release and new external output required")

    def safe(items):
        if any(Path(p["path"]).resolve() in protected for p in items):
            raise ValueError("Deferred calibration files cannot serve as registration inputs")

    inputs = [registration, spec["role_plan"], *(spec[key] for key in MODEL_FILES)]
    inputs += [spec[role][key] for role in ROLES for key in ("commitment", "blocks")]
    safe(inputs)
    roles = read(spec["role_plan"])
    if roles["policy"] != PLAN or roles["raw_labels_read"] is not False:
        raise ValueError("Label-blind role plan required")
    observation_pin = roles["observations"]
    safe([observation_pin])
    observations = read(observation_pin)
    inputs.append(observation_pin)
    dev_root = Path(spec["development"]["path"]).resolve().parent
    if any(dev_root in p.parents for p in protected):
        raise ValueError("Deferred files cannot be inside the development package")
    scene, record, dev_files = load_development(spec["development"])
    safe(dev_files)
    inputs.extend(dev_files)
    available = {b["tile_id"]: b for b in roles["blocks"]}
    observed = {b["tile_id"]: b for b in observations["tiles"]}
    if (roles["scene"] != scene.dataset or observations["scene"] != scene.dataset or roles["classes"] != record["classes"]
            or observations["status"] != "candidates_prepared"
            or roles["geometry_sha256"] != observations["plan_sha256"] or roles["geometry_sha256"] != scene.plan["plan_hash"]
            or roles["development_sha256"] != observations["development_sha256"]
            or roles["development_sha256"] != record["validation_hash"] or observations["raw_labels_read"] is not False
            or len(available) != len(roles["blocks"]) or len(observed) != len(observations["tiles"])
            or available.keys() != observed.keys() or available.keys() != set(unused_tiles(scene.plan, record))):
        raise ValueError("Role plan and observations must match the isolated development geometry")

    blocks, commitments = [], {}
    for role in ROLES:
        release = spec[role]
        commitment = read(release["commitment"])
        if (set(commitment) != COMMITMENT_FIELDS or commitment["policy"] != COMMITMENT or commitment["role"] != role
                or commitment["plan_sha256"] != spec["role_plan"]["sha256"]
                or commitment["blocks_sha256"] != release["blocks"]["sha256"]
                or not all(pin(commitment[k]) for k in COMMITMENT_FIELDS - {"policy", "role"})):
            raise ValueError("Role-specific commitment and block pin required")
        commitments[role] = commitment
        selected = read(release["blocks"])
        if not isinstance(selected, list) or not selected:
            raise ValueError("Nonempty sparse block list required for each role")
        for block in selected:
            candidate = available.get(block["tile_id"])
            if (candidate is None or block["role"] != role or candidate["role"] != role or block["id"] != candidate["id"]
                    or candidate["box"] != observed[block["tile_id"]]["box"]
                    or candidate["box"] != scene.plan["tiles"][block["tile_id"]]["box"]):
                raise ValueError("Released positions escape the frozen role plan")
        blocks.extend(selected)
    if commitments["calibration"]["labels_sha256"] != spec["calibration"]["labels"]["sha256"]:
        raise ValueError("Deferred calibration label pin differs from custody")
    layout(scene.plan, record, blocks)

    history = read(spec["full_history"])
    if history["history_evidence_sha256"] != spec["history_evidence"]["sha256"]:
        raise ValueError("Full history evidence differs")
    _read(spec["history_evidence"])
    full = _load_source(spec["full_weights"], spec["full_metadata"])
    light = load_trained_light(full, spec["light"]["path"], expected_sha256=spec["light"]["sha256"],
                               expected_training_sha256=spec["light_training_sha256"],
                               expected_contract_hash=spec["light_contract_hash"])
    identity = validate_full_history(full, scene, record, history)
    light_spec = light.specification
    if light.validate(full) != scene.dataset or light_spec["training_spec"]["identity"] != identity:
        raise ValueError("Light training and selection footprint differs")

    tiles = {b["tile_id"]: observed[b["tile_id"]]["file"] for b in blocks}
    safe(tiles.values())
    inputs.extend(tiles.values())
    load_held_aside_tiles(scene, tiles)
    requests = build_requests(scene, blocks, spec["scale"])
    calibration = {b["id"]: b for b in blocks if b["role"] == "calibration"}
    hashes = {role: commitments[role]["labels_content_sha256"] for role in ROLES}
    admission = dict(policy=POLICY, history_sha256=digest(history), light_spec_sha256=digest(light_spec),
                     light_artifact_sha256=spec["light"]["sha256"], blocks=blocks, request_sha256=request_hash(requests),
                     labels_sha256=hashes["confirmation"], scale=spec["scale"], benchmark=spec["benchmark"])
    gate_blocks = [dict(id=f"development-{b['id']}", scene=scene.dataset, condition="full", context=b["box"],
                        role="training" if b["role"] == "devtrain" else "selection", label_indices=[])
                   for b in record["roles"] if b["role"] in ("devtrain", "devval")]
    gate_blocks.extend(dict(id=b["id"], scene=scene.dataset, condition="full", role=b["role"],
                           context=available[b["tile_id"]]["box"], label_indices=b["label_indices"]) for b in blocks)
    gate = dict(schema=SPARSE_PROTOCOL, **spec["settings"], full_state=history["full_state"],
                light_artifact=spec["light"]["sha256"], classes=scene.classes, blocks=gate_blocks, data_hashes=hashes,
                admission_hash=digest(admission), calibration_request_hash=request_hash([r for r in requests if r["id"] in calibration]))
    GateProtocol(gate, digest(gate))
    validate_gate_inputs(gate, scene, history, admission)
    verify_inputs(inputs, code, out)

    out.mkdir()
    write_json(out / "admission.json", admission)
    write_json(out / "gate-protocol.json", gate)
    base = {key: spec[key] for key in (*MODEL_FILES, *LIGHT_PINS)}
    base.update(admission=file_pin(out / "admission.json"), gate_protocol=file_pin(out / "gate-protocol.json"))
    cal_tiles = {b["tile_id"] for b in calibration.values()}
    tile_files = [dict(tile_id=i, file=p) for i, p in sorted(tiles.items())]
    write_json(out / "calibration-registration.json", dict(base,
               calibration_tiles=[t for t in tile_files if t["tile_id"] in cal_tiles],
               **{f"calibration_{key}": spec["calibration"][key] for key in ("receipt", "blocks", "labels")}))
    write_json(out / "confirmation-handoff.json", dict(inputs=dict(base, tiles=tile_files),
               confirmation_labels_sha256=commitments["confirmation"]["labels_sha256"],
               required_after_freeze=["gate_freeze", "fixed", "confirmation_labels", "curve_protocol", "comparison"]))
    receipt = dict(status="registered", gate_protocol_hash=digest(gate), admission_hash=digest(admission),
                   calibration=file_pin(out / "calibration-registration.json"), confirmation=file_pin(out / "confirmation-handoff.json"),
                   calibration_tiles=len(calibration), confirmation_tiles=len(blocks)-len(calibration),
                   held_aside_labels_read=False, model_forwards=0, optimizer_updates=0, query=0)
    write_json(out / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument(f"--{key}", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux gate-registration checkout required")
    run(dict(path=args.registration, sha256=args.registration_sha256), external_directory(args.out), code)


if __name__ == "__main__":
    main()
