"""Authenticate a paired fallback without changing the historical light artifact."""

import json
from pathlib import Path

from inference.trained_light import load_trained_light
from model.research_config import digest
from transfer.g3_training import state_hash
from transfer.registered_development import _load_source, _read
from transfer.registered_wp11_learning import RELEASE


MODEL_FILES = ("development", "full_weights", "full_metadata", "full_history", "history_evidence", "light")
LIGHT_PINS = ("light_training_sha256", "light_contract_hash")
SOURCE_FILES = ("source_weights", "source_metadata", "source_history", "source_evidence")
MATCHED = "matched_release"


def forbid_aliases(pins, forbidden):
    for candidate in pins:
        path = Path(candidate["path"]).resolve()
        if any(candidate["sha256"] == p["sha256"] or
               ("path" in p and path == Path(p["path"]).resolve()) for p in forbidden):
            raise ValueError("Deferred role labels cannot be used as model or history inputs")


def load_pair(spec, scene, record, *, forbidden=()):
    """Authenticate legacy endpoints or the exact opt-in matched release."""
    from evaluation.wp11_confirmation import validate_full_history

    pins = [spec[key] for key in MODEL_FILES]
    paired = MATCHED in spec
    if paired:
        pins.append(spec[MATCHED])
    forbid_aliases(pins, forbidden)
    if paired:
        release = json.loads(_read(spec[MATCHED]))
        fields = {*MODEL_FILES, *LIGHT_PINS, *SOURCE_FILES, "registration", "policy", "commit",
                  "source_state", "full_state", "light_spec_sha256"}
        if (set(release) != fields or release["policy"] != RELEASE
                or any(release[k] != spec[k] for k in (*MODEL_FILES, *LIGHT_PINS))):
            raise ValueError("Matched endpoints differ from their audited release")
        extra = [release[key] for key in (*SOURCE_FILES, "registration")]
        forbid_aliases(extra, forbidden)
        pins.extend(extra)
    full = _load_source(spec["full_weights"], spec["full_metadata"])
    history = json.loads(_read(spec["full_history"]))
    if history["history_evidence_sha256"] != spec["history_evidence"]["sha256"]:
        raise ValueError("Full history evidence differs")
    evidence = _read(spec["history_evidence"])
    if paired:
        source = _load_source(release["source_weights"], release["source_metadata"])
        old_history = json.loads(_read(release["source_history"]))
        if old_history["history_evidence_sha256"] != release["source_evidence"]["sha256"]:
            raise ValueError("Original source history evidence differs")
        _read(release["source_evidence"])
        validate_full_history(source, scene, record, old_history)
        registered = json.loads(_read(release["registration"]))
        audit = json.loads(evidence)
        ancestry = audit["label_use"]
        if (audit["status"] != "audited" or audit["policy"] != RELEASE or audit["commit"] != release["commit"]
                or audit["registration"] != release["registration"] or registered["commit"] != release["commit"]
                or audit["exact_continuous_resume_completed"] is not True
                or audit["actual_updates"] != registered["total_updates"]
                or any(ancestry[key] is not False for key in ("read_calibration", "read_confirmation", "original_gt_read"))
                or ancestry["query"] != 0 or ancestry["development"] != release["development"]
                or ancestry["source_history"] != release["source_history"]
                or ancestry["source_history_evidence"] != release["source_evidence"]
                or ancestry["source_full_usage"] != old_history["tile_usage"]
                or history["tile_usage"]["fit"] != old_history["tile_usage"]["fit"]
                or history["tile_usage"]["selection"] != old_history["tile_usage"]["selection"]
                or history["tile_usage"]["training"] != sorted(set(old_history["tile_usage"]["training"]) |
                                                              set(ancestry["optimizer_tiles"]))
                or release["source_state"] != state_hash(source) or release["source_state"] != registered["source_state"]
                or release["full_state"] != state_hash(full) or release["full_state"] != audit["full_state"]):
            raise ValueError("Matched training and inherited label history are not authenticated")
    else:
        source = full
    validate_full_history(full, scene, record, history)
    light = load_trained_light(source, spec["light"]["path"], expected_sha256=spec["light"]["sha256"],
                               expected_training_sha256=spec["light_training_sha256"],
                               expected_contract_hash=spec["light_contract_hash"])
    if paired:
        if digest(light.specification) != release["light_spec_sha256"]:
            raise ValueError("Trained light specification differs from the release")
        light.bind_fallback(source, full, expected_fallback_state=release["full_state"])
    return full, light, history, pins
