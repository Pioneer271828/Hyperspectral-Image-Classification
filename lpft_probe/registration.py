"""Strict, read-only preflight for bounded LP-FT experiments.

This module never constructs a model, takes an update, or creates an output
directory. It authenticates the plan and reference launch before a future
consumer may allocate resources. Passing preflight is not training acceptance.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from g4_matched_target_evaluation import new_output
from g4_matched_target_execution import unchanged_producer
from g4_matched_target_study import METHODS, MODES, SEEDS, select_case
from joint_baseline import NAMES
from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code

from .g4_short import REFERENCE_LAUNCH_SHA256
from .policy import LPFTPlan


POLICY = "side_lpft_short_registration_v1"
FIELDS = {"policy", "producer_commit", "reference_launch", "out_dir", "cases",
          "plan", "saved_tensors", "planned_total_updates", "estimated_seconds",
          "query_labels_accessed", "selection"}


def _unique(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("Duplicate JSON field: " + name)
        value[name] = item
    return value


def _read(path, sha256):
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("Explicit SHA256 required")
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Input SHA256 mismatch: " + str(path))
    value = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return path, value


def validate(value, expected_commit):
    """Validate the entire budget before any file, model, or GPU operation."""
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Exact short-registration fields required")
    if (not isinstance(expected_commit, str) or
            re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None or
            value["producer_commit"] != expected_commit or value["policy"] != POLICY):
        raise ValueError("Registration code or policy mismatch")
    plan = value["plan"]
    if not isinstance(plan, dict) or set(plan) != {
            "total_steps", "probe_steps", "head_lr", "backbone_lr", "weight_decay"}:
        raise ValueError("Exact two-stage budget and optimizer fields required")
    plan = LPFTPlan(**plan)
    if plan.total_steps > 12:
        raise ValueError("Short registration permits at most twelve updates per case")
    cases = value["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 3:
        raise ValueError("One to three explicit short cases required")
    identities = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"target", "seed", "mode", "gpu"}:
            raise ValueError("Exact target/seed/mode/GPU case fields required")
        if (case["target"] not in NAMES or type(case["seed"]) is not int or
                case["seed"] not in SEEDS or case["mode"] not in MODES or
                type(case["gpu"]) is not int or case["gpu"] < 0):
            raise ValueError("Unsupported short case or physical GPU")
        identity = (case["target"], case["seed"], case["mode"])
        if identity in identities:
            raise ValueError("Duplicate short case")
        identities.add(identity)
    budget = len(cases) * plan.total_steps
    if type(value["planned_total_updates"]) is not int or value["planned_total_updates"] != budget:
        raise ValueError("Aggregate update budget disagrees with explicit cases")
    seconds = value["estimated_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Finite positive time estimate required")
    if (value["saved_tensors"] not in ("device", "cpu") or
            value["query_labels_accessed"] is not False or
            value["selection"] != "none_endpoint_only"):
        raise ValueError("Memory, query or selection policy differs")
    reference = value["reference_launch"]
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"} or
            reference["sha256"] != REFERENCE_LAUNCH_SHA256):
        raise ValueError("Frozen Issue184 reference launch required")
    for name, path in (("reference", reference["path"]), ("output", value["out_dir"])):
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("Absolute " + name + " path required")
    return copy.deepcopy(value), plan


def preflight(path, sha256, *, expected_commit):
    """Authenticate plan/launch, code and output separation without mutations."""
    path, value = _read(path, sha256)
    value, plan = validate(value, expected_commit)
    code = code_snapshot()
    if code["dirty"] or code["commit"] != expected_commit:
        raise ValueError("Clean expected consumer commit required")
    reference = value["reference_launch"]
    launch_path, launch = _read(reference["path"], reference["sha256"])
    unchanged_producer(launch["code"], code)
    out = new_output(value["out_dir"], launch, extra_owners=(path.parent,))
    if out.exists():
        raise ValueError("New, unused output directory required")
    roles = []
    for case in value["cases"]:
        selected = select_case(launch, launch["code"], case["target"], case["seed"],
                               METHODS[1], case["mode"], phase="full_campaign")
        roles.append({**case, "reference_case_sha256": digest(selected)})
    # Recheck authenticated bytes and code before publishing the read-only result.
    _read(path, sha256)
    _read(launch_path, reference["sha256"])
    require_frozen_code(code)
    return {"status": "short_registration_preflight_passed", "registration_sha256": sha256,
            "producer_commit": expected_commit, "source_hash": code["source_hash"],
            "planned_total_updates": len(roles) * plan.total_steps, "cases": roles,
            "plan": plan.contract(), "saved_tensors": value["saved_tensors"],
            "out_dir": str(out), "optimizer_updates": 0, "query_labels_accessed": False,
            "source_context_runtime_check": "required_before_each_actual_case",
            "gpu_availability": "not_checked", "training_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    result = preflight(args.registration, args.registration_sha256,
                       expected_commit=args.expected_commit)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
