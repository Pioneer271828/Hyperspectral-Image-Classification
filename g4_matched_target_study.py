"""Matched explicit ordered full target profiles; no query or training entry here."""
import copy
import json
from pathlib import Path

import torch
from joint_baseline import NAMES
from joint_training import validation_epochs
from loadData.dataset_registry import file_hash
from m4_full_target_runtime import FullSourceTargetRuntime, settings as active_settings
from m4_full_transfer import initialize_full_target
from model.research_config import digest
from none_full_target_runtime import NoneFullSourceTargetRuntime, settings as none_settings
from none_full_transfer import initialize_none_full_target
from research_runtime import ROOT, tensor_hash
from target_initialization import METHODS, domain_keys, state_id

MODES = ("none", "capacity_control", "x_residual")
SEEDS = (64, 512, 2048)
EXPECTED = {
    "protocol": "g4_matched_ordered_full_target_v2", "targets": list(NAMES), "modes": list(MODES),
    "model_seeds": list(SEEDS), "methods": list(METHODS[:2]), "support_seed": 64,
    "support_per_class": 10, "validation_per_class": 5, "head_seed_offset": 1000,
    "epochs": 1499, "steps_per_epoch": 1, "validation_interval": 30, "validation_events": 51,
    "updates_per_case": 1500, "none_cases": 18, "active_cases": 36, "total_cases": 54,
    "planned_full_optimizer_updates": 81000,
    "selection": "minimum_calibrated_fused_validation_nll_first_tie",
    "target_arithmetic": "ordered_calibration_and_validation_v2", "training_arithmetic_changed": False,
    "query_labels_during_training": False, "test_scoring_during_training": False,
    "probability_rtol": 1e-5, "probability_atol": 1e-6, "decision_changes": 0, "ordered_reloads_exact": True,
    "none_short_receipt_sha256": "1a34b9830f8333fe21aab312495b4cc95bbe272d3db50fddfdb690de08a1170c",
    "active_short_receipt_sha256": "abc7c5a5c464f4cf976f07b50f5a48ece23f44901b5d29cfc0da88c100fea809",
    "none_source_producer": "52f32978419cbeda216a90bfb0ed76c393ef4f5d",
    "active_source_producer": "01c29c82e92ce0a529749c53962202e9bf0dc5fe",
    "none_source_audit_sha256": "6ff38d1915c53286c1800832234afd3991cc879f029317c6dc01a70e39d4aa0b",
    "active_source_audit_sha256": "8e0617ec639f75f042d6037702269e96512335d41d4ed2c3ccb2c24921ef7d6f",
}
STUDY_PATH = ROOT / "research/experiments/g4_matched_target_full_v2.json"


def load_study(path=STUDY_PATH):
    value = json.loads(Path(path).read_text())
    if value != EXPECTED:
        raise ValueError("Register a new version for any full target study change")
    return value


def grid():
    return [(target, seed, method, mode) for target in NAMES for seed in SEEDS
            for method in METHODS[:2] for mode in MODES]


def case_name(target, seed, method, mode):
    if type(seed) is not int or (target, seed, method, mode) not in grid():
        raise ValueError("Registered matched full target role required")
    return target + "-s" + str(seed) + "-" + mode + "-" + method


def settings(target, seed, method, mode, *, device="cuda"):
    case_name(target, seed, method, mode)
    return (none_settings(target, method, seed, full=True, device=device) if mode == "none" else
            active_settings(target, method, mode, seed, full=True, device=device, ordered=True))


def training_contract(value):
    """Strip only declared architecture/identity differences, retaining all training semantics."""
    result = copy.deepcopy(value)
    for key in ("protocol", "full_source_initialization", "m4_mode"):
        result.pop(key, None)
    result["config"]["name"] = "matched_target_identity"
    result["config"]["model"]["region_residual_mode"] = "declared_mode"
    return result


def assert_matched_profiles(target, seed, method, *, device="cuda"):
    values = {mode: settings(target, seed, method, mode, device=device) for mode in MODES}
    reference = training_contract(values["none"])
    for mode, value in values.items():
        if (training_contract(value) != reference or value["optimizer_steps"] != 1500
                or list(validation_epochs(value)) != list(range(0, 1499, 30)) + [1499]
                or value["test_scoring"] is not False or value["support_seed"] != 64
                or (value["support_per_class"], value["validation_per_class"]) != (10, 5)):
            raise ValueError("Matched full training/calibration/selection contract differs: " + mode)
    return digest(reference)


def initialize(target, seed, method, mode, *, source_spec=None, device="cpu"):
    case_name(target, seed, method, mode)
    kwargs = {"model_seed": seed, "head_seed": seed + 1000, "source_spec": source_spec, "device": device}
    return (initialize_none_full_target(target, method, **kwargs) if mode == "none" else
            initialize_full_target(target, method, mode, **kwargs))


def runtime(case, items, *, directory=None):
    cls = NoneFullSourceTargetRuntime if case["mode"] == "none" else FullSourceTargetRuntime
    return cls(case["settings"], items, directory or case["out"], source_spec=case["source_spec"],
               initial_state_sha256=case["initial_state_sha256"])


def fresh_state(model):
    keys = {"backbone." + n for n in domain_keys(model.backbone)}
    keys.update(n for n in model.state_dict() if n.startswith(("heads.", "hgnn_heads.", "cnn_heads.")))
    return {n: value for n, value in model.state_dict().items() if n in keys}


def common_fresh_initials(target, seed):
    baseline, _ = initialize(target, seed, METHODS[0], "none")
    common = baseline.state_dict()
    fresh = fresh_state(baseline)
    records = {}
    for mode in MODES:
        model, _ = initialize(target, seed, METHODS[0], mode)
        state = model.state_dict()
        if (any(n not in state or not torch.equal(v, state[n]) for n, v in common.items())
                or set(fresh_state(model)) != set(fresh)
                or any(not torch.equal(v, fresh_state(model)[n]) for n, v in fresh.items())):
            raise ValueError("None/active common fresh initialization differs")
        records[mode] = state_id(model)
        del model
    return {"target": target, "seed": seed, "common_state_tensors": len(common),
            "common_state_sha256": state_id(baseline), "mode_initial_states": records,
            "fresh_state_sha256": digest({n: tensor_hash(v) for n, v in fresh.items()})}


def verify_fresh_target(model, proof):
    if digest({n: tensor_hash(v) for n, v in fresh_state(model).items()}) != proof["fresh_state_sha256"]:
        raise ValueError("Transferred target heads/DomainBN differ from paired fresh state")


def select_case(launch, code, target, seed, method, mode, *, phase):
    key = (target, seed, method, mode)
    pairs = grid()
    root = Path(launch["out_dir"]).resolve()
    keys = [(c["target"], c["seed"], c["method"], c["mode"]) for c in launch["cases"]]
    if (launch["study"] != load_study() or launch["study_file_sha256"] != file_hash(STUDY_PATH)
            or launch["code"] != code or code["dirty"] or key not in pairs or len(keys) != 54 or set(keys) != set(pairs)
            or launch["phase"] != phase or phase not in ("engineering_zero_update", "full_campaign")
            or launch["planned_full_optimizer_updates"] != 81000 or launch["query_labels_accessed"] is not False
            or launch["test_scoring"] is not False
            or len(launch["common_initials"]) != 9
            or {(p["target"], p["seed"]) for p in launch["common_initials"]} != {(n, s) for n in NAMES for s in SEEDS}
            or (phase == "engineering_zero_update" and (launch["optimizer_updates"] != 0 or launch["execution_enabled"] is not False))
            or (phase == "full_campaign" and (launch["optimizer_updates"] != 81000 or launch["execution_enabled"] is not True))):
        raise ValueError("Complete frozen54 matched full study and explicit execution phase required")
    contexts = {}
    for case in launch["cases"]:
        t, s, m, md = case["target"], case["seed"], case["method"], case["mode"]
        name = case_name(t, s, m, md)
        proof = next(p for p in launch["common_initials"] if (p["target"], p["seed"]) == (t, s))
        if (case["name"] != name or case["settings"] != settings(t, s, m, md)
                or Path(case["out"]).resolve() != root / "cases" / name
                or case["training_contract_sha256"] != assert_matched_profiles(t, s, m)
                or case["fresh_state_sha256"] != proof["fresh_state_sha256"]
                or case["initial_state_sha256"] != case["initialization"]["initial_state_sha256"]
                or (m == METHODS[0]) != (not case["source_spec"])
                or case["target_context"]["query_labels_accessed"] is not False):
            raise ValueError("Frozen full target role/settings/initialization/source differs")
        paired = (case["inputs"], case["target_context"], case["graph_binding"])
        if t in contexts and contexts[t] != paired:
            raise ValueError("Full comparisons use different target labels/observation/graph")
        contexts[t] = paired
        if m != METHODS[0]:
            binding = case["initialization"]["source"]
            if (binding["target"], binding["seed"], binding["mode"]) != (t, s, md):
                raise ValueError("Full source target/model-seed/mode binding differs")
    return launch["cases"][keys.index(key)]
