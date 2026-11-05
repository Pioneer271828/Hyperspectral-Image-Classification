"""Freeze all six engineering targets before any update; train/validation only."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

from joint_baseline import NAMES
from joint_training import run_training
from loadData.dataset_registry import file_hash
from model.research_config import digest
from none_full_target_runtime import NoneFullSourceTargetRuntime, settings
from none_full_transfer import SOURCE_FILES, initialize_none_full_target, verify_spec_files
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.train_joint_m4 import graph_binding
from target_data_context import load_target_context
from target_initialization import METHODS

PROFILE = "g4_none_full_target_short_launch_v2"


def source_spec(root, audit, target):
    folder = root / "cases" / (target + "-s64")
    spec = {"source_path": str(folder / "primary.pt"),
            "source_weight_sha256": file_hash(folder / "primary.pt"),
            "source_sidecar_sha256": file_hash(folder / "primary.pt.json")}
    for field, path in {"source_result": folder / "result.json", "source_launch": root / "launch.json",
        "source_acceptance": root / "training-acceptance.json", "source_independent_audit": audit,
        "source_protected_manifest": root / "protected-files.json"}.items():
        spec[field], spec[field + "_sha256"] = str(path), file_hash(path)
    return spec


def select_case(launch, code, target, method):
    pairs = {(n, m) for n in NAMES for m in METHODS[:2]}
    root = external_directory(launch["out_dir"])
    if (launch["protocol"] != PROFILE or launch["code"] != code or code["dirty"]
            or launch["optimizer_updates"] != 72 or launch["query_labels_accessed"] is not False
            or launch["test_scoring"] is not False or len(launch["cases"]) != 6
            or {(c["target"], c["method"]) for c in launch["cases"]} != pairs
            or (target, method) not in pairs):
        raise ValueError("All six distinct frozen short target roles required")
    for case in launch["cases"]:
        n, m = case["target"], case["method"]
        name = n + "-" + m
        if (case["name"] != name or case["settings"] != settings(n, m, device="cuda")
                or Path(case["out"]).resolve() != root / "cases" / name
                or not re.fullmatch("[0-9a-f]{64}", case["initial_state_sha256"])
                or set(case["graph_binding"]) != {n}
                or (m == METHODS[0]) != (not case["source_spec"])):
            raise ValueError("Frozen target settings, source role, initial state or graph differs")
        if case["source_spec"]:
            verify_spec_files(case["source_spec"])
    return next(c for c in launch["cases"] if (c["target"], c["method"]) == (target, method))


def load_case_data(case):
    inputs = case["inputs"]
    data, provenance, guard = load_target_context(case["target"], inputs["observation"], inputs["record"],
        inputs["record_sha256"], inputs["family"], inputs["family_sha256"], per_class=10, support_seed=64)
    if provenance != case["target_context"] or graph_binding(data) != case["graph_binding"]:
        raise ValueError("Frozen target labels/observation/graph differ")
    return data, provenance, guard


def prepare_launch(source, audit, observations, families, out, code):
    out = external_directory(out)
    out.mkdir(exist_ok=False)
    (out / "logs").mkdir()
    cases = []
    for target in NAMES:
        observation = Path(observations) / target
        family = Path(families) / (target + "-64") / "curator-family.json"
        inputs = {"observation": str(observation / "observation.pt"), "record": str(observation / "record.json"),
                  "record_sha256": file_hash(observation / "record.json"),
                  "family": str(family), "family_sha256": file_hash(family)}
        data, context, guard = load_target_context(target, inputs["observation"], inputs["record"],
            inputs["record_sha256"], inputs["family"], inputs["family_sha256"], per_class=10, support_seed=64)
        spec = source_spec(Path(source), Path(audit), target)
        for method in METHODS[:2]:
            used = {} if method == METHODS[0] else spec
            model, initial = initialize_none_full_target(target, method, model_seed=64, head_seed=1064,
                                                        source_spec=used, device="cpu")
            name = target + "-" + method
            cases.append({"name": name, "target": target, "method": method, "out": str(out / "cases" / name),
                "settings": settings(target, method, device="cuda"), "inputs": inputs,
                "source_spec": copy.deepcopy(used), "target_context": context, "graph_binding": graph_binding(data),
                "initial_state_sha256": initial["initial_state_sha256"], "initialization": initial})
            del model
        guard()
    launch = {"protocol": PROFILE, "code": code, "out_dir": str(out), "cases": cases,
              "optimizer_updates": 72, "query_labels_accessed": False, "test_scoring": False}
    select_case(launch, code, NAMES[0], METHODS[0])
    require_frozen_code(code)
    write_json(out / "launch.json", launch)
    return launch


def worker_main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "launch-plan", "launch-sha256", "target", "method"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected remote commit required")
    if file_hash(args.launch_plan) != args.launch_sha256:
        raise ValueError("Pinned full none target launch required")
    launch = json.loads(Path(args.launch_plan).read_text())
    case = select_case(launch, code, args.target, args.method)
    data, provenance, guard = load_case_data(case)
    provenance.update(source_spec=case["source_spec"], none_full_target_launch={"protocol": PROFILE,
        "launch_sha256": args.launch_sha256, "case": case["name"],
        "settings_sha256": digest(case["settings"]), "initial_state_sha256": case["initial_state_sha256"]})
    pins = provenance["protected_files"]
    pins[args.launch_plan] = args.launch_sha256
    spec = case["source_spec"]
    if spec:
        for field in SOURCE_FILES:
            pins[spec[field]] = spec["source_weight_sha256" if field == "source_path" else field + "_sha256"]
        pins[spec["source_path"] + ".json"] = spec["source_sidecar_sha256"]
    def finish():
        guard()
        require_frozen_code(code)
        if any(file_hash(p) != h for p, h in pins.items()):
            raise ValueError("Protected full none source/target/launch changed")
    finish()
    value = run_training(case["settings"], data, case["out"], provenance, code, device="cuda",
        resume=args.resume, stop_after=args.stop_after, finish_guard=finish,
        runtime_adapter=NoneFullSourceTargetRuntime(case["settings"], data, case["out"],
            source_spec=spec, initial_state_sha256=case["initial_state_sha256"]))
    print(json.dumps({k: value[k] for k in ("operation", "updates_this_call")}), flush=True)
