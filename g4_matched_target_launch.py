"""Accepted short gates and full54 zero-update identity/paired-context preflight."""
import argparse
import json
from pathlib import Path
import sys
import time

from g4_matched_target_study import (EXPECTED, METHODS, MODES, SEEDS, STUDY_PATH, assert_matched_profiles,
    case_name, common_fresh_initials, grid, initialize, load_study, select_case, settings, verify_fresh_target)
from joint_baseline import NAMES
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.train_joint_m4 import graph_binding
from target_data_context import load_target_context


def files(root):
    return {str(p): file_hash(p) for p in Path(root).rglob("*") if p.is_file()}


def accepted_short(root, mode_group, consumer):
    root = Path(root).resolve(strict=True)
    if mode_group not in ("none", "active"):
        raise ValueError("Explicit accepted short cohort required")
    sha = EXPECTED[mode_group + "_short_receipt_sha256"]
    if file_hash(root / "receipt.json") != sha:
        raise ValueError("Pinned accepted real short receipt differs")
    receipt, launch = [json.loads((root / name).read_text()) for name in ("receipt.json", "launch.json")]
    code = receipt["code"]
    count, updates = (6, 72) if mode_group == "none" else (12, 144)
    if (receipt["status"] != "passed" or receipt["updates"] != updates or len(receipt["cases"]) != count
            or len(launch["cases"]) != count or launch["code"] != code or code["dirty"] or consumer["dirty"]
            or code["source_hash"] != digest(code["files"]) or consumer["source_hash"] != digest(consumer["files"])
            or any(consumer["files"].get(p) != h for p, h in code["files"].items())
            or receipt["query_labels_accessed"] is not False or receipt["test_scoring"] is not False):
        raise ValueError("Accepted short cohort/implementation/label permissions differ")
    if "launch_sha256" in receipt and receipt["launch_sha256"] != file_hash(root / "launch.json"):
        raise ValueError("Accepted short launch changed")
    names = {c["name"] for c in launch["cases"]}
    if len(names) != count or {c["name"] for c in receipt["cases"]} != names:
        raise ValueError("Accepted real short role coverage differs")
    metrics = recalls = repeats = 0
    for row in receipt["cases"]:
        case = next(c for c in launch["cases"] if c["name"] == row["name"])
        folder = root / "cases" / case["name"]
        result = json.loads((folder / "result.json").read_text())
        if (row["updates"] != 12 or result["status"] != "completed" or result["optimizer_steps"] != 12
                or result["start_code"] != code or result["settings"] != case["settings"]):
            raise ValueError("Accepted short execution/result identity differs")
        required = {"identity.json", "progress.json", "primary.pt", "primary.pt.json", "last.pt", "last.pt.json",
                    "selected-probabilities.pt", "trace.json", "history.json"}
        if not required <= set(result["artifacts"]):
            raise ValueError("Accepted short canonical artifacts missing")
        for name, h in result["artifacts"].items():
            if Path(name).name != name or file_hash(folder / name) != h:
                raise ValueError("Accepted short artifact changed")
        if "result_sha256" in row and row["result_sha256"] != file_hash(folder / "result.json"):
            raise ValueError("Accepted short result hash changed")
        score = row.get("independent", row)
        metrics += score["head_metrics"]
        recalls += score["class_recalls"]
        repeats += row.get("exact_reload_repeats", row.get("strict_reload_repeats", 0))
    if (metrics, recalls, repeats) != (count * 6, count * 80, count * 3):
        raise ValueError("Accepted short independent metric/recall/reload coverage differs")
    return {"root": str(root), "receipt_sha256": sha, "producer_commit": code["commit"],
            "launch_sha256": file_hash(root / "launch.json"), "cases": count, "updates": updates}


def source_spec(root, audit, target, seed, mode):
    folder = root / "cases" / (target + "-s" + str(seed) + ("" if mode == "none" else "-" + mode))
    spec = {"source_path": str(folder / "primary.pt"), "source_weight_sha256": file_hash(folder / "primary.pt"),
            "source_sidecar_sha256": file_hash(folder / "primary.pt.json")}
    paths = {"source_result": folder / "result.json", "source_launch": root / "launch.json",
             "source_acceptance": root / "training-acceptance.json", "source_independent_audit": audit}
    if mode == "none":
        paths["source_protected_manifest"] = root / "protected-files.json"
    for field, path in paths.items():
        spec[field], spec[field + "_sha256"] = str(path), file_hash(path)
    return spec


def prepare_preflight(*, none_source, active_source, none_audit, active_audit, none_short, active_short,
                      observation_root, family_root, out, code):
    paths = {k: Path(v).resolve(strict=True) for k, v in {"none_source": none_source, "active_source": active_source,
        "none_audit": none_audit, "active_audit": active_audit, "none_short": none_short, "active_short": active_short}.items()}
    out = external_directory(out)
    roots = [paths[n] for n in ("none_source", "active_source", "none_short", "active_short")]
    roots += [paths[n].parent for n in ("none_audit", "active_audit")]
    if any(out == p or p in out.parents or out in p.parents for p in roots):
        raise ValueError("Independent full preflight output required")
    for group in ("none", "active"):
        if file_hash(paths[group + "_audit"]) != EXPECTED[group + "_source_audit_sha256"]:
            raise ValueError("Accepted full source audit changed")
    accepted = {g: accepted_short(paths[g + "_short"], g, code) for g in ("none", "active")}
    protected = {p: h for root in roots for p, h in files(root).items()}
    # This manifest is part of the accepted none short cohort and pins the old
    # source/target trees as well as actual input files. It is not a label reader.
    protected.update(json.loads((paths["none_short"] / "protected-files.json").read_text()))
    if any(file_hash(p) != h for p, h in protected.items()):
        raise ValueError("Accepted source/short/historical/input changed before full preflight")
    out.mkdir(exist_ok=False)
    started, cases, proofs = time.monotonic(), [], []
    for target in NAMES:
        observation = Path(observation_root) / target
        family = Path(family_root) / (target + "-64") / "curator-family.json"
        inputs = {"observation": str(observation / "observation.pt"), "record": str(observation / "record.json"),
                  "record_sha256": file_hash(observation / "record.json"),
                  "family": str(family), "family_sha256": file_hash(family)}
        data, context, guard = load_target_context(target, inputs["observation"], inputs["record"],
            inputs["record_sha256"], inputs["family"], inputs["family_sha256"], per_class=10, support_seed=64)
        protected.update(context["protected_files"])
        for seed in SEEDS:
            proof = common_fresh_initials(target, seed)
            proofs.append(proof)
            for method in METHODS[:2]:
                contract = assert_matched_profiles(target, seed, method)
                for mode in MODES:
                    group = "none" if mode == "none" else "active"
                    spec = {} if method == METHODS[0] else source_spec(paths[group + "_source"], paths[group + "_audit"], target, seed, mode)
                    model, initial = initialize(target, seed, method, mode, source_spec=spec)
                    verify_fresh_target(model, proof)
                    name = case_name(target, seed, method, mode)
                    cases.append({"name": name, "target": target, "seed": seed, "method": method, "mode": mode,
                        "out": str(out / "cases" / name), "settings": settings(target, seed, method, mode),
                        "source_spec": spec, "initialization": initial, "initial_state_sha256": initial["initial_state_sha256"],
                        "fresh_state_sha256": proof["fresh_state_sha256"], "inputs": inputs, "target_context": context,
                        "graph_binding": graph_binding(data), "training_contract_sha256": contract})
                    del model
            write_json(out / "progress.json", {"frozen_cases": len(cases), "optimizer_updates": 0})
        guard()
        del data
    launch = {"study": load_study(), "study_file_sha256": file_hash(STUDY_PATH), "code": code,
        "out_dir": str(out), "phase": "engineering_zero_update", "execution_enabled": False,
        "optimizer_updates": 0, "planned_full_optimizer_updates": 81000, "short_acceptances": accepted,
        "common_initials": proofs, "cases": cases, "query_labels_accessed": False, "test_scoring": False}
    select_case(launch, code, *grid()[0], phase="engineering_zero_update")
    require_frozen_code(code)
    if any(file_hash(p) != h for p, h in protected.items()):
        raise ValueError("Full preflight mutated protected source/short/input")
    write_json(out / "launch.json", launch)
    write_json(out / "protected-files.json", protected)
    write_json(out / "receipt.json", {"status": "engineering_zero_update_passed", "code": code,
        "launch_sha256": file_hash(out / "launch.json"), "study_file_sha256": file_hash(STUDY_PATH),
        "cases": 54, "common_initial_proofs": 9, "optimizer_updates": 0, "execution_enabled": False,
        "planned_full_optimizer_updates": 81000, "full_training_completed": False, "query_labels_accessed": False,
        "test_scoring": False, "protected_files_and_inputs": len(protected), "elapsed_seconds": time.monotonic() - started})
    return launch


def preflight_main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "none-source", "active-source", "none-audit", "active-audit", "none-short",
                 "active-short", "observation-root", "family-root", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected remote commit required")
    values = vars(args)
    values.pop("expected_commit")
    prepare_preflight(**values, code=code)
