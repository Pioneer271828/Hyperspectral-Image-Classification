"""Register the fixed Issue 184 three-scene matrix and authenticate its evidence."""
from pathlib import Path
import time

from evaluation import frozen_probe_history as history, frozen_probe_registry as reg
from g4_matched_target_evaluation import new_output
from g4_support_context import check_files
from model.research_config import digest
from research_runtime import require_frozen_code, write_json


POLICY = "wp06_allscene_frozen_probe_campaign_v1"
CLASSES = {"paviaU": 9, "xuzhou": 9, "whu-honghu": 22}


def merge_pins(*sources):
    merged = {}
    for pins in sources:
        for path, sha in pins.items():
            reg.require(merged.setdefault(path, sha) == sha, f"Conflicting source fingerprint: {path}")
    return merged


def case_scope(scope, case):
    return {**scope, "probe_plan": scope["probe_plans"][case["target"]]}


def authenticate(freeze, freeze_sha, analysis, analysis_sha, index, index_sha, history_paths, missing, *, synthetic):
    started = time.monotonic()
    paths, historical = history.open_history(*history_paths, synthetic=synthetic)
    prior = historical["scope"]
    historical_target = "xuzhou" if synthetic else "paviaU"
    new_targets = CLASSES.keys() - {historical_target}
    reg.require(isinstance(missing, (list, tuple)) and len(missing) == len(set(missing))
                and set(missing) < new_targets, "Explicit missing new scene(s) must leave at least one complete new scene")
    present = {scene: count for scene, count in CLASSES.items() if scene in new_targets and scene not in missing}
    new_entries = reg.capture_index(index, index_sha, (18*len(present),))
    old_entries = reg.capture_index(prior["capture_index"], prior["capture_index_sha256"], (18,))
    reg.require((str(Path(freeze).resolve()), freeze_sha, str(Path(analysis).resolve()), analysis_sha)
        == (prior["freeze"], prior["freeze_sha256"], prior["analysis"], prior["analysis_sha256"]),
        "Historical and new cohorts must use the same frozen authority")
    frozen, launch, boundary, producer = reg.authenticated_boundary(freeze, freeze_sha, analysis, analysis_sha,
                                                                     old_entries, synthetic=synthetic)
    ordered = frozen["models"]
    roles = {row["name"] for target in CLASSES for row in reg.grid_rows(frozen, target)}
    reg.require(len(ordered) == 54 and {row["name"] for row in ordered} == roles, "Exact 54-case frozen roster required")
    old_cases, old_pins = reg.authenticate_captures(frozen, launch, boundary, old_entries, producer,
        {historical_target: CLASSES[historical_target]}, freeze_sha, analysis_sha, synthetic=synthetic)
    old_pins[str(Path(prior["capture_index"]).resolve(strict=True))] = prior["capture_index_sha256"]
    identity, reused, historical_pins = history.verify_history(paths, historical, old_cases, old_pins)
    new_cases, new_pins = reg.authenticate_captures(frozen, launch, boundary, new_entries, producer, present,
                                                    freeze_sha, analysis_sha, synthetic=synthetic)
    captured = {case["name"]: (case, bound, local) for case, bound, local in (*old_cases, *new_cases)}
    pins = merge_pins(new_pins, historical_pins, {str(Path(index).resolve(strict=True)): index_sha})
    cases = []
    for row in ordered:
        name, scene = row["name"], row["target"]
        if scene in missing:
            case = {k: row[k] for k in ("name", "target", "seed", "mode", "method", "case_sha256",
                                        "primary_sha256", "selected_step", "family_hash")}
            cases.append(({**case, "disposition": "missing_capture"}, None, {}))
            continue
        case, bound, local = captured[name]
        disposition = "historical" if name in reused else "new"
        cases.append(({**case, "disposition": disposition}, bound, local))
    return launch, cases, pins, identity, reused, time.monotonic()-started


def data(freeze, freeze_sha, analysis, analysis_sha, index, index_sha, code, cases, pins, identity,
         missing, *, synthetic, steps):
    reg.require(type(steps) is int and (4 < steps <= reg.PLAN["steps"] if synthetic else steps == reg.PLAN["steps"]),
                "Fixed all-scene probe budget required")
    paths = {"full": [steps], "split": [2, 4, steps], "completed": [steps]} if synthetic else reg.PATHS
    updates = 2*len(reg.LAYERS)*steps
    historical = sum(case["disposition"] == "historical" for case, _, _ in cases)
    new = sum(case["disposition"] == "new" for case, _, _ in cases)
    reg.require(identity["probe_updates"] == historical*updates, "Historical and new probe update plans differ")
    return {"policy": POLICY, "commit": code["commit"], "source_hash": code["source_hash"], "synthetic": synthetic,
            "freeze": str(Path(freeze).resolve(strict=True)), "freeze_sha256": freeze_sha,
            "analysis": str(Path(analysis).resolve(strict=True)), "analysis_sha256": analysis_sha,
            "capture_index": str(Path(index).resolve(strict=True)), "capture_index_sha256": index_sha,
            "history": identity, "cases": [case for case, _, _ in cases], "missing_capture_scenes": sorted(missing),
            "protected_file_count": len(pins), "protected_files_sha256": digest(pins),
            "scene_classes": CLASSES, "layers": list(reg.LAYERS), "paths": paths,
            "probe_plans": {scene: {**reg.PLAN, "classes": classes, "steps": steps} for scene, classes in CLASSES.items()},
            "columns": {stage: reg.probe_columns(64, 64, width) for stage, width in (("stage1", 128), ("stage2", 64))},
            "validation": "every_step_first_minimum_val_nll", "optimizer": "cpu_fp64_full_batch_sgd_zero_initialization",
            "expected_probe_updates": len(cases)*updates, "reused_probe_updates": historical*updates,
            "planned_new_probe_updates": new*updates, "planned_new_cases": new,
            "raw_gt_deserialized": False, "query_labels_accessed": False, "backbone_updates": 0, "bn_updates": 0}


def output(out, launch, scope, *other):
    owners = [Path(scope[key]).parent for key in ("freeze", "analysis", "capture_index")]
    owners.extend(Path(item["path"]).parent for item in scope["history"]["sources"].values())
    owners.extend(Path(case["capture_receipt"]).parent for case in scope["cases"] if "capture_receipt" in case)
    return new_output(out, launch, (*owners, *other))


def prepare(freeze, freeze_sha, analysis, analysis_sha, index, index_sha, history_paths, out, code,
            *, missing=(), synthetic=False, steps=500):
    require_frozen_code(code)
    launch, cases, pins, identity, _, elapsed = authenticate(freeze, freeze_sha, analysis, analysis_sha,
        index, index_sha, history_paths, missing, synthetic=synthetic)
    scope = data(freeze, freeze_sha, analysis, analysis_sha, index, index_sha, code,
                 cases, pins, identity, missing, synthetic=synthetic, steps=steps)
    out = output(out, launch, scope)
    check_files(pins)
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    write_json(out/"scope.json", {**scope, "input_authentication_seconds": elapsed})
    return out/"scope.json"


def load(scope_path, scope_sha, code):
    require_frozen_code(code)
    scope = reg.pinned_json(scope_path, scope_sha)
    reg.require((scope["policy"], scope["commit"], scope["source_hash"]) == (POLICY, code["commit"], code["source_hash"]),
                "Registered all-scene consumer source differs")
    freeze, analysis, index = (scope[k] for k in ("freeze", "analysis", "capture_index"))
    freeze_sha, analysis_sha, index_sha = (scope[k+"_sha256"] for k in ("freeze", "analysis", "capture_index"))
    history_paths = [scope["history"]["sources"][k]["path"] for k in ("scope", "execution", "review")]
    missing, synthetic = scope["missing_capture_scenes"], scope["synthetic"]
    launch, cases, pins, identity, reused, elapsed = authenticate(freeze, freeze_sha, analysis, analysis_sha,
        index, index_sha, history_paths, missing, synthetic=synthetic)
    steps = scope["probe_plans"]["paviaU"]["steps"] if synthetic else reg.PLAN["steps"]
    expected = data(freeze, freeze_sha, analysis, analysis_sha, index, index_sha, code,
                    cases, pins, identity, missing, synthetic=synthetic, steps=steps)
    reg.require({k: v for k, v in scope.items() if k != "input_authentication_seconds"} == expected,
                "Registered all-scene roster, identity, plan or budget changed")
    pins = merge_pins(pins, {str(Path(scope_path).resolve(strict=True)): scope_sha})
    return scope, launch, cases, pins, reused, elapsed
