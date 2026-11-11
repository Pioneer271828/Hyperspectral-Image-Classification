"""Register the complete PaviaU frozen-model probe submatrix."""
from itertools import product
from pathlib import Path
import time

import torch

from evaluation import authenticated_probe_pack as packs, frozen_representation_capture as capture
from evaluation.linear_probe import POLICY as LINEAR_POLICY, fingerprint
from g4_matched_target_evaluation import new_output
from g4_matched_target_study import METHODS, MODES, case_name
from g4_support_context import check_files, read
from model.research_config import digest
from research_runtime import hash_file, require_frozen_code, write_json


POLICY = "wp06_full_pavia_frozen_probe_campaign_v1"
TARGET = "paviaU"
SEEDS = (64, 512, 2048)
LAYERS = ("stage1_cnn", "stage1_graph", "stage2_cnn", "stage2_graph")
FREEZE_SHA = "7dd7e87334766d74005d14345e994fc2e700e109f6d8141b462004831496b48c"
ANALYSIS_SHA = "9b9d2e1390fa54915cc34bac87e6826cc5cb138cc746ae1c5012bead8fec441b"
PLAN = {"policy": LINEAR_POLICY, "classes": 9, "width": 64, "seed": 64, "steps": 500,
        "lr": .1, "l2": 1e-4, "scale_floor": 1e-12}
PATHS = {"full": [500], "split": [2, 250, 500], "completed": [500]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(path, sha):
    path = Path(path).resolve(strict=True)
    require(isinstance(sha, str) and len(sha) == 64 and hash_file(path) == sha, f"Pinned JSON differs: {path}")
    return read(path)


def probe_columns(width, seed, original):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    return list(range(original)) if width == original else torch.randperm(original, generator=rng)[:width].sort().values.tolist()


def grid_rows(frozen, target):
    rows = [row for row in frozen["models"] if row["target"] == target]
    require(len(rows) == 18 and len({row["name"] for row in rows}) == 18
            and {(r["seed"], r["method"], r["mode"]) for r in rows} == set(product(SEEDS, METHODS[:2], MODES))
            and all(r["name"] == case_name(target, r["seed"], r["method"], r["mode"]) for r in rows),
            "Exact ordered 18-case frozen role matrix required")
    return rows


def capture_plan(bound, frozen_row, synthetic):
    path, receipt, _, family, releases, overlay, selected, _ = bound
    require(selected == frozen_row and family["family_hash"] == selected["family_hash"]
            and receipt["case"] == selected["name"]
            and receipt["evidence"]["stage_calls"] == dict.fromkeys(capture.STAGES, 1),
            "Capture stage, authorized family or selected model role differs")
    for role, expected in (("train", 10), ("val", 5)):
        require(torch.bincount(overlay[role+"_labels"], minlength=9).tolist() == [expected]*9,
                "Exact K10/V5 released class support required")
    summaries = [read(path.parent / f"stage{stage}-summary.json") for stage in (1, 2)]
    if not synthetic:
        for summary in summaries:
            require(summary["maximum_regions"] == 1024 and summary["seed"] == 64
                    and summary["sampled_regions"] <= 1024
                    and all(summary[branch]["variance_floor"] == 1e-24 for branch in ("cnn", "graph")),
                    "Historical stage summary sampling contract differs")
    return {**{k: selected[k] for k in ("name", "target", "seed", "mode", "method", "case_sha256",
                                      "primary_sha256", "selected_step", "family_hash")},
            "capture_receipt": str(path), "capture_receipt_sha256": hash_file(path),
            "released_rows_sha256": fingerprint(overlay),
            "phase_hashes": {key: digest(value) for key, value in releases.items()}}


def authenticate_inputs(freeze, freeze_sha, analysis, analysis_sha, index, index_sha, *, synthetic=False):
    started = time.monotonic()
    target = "xuzhou" if synthetic else TARGET
    require(synthetic or (freeze_sha, analysis_sha) == (FREEZE_SHA, ANALYSIS_SHA),
            "Exact accepted Issue184 freeze and analysis required")
    index_data = pinned_json(index, index_sha)
    require(set(index_data) == {"captures"} and isinstance(index_data["captures"], list)
            and len(index_data["captures"]) == 18, "Exact capture index required")
    entries = {}
    for row in index_data["captures"]:
        require(set(row) == {"case", "receipt", "sha256", "origin"} and row["origin"] in ("captured", "reused")
                and row["case"] not in entries, "Duplicate or unregistered capture index row")
        entries[row["case"]] = row
    first = next(iter(entries.values()))
    _, first_receipt = packs.read_capture_receipt(first["receipt"], first["sha256"])
    frozen, _, launch, boundary = capture._authenticate_boundary(freeze, freeze_sha, analysis, analysis_sha,
                                                                  first_receipt["code"])
    expected = grid_rows(frozen, target)
    require(set(entries) == {row["name"] for row in expected}, "Capture index does not cover the entire frozen matrix")
    pins = {**boundary, str(Path(index).resolve(strict=True)): index_sha}
    cases, seen_paths, common = [], set(), None
    for frozen_row in expected:
        name = frozen_row["name"]
        item = entries[name]
        path, receipt = packs.read_capture_receipt(item["receipt"], item["sha256"])
        require(path.name == "receipt.json" and path.parent not in seen_paths and receipt["case"] == name
                and (receipt["freeze_sha256"], receipt["analysis_sha256"]) == (freeze_sha, analysis_sha)
                and receipt["code"]["files"] == first_receipt["code"]["files"],
                "Capture is reused across roles or from a different frozen producer/boundary")
        seen_paths.add(path.parent)
        case, record, _, _, local = capture._authenticate_case(frozen, launch, boundary, name)
        require(all(boundary.get(p) == sha for p, sha in local.items()), "Case files are outside full authentication")
        bound = packs.bind_capture(path, item["sha256"], receipt, frozen, case, record, boundary, local_pins=local)
        plan = capture_plan(bound, frozen_row, synthetic)
        identity = (plan["released_rows_sha256"], plan["phase_hashes"], receipt["evidence"]["observation"])
        if common is None:
            common = identity
        require(common == identity, "PaviaU roles do not use one ordered label/observation release")
        local = {**local, **packs.capture_files(bound)}
        family = str(Path(case["inputs"]["family"]).resolve(strict=True))
        local[family] = boundary[family]
        for p, sha in local.items():
            require(p not in pins or pins[p] == sha, "Conflicting batch input pin")
            pins[p] = sha
        cases.append(({**plan, "origin": item["origin"]}, bound, local))
    return launch, cases, pins, time.monotonic()-started


def registry_data(*, freeze, freeze_sha, analysis, analysis_sha, index, index_sha, code, plan, paths,
                  synthetic, cases, pins):
    if synthetic:
        require(plan["classes"] == 9 and paths["full"] == [plan["steps"]]
                and paths["split"] == [2, 4, plan["steps"]] and plan["steps"] > 4
                and paths["completed"] == [plan["steps"]], "Bounded synthetic probe path differs")
    else:
        require(plan == PLAN and paths == PATHS, "Real probe plan and paired budget are fixed")
    return {"policy": POLICY, "commit": code["commit"], "source_hash": code["source_hash"], "synthetic": synthetic,
            "freeze": str(Path(freeze).resolve()), "freeze_sha256": freeze_sha,
            "analysis": str(Path(analysis).resolve()), "analysis_sha256": analysis_sha,
            "capture_index": str(Path(index).resolve()), "capture_index_sha256": index_sha,
            "protected_file_count": len(pins), "protected_files_sha256": digest(pins),
            "cases": [case for case, _, _ in cases], "layers": list(LAYERS), "probe_plan": plan,
            "columns": {stage: probe_columns(plan["width"], plan["seed"], width)
                        for stage, width in (("stage1", 128), ("stage2", 64))},
            "paths": paths, "validation": "every_step_first_minimum_val_nll",
            "planned_probe_updates": len(cases)*len(LAYERS)*plan["steps"]*2,
            "optimizer": "cpu_fp64_full_batch_sgd_zero_initialization",
            "raw_gt_deserialized": False, "query_labels_accessed": False, "backbone_updates": 0}


def prepare(freeze, freeze_sha, analysis, analysis_sha, index, index_sha, out, code, *,
            synthetic=False, plan=None, paths=None):
    require_frozen_code(code)
    plan, paths = (PLAN if plan is None else plan), (PATHS if paths is None else paths)
    launch, cases, pins, elapsed = authenticate_inputs(freeze, freeze_sha, analysis, analysis_sha, index, index_sha,
                                                         synthetic=synthetic)
    extras = (Path(freeze).resolve().parent, Path(analysis).resolve().parent, Path(index).resolve().parent,
              *(Path(case["capture_receipt"]).parent for case, _, _ in cases))
    out = new_output(out, launch, extras)
    value = registry_data(freeze=freeze, freeze_sha=freeze_sha, analysis=analysis, analysis_sha=analysis_sha,
                          index=index, index_sha=index_sha, code=code, plan=plan, paths=paths,
                          synthetic=synthetic, cases=cases, pins=pins)
    check_files(pins)
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    write_json(out / "scope.json", {**value, "input_authentication_seconds": elapsed})
    return out / "scope.json"


def load(scope_path, scope_sha, code):
    require_frozen_code(code)
    scope = pinned_json(scope_path, scope_sha)
    require(scope["policy"] == POLICY and scope["commit"] == code["commit"]
            and scope["source_hash"] == code["source_hash"], "Registered consumer source differs")
    launch, cases, pins, elapsed = authenticate_inputs(scope["freeze"], scope["freeze_sha256"], scope["analysis"],
        scope["analysis_sha256"], scope["capture_index"], scope["capture_index_sha256"], synthetic=scope["synthetic"])
    expected = registry_data(freeze=scope["freeze"], freeze_sha=scope["freeze_sha256"], analysis=scope["analysis"],
        analysis_sha=scope["analysis_sha256"], index=scope["capture_index"], index_sha=scope["capture_index_sha256"],
        code=code, plan=scope["probe_plan"], paths=scope["paths"], synthetic=scope["synthetic"], cases=cases, pins=pins)
    require({k: v for k, v in scope.items() if k != "input_authentication_seconds"} == expected,
            "Registered matrix, capture identity, plan or paths changed")
    pins[str(Path(scope_path).resolve(strict=True))] = scope_sha
    return scope, launch, cases, pins, elapsed
