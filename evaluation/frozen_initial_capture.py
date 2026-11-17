"""Capture the exact pre-adaptation state under the original frozen producer."""
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from evaluation import frozen_representation_capture as original
from g4_matched_target_evaluation import new_output
from g4_matched_target_study import METHODS, NAMES, SEEDS, common_fresh_initials, grid, initialize, verify_fresh_target
from g4_support_context import check_files
from joint_baseline import _item_fingerprint
from loadData.dataset_registry import DATASETS
from model.research_config import digest
from research_runtime import ROOT, hash_file, require_frozen_code, write_json
from target_bn_calibration import OBS_KEYS
from target_initialization import state_id


BASE = "f090ca302192a065de852c43e07e4f3beaeea6b4"
BASE_SOURCE = "996bd5580863976662010288d2b67cc05569781c6d3a422f5e3bc3cb0c61f324"
FREEZE_SHA = "7dd7e87334766d74005d14345e994fc2e700e109f6d8141b462004831496b48c"
ANALYSIS_SHA = "9b9d2e1390fa54915cc34bac87e6826cc5cb138cc746ae1c5012bead8fec441b"
POLICY = "e0_frozen_full54_pre_adaptation_capture_v1"
ADDITIONS = {"evaluation/frozen_initial_capture.py", "scripts/export_frozen_initial_capture.py",
             "tests/test_frozen_initial_capture.py", "research/reports/WP06_历史初态采集桥接_20261111.md"}
OPTIONS = {"maximum_regions": 1024, "seed": 64, "variance_floor": 1e-24}


def require(valid, reason):
    if not valid:
        raise ValueError(reason)


def verify_source(code):
    require_frozen_code(code)
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, code["commit"]], cwd=ROOT, check=True)
    paths = subprocess.check_output(["git", "-c", "core.quotepath=false", "diff", "--no-renames", "--name-status",
                                     BASE, code["commit"]], cwd=ROOT, text=True, encoding="utf-8").splitlines()
    require(set(paths) <= {"A\t"+name for name in ADDITIONS},
            "Historical source must only add the dedicated capture bridge")
    files = {path: sha for path, sha in code["files"].items() if path not in ADDITIONS}
    require(digest(files) == BASE_SOURCE, "Original f090 producer files differ")


def common_proofs(launch):
    roster = [(c["target"], c["seed"], c["method"], c["mode"]) for c in launch["cases"]]
    rows = launch["common_initials"]
    proofs = {(row["target"], row["seed"]): row for row in rows}
    require(len(roster) == 54 and set(roster) == set(grid())
        and len(rows) == len(proofs) == 9 and set(proofs) == {(t, s) for t in NAMES for s in SEEDS}
        and launch["phase"] == "full_campaign" and launch["execution_enabled"] is True
        and launch["optimizer_updates"] == launch["planned_full_optimizer_updates"] == 81000
        and launch["test_scoring"] is launch["query_labels_accessed"] is False,
        "Complete original54 and nine common fresh states required")
    for (target, seed), proof in proofs.items():
        with torch.random.fork_rng(devices=[]):
            require(common_fresh_initials(target, seed) == proof, "Rebuilt common fresh model identity differs")
    return proofs


def source_pins(case, pins):
    spec = case["source_spec"]
    if case["method"] == METHODS[0]:
        require(not spec and case["initialization"]["source"] is None, "Scratch source must be absent")
        return {}
    fields = {"source_result", "source_launch", "source_acceptance", "source_independent_audit"}
    if case["mode"] == "none":
        fields.add("source_protected_manifest")
    expected = {k+"_sha256" for k in fields} | fields | {"source_path", "source_weight_sha256", "source_sidecar_sha256"}
    require(set(spec) == expected, "Exact historical source fields required")
    sources = {spec[k]: spec[k+"_sha256"] for k in fields}
    sources.update({spec["source_path"]: spec["source_weight_sha256"],
                    spec["source_path"]+".json": spec["source_sidecar_sha256"]})
    source = case["initialization"]["source"]
    require((source["target"], source["seed"], source["mode"]) == (case["target"], case["seed"], case["mode"])
            and all(pins.get(p) == sha for p, sha in sources.items()),
            "Source role, artifact or source audit differs from the frozen boundary")
    return sources


def identity(case, proof):
    init = case["initialization"]
    require(case["target_context"]["query_labels_accessed"] is False
        and init["head_seed"] == case["seed"]+1000 and init["model_seed"] == case["seed"]
        and init["initial_device"] == "cpu" and init["optimizer_updates"] == 0
        and init["source_heads_copied"] is init["source_domain_state_copied"] is False
        and init["model_test_scoring"] is False and case["initial_state_sha256"] == init["initial_state_sha256"]
        and case["fresh_state_sha256"] == proof["fresh_state_sha256"],
        "Original step-zero state, seed, fresh domain or label permissions differ")
    if case["method"] == METHODS[0]:
        require(case["initial_state_sha256"] == proof["mode_initial_states"][case["mode"]],
                "Scratch identity differs from the common mode proof")


def rebuild(case, proof):
    identity(case, proof)
    with torch.random.fork_rng(devices=[]):
        model, evidence = initialize(case["target"], case["seed"], case["method"], case["mode"],
                                     source_spec=case["source_spec"], device="cpu")
        require(evidence == case["initialization"] and state_id(model) == case["initial_state_sha256"],
                "Historical initializer did not recreate the exact initial state and receipt")
        verify_fresh_target(model, proof)
    return model


def capture_case(case, proof, record, observation_path):
    py_rng, np_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            model = rebuild(case, proof).eval()
            item = torch.load(observation_path, map_location="cpu", weights_only=True)
            require(set(item) == original.ARCHIVE_KEYS and _item_fingerprint(item) == record["prepared_fields"]
                and item["name"] == case["target"] and item["class_num"] == DATASETS[case["target"]][2],
                "Exact label-free observation archive required")
            observation = {key: item[key] for key in OBS_KEYS}
            rng, np_before, py_before = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
            stages, _, evidence = original.capture(model, case["target"], observation)
            np_after = np.random.get_state()
            require(state_id(model) == evidence["state_sha256"] == case["initial_state_sha256"]
                and evidence["registered_state_flags_rng_inputs_unchanged"] is True
                and torch.equal(rng, torch.get_rng_state()) and py_before == random.getstate()
                and np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1])
                and np_before[2:] == np_after[2:], "Step-zero capture changed model or RNG")
            return stages, evidence
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)


def export(freeze, analysis, out, code, *, initialization_only=False):
    verify_source(code)
    frozen, _, launch, pins = original._authenticate_boundary(freeze, FREEZE_SHA, analysis, ANALYSIS_SHA, code)
    proofs = common_proofs(launch)
    out = new_output(out, launch, (Path(freeze).resolve().parent, Path(analysis).resolve().parent))
    require(not out.exists(), "Fresh external capture directory required")
    plans = []
    for row in frozen["models"]:
        name = row["name"]
        case, record, obs, _, local = original._authenticate_case(frozen, launch, pins, name)
        proof = proofs[(case["target"], case["seed"])]
        sources = source_pins(case, pins)
        identity(case, proof)
        require(all(pins.get(p) == sha for p, sha in local.items()), "Observation and model pins differ")
        plans.append((row, case, record, obs, {**local, **sources}, proof))
    out.mkdir(exist_ok=False)
    cases = [{"case": row["name"], "status": "not_run"} for row, *_ in plans]
    record = {"policy": POLICY, "status": "running", "producer": code, "base_commit": BASE, "base_source_hash": BASE_SOURCE,
              "freeze_sha256": FREEZE_SHA, "analysis_sha256": ANALYSIS_SHA, "boundary_sha256": digest(pins),
              "cases": cases, "protected_files": pins, "options": OPTIONS, "optimizer_updates": 0,
              "bn_updates": 0, "raw_labels_deserialized": False, "new_query_release": False,
              "initialization_only": initialization_only}
    progress = out/"receipt.json"
    write_json(progress, record)
    started = time.monotonic()
    for status, (row, case, observed, obs, local, proof) in zip(cases, plans, strict=True):
        tick = time.monotonic()
        try:
            check_files(local)
            if initialization_only:
                rebuild(case, proof)
                status.update(status="verified_initialization", model_forwards=0)
            else:
                stages, evidence = capture_case(case, proof, observed, obs)
                summaries = original._summaries(stages, **OPTIONS)
                folder = out/row["name"]
                folder.mkdir(exist_ok=False)
                for stage, value in stages.items():
                    torch.save(value, folder/(stage+"-features.pt"))
                    write_json(folder/(stage+"-summary.json"), summaries[stage])
                del stages
                local_receipt = {k: record[k] for k in ("policy", "freeze_sha256", "analysis_sha256",
                    "boundary_sha256", "options", "optimizer_updates", "bn_updates", "raw_labels_deserialized", "new_query_release")}
                local_receipt.update(status="passed", case=row["name"], stage_role="pre_adaptation", model_forwards=1,
                    producer_commit=code["commit"], producer_source_hash=code["source_hash"],
                    case_sha256=row["case_sha256"], initial_state_sha256=case["initial_state_sha256"], selected_step=0,
                    initialization=case["initialization"], source_spec=case["source_spec"], common_fresh=proof,
                    graph_binding=case["graph_binding"], evidence=evidence,
                    outputs={p.name: hash_file(p) for p in folder.iterdir()})
                check_files(local)
                path = folder/"receipt.json"
                write_json(path, local_receipt)
                status.update(status="captured", receipt=str(path), sha256=hash_file(path), model_forwards=1)
            check_files(local)
        except Exception as error:
            status.update(status="failed", error=repr(error))
        status["wall_seconds"] = time.monotonic()-tick
        write_json(progress, record)
        print(status["case"], status["status"], flush=True)
    check_files(pins)
    verify_source(code)
    record.update(status="passed" if all(row["status"] in ("captured", "verified_initialization") for row in cases)
                  else "incomplete_or_failed", model_forwards=sum(row.get("model_forwards", 0) for row in cases),
                  protected_files_unchanged=True, wall_seconds=time.monotonic()-started)
    write_json(progress, record)
    return progress, record["status"]
