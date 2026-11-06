"""Observe original frozen stages; full54/analysis gates precede archive loads."""
import json
import math
from pathlib import Path
import time

import torch

from evaluation.representation_redundancy import aligned_summary
from g4_matched_target_analysis import POLICY as ANALYSIS_POLICY, PLAN_SHA256, PRODUCER, PRODUCER_SOURCE
from g4_matched_target_evaluation import new_output, require_frozen_boundary
from g4_matched_target_execution import unchanged_producer
from g4_matched_target_study import case_name, grid
from g4_support_context import check_files, read
from joint_baseline import _item_fingerprint, forward
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import OrderedValidation
from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCNBackbone
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import require_frozen_code, write_json
from target_bn_calibration import GRAPH_POLICY, OBS_KEYS, observation_id
from target_initialization import state_id

ANALYSIS_COMMIT = "bdb4b3f14cc108e632b986ad7bffec7cd1a70bfc"
ANALYSIS_SOURCE = "790a5e39400419ec9d51e06f56875a62520b348d4604c4a34f3902c2286e98ee"
POLICY = "e0_frozen_full54_stage_capture_v1"
BATCH_POLICY = "e0_frozen_full54_stage_summary_v1"
ARCHIVE_KEYS = OBS_KEYS | {"name", "height", "width", "bands", "class_num"}
STAGES = {"cnn_stage1": ("stage1", "cnn", 128), "graph_stage1": ("stage1", "graph", 128),
          "cnn_stage2": ("stage2", "cnn", 64), "graph_stage2": ("stage2", "graph", 64)}


def _devices(model):
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1:
        raise ValueError("Isolated single-device frozen model required")
    device = next(iter(devices))
    return [device.index] if device.type == "cuda" else []


@torch.no_grad()
def capture(model, target, observation):
    if (not isinstance(model, MultiDatasetVCASAHGCN) or tuple(model.heads) != (target,)
            or any(m.training for m in model.modules()) or any(p.grad is not None for p in model.parameters())
            or model.backbone.graph_mode != "static" or model.backbone.norm_type != "domain_bn"):
        raise ValueError("Isolated fully eval static single-target model without pending gradients required")
    identity = observation_id(observation)
    backbone = model.backbone
    originals = {}
    for name in STAGES:
        method = getattr(backbone, name)
        if name in backbone.__dict__ or getattr(method, "__func__", None) is not getattr(VCASAHGCNBackbone, name):
            raise ValueError("Original unmodified stage methods required")
        originals[name] = method
    before = state_id(model); flags = [m.training for m in model.modules()]
    cpu_rng = torch.get_rng_state().clone(); devices = _devices(model)
    cuda_rng = {d: torch.cuda.get_rng_state(d).clone() for d in devices}
    captured = {"stage1": {}, "stage2": {}}; calls = {}
    def observe(name, original):
        stage, branch, width = STAGES[name]
        def invoke(*args, **kwargs):
            if calls.get(name, 0):
                raise ValueError("Each original stage must execute exactly once")
            value = original(*args, **kwargs); calls[name] = 1
            count = observation["Q"].shape[0 if branch == "cnn" else 1]
            if (not torch.is_tensor(value) or value.shape != (count, width) or not torch.isfinite(value).all()):
                raise ValueError("Actual original stage shape/finite contract differs")
            captured[stage][branch] = value.detach().cpu().clone()
            return value
        return invoke
    try:
        for name, original in originals.items(): setattr(backbone, name, observe(name, original))
        with torch.random.fork_rng(devices=devices), OrderedValidation():
            predictions = tuple(p.detach().cpu().clone() for p in forward(model, target, observation))
    finally:
        for name in originals:
            if name in backbone.__dict__: delattr(backbone, name)
        if (state_id(model) != before or [m.training for m in model.modules()] != flags
                or observation_id(observation) != identity or not torch.equal(cpu_rng, torch.get_rng_state())
                or any(not torch.equal(rng, torch.cuda.get_rng_state(d)) for d, rng in cuda_rng.items())):
            raise ValueError("Read-only stage capture changed registered state/flags/input/RNG")
    if set(calls) != set(STAGES):
        raise ValueError("Incomplete original stage observation")
    for value in captured.values(): value["Q"] = observation["Q"].detach().cpu().clone()
    return captured, predictions, {"state_sha256": before, "observation": identity,
        "stage_calls": calls, "registered_state_flags_rng_inputs_unchanged": True,
        "scope": "full_target_scene_transductive", "ephemeral_non_tensor_cache_immutability_claim": False}


def _authenticate_boundary(freeze_path, freeze_sha256, analysis_path, analysis_sha256, code):
    paths = [Path(p).resolve(strict=True) for p in (freeze_path, analysis_path)]
    pins = {str(p): sha for p, sha in zip(paths, (freeze_sha256, analysis_sha256))}
    check_files(pins); frozen, analysis = map(read, paths)
    # The analysis pins below cover the full freeze manifest, including its live boundary.
    require_frozen_boundary(frozen, full_integrity=False)
    producer = frozen["producer"]
    if (producer["commit"] != PRODUCER or producer["source_hash"] != PRODUCER_SOURCE
            or frozen["code"]["commit"] != PRODUCER or frozen["code"]["source_hash"] != PRODUCER_SOURCE
            or analysis["status"] != "passed" or analysis["policy"] != ANALYSIS_POLICY
            or analysis["code"]["commit"] != ANALYSIS_COMMIT or analysis["code"]["source_hash"] != ANALYSIS_SOURCE
            or analysis["plan_sha256"] != PLAN_SHA256
            or analysis["freeze_sha256"] != freeze_sha256
            or (analysis["models_analyzed"], analysis["heads_analyzed"], analysis["groups"]) != (54, 162, 18)
            or analysis["new_optimizer_updates"] != 0 or analysis["raw_labels_deserialized"] is not False
            or analysis["model_inference_executed"] is not False or analysis["new_query_release"] is not False
            or analysis["input_query_metrics_consumed"] is not True):
        raise ValueError("Completed original331 full54 and accepted resource-aware analysis required before feature/model load")
    unchanged_producer(producer, code); unchanged_producer(producer, frozen["code"])
    unchanged_producer(analysis["code"], code)
    if (any(analysis["protected_files"].get(p) != sha for p, sha in frozen["protected_files"].items())
            or set(analysis["outputs"]) != {"analysis.json", "report.md"}):
        raise ValueError("Analysis inputs/outputs do not cover accepted frozen boundary")
    for p, sha in analysis["protected_files"].items():
        if p in pins and pins[p] != sha: raise ValueError("Protected identity collision")
        pins[p] = sha
    for filename, sha in analysis["outputs"].items():
        p = str(paths[1].parent / filename)
        if p in pins and pins[p] != sha: raise ValueError("Protected output identity collision")
        pins[p] = sha
    check_files(pins)
    root = Path(frozen["training_root"]).resolve(strict=True); launch_path = root / "launch.json"
    if pins.get(str(launch_path)) != frozen["launch_sha256"]:
        raise ValueError("Pinned original launch required")
    launch = read(launch_path)
    if launch["code"] != producer or len(launch["cases"]) != 54:
        raise ValueError("Original complete launch identity differs")
    return frozen, analysis, launch, pins


def _authenticate_case(frozen, launch, pins, name):
    matches = [c for c in launch["cases"] if c["name"] == name]
    model_rows = [m for m in frozen["models"] if m["name"] == name]
    if len(matches) != 1 or len(model_rows) != 1 or digest(matches[0]) != model_rows[0]["case_sha256"]:
        raise ValueError("Case must match exactly one frozen launch identity")
    case = matches[0]; record_path = Path(case["inputs"]["record"]).resolve(strict=True)
    observation_path = Path(case["inputs"]["observation"]).resolve(strict=True)
    if pins.get(str(record_path)) != case["inputs"]["record_sha256"]:
        raise ValueError("Pinned label-free observation record required")
    record = read(record_path)
    if (record["policy"] != GRAPH_POLICY or record["graph_fit_labels"] != 0 or record["target"] != case["target"]
            or set(record["prepared_fields"]) != ARCHIVE_KEYS
            or pins.get(str(observation_path)) != record["observation_sha256"]):
        raise ValueError("Accepted label-free archive identity required before tensor load")
    primary = Path(frozen["training_root"]) / "cases" / name / "primary.pt"
    if Path(case["out"]).resolve() != primary.parent.resolve():
        raise ValueError("Case output differs from original frozen model directory")
    case_pins = {str(record_path): pins[str(record_path)], str(observation_path): pins[str(observation_path)],
                 str(primary): model_rows[0]["primary_sha256"], str(primary) + ".json": model_rows[0]["sidecar_sha256"]}
    return case, record, observation_path, primary, case_pins


def authenticate(freeze_path, freeze_sha256, analysis_path, analysis_sha256, name, code):
    frozen, analysis, launch, pins = _authenticate_boundary(freeze_path, freeze_sha256, analysis_path, analysis_sha256, code)
    case, record, observation_path, _, _ = _authenticate_case(frozen, launch, pins, name)
    return frozen, analysis, launch, case, record, observation_path, pins


def _capture_options(device, maximum_regions, seed, variance_floor):
    if (type(maximum_regions) is not int or not 1 <= maximum_regions <= 8192 or type(seed) is not int
            or not 0 <= seed < 2**63 or isinstance(variance_floor, bool)
            or not isinstance(variance_floor, (int, float)) or not math.isfinite(variance_floor) or variance_floor < 0):
        raise ValueError("Predeclared bounded common sampling and variance floor required")
    chosen_device = torch.device(device)
    if chosen_device.type not in ("cpu", "cuda"): raise ValueError("CPU or explicitly assigned CUDA device required")
    return chosen_device


def _capture_case(case, record, observation_path, primary, *, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        item = torch.load(observation_path, map_location="cpu", weights_only=True)
        if (set(item) != ARCHIVE_KEYS or _item_fingerprint(item) != record["prepared_fields"]
                or item["name"] != case["target"] or item["class_num"] != DATASETS[case["target"]][2]):
            raise ValueError("Original label-free archive content differs")
        model = load_model_checkpoint(primary, device=device)
        observation = {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items() if k in OBS_KEYS}
        stages, _, evidence = capture(model, case["target"], observation)
        del model
    return stages, evidence


def _summaries(stages, maximum_regions, seed, variance_floor):
    return {stage: aligned_summary(value["cnn"], value["graph"], value["Q"],
        maximum_regions=maximum_regions, seed=seed, variance_floor=variance_floor) for stage, value in stages.items()}


def export_case(freeze_path, freeze_sha256, analysis_path, analysis_sha256, name, out, code, *,
                device="cpu", maximum_regions=1024, seed=64, variance_floor=1e-24):
    device = _capture_options(device, maximum_regions, seed, variance_floor)
    frozen, _, launch, pins = _authenticate_boundary(freeze_path, freeze_sha256, analysis_path, analysis_sha256, code)
    case, record, observation_path, primary, case_pins = _authenticate_case(frozen, launch, pins, name)
    out = new_output(out, launch, (Path(freeze_path).resolve().parent, Path(analysis_path).resolve().parent))
    if out.exists(): raise ValueError("New sibling diagnostic output required")
    check_files(case_pins)
    stages, evidence = _capture_case(case, record, observation_path, primary, device=device)
    summaries = _summaries(stages, maximum_regions, seed, variance_floor)
    check_files(pins); require_frozen_code(code)
    out.mkdir()
    for stage, value in stages.items():
        torch.save(value, out / (stage + "-features.pt"))
        write_json(out / (stage + "-summary.json"), summaries[stage])
    result = {"status": "passed", "policy": POLICY, "code": code, "case": name,
        "freeze_sha256": freeze_sha256, "analysis_sha256": analysis_sha256,
        "scope": "full_target_scene_transductive_not_independent_train_validation_blocks",
        "evidence": evidence, "protected_files": pins, "files_verified_unchanged": len(pins),
        "outputs": {p.name: file_hash(p) for p in out.iterdir()}, "model_forwards": 1, "optimizer_updates": 0,
        "bn_calibration_updates": 0, "new_query_release": False, "raw_labels_deserialized": False,
        "channel_selection": False, "all54_diagnostic_cases_completed_by_this_call": False}
    check_files(pins); require_frozen_code(code); write_json(out / "receipt.json", result)
    return result


def export_batch(freeze_path, freeze_sha256, analysis_path, analysis_sha256, out, code, *,
                 device="cpu", maximum_regions=1024, seed=64, variance_floor=1e-24):
    """Run one forward per accepted role, retaining only the regional statistics."""
    started = time.monotonic()
    device = _capture_options(device, maximum_regions, seed, variance_floor)
    frozen, _, launch, pins = _authenticate_boundary(freeze_path, freeze_sha256, analysis_path, analysis_sha256, code)
    plans = [_authenticate_case(frozen, launch, pins, case_name(*role)) for role in grid()]
    if any(pins.get(path) != sha for *_, case_pins in plans for path, sha in case_pins.items()):
        raise ValueError("Every case input must be in the verified full integrity manifest")
    out = new_output(out, launch, (Path(freeze_path).resolve().parent, Path(analysis_path).resolve().parent))
    require_frozen_code(code)
    out.mkdir(exist_ok=False)
    authenticated_seconds = time.monotonic() - started
    rows, samples = [], {}
    for case, record, observation, primary, case_pins in plans:
        case_started = time.monotonic()
        check_files(case_pins)
        stages, evidence = _capture_case(case, record, observation, primary, device=device)
        summaries = _summaries(stages, maximum_regions, seed, variance_floor)
        del stages
        for stage, summary in summaries.items():
            key = (case["target"], stage)
            sample = (case_pins[str(observation)], summary["sampled_region_indices"], summary["sampled_region_masses"])
            if samples.setdefault(key, sample) != sample:
                raise ValueError("All matched roles must use the same observation and sampled regions")
        check_files(case_pins)
        row = {**{key: case[key] for key in ("name", "target", "mode", "method", "seed")},
               "primary_sha256": case_pins[str(primary)], "evidence": evidence,
               "elapsed_seconds": time.monotonic() - case_started}
        output = out / (case["name"] + ".json")
        write_json(output, {**row, "stages": summaries})
        rows.append({**row, "summary": output.name, "summary_sha256": file_hash(output)})
    check_files(pins)
    check_files({str(out / row["summary"]): row["summary_sha256"] for row in rows})
    require_frozen_code(code)
    result = {"status": "passed", "policy": BATCH_POLICY, "code": code,
              "freeze_sha256": freeze_sha256, "analysis_sha256": analysis_sha256, "device": str(device),
              "scope": "full_target_scene_transductive_not_independent_train_validation_blocks",
              "sample": {"maximum_regions": maximum_regions, "seed": seed, "variance_floor": variance_floor},
              "protected_files": pins, "files_verified_unchanged": len(pins), "cases": rows,
              "model_forwards": len(rows), "optimizer_updates": 0, "bn_calibration_updates": 0,
              "new_query_release": False, "raw_labels_deserialized": False, "channel_selection": False,
              "all54_diagnostic_cases_completed_by_this_call": True,
              "authenticated_seconds": authenticated_seconds, "elapsed_seconds": time.monotonic() - started}
    write_json(out / "receipt.json", result)
    return result
