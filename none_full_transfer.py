"""Accepted ordered none source cohort to fresh target heads and domain state."""
import copy
import json
from pathlib import Path
import re

import torch
from g4_none_ordered_source import NoneOrderedSourceRuntime
from g4_none_source_launch import launch_binding, select_bound_case
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import POLICY
from model.checkpoint_io import load_model_checkpoint
from research.reproduction.m4g4b.none_source_full_audit import acceptance_contract
from research_runtime import code_snapshot
from target_initialization import METHODS, domain_keys, initialize_target, state_id

SOURCE_PRODUCER = "52f32978419cbeda216a90bfb0ed76c393ef4f5d"
SOURCE_BINDING = "g4_none_full_source_binding_v2"
SOURCE_FILES = ("source_path", "source_result", "source_launch", "source_acceptance",
                "source_independent_audit", "source_protected_manifest")
SPEC_FIELDS = {"source_path", "source_weight_sha256", "source_sidecar_sha256"} | {
    name for field in SOURCE_FILES[1:] for name in (field, field + "_sha256")}


def verify_spec_files(spec):
    if set(spec) != SPEC_FIELDS:
        raise ValueError("Explicit full none cohort, protected manifest, result and checkpoint pair required")
    for field in SOURCE_FILES:
        sha = spec["source_weight_sha256" if field == "source_path" else field + "_sha256"]
        if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha) or file_hash(spec[field]) != sha:
            raise ValueError("Pinned full none source file changed")
    if file_hash(str(spec["source_path"]) + ".json") != spec["source_sidecar_sha256"]:
        raise ValueError("Full none source sidecar changed")


def full_none_source(spec, target, seed):
    verify_spec_files(spec)
    launch, accepted, audited, result = [json.loads(Path(spec[k]).read_text()) for k in
        ("source_launch", "source_acceptance", "source_independent_audit", "source_result")]
    producer = launch["code"]
    entries = acceptance_contract(launch, accepted, code_snapshot(), SOURCE_PRODUCER, spec["source_launch_sha256"])
    names = set(entries)
    if (audited["status"] != "passed" or audited["producer"] != producer
            or audited["launch_sha256"] != spec["source_launch_sha256"]
            or audited["acceptance_sha256"] != spec["source_acceptance_sha256"]
            or audited["protected_manifest_sha256"] != spec["source_protected_manifest_sha256"]
            or audited["new_optimizer_updates"] != 0 or audited["query_labels_accessed"] is not False
            or audited["test_scoring"] is not False or len(audited["cases"]) != 9
            or {x["name"] for x in audited["cases"]} != names
            or sum(x["metrics"] for x in audited["cases"]) != 108
            or sum(x["recalls"] for x in audited["cases"]) != 1440
            or sum(x["exact_reload_repeats"] for x in audited["cases"]) != 27):
        raise ValueError("Complete independent ordered none source acceptance required")
    case = select_bound_case(launch, producer, target, seed, full=True)
    folder = Path(case["out"]).resolve()
    root = folder.parent.parent
    if (Path(spec["source_path"]).resolve() != folder / "primary.pt"
            or Path(spec["source_result"]).resolve() != folder / "result.json"
            or Path(launch["out_dir"]).resolve() != root
            or any(Path(spec[k]).resolve() != root / name for k, name in
                   (("source_launch", "launch.json"), ("source_acceptance", "training-acceptance.json"),
                    ("source_protected_manifest", "protected-files.json")))):
        raise ValueError("Full none source role/path differs from frozen case")
    binding = launch_binding(launch, case, spec["source_launch_sha256"])
    provenance = {**case["source_context"], "none_source_launch": binding}
    entry = entries[case["name"]]
    checked = next(x for x in audited["cases"] if x["name"] == case["name"])
    if (entry["result_sha256"] != spec["source_result_sha256"]
            or entry["primary_sha256"] != spec["source_weight_sha256"]
            or entry["sidecar_sha256"] != spec["source_sidecar_sha256"]
            or checked["result_sha256"] != spec["source_result_sha256"]
            or result["status"] != "completed" or result["optimizer_steps"] != 1002
            or result["start_code"] != producer or result["settings"] != case["settings"]
            or result["identity"]["prepared"] != provenance
            or result["selected_step"] != entry["selected_step"] or result["selected_step"] != checked["selected_step"]
            or result["artifacts"]["primary.pt"] != spec["source_weight_sha256"]
            or result["artifacts"]["primary.pt.json"] != spec["source_sidecar_sha256"]):
        raise ValueError("Full none source result/budget/model pair differs")
    with torch.random.fork_rng(devices=[]):
        source = load_model_checkpoint(spec["source_path"], device="cpu")
    meta = source.training_config
    NoneOrderedSourceRuntime().validate(meta["joint_runtime"])
    if (meta.get("source_validation_arithmetic") != POLICY or meta.get("none_source_launch") != binding
            or meta["training_code"] != producer or meta["research_config"] != case["settings"]["config"]
            or meta["provenance"] != provenance or meta["joint_runtime"].get("executed_optimizer_steps") != 1002
            or meta["joint_runtime"].get("selected_step") != result["selected_step"]
            or meta["checkpoint_optimizer_step"] != result["selected_step"]
            or tuple(source.heads) != tuple(case["settings"]["source_tasks"])
            or provenance.get("policy") != "selective_acc1p_source_v1"
            or provenance.get("target_payload_accessed") is not False
            or not re.fullmatch("[0-9a-f]{64}", provenance.get("source_identity", ""))
            or any(source.heads[n].out_features != DATASETS[n][2] for n in source.heads)
            or any(target in name.split(".") for name in source.state_dict())
            or any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
                   for v in source.state_dict().values())):
        raise ValueError("Full none selected checkpoint provenance differs")
    verify_spec_files(spec)
    return source, {"protocol": SOURCE_BINDING, "mode": "none", "target": target, "seed": seed,
                    "selected_step": result["selected_step"], "producer": SOURCE_PRODUCER,
                    "source_graph_binding": case["graph_binding"], "source_launch_binding": binding,
                    "pins": copy.deepcopy(spec)}


def initialize_none_full_target(target, method, *, model_seed, head_seed, source_spec=None, device="cpu"):
    if method not in METHODS[:2] or (method == METHODS[0]) != (not source_spec):
        raise ValueError("Registered none scratch/full-finetune source role required")
    model, record = initialize_target(target, METHODS[0], model_seed=model_seed, head_seed=head_seed, device=device)
    if source_spec:
        source, binding = full_none_source(source_spec, target, model_seed)
        before = state_id(source)
        source_config, target_config = [dict(m.export_config()["backbone_kwargs"]) for m in (source, model)]
        for value in (source_config, target_config):
            value.pop("dataset_names", None)
        if source_config != target_config:
            raise ValueError("Full none source/target architecture differs")
        fresh = domain_keys(model.backbone)
        src, dst = source.backbone.state_dict(), model.backbone.state_dict()
        shared = set(dst) - fresh
        if shared != set(src) - domain_keys(source.backbone):
            raise ValueError("Full none shared backbone keys differ")
        for name in sorted(shared):
            if src[name].shape != dst[name].shape or src[name].dtype != dst[name].dtype:
                raise ValueError("Full none shared tensor shape/dtype differs")
            dst[name] = src[name].detach().clone()
        model.backbone.load_state_dict(dst, strict=True)
        if state_id(source) != before:
            raise ValueError("Full none source mutated during initialization")
        record.update(source=binding, copied_backbone_keys=sorted(shared), source_graph_context="source_train_label_lda_slic")
        verify_spec_files(source_spec)
    # Retain the accepted fresh-domain initialization contract consumed by the
    # unchanged target BN calibrator; bind the new source protocol separately.
    record.update(full_source_initialization=SOURCE_BINDING, method=method, initial_device=str(torch.device(device)),
                  initial_state_sha256=state_id(model))
    model = model.to(device).train()
    model.training_config = {"target_initialization": copy.deepcopy(record)}
    return model, record
