"""Full ordered-source initialization for matched M4 target fine-tuning."""
import json
import re
from pathlib import Path

import torch

from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import M4SourceOrderedRuntime, POLICY, MODES, select_case
from m4_transfer import initialize_m4_target
from model.checkpoint_io import load_model_checkpoint
from target_initialization import METHODS, domain_keys, state_id

SOURCE_PRODUCER = "01c29c82e92ce0a529749c53962202e9bf0dc5fe"


def full_source(spec, target, seed, mode):
    required = {"source_path", "source_weight_sha256", "source_sidecar_sha256",
                "source_result", "source_result_sha256", "source_launch", "source_launch_sha256",
                "source_acceptance", "source_acceptance_sha256",
                "source_independent_audit", "source_independent_audit_sha256"}
    if set(spec) != required:
        raise ValueError("Explicit full source batch, result and weight pins required")
    for field in ("source_path", "source_result", "source_launch", "source_acceptance", "source_independent_audit"):
        digest = spec["source_weight_sha256" if field == "source_path" else field + "_sha256"]
        if file_hash(spec[field]) != digest:
            raise ValueError("Pinned full source file changed")
    path = Path(spec["source_path"]).resolve(strict=True)
    sidecar = Path(str(path) + ".json")
    if file_hash(sidecar) != spec["source_sidecar_sha256"]:
        raise ValueError("Full source sidecar changed")
    launch = json.loads(Path(spec["source_launch"]).read_text())
    accepted = json.loads(Path(spec["source_acceptance"]).read_text())
    result = json.loads(Path(spec["source_result"]).read_text())
    audited = json.loads(Path(spec["source_independent_audit"]).read_text())
    producer = launch["code"]
    if (producer["commit"] != SOURCE_PRODUCER or producer["dirty"] or accepted["code"] != producer
            or accepted["status"] != "passed" or accepted["optimizer_updates"] != 18036
            or len(accepted["cases"]) != 18 or accepted["launch_sha256"] != spec["source_launch_sha256"]
            or audited["status"] != "passed" or audited["producer"] != producer
            or audited["acceptance_sha256"] != spec["source_acceptance_sha256"]
            or audited["launch_sha256"] != spec["source_launch_sha256"]
            or len(audited["cases"]) != 18):
        raise ValueError("Complete accepted v2 source batch required")
    case = select_case(launch, producer, target, seed, mode)
    folder = Path(case["out"]).resolve(strict=True)
    if (path != folder / "primary.pt" or Path(spec["source_result"]).resolve() != folder / "result.json"
            or Path(spec["source_launch"]).resolve() != folder.parent.parent / "launch.json"
            or Path(spec["source_acceptance"]).resolve() != folder.parent.parent / "training-acceptance.json"):
        raise ValueError("Source paths differ from frozen case")
    entry = next(c for c in accepted["cases"] if c["name"] == case["name"])
    if sum(c["name"] == case["name"] for c in audited["cases"]) != 1:
        raise ValueError("Independent source audit case missing")
    if (entry["result_sha256"] != spec["source_result_sha256"]
            or result["status"] != "completed" or result["start_code"] != producer
            or result["optimizer_steps"] != 1002 or result["settings"] != case["settings"]
            or result["identity"]["prepared"] != case["provenance"]
            or result["artifacts"]["primary.pt"] != spec["source_weight_sha256"]
            or result["artifacts"]["primary.pt.json"] != spec["source_sidecar_sha256"]):
        raise ValueError("Source result/role/weights differ")
    with torch.random.fork_rng(devices=[]):
        source = load_model_checkpoint(path, device="cpu")
    meta = source.training_config
    M4SourceOrderedRuntime().validate(meta["joint_runtime"])
    if (meta["source_validation_arithmetic"] != POLICY or meta["training_code"] != producer
            or meta["research_config"] != case["settings"]["config"]
            or meta["provenance"] != case["provenance"]
            or meta["joint_runtime"]["executed_optimizer_steps"] != 1002
            or meta["joint_runtime"]["selected_step"] != result["selected_step"]
            or meta["checkpoint_optimizer_step"] != result["selected_step"]
            or tuple(source.heads) != tuple(case["settings"]["source_tasks"])
            or meta["provenance"].get("policy") != "selective_acc1p_source_v1"
            or not re.fullmatch("[0-9a-f]{64}", meta["provenance"].get("source_identity", ""))
            or any(source.heads[n].out_features != DATASETS[n][2] for n in source.heads)
            or any(target in n.split(".") for n in source.state_dict())
            or any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
                   for v in source.state_dict().values())):
        raise ValueError("Full source checkpoint provenance differs")
    return source, {"protocol": "g4_m4_full_source_binding_v2", "mode": mode,
                    "target": target, "seed": seed, "selected_step": result["selected_step"],
                    "producer": SOURCE_PRODUCER, "pins": dict(spec),
                    "source_graph_binding": case["graph_binding"]}


def initialize_full_target(target, method, mode, *, model_seed, head_seed, source_spec=None, device="cpu"):
    if method not in METHODS[:2] or mode not in MODES:
        raise ValueError("Registered full scratch/fine-tune methods required")
    if (method == METHODS[0]) != (not source_spec):
        raise ValueError("Scratch/source role differs")
    # Reuse the accepted active-M4 initializer for exactly the same fresh target
    # architecture, head seeds and domain state before copying shared weights.
    model, record = initialize_m4_target(target, METHODS[0], mode, model_seed=model_seed,
                                         head_seed=head_seed, device=device)
    if source_spec:
        source, binding = full_source(source_spec, target, model_seed, mode)
        before = state_id(source)
        source_config = dict(source.export_config()["backbone_kwargs"])
        target_config = dict(model.export_config()["backbone_kwargs"])
        source_config.pop("dataset_names", None)
        target_config.pop("dataset_names", None)
        if source_config != target_config:
            raise ValueError("Source/target architecture differs")
        fresh = domain_keys(model.backbone)
        src, dst = source.backbone.state_dict(), model.backbone.state_dict()
        shared = set(dst) - fresh
        if shared != set(src) - domain_keys(source.backbone):
            raise ValueError("Shared source/target keys differ")
        for name in sorted(shared):
            if src[name].shape != dst[name].shape or src[name].dtype != dst[name].dtype:
                raise ValueError("Shared tensor shape/dtype differs")
            dst[name] = src[name].detach().clone()
        model.backbone.load_state_dict(dst, strict=True)
        if state_id(source) != before:
            raise ValueError("Source mutated during initialization")
        record.update(source=binding, copied_backbone_keys=sorted(shared),
                      source_graph_context="source_train_label_lda_slic")
    record.update(method=method, initial_device=str(torch.device(device)))
    model = model.to(device).train()
    record["initial_state_sha256"] = state_id(model)
    model.training_config = {"target_initialization": record}
    if source_spec:
        for field in ("source_path", "source_result", "source_launch", "source_acceptance", "source_independent_audit"):
            digest = source_spec["source_weight_sha256" if field == "source_path" else field + "_sha256"]
            if file_hash(source_spec[field]) != digest:
                raise ValueError("Source changed during initialization")
        if file_hash(str(source_spec["source_path"]) + ".json") != source_spec["source_sidecar_sha256"]:
            raise ValueError("Source sidecar changed during initialization")
    return model, record

