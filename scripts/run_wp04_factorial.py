"""Register and verify the fixed-Q/A WP04 scratch factorial through the bounded target runtime."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, redirect_stdout
import csv
import ctypes
import json
from pathlib import Path
import os
import resource
import subprocess
import sys
from time import perf_counter, sleep
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import numpy as np
import torch

import g4_edge_target as edge
import g4_edge_target_runtime as edge_runtime
import g4_branch_local_target as local_control
import g4_branch_local_target_runtime as local_runtime
import g4_interaction_target as interaction
import g4_interaction_target_runtime as interaction_runtime
import g4_normalized_edge_target as normalization
import g4_normalized_edge_target_runtime as normalization_runtime
from evaluation.classification_metrics import metrics_from_confusion
from g4_position_fusion_target_runtime import prepare
from g4_support_context import prepare_contexts, read, validate_manifest
from joint_training import _tree_fingerprint, run_training
from model.region_pixel_alignment import broadcast_regions, pool_regions
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, tensor_hash, write_json
from target_initialization import METHODS
from target_training import TargetRuntime, observation
from g4_position_fusion_target import calibrate, initialize as initialize_position
from target_initialization import state_id
from transfer import p4_runtime_device as p4
from transfer.shared_gpu_lease import physical_lease


SEEDS = (64, 512, 2048)
ARMS = {
    "B0": ("none", "fixed"),
    "R": ("x_residual", "fixed"),
    "F": ("none", "position_gate"),
    "R+F": ("x_residual", "position_gate"),
    "Bcap": ("capacity_control", "fixed"),
    "Bcap+F": ("capacity_control", "position_gate"),
}
HEADS = {"fused": "heads", "graph": "hgnn_heads", "cnn": "cnn_heads"}
EXTRA = ("backbone.region_residual.", "backbone.position_fusion.")
STEPS = {"whole": 12, "prefix": 2, "resume": 10, "completed": 0}
CONTRASTS = {
    "R-B0": {"R": 1, "B0": -1},
    "F-B0": {"F": 1, "B0": -1},
    "R+F-B0": {"R+F": 1, "B0": -1},
    "R+F-R-F+B0": {"R+F": 1, "R": -1, "F": -1, "B0": 1},
    "R-Bcap": {"R": 1, "Bcap": -1},
    "R+F-(Bcap+F)": {"R+F": 1, "Bcap+F": -1},
    "Bcap+F-Bcap-F+B0": {"Bcap+F": 1, "Bcap": -1, "F": -1, "B0": 1},
}
SCHEMA = "wp04_fixed_graph_scratch_factorial_v1"
REAL_CATALOG_SHA = "a2c23f9b60ccf094008e6ac4c7bcd4aa289dece6e8b3993e91431ee0a75c502e"
CONTROLS = {
    "directions": {"off": "none", "C2G": "cnn_to_graph", "G2C": "graph_to_cnn", "both": "gated"},
    "edges": {"off": "none", "I": "gated", "Wg": "graph", "Wj": "joint", "I+Wg": "gated_graph",
              "I7": "add_joint", "I8": "gated_joint", "I9": "global_joint"},
}
CONTROL_CONTRASTS = {
    "directions": (("C2G", "off"), ("G2C", "off"), ("both", "off"), ("both", "C2G"), ("both", "G2C")),
    "edges": (("I", "off"), ("Wg", "off"), ("Wj", "off"), ("Wj", "Wg"), ("I+Wg", "I"),
              ("I8", "I"), ("I8", "I+Wg"), ("I7", "I8"), ("I9", "I8")),
}
ADAPTERS = {
    "directions": (interaction_runtime.prepare, interaction, "interaction_binding"),
    "edges": (edge_runtime.prepare, edge, "edge_binding"),
}
REAL_PAIR_SCOPE = "real_dev_c2g_paired"
REAL_PAIR = ("off", "C2G")
REAL_PAIR_UPDATES = len(SEEDS) * len(REAL_PAIR) * (STEPS["whole"] + STEPS["prefix"] + STEPS["resume"])
DIRECTION_SCOPE = "real_dev_direction_capacity"
FACTORIAL_SCOPE = "real_dev_fixed_qa_residual_fusion"
EDGE_WEIGHT_SCOPE = "real_dev_fixed_candidates_edge_weights"
EDGE_WEIGHT_GROUP = "edge_weights"
EDGE_WEIGHT_CONDITIONS = {"off": "none", "norm": "row_normalized", "Wg": "graph", "Wj": "joint"}
EDGE_WEIGHT_PARAMS = {"off": 61696, "norm": 61696, "Wg": 70144, "Wj": 78592}
EDGE_WEIGHT_CONTRASTS = (("norm", "off"), ("Wg", "norm"), ("Wj", "norm"),
                         ("Wj", "Wg"), ("Wg", "off"), ("Wj", "off"))
EDGE_WEIGHT_PRIMARY = tuple(f"{a}-{b}" for a, b in EDGE_WEIGHT_CONTRASTS[:3])
MESSAGE_SCOPE = "real_dev_joint_message_correspondence"
MESSAGE_GROUP = "message_correspondence"
MESSAGE_CONDITIONS = {"I8": "gated_joint", "I9-message": "message_global_joint",
                      "Wg65": "capacity_matched_graph"}
MESSAGE_PARAMS = {"I8": 137090, "I9-message": 137090, "Wg65": 78592}
MESSAGE_CONTRASTS = (("I8", "I9-message"),)
EDGE_POINT_GROUPS = {EDGE_WEIGHT_GROUP: EDGE_WEIGHT_CONDITIONS, MESSAGE_GROUP: MESSAGE_CONDITIONS}
EDGE_POINT_SCOPES = (EDGE_WEIGHT_SCOPE, MESSAGE_SCOPE)
POINT_SCOPES = (FACTORIAL_SCOPE, *EDGE_POINT_SCOPES)
REAL_SCOPES = (REAL_PAIR_SCOPE, DIRECTION_SCOPE, *POINT_SCOPES)
FACTORIAL_PARAMS = {"B0": 61696, "R": 69952, "F": 61825, "R+F": 70081, "Bcap": 69952, "Bcap+F": 70081}
FACTORIAL_EFFECTS = {
    **CONTRASTS,
    "R+F-F": {"R+F": 1, "F": -1},
    "R+F-R": {"R+F": 1, "R": -1},
    "Bcap-B0": {"Bcap": 1, "B0": -1},
    "Bcap+F-F": {"Bcap+F": 1, "F": -1},
    "Bcap+F-Bcap": {"Bcap+F": 1, "Bcap": -1},
    "R-main": {"R": .5, "B0": -.5, "R+F": .5, "F": -.5},
    "F-main": {"F": .5, "B0": -.5, "R+F": .5, "R": -.5},
}
PRIOR_CODE = "3b383a59d644b17e6427d5a8656ce22d971c6d82"
PRIOR_EVIDENCE = "955c50b81d729ea0a811789eac78cde59705b446530c1e68f0ddff8ebfdc295f"
NEW_DIRECTIONS = {"add": "add", "both": "gated", "G2C": "graph_to_cnn",
                  "local-add": "local_add", "local-both": "branch_local",
                  "local-C2G": "local_graph", "local-G2C": "local_cnn"}
MATCHES = (("add", "local-add"), ("both", "local-both"), ("C2G", "local-C2G"), ("G2C", "local-G2C"))
DIRECTION_PARAMS = {"off": 61696, "C2G": 90945, "G2C": 90945, "add": 94976, "both": 120194}
DIRECTION_PARAMS.update({control: DIRECTION_PARAMS[arm] for arm, control in MATCHES})


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(code):
    return {k: code[k] for k in ("commit", "source_hash")}


def pinned(path):
    return {"path": str(path), "sha256": hash_file(path)}


def verify(pin):
    path = Path(pin["path"])
    require(path.resolve(strict=True) == path and hash_file(path) == pin["sha256"], "Pinned input changed")
    return path


def roster(group="factorial"):
    if group in EDGE_POINT_GROUPS:
        return [{"seed": seed, "arm": arm, "condition": condition}
                for seed in SEEDS for arm, condition in EDGE_POINT_GROUPS[group].items()]
    if group in CONTROLS:
        return [{"seed": seed, "arm": arm, "condition": condition}
                for seed in SEEDS for arm, condition in CONTROLS[group].items()]
    require(group == "factorial", "Unknown WP04 group")
    return [{"seed": seed, "arm": arm, "mode": mode, "fusion": fusion}
            for seed in SEEDS for arm, (mode, fusion) in ARMS.items()]


def plan(catalog, code, scope, *, arithmetic=None, group="factorial"):
    require(arithmetic in (None, p4.CUDA_FP64_WEIGHT), "Unknown WP04 arithmetic")
    value = {"schema": SCHEMA, "scope": scope, "code": identity(code), "catalog": pinned(catalog),
            "target": "xuzhou", "support_seed": 512, "per_class": 1, "cases": roster(group),
            "phase_updates": STEPS, "method": METHODS[0], "graph": "input_Q_and_A_fixed",
            "selection": "first_min_fused_validation_nll", "real_updates_authorized": 0,
            "query_access_authorized": False,
            "full_source_ft": "separate_nonfactorial_scope_different_pretrained_source_by_mode"}
    if arithmetic is not None:
        value["arithmetic"] = arithmetic
    if group in CONTROLS:
        value["group"] = group
        if group == "edges":
            value["graph"] = "fixed_Q_and_A_candidate_support_opt_in_learned_A_values"
    if group == EDGE_WEIGHT_GROUP:
        value.update(group=group, graph="fixed_Q_and_A_candidates_stage2_weight_control")
    if group == MESSAGE_GROUP:
        value.update(group=group, graph="fixed_Q_and_A_candidates_regional_joint_edge_alignment")
    return value


def registration(args, code):
    path = verify({"path": str(Path(args.registration[0]).resolve()), "sha256": args.registration[1]})
    value = read(path)
    catalog = verify(value["catalog"])
    if value["scope"] in REAL_SCOPES:
        published = value["published_catalog"]
        verify(published)
        require(published["sha256"] == REAL_CATALOG_SHA
                and catalog == path.parent / "rebound-contexts" / "manifest.json"
                and value == real_plan(catalog, published, code, [value["devices"][str(seed)] for seed in SEEDS],
                                       prior=value.get("prior"), factorial=value["scope"] == FACTORIAL_SCOPE,
                                       edge_weights=value["scope"] == EDGE_WEIGHT_SCOPE,
                                       messages=value["scope"] == MESSAGE_SCOPE),
                "Real paired development registration or publication differs")
        return value
    require(value == plan(catalog, code, "synthetic_only", arithmetic=value.get("arithmetic"),
                          group=value.get("group", "factorial"))
            and catalog == path.parent / "contexts" / "manifest.json",
            "Only the generated synthetic fixture registration can run optimization")
    return value


def real_plan(catalog, published, code, devices, *, prior=None, factorial=False, edge_weights=False, messages=False):
    require(len(devices) == len(SEEDS) and len(set(devices)) == len(SEEDS)
            and all("GPU-" + str(UUID(gpu.removeprefix("GPU-"))) == gpu for gpu in devices),
            "Assign one distinct physical GPU UUID to each paired seed")
    base = plan(catalog, code, REAL_PAIR_SCOPE, arithmetic=p4.CUDA_FP64_WEIGHT, group="directions")
    base = {**base, "cases": [c for c in base["cases"] if c["arm"] in REAL_PAIR],
            "published_catalog": published, "devices": dict(zip(map(str, SEEDS), devices, strict=True)),
            "real_updates_authorized": REAL_PAIR_UPDATES, "resource_policy": "actual_cuda_allocation",
            "role_scope": "published_train_dev_only", "execution": str(catalog.parent.parent / "execution")}
    if factorial:
        require(prior is None, "The position-fusion matrix requires its matched position runtime throughout")
        return {**base, "scope": FACTORIAL_SCOPE, "group": "factorial", "graph": "input_Q_and_A_fixed",
                "cases": roster("factorial"), "parameters_by_arm": FACTORIAL_PARAMS,
                "primary_comparisons": ["R-Bcap", "R+F-(Bcap+F)", "F-B0", "R+F-R"],
                "real_updates_authorized": len(SEEDS)*len(ARMS)*(STEPS["whole"]+STEPS["prefix"]+STEPS["resume"])}
    if edge_weights:
        require(prior is None, "Edge weight controls require a fresh, matched baseline")
        cases = roster(EDGE_WEIGHT_GROUP)
        return {**base, "scope": EDGE_WEIGHT_SCOPE, "group": EDGE_WEIGHT_GROUP,
                "graph": "fixed_Q_and_A_candidates_stage2_weight_control",
                "cases": cases, "parameters_by_arm": EDGE_WEIGHT_PARAMS,
                "primary_comparisons": list(EDGE_WEIGHT_PRIMARY),
                "real_updates_authorized": len(cases) * sum(STEPS.values())}
    if messages:
        require(prior is None, "Use the registered joint message and capacity controls")
        cases = roster(MESSAGE_GROUP)
        return {**base, "scope": MESSAGE_SCOPE, "group": MESSAGE_GROUP,
                "graph": "fixed_Q_and_A_candidates_regional_joint_edge_alignment",
                "cases": cases, "parameters_by_arm": MESSAGE_PARAMS,
                "primary_comparisons": ["I8-I9-message", "Wj-Wg65"],
                "real_updates_authorized": len(cases) * sum(STEPS.values())}
    if prior is None:
        return base
    require(prior["sha256"] == PRIOR_EVIDENCE, "Expected the independently accepted I0/I3 evidence")
    return {**base, "scope": DIRECTION_SCOPE, "prior": prior, "parameters_by_arm": DIRECTION_PARAMS,
            "cases": [{"seed": seed, "arm": arm, "condition": condition}
                      for seed in SEEDS for arm, condition in NEW_DIRECTIONS.items()],
            "real_updates_authorized": len(SEEDS)*len(NEW_DIRECTIONS)*(STEPS["whole"]+STEPS["prefix"]+STEPS["resume"])}


def check_real_lane(reg, seed, gpu, lane):
    if reg["scope"] in REAL_SCOPES:
        require(gpu == reg["devices"].get(str(seed)) and lane == Path(reg["execution"]) / str(seed),
                "Production worker does not match its registered seed, GPU or output directory")


def bound(reg, case, out, device):
    common = dict(target=reg["target"], support_seed=reg["support_seed"], per_class=reg["per_class"],
                  method=reg["method"], model_seed=case["seed"], device=device)
    paths = reg["catalog"]["path"], reg["catalog"]["sha256"], out
    group = reg.get("group", "factorial")
    if group == "directions" and case["condition"] in local_control.LOCAL_MATCHES:
        return local_runtime.prepare(*paths, condition=case["condition"], **common)
    if group in EDGE_POINT_GROUPS:
        runtime, _, _ = edge_weight_adapter(case["condition"])
        return runtime(*paths, condition=case["condition"], **common)
    if group in ADAPTERS:
        return ADAPTERS[group][0](*paths, condition=case["condition"], **common)
    return prepare(*paths, mode=case["mode"], condition=case["fusion"],
                   exact_capacity=reg["scope"] == FACTORIAL_SCOPE and case["mode"] == "capacity_control", **common)


def edge_weight_adapter(condition):
    if condition == "row_normalized":
        return normalization_runtime.prepare, normalization, "normalization_binding"
    return edge_runtime.prepare, edge, "edge_binding"


def attest(gpu):
    lib = ctypes.CDLL("libcuda.so.1")
    ordinal, value = ctypes.c_int(), (ctypes.c_ubyte * 16)()
    require(lib.cuInit(0) == 0 and lib.cuDeviceGet(ctypes.byref(ordinal), 0) == 0
            and lib.cuDeviceGetUuid(ctypes.byref(value), ordinal) == 0
            and "GPU-" + str(UUID(bytes=bytes(value))) == gpu, "Actual CUDA device differs")
    props = torch.cuda.get_device_properties(0)
    return {"uuid": gpu, "name": props.name, "memory_total_bytes": props.total_memory, "cuda": torch.version.cuda}


def hardware(gpu, started):
    if gpu:
        torch.cuda.synchronize()
        return {"wall_seconds": perf_counter()-started, "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(0)}
    return {"wall_seconds": perf_counter()-started, "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def model_snapshot(model, group="factorial"):
    tensors = {k: tensor_hash(v) for k, v in model.state_dict().items()}
    if group in CONTROLS or group in EDGE_POINT_GROUPS:
        prefixes = edge.PREFIXES if group in ("edges", *EDGE_POINT_GROUPS) else (interaction.PREFIX,)
        return {"shared": digest({k: v for k, v in tensors.items() if not k.startswith(prefixes)}),
                "additional_tensors": {k: v for k, v in tensors.items() if k.startswith(prefixes)},
                "parameters": sum(p.numel() for p in model.parameters()),
                "learned_parameters": {k: p.numel() for k, p in model.named_parameters() if k.startswith(prefixes)}}
    return {"shared": digest({k: v for k, v in tensors.items() if not k.startswith(EXTRA)}),
            "residual": {k: v for k, v in tensors.items() if k.startswith(EXTRA[0])},
            "fusion": {k: v for k, v in tensors.items() if k.startswith(EXTRA[1])},
            "parameters": sum(p.numel() for p in model.parameters()),
            "residual_parameters": sum(p.numel() for k, p in model.named_parameters() if k.startswith(EXTRA[0])),
            "fusion_parameters": sum(p.numel() for k, p in model.named_parameters() if k.startswith(EXTRA[1]))}


def move_items(items, device):
    return {name: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
            for name, item in items.items()}


def validation_evidence(probabilities, item, recorded, *, keep_correctness=False, keep_points=False):
    data = {}
    truth = item["val_labels"].cpu().numpy()
    classes = item["class_num"]
    for head in HEADS:
        scores = probabilities[head].cpu().numpy()
        matrix = np.bincount(truth * classes + scores.argmax(1), minlength=classes**2).reshape(classes, classes)
        stats = metrics_from_confusion(matrix, range(1, classes+1))
        trace = recorded[head]
        expected_nll = -np.log(np.maximum(scores[np.arange(len(truth)), truth], 1e-8)).mean(dtype=np.float64)
        require(stats["confusion_matrix"] == trace["confusion"]
                and abs(stats["oa"]-trace["oa"]) < 1e-12 and abs(stats["aa"]-trace["aa"]) < 1e-12
                and abs(float(expected_nll)-trace["nll"]) < 1e-6
                and [c["support"] for c in stats["per_class"]] == [5] * classes,
                "Independent validation confusion, count, or NLL differs")
        data[head] = {"oa": stats["oa"], "aa": stats["aa"], "nll": trace["nll"],
                      "classes": [{**c, "correct": int(matrix[i, i])} for i, c in enumerate(stats["per_class"])],
                      "probability_sha256": tensor_hash(probabilities[head])}
        if keep_correctness:
            correct = scores.argmax(1) == truth
            data[head]["correctness"] = [correct[truth == label].tolist() for label in range(classes)]
        if keep_points:
            actual = scores[np.arange(len(truth)), truth]
            other = scores.copy()
            other[np.arange(len(truth)), truth] = -np.inf
            data[head]["confusion"] = matrix.tolist()
            data[head]["points"] = [
                {"class_id": int(label) + 1, "point": ordinal + 1,
                 "predicted": int(scores[index].argmax()) + 1,
                 "true_probability": float(actual[index]),
                 "margin": float(actual[index] - other[index].max())}
                for label in range(classes)
                for ordinal, index in enumerate(np.flatnonzero(truth == label))
            ]
    return data


def terminal(path):
    progress = read(path / "progress.json")
    terminal_path = path / "state-000012.pt"
    require(progress["cursor"] == 12 and progress["state_file"] == terminal_path.name
            and progress["state_sha256"] == hash_file(terminal_path), "Terminal checkpoint differs")
    return torch.load(terminal_path, map_location="cpu", weights_only=True)


def control_diagnostics(group, case, model, item, runtime, probabilities):
    if group == "directions":
        return {"initial_head_sha256": {k: tensor_hash(v) for k, v in probabilities.items()},
                "configuration": (local_control.configuration(case["condition"])
                                  if case["condition"] in local_control.LOCAL_MATCHES
                                  else interaction.configuration(case["condition"]))}
    if group in EDGE_POINT_GROUPS:
        return edge_weight_diagnostics(case, model, item, runtime)
    if group != "edges":
        return {}
    config = edge.configuration(case["condition"])
    result = {"configuration": config}
    if config["graph_update_mode"] == "none":
        return result
    with torch.no_grad():
        info = edge.forward_observation(model, observation(model, item),
            expected_binding=runtime.edge_binding, diagnostics=True)[-1]["interaction"]
    initial = item["A"].to_sparse_coo().coalesce()
    indices = initial.indices()[:, initial.values() > 0]
    original = {tuple(pair) for pair in indices.T.tolist()}
    occupied = {row for row, _ in original}
    fallback = {(row, row) for row in range(initial.shape[0]) if row not in occupied}
    observed = {tuple(pair) for pair in info["edge_indices"].T.tolist()}
    require(observed == original | fallback, "Learned A changed candidate support")
    before, after = info["edge_weights_before"], info["edge_weights_after"]
    result.update(candidate_edges=len(original), isolated_fallback_edges=len(fallback), actual_edges=len(observed),
                  updated_weight_elements=int(torch.count_nonzero(after-before)), maximum_change=float((after-before).abs().max()))
    return result


def edge_weight_diagnostics(case, model, item, runtime):
    condition = case["condition"]
    if condition == "none":
        require(model.backbone.edge_updater is None, "Off arm acquired an edge updater")
        return {"condition": condition, "stage2_operator": "original_input_A"}
    obs = observation(model, item)
    path = {}
    with torch.no_grad():
        if condition == "row_normalized":
            normalization.verify_initialization_binding(model, runtime.normalization_binding)
            captured = []
            hook = model.backbone.edge_updater.register_forward_hook(lambda _m, _a, output: captured.append(output[1]))
            try:
                model(runtime.normalization_binding["target"], obs["net_input"], obs["Q"], obs["A"],
                      wavelengths=obs["wavelengths"], fwhm=obs["fwhm"], superpixel_scale=obs["superpixel_scale"],
                      return_graph_info=True, return_diagnostics=True)
            finally:
                hook.remove()
            require(len(captured) == 1, "Normalization was not applied exactly once in the real schedule")
            info = captured[0]
        elif case["arm"] in MESSAGE_CONDITIONS:
            info, path = joint_message_path(model, obs, runtime)
        else:
            result = edge.forward_observation(model, obs, expected_binding=runtime.edge_binding, diagnostics=True)
            info = result[-1]["interaction"]
    a = item["A"].to_sparse_coo().coalesce()
    edges = {tuple(row) for row in a.indices()[:, a.values() > 0].T.tolist()}
    occupied = {row for row, _ in edges}
    fallback = {(row, row) for row in range(a.shape[0]) if row not in occupied}
    support = {tuple(row) for row in info["edge_indices"].T.tolist()}
    require(support == edges | fallback, "Stage two changed fixed candidate support")
    prior, weights = info["edge_weights_before"], info["edge_weights_after"]
    delta = (weights - prior).abs()
    changes = int(torch.count_nonzero(delta))
    require(bool(torch.isfinite(weights).all() and (weights > 0).all())
            and (changes == 0) == (condition == "row_normalized"), "Unexpected normalized or learned edge weights")
    return {"condition": condition, "stage2_operator": "normalized_prior" if condition == "row_normalized"
            else "learned_prior_mixture", "nodes": a.shape[0], "candidate_edges": len(edges),
            "isolated_fallback_edges": len(fallback), "actual_edges": len(support),
            "support_sha256": tensor_hash(info["edge_indices"]), "prior_sha256": tensor_hash(prior),
            "weights_sha256": tensor_hash(weights), "changed_weights": changes,
            "mean_abs_change_from_prior": float(delta.mean()), "max_abs_change_from_prior": float(delta.max()), **path}


def joint_message_path(model, obs, runtime):
    backbone = model.backbone
    messages, edge_update = backbone.interaction, backbone.edge_updater
    condition = runtime.edge_binding["condition"]
    captures = {}

    def capture(name):
        def receive(_module, inputs, output):
            require(name not in captures, "Unexpected repeated message or edge update")
            captures[name] = (inputs, output)
        return receive

    handles = [edge_update.register_forward_hook(capture("edges"))]
    if messages is not None:
        handles.extend(getattr(messages, name).register_forward_hook(capture(name))
                       for name in ("graph_to_cnn", "cnn_to_graph"))
    try:
        result = edge.forward_observation(model, obs, expected_binding=runtime.edge_binding, diagnostics=True)
    finally:
        for handle in handles:
            handle.remove()
    info = result[-1]["interaction"]
    (edges_regions, adjacency, aligned_pixels), _ = captures["edges"]
    require(tensor_hash(adjacency) == tensor_hash(obs["A"]), "Edge updater consumed a different candidate graph")
    rows = edges_regions.shape[0]
    channels, width = edges_regions.shape[1], edge_update.embed_dim
    edge_path = {"mode": edge_update.mode, "feature_channels": edge_update.query.in_features,
                 "embedding_channels": width, "projection_dense_macs": 2 * rows * edge_update.query.in_features * width,
                 "edge_dot_macs": info["edge_indices"].shape[1] * width,
                 "region_input": tensor_hash(edges_regions), "aligned_cnn_input": None}
    if messages is None:
        require(edge_update.mode == "graph" and aligned_pixels is None and not set(captures) - {"edges"},
                "Graph-only control consumed a CNN feature or message")
        return info, {"message_paths": {}, "edge_path": edge_path}

    (pixels, graph_source), (pixels_after, _) = captures["graph_to_cnn"]
    (regions, cnn_source), (regions_after, _) = captures["cnn_to_graph"]
    q = obs["Q"]
    global_messages = condition == "message_global_joint"
    graph_expected = regions.mean(0, keepdim=True).expand_as(pixels) if global_messages else broadcast_regions(q, regions)
    cnn_expected = pixels.mean(0, keepdim=True).expand_as(regions) if global_messages else pool_regions(q, pixels)
    require(torch.equal(graph_source, graph_expected) and torch.equal(cnn_source, cnn_expected)
            and torch.equal(edges_regions, regions_after) and torch.equal(aligned_pixels, pool_regions(q, pixels_after)),
            "The message sources or the region-aligned joint update differed from the registered tensor path")
    edge_path["aligned_cnn_input"] = tensor_hash(aligned_pixels)
    path = {"message_paths": {
        "correspondence": "global_mean" if global_messages else "region_Q",
        "graph_to_cnn": {"target": tensor_hash(pixels), "source": tensor_hash(graph_source),
                         "output": tensor_hash(pixels_after), "shape": list(graph_source.shape)},
        "cnn_to_graph": {"target": tensor_hash(regions), "source": tensor_hash(cnn_source),
                         "output": tensor_hash(regions_after), "shape": list(cnn_source.shape)}},
        "edge_path": edge_path}
    return info, path


def component_activity(group, trace):
    labels = {"directions": ("interaction", "directions"), "edges": ("edge_components", "components")}
    if group not in labels:
        return {}
    key, members = labels[group]
    if group == "directions" and "branch_local_control" in trace[0]:
        key, members = "branch_local_control", "slots"
    values = {}
    for name in trace[0][key][members]:
        rows = [row[key][members][name] for row in trace]
        if rows[0] is None:
            require(all(row is None for row in rows), "Inactive mechanism acquired an optimizer step")
            values[name] = None
            continue
        require(all(row is not None for row in rows) and max(row["gradient_norm"] for row in rows) > 0
                and sum(row["changed_elements"] for row in rows) > 0, "Active mechanism did not learn")
        values[name] = {"parameters": rows[0]["parameters"],
                        "peak_gradient_norm": max(row["gradient_norm"] for row in rows),
                        "changed_elements_across_steps": sum(row["changed_elements"] for row in rows)}
    return values


def audit_case(reg, case, out, gpu, *, real=False):
    device = "cuda:0" if gpu else "cpu"
    split = out / "split"
    opts, cpu_items, _, _, runtime, guard = bound(reg, case, split, device)
    target = reg["target"]
    group = reg.get("group", "factorial")
    item = cpu_items[target]
    inputs = {k: tensor_hash(item[k]) for k in ("net_input", "Q", "A", "train_indices", "train_labels", "val_indices", "val_labels")}
    model = runtime.build_model(ResearchConfig.from_dict(opts["config"]), {target: item["class_num"]}).to(device)
    init = model_snapshot(model, group)
    require("test_labels" not in item and opts["optimizer_steps"] == 12, "Foreign input role or budget")
    items = move_items(cpu_items, device)

    def calibrated(model):
        if group == "directions" and case["condition"] in local_control.LOCAL_MATCHES:
            calibrator, binding = local_control.calibrate, runtime.branch_local_binding
        elif group in EDGE_POINT_GROUPS:
            _, factory, binding_key = edge_weight_adapter(case["condition"])
            calibrator, binding = factory.calibrate, getattr(runtime, binding_key)
        elif group in ADAPTERS:
            _, factory, binding_key = ADAPTERS[group]
            calibrator, binding = factory.calibrate, getattr(runtime, binding_key)
        else:
            calibrator, binding = calibrate, runtime.position_binding
        return calibrator(model, observation(model, items[target]), expected_binding=binding,
                          source_state_sha256=state_id(model))[0]

    diagnostic = {}
    strict_replay = {}
    if real or group in CONTROLS or group in EDGE_POINT_GROUPS:
        before = state_id(model)
        candidate = calibrated(model)
        checked, probs = runtime.validation_snapshot(candidate, items)
        if real and reg.get("arithmetic"):
            # The production validation wrapper uses its own ordered FP32 mode.
            # Use the same calibrated candidate for a separate P4 forward.
            _, strict_probs = TargetRuntime.validation_snapshot(runtime, candidate, items)
            for head in HEADS:
                original = probs[target]["val"][head]
                replay = strict_probs[target]["val"][head]
                require(torch.allclose(replay, original, rtol=1e-5, atol=1e-6),
                        "Strict real head differs from ordered validation")
                strict_replay[head] = {
                    "max_abs_difference": float((replay - original).abs().max()),
                    "classification_changes": int((replay.argmax(1) != original.argmax(1)).sum()),
                }
        diagnostic = control_diagnostics(group, case, candidate, items[target], runtime, probs[target]["val"])
        require(state_id(model) == before, "Zero-update audit changed initial model")
        del candidate

    if real:
        guard()
        return {"case": case, "inputs": inputs, "init": init, "optimizer_updates": 0,
                **({"diagnostics": diagnostic} if diagnostic else {}),
                **({"strict_replay": strict_replay} if strict_replay else {}),
                "step_zero": validation_evidence(probs[target]["val"], item, checked["metrics"][target]["val"])}

    first, last = terminal(out / "whole"), terminal(split)
    fields = ("model", "optimizer", "scheduler", "history", "trace", "rng", "best_model", "best_probabilities")
    if reg.get("arithmetic"):
        one_a, one_b = (torch.load(out / route / "state-000001.pt", map_location="cpu", weights_only=True)
                        for route in ("whole", "split"))
        require(all(_tree_fingerprint(one_a[k]) == _tree_fingerprint(one_b[k]) for k in fields),
                "Independent first-step training states differed")
    require(all(_tree_fingerprint(first[k]) == _tree_fingerprint(last[k]) for k in fields),
            "Whole and split training states differed")
    minimum = min(last["history"], key=lambda row: row["val_loss"])
    selected_step = minimum["step"]
    require(selected_step == last["best_step"] == read(split / "result.json")["selected_step"],
            "First-min selection changed")
    changes = {}
    for head, prefix in HEADS.items():
        names = [k for k in last["model"] if k.startswith(prefix + ".")]
        norms = [row["gradient_norms"][head + "_head"] for row in last["trace"]]
        changed = sum(int(torch.count_nonzero(last["model"][k] != model.state_dict()[k].detach().cpu())) for k in names)
        require(len(norms) == 12 and all(np.isfinite(norms)) and max(norms) > 0 and changed > 0,
                "One of the three heads did not receive gradient and update")
        changes[head] = {"nonzero_gradient_steps": sum(v > 0 for v in norms), "changed_elements": changed}
    raw = split / "raw-validation-000012.pt"
    require(_tree_fingerprint(last["model"]) == _tree_fingerprint(torch.load(raw, weights_only=True, map_location="cpu")),
            "Final calibration used the wrong raw state")
    model.load_state_dict(last["model"], strict=True)
    final_model = calibrated(model)
    _, probs = runtime.validation_snapshot(final_model, items)
    point_options = {"keep_correctness": reg["scope"] in POINT_SCOPES,
                     "keep_points": reg["scope"] in EDGE_POINT_SCOPES}
    trained = {}
    if group in EDGE_POINT_GROUPS:
        trained["last"] = edge_weight_diagnostics(case, final_model, items[target], runtime)
    final = validation_evidence(probs[target]["val"], item, last["history"][-1]["metrics"][target]["val"],
                                **point_options)
    del final_model
    primary = runtime.load_selected(split / "primary.pt", items, split / f"raw-validation-{selected_step:06d}.pt", device=device)
    _, probs = runtime.validation_snapshot(primary, items)
    selected = validation_evidence(probs[target]["val"], item, minimum["metrics"][target]["val"], **point_options)
    if group in EDGE_POINT_GROUPS:
        trained["selected"] = edge_weight_diagnostics(case, primary, items[target], runtime)
    guard()
    costs = {phase: read(out / f"{phase}.json") for phase in STEPS}
    require([costs[phase]["updates"] for phase in STEPS] == list(STEPS.values()), "Wrong phase updates")
    activity = {}
    if reg["scope"] == FACTORIAL_SCOPE:
        trace = last["trace"]
        for name, active, gradients, elements in (
                ("residual_or_control", case["mode"] != "none",
                 [r.get("m4_parameter_gradient_norm", 0.) for r in trace],
                 [r.get("m4_changed_elements", 0) for r in trace]),
                ("position_gate", case["fusion"] == "position_gate",
                 [r["position_fusion"]["gradient_norm"] for r in trace],
                 [r["position_fusion"]["changed_elements"] for r in trace])):
            require(all(np.isfinite(gradients)) and (max(gradients) > 0) == active
                    and (sum(elements) > 0) == active, "Declared factorial mechanism did not match its update trace")
            activity[name] = {"active": active, "peak_gradient_norm": max(gradients), "changed_elements": sum(elements)}
    if group in EDGE_POINT_GROUPS:
        if case["condition"] == "row_normalized":
            require(all(row["prior_normalization"]["normalization_parameters"] == 0
                        and row["prior_normalization"]["changed_elements"] == 0 for row in last["trace"]),
                    "Parameter-free normalization acquired trainable state")
        else:
            activity = component_activity("edges", last["trace"])
    return {"case": case, "inputs": inputs, "init": init, "steps": {"selected": selected_step, "last": 12},
            "endpoints": {"selected": selected, "last": final}, "head_updates": changes, "phases": costs,
            "exact_whole_split": True, "resume_completed_updates": 0, "audit_updates": 0,
            "trace_proof": digest(last["trace"]),
            **({"learned_components": activity} if reg["scope"] == FACTORIAL_SCOPE else {}),
            **({"diagnostics": {"initial": diagnostic, **trained}, "learned_components": activity}
               if group in EDGE_POINT_GROUPS else {}),
            **({"diagnostics": diagnostic, "learned_components": component_activity(group, last["trace"])}
               if group in CONTROLS else {}),
            **({"exact_independent_step_one": True} if reg.get("arithmetic") else {})}


def paired_initialization(rows, group="factorial"):
    arms = (MESSAGE_CONDITIONS if group in (MESSAGE_GROUP, MESSAGE_SCOPE) else
            EDGE_WEIGHT_CONDITIONS if group in (EDGE_WEIGHT_GROUP, EDGE_WEIGHT_SCOPE) else DIRECTION_PARAMS if group == DIRECTION_SCOPE
            else REAL_PAIR if group == REAL_PAIR_SCOPE else CONTROLS.get(group, ARMS))
    for seed in SEEDS:
        cases = [row for row in rows if row["case"]["seed"] == seed]
        require({row["case"]["arm"] for row in cases} == set(arms)
                and len({row["init"]["shared"] for row in cases}) == 1
                and len({digest(row["inputs"]) for row in cases}) == 1, "Paired input or shared initialization differs")
        control = {r["case"]["arm"]: r for r in cases}
        by_arm = {r["case"]["arm"]: r["init"] for r in cases}
        if group in (MESSAGE_GROUP, MESSAGE_SCOPE):
            require({arm: init["parameters"] for arm, init in by_arm.items()} == MESSAGE_PARAMS
                    and by_arm["I8"] == by_arm["I9-message"],
                    "Joint message pair changed its complete initial tensors or capacity")
            endpoints = ("initial", "selected", "last") if group == MESSAGE_SCOPE else (None,)
            for endpoint in endpoints:
                diagnostics = [control[arm]["diagnostics"] for arm in arms]
                if endpoint is not None:
                    diagnostics = [d[endpoint] for d in diagnostics]
                require(len({(d["support_sha256"], d["prior_sha256"]) for d in diagnostics}) == 1,
                        "Joint message controls changed candidates or the prior")
            continue
        if group == EDGE_WEIGHT_SCOPE:
            require({arm: init["parameters"] for arm, init in by_arm.items()} == EDGE_WEIGHT_PARAMS
                    and not by_arm["off"]["additional_tensors"]
                    and set(by_arm["norm"]["additional_tensors"]) == {normalization.PREFIX + "protocol_version"},
                    "Edge controls changed parameter count or the normalization-only state")
            for endpoint in ("initial", "selected", "last"):
                signatures = {(d["support_sha256"], d["prior_sha256"])
                              for arm in ("norm", "Wg", "Wj")
                              for d in (control[arm]["diagnostics"][endpoint],)}
                require(len(signatures) == 1, "Learned arms used a different candidate support or normalized prior")
            continue
        if group in ("directions", REAL_PAIR_SCOPE, DIRECTION_SCOPE):
            def head(arm, name):
                return control[arm]["diagnostics"]["initial_head_sha256"][name]
            require(head("C2G", "cnn") == head("off", "cnn")
                    and (group == REAL_PAIR_SCOPE or head("G2C", "graph") == head("off", "graph")),
                    "A disabled message direction changed the non-receiving branch")
            if group == DIRECTION_SCOPE:
                require({arm: init["parameters"] for arm, init in by_arm.items()} == DIRECTION_PARAMS,
                        "Measured parameters differ from the registered direction matrix")
                for arm, capacity in MATCHES:
                    reference, local = by_arm[arm], by_arm[capacity]
                    require(reference["learned_parameters"] == local["learned_parameters"]
                            and reference["additional_tensors"] == {
                                n: v for n, v in local["additional_tensors"].items()
                                if n != interaction.PREFIX + "protocol_version"},
                            "Local control differs in trainable capacity or initial tensors")
                require(head("local-C2G", "cnn") == head("off", "cnn")
                        and head("local-G2C", "graph") == head("off", "graph"),
                        "Local control changed the untouched branch")
            continue
        if group == "edges":
            require(by_arm["I8"]["additional_tensors"] == by_arm["I9"]["additional_tensors"]
                    and by_arm["I8"]["learned_parameters"] == by_arm["I9"]["learned_parameters"],
                    "Regional/global joint comparison lost exact capacity and initialization")
            continue
        for base, treatment in (("B0", "F"), ("R", "R+F"), ("Bcap", "Bcap+F")):
            a, b = (by_arm[k] for k in (base, treatment))
            require(a["residual"] == b["residual"] and b["parameters"]-a["parameters"] == 129
                    and b["fusion"] == by_arm["F"]["fusion"], "Gate/state or extra parameter pairing differs")
        require(by_arm["R"]["parameters"] - by_arm["B0"]["parameters"] == 8256
                and by_arm["Bcap"]["parameters"] - by_arm["B0"]["parameters"]
                == (8256 if group == FACTORIAL_SCOPE else 8320), "Capacity control differs")
        if group == FACTORIAL_SCOPE:
            require({arm: init["parameters"] for arm, init in by_arm.items()} == FACTORIAL_PARAMS,
                    "Measured factorial arm capacity differs")
    require(len({digest(row["inputs"]) for row in rows}) == 1, "Comparison changed support across seeds")
    require(len({row["init"]["shared"] for row in rows}) == len(SEEDS), "Missing independent model seeds")


def make_tables(rows, group="factorial"):
    endpoints, per_class, paired, paired_class, costs = [], [], [], [], []
    selected = {}
    pairs = (("C2G", "off"),) if group == REAL_PAIR_SCOPE else CONTROL_CONTRASTS.get(group, ())
    if group == EDGE_WEIGHT_SCOPE:
        pairs = EDGE_WEIGHT_CONTRASTS
    if group == MESSAGE_SCOPE:
        pairs = MESSAGE_CONTRASTS
    if group == DIRECTION_SCOPE:
        pairs = (*MATCHES, *((arm, "off") for arm in DIRECTION_PARAMS if arm != "off"),
                 ("both", "add"), ("both", "C2G"), ("both", "G2C"), ("C2G", "G2C"))
    contrasts = CONTRASTS if group == "factorial" else {f"{a}-{b}": {a: 1, b: -1} for a, b in pairs}
    if group == FACTORIAL_SCOPE:
        contrasts = FACTORIAL_EFFECTS
    changes = []
    for row in rows:
        case = row["case"]
        for endpoint, heads in row["endpoints"].items():
            for name, score in heads.items():
                context = dict(seed=case["seed"], arm=case["arm"], endpoint=endpoint, head=name)
                selected[case["seed"], endpoint, name, case["arm"]] = score
                endpoints.append({**context, "step": row["steps"][endpoint], "oa": score["oa"], "aa": score["aa"], "nll": score["nll"],
                                  "parameters": row["init"]["parameters"]})
                per_class.extend({**context, **{k: label[k] for k in ("class_id", "correct", "support", "recall")}}
                                 for label in score["classes"])
        for phase, cost in row["phases"].items():
            costs.append(dict(seed=case["seed"], arm=case["arm"], phase=phase, updates=cost["updates"],
                              wall_seconds=cost["wall_seconds"], peak_allocated_bytes=cost.get("max_allocated_bytes"),
                              peak_reserved_bytes=cost.get("max_reserved_bytes"), peak_rss_bytes=cost.get("max_rss_bytes"),
                              hardware=cost["gpu"]["uuid"] if cost["gpu"] else "cpu"))
    for seed in SEEDS:
        for endpoint in ("selected", "last"):
            for head in HEADS:
                for comparison, design in contrasts.items():
                    context = dict(seed=seed, endpoint=endpoint, head=head, comparison=comparison)
                    arms = [(selected[seed, endpoint, head, arm], v) for arm, v in design.items()]
                    deltas = []
                    for classes in zip(*(a["classes"] for a, _ in arms), strict=True):
                        require(len({(c["class_id"], c["support"]) for c in classes}) == 1, "Different class denominators")
                        delta = sum(c["recall"]*sign for (_, sign), c in zip(arms, classes, strict=True))
                        deltas.append(delta)
                        paired_class.append(dict(**context, class_id=classes[0]["class_id"], denominator=classes[0]["support"],
                            correct_by_arm=json.dumps({arm: c["correct"] for arm, c in zip(design, classes, strict=True)}),
                            delta_recall=delta))
                    paired.append({**context, **{f"delta_{k}": sum(a[k]*w for a, w in arms) for k in ("oa", "aa", "nll")},
                                   "minimum_class_recall_difference": min(deltas), "negative_class_differences": sum(d < -1e-12 for d in deltas)})
                    if group in POINT_SCOPES and tuple(design.values()) == (1, -1):
                        a, b = (arm["correctness"] for arm, _ in arms)
                        for label, (now, old) in enumerate(zip(a, b, strict=True), 1):
                            transitions = list(zip(now, old, strict=True))
                            changes.append(dict(**context, class_id=label, support=len(transitions),
                                retained=sum(n and p for n, p in transitions), gained=sum(n and not p for n, p in transitions),
                                lost=sum(p and not n for n, p in transitions)))
    products = {"endpoints": endpoints, "per-class": per_class, "paired": paired, "paired-per-class": paired_class, "costs": costs}
    if group in POINT_SCOPES:
        products["correct-point-transitions"] = changes
    return products


def stage(args, code, reg, out):
    case = next(c for c in reg["cases"] if (c["seed"], c["arm"]) == (args.seed, args.arm))
    gpu, phase = args.gpu, args.phase
    check_real_lane(reg, args.seed, gpu, out.parent)
    p4.check_inherited_lease(args.matrix_fd, out.parent / ".matrix.lock")
    require(out == out.parent / f"{case['seed']}-{case['arm']}" and
            not (out / f"{phase}.json").exists(), "Wrong or already committed operation")
    numeric = reg.get("arithmetic")
    require(numeric is None or gpu is not None, "Strict arithmetic requires a registered physical GPU")
    calls = Counter()
    compute = (p4.arithmetic(numeric, gpu, kernel_calls=calls,
                              enforce_resource_thresholds=reg["scope"] not in REAL_SCOPES)
               if numeric else nullcontext(None))
    lease = physical_lease(gpu, inherited_fd=args.lease_fd) if gpu else nullcontext()
    with lease, compute as details:
        inventory = (p4.snapshot(gpu) if numeric else p4.gate(gpu)) if gpu else None
        device = "cuda:0" if gpu else "cpu"
        card = attest(gpu) if gpu else None
        if phase == "audit":
            audit = audit_case(reg, case, out, gpu)
            if numeric:
                audit["arithmetic"] = details
                audit["kernel_calls"] = dict(calls)
            write_json(out / "audit.json", audit)
            return
        if gpu:
            torch.cuda.reset_peak_memory_stats(0)
        started = perf_counter()
        target = out / ("whole" if phase == "whole" else "split")
        opts, items, provenance, _, runtime, guard = bound(reg, case, target, device)
        if numeric:
            provenance["registered_wp04_cuda_arithmetic"] = details
        with redirect_stdout(sys.stderr):
            result = run_training(opts, items, target, provenance, code, runtime_adapter=runtime, finish_guard=guard,
                                  device=device, resume=phase in ("resume", "completed"),
                                  stop_after=2 if phase == "prefix" else None)
        require(result["updates_this_call"] == STEPS[phase], "Different optimizer cursor")
        cost = hardware(gpu, started)
        if numeric:
            cost["arithmetic"] = details
            cost["kernel_calls"] = dict(calls)
            require(phase == "completed" or calls["sparse_mm"] > 0,
                    "Strict path did not intercept the sparse training operator")
        write_json(out / f"{phase}.json", {**cost, "updates": STEPS[phase], "gpu": card, "resources": inventory,
                   "torch": str(torch.__version__), "scope": "phase_wall_including_prepare_checkpoint_and_verification"})


def run(args, code, reg, out):
    gpu = args.gpu
    check_real_lane(reg, args.seed, gpu, out)
    out.mkdir(exist_ok=True)
    require(reg.get("arithmetic") is None or gpu, "Strict arithmetic requires a registered physical GPU")
    chosen = [c for c in reg["cases"] if (args.seed is None or c["seed"] == args.seed)
              and (args.arm is None or c["arm"] == args.arm)]
    require(chosen, "No selected registered case")
    lock = physical_lease(gpu) if gpu else nullcontext()
    with p4.lease(out / ".matrix.lock") as matrix_lock, lock as lease:
        owner = {"registration": args.registration, "gpu": gpu}
        path = out / "owner.json"
        if path.exists():
            require(read(path) == owner, "Output belongs to another matrix or device")
        else:
            write_json(path, owner)
        for case in chosen:
            directory = out / f"{case['seed']}-{case['arm']}"
            directory.mkdir(exist_ok=True)
            for phase in (*STEPS, "audit"):
                if (directory / f"{phase}.json").exists():
                    continue
                cmd = [sys.executable, "-B", __file__, "--expected-commit", code["commit"], "stage",
                       "--registration", *args.registration, "--out", str(directory), "--seed", str(case["seed"]),
                       "--arm", case["arm"], "--phase", phase, "--matrix-fd", str(matrix_lock.fileno())]
                fd = (matrix_lock.fileno(),)
                if gpu:
                    fd += (lease.fileno(),)
                    cmd += ["--gpu", gpu, "--lease-fd", str(lease.fileno())]
                started = perf_counter()
                with (directory / f"{phase}.log").open("ab") as log:
                    while True:
                        result = subprocess.run(cmd, check=False, pass_fds=fd, stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode == 0:
                            break
                        if result.returncode != 75 or perf_counter()-started >= args.resource_wait_seconds:
                            result.check_returncode()
                        sleep(5)
        if args.seed is not None:
            return
        rows = [read(out / f"{c['seed']}-{c['arm']}" / "audit.json") for c in reg["cases"]]
        group = reg.get("group", "factorial")
        paired_initialization(rows, group)
        products = make_tables(rows, group)
        if (out / "evidence.json").exists():
            previous = read(out / "evidence.json")
            require(previous["case_pins"] == [pinned(out / f"{c['seed']}-{c['arm']}" / "audit.json") for c in reg["cases"]],
                    "Existing audited rows differ")
            for pin in previous["tables"]:
                verify(pin)
            print(json.dumps({"status": "reused", "evidence": pinned(out / "evidence.json")}))
            return
        for name, data in products.items():
            with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        write_json(out / "evidence.json", {"schema": SCHEMA, "scope": reg["scope"], "device": gpu or "cpu",
                   "code": identity(code), "registration": dict(zip(("path", "sha256"), args.registration, strict=True)),
                   "tables": [pinned(out / f"{name}.csv") for name in products], "cases": len(rows),
                   "case_pins": [pinned(out / f"{c['seed']}-{c['arm']}" / "audit.json") for c in reg["cases"]],
                   "counts": {name: len(data) for name, data in products.items()}, "synthetic_updates": len(rows)*24,
                   "reporting_scope": "engineering_fixture_only", "real_updates": 0, "query_access": False,
                   **({"group": group} if group in CONTROLS else {}),
                   **({"arithmetic": reg["arithmetic"], "strict_qualification": True} if reg.get("arithmetic") else {})})
        print(json.dumps({"status": "passed", "evidence": pinned(out / "evidence.json")}))


def rebind(args, code, out):
    published = verify({"path": str(Path(args.manifest[0]).resolve()), "sha256": args.manifest[1]})
    require(args.manifest[1] == REAL_CATALOG_SHA, "Expected the independently admitted Xuzhou context")
    out.mkdir(exist_ok=False)
    original = read(published)
    validate_manifest(original, published.parent)
    checksum = original["input_spec_sha256"]
    sources = [path for path, sha in original["protected_files"].items() if sha == checksum]
    require(len(sources) == 1, "Missing original support-context input registration")
    rebound = out / "rebound-contexts"
    updated = prepare_contexts(sources[0], checksum, rebound, code)
    require(updated["observations"] == original["observations"]
            and updated["protected_files"] == original["protected_files"]
            and [{k: v for k, v in row.items() if k != "sha256"} for row in updated["contexts"]]
            == [{k: v for k, v in row.items() if k != "sha256"} for row in original["contexts"]],
            "Rebound support roles, graph or observations changed")
    static_proof = pinned(published)
    return rebound / "manifest.json", static_proof


def probe(args, code, out):
    catalog, static_proof = rebind(args, code, out)
    arithmetic = p4.CUDA_FP64_WEIGHT if args.strict_cuda else None
    require(arithmetic is None or args.gpu, "Strict real probe requires a registered physical GPU")
    reg = plan(catalog, code, "real_zero_update_draft", arithmetic=arithmetic, group=args.group)
    rows = []
    leases = physical_lease(args.gpu) if args.gpu else nullcontext()
    calls = Counter()
    compute = p4.arithmetic(arithmetic, args.gpu, kernel_calls=calls) if arithmetic else nullcontext(None)
    with leases, compute as numeric:
        if args.gpu:
            if not arithmetic:
                p4.gate(args.gpu)
            attest(args.gpu)
        for case in reg["cases"]:
            start = perf_counter()
            sparse_calls_before = calls["sparse_mm"]
            row = audit_case(reg, case, out / f"{case['seed']}-{case['arm']}", args.gpu, real=True)
            if arithmetic:
                row["strict_sparse_mm_calls"] = calls["sparse_mm"] - sparse_calls_before
                require(row["strict_sparse_mm_calls"] > 0, "Strict real replay did not call the P4 sparse operator")
            rows.append({**row, **hardware(args.gpu, start)})
    paired_initialization(rows, args.group)
    write_json(out / "registration-draft.json", {**reg, "candidate_updates_after_separate_release": len(rows)*12,
               "phase_updates": {key: STEPS[key] for key in ("prefix", "resume", "completed")},
               "published_catalog": static_proof,
               "execution_enabled": False, "approval": None})
    write_json(out / "evidence.json", {"schema": SCHEMA, "scope": reg["scope"], "code": identity(code),
               "cases": rows, "registration_draft": pinned(out / "registration-draft.json"),
               "published_catalog": static_proof, "rebound_contexts_verified": 27,
               "real_updates": 0, "query_access": False, "fixed_graph": args.group != "edges", "device": args.gpu or "cpu",
                   **({"group": args.group} if args.group in (*CONTROLS, MESSAGE_GROUP) else {}),
               **({"arithmetic": numeric, "kernel_calls": dict(calls)} if arithmetic else {})})
    verify(static_proof)


def prior_directions(reg, code):
    path = verify(reg["prior"])
    previous = read(path)
    old = read(verify(previous["registration"]))
    expected_contract = ("catalog", "published_catalog", "target", "support_seed", "per_class", "phase_updates",
                         "method", "graph", "selection", "arithmetic", "devices", "role_scope", "resource_policy")
    require(previous["scope"] == old["scope"] == REAL_PAIR_SCOPE
            and previous["code"] == old["code"] and old["code"]["commit"] == PRIOR_CODE
            and previous["real_updates"] == old["real_updates_authorized"] == 144
            and previous["research_updates"] == 72 and previous["query_access"] is False
            and all(old[k] == reg[k] for k in expected_contract if k != "catalog"),
            "Accepted I0/I3 contract differs")
    old_catalog = read(verify(old["catalog"]))
    new_catalog = read(verify(reg["catalog"]))
    for k in ("input_spec_sha256", "observations", "protected_files"):
        require(old_catalog[k] == new_catalog[k], "Historical data binding changed")
    subprocess.run(["git", "diff", "--exit-code", PRIOR_CODE, code["commit"], "--",
                    "model", "g4_interaction_target.py", "g4_interaction_target_runtime.py", "target_training.py",
                    "g4_support_context.py", "g4_support_target.py", "joint_training.py",
                    "transfer/p4_runtime_device.py", "transfer/shared_gpu_lease.py"], check=True)
    require(len(previous["case_pins"]) == len(SEEDS)*len(REAL_PAIR), "Historical matrix incomplete")
    rows = [read(verify(pin)) for pin in previous["case_pins"]]
    require([row["case"] for row in rows] == old["cases"]
            and all(row["exact_independent_step_one"] and row["exact_whole_split"]
                    and row["resume_completed_updates"] == row["audit_updates"] == 0
                    for row in rows), "Historical case receipt or restoration differs")
    return rows, previous["case_pins"]


def dispatch_real(args, code, reg, out):
    scope = reg["scope"]
    require(scope in REAL_SCOPES and out == Path(reg["execution"]),
            "Expected the registered paired development output")
    reused_rows, reused_pins = prior_directions(reg, code) if scope == DIRECTION_SCOPE else ([], [])
    out.mkdir(exist_ok=True)

    def lane(seed):
        gpu = reg["devices"][str(seed)]
        command = [sys.executable, "-B", __file__, "--expected-commit", code["commit"], "run",
                   "--registration", *args.registration, "--out", str(out / str(seed)), "--seed", str(seed),
                   "--gpu", gpu]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUBLAS_WORKSPACE_CONFIG": p4.WORKSPACE}
        with (out / f"{seed}.log").open("ab") as log:
            return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

    with p4.lease(out / ".dispatch.lock"):
        with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
            exits = dict(zip(map(str, SEEDS), pool.map(lane, SEEDS), strict=True))
        receipt = {"lane_exit_codes": exits, "devices": reg["devices"]}
        if any(exits.values()):
            failed = out / f"failed-{os.getpid()}.json"
            write_json(failed, receipt)
            raise RuntimeError(f"Incomplete paired run, see {failed} and each lane log")
        rows = []
        case_pins = []
        for case in reg["cases"]:
            path = out / str(case["seed"]) / f"{case['seed']}-{case['arm']}" / "audit.json"
            rows.append(read(path))
            case_pins.append(pinned(path))
        paired_initialization([*reused_rows, *rows], scope)
        actual_updates = sum(phase["updates"] for row in rows for phase in row["phases"].values())
        require(actual_updates == reg["real_updates_authorized"], "Real recovery budget differs")
        require_frozen_code(code)
        evidence = out / "evidence.json"
        if evidence.exists():
            previous = read(evidence)
            require(previous["case_pins"] == case_pins, "Paired case evidence changed")
            require(previous.get("reused_case_pins", []) == reused_pins, "Reused evidence changed")
            for pin in previous["tables"]:
                verify(pin)
        else:
            products = make_tables([*reused_rows, *rows], scope)
            for name, data in products.items():
                with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                    writer.writeheader()
                    writer.writerows(data)
            write_json(evidence, {**receipt, "code": identity(code), "scope": reg["scope"],
                       "registration": dict(zip(("path", "sha256"), args.registration, strict=True)),
                       "case_pins": case_pins, "tables": [pinned(out / f"{name}.csv") for name in products],
                       "real_updates": actual_updates, "research_updates": len(rows) * STEPS["whole"],
                       **({"reused_case_pins": reused_pins, "reused_updates": REAL_PAIR_UPDATES}
                          if scope == DIRECTION_SCOPE else {}),
                       "query_access": False})
        print(json.dumps(pinned(evidence)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    for kind in ("prepare", "run", "stage", "probe", "prepare-real-pairs", "dispatch-real-pairs"):
        selector = sub.add_parser(kind)
        selector.add_argument("--out", required=True, type=Path)
        if kind in ("run", "stage", "dispatch-real-pairs"):
            selector.add_argument("--registration", required=True, nargs=2)
        if kind in ("run", "stage"):
            selector.add_argument("--seed", type=int, choices=SEEDS, required=kind == "stage")
            selector.add_argument("--arm", choices=sorted(set(ARMS).union(DIRECTION_PARAMS, EDGE_WEIGHT_CONDITIONS,
                                                                   MESSAGE_CONDITIONS,
                                                                   *CONTROLS.values())),
                                  required=kind == "stage")
        if kind in ("run", "stage", "probe"):
            selector.add_argument("--gpu")
        if kind in ("prepare", "probe"):
            selector.add_argument("--strict-cuda", action="store_true")
            groups = ("factorial", *CONTROLS, MESSAGE_GROUP) if kind == "probe" else ("factorial", *CONTROLS)
            selector.add_argument("--group", choices=groups, default="factorial")
        if kind == "run":
            selector.add_argument("--resource-wait-seconds", type=float, default=0)
        if kind == "stage":
            selector.add_argument("--lease-fd", type=int)
            selector.add_argument("--matrix-fd", required=True, type=int)
            selector.add_argument("--phase", required=True, choices=(*STEPS, "audit"))
        if kind in ("probe", "prepare-real-pairs"):
            selector.add_argument("--manifest", nargs=2, required=True)
        if kind == "prepare-real-pairs":
            selector.add_argument("--devices", nargs=len(SEEDS), required=True)
            flavor = selector.add_mutually_exclusive_group()
            flavor.add_argument("--prior", nargs=2, metavar=("ACCEPTANCE", "SHA256"))
            flavor.add_argument("--factorial", action="store_true")
            flavor.add_argument("--edge-weights", action="store_true")
            flavor.add_argument("--messages", action="store_true")
    args = parser.parse_args()
    code = code_snapshot()
    require(sys.platform == "linux" and __debug__ and code["commit"] == args.expected_commit, "Expected Linux frozen revision")
    require_frozen_code(code)
    torch.set_num_threads(1)
    out = external_directory(args.out)
    if args.action == "prepare":
        from test_g4_support_context import prepared

        out.mkdir()
        if args.group == "edges":
            from test_g4_edge_target_runtime import fixture

            catalog, _, _ = fixture(out)
        else:
            _, _, directory, _ = prepared(out)
            catalog = directory / "manifest.json"
        path = out / "registration.json"
        write_json(path, plan(catalog, code, "synthetic_only", group=args.group,
                              arithmetic=p4.CUDA_FP64_WEIGHT if args.strict_cuda else None))
        print(json.dumps(pinned(path)))
    elif args.action == "probe":
        probe(args, code, out)
    elif args.action == "prepare-real-pairs":
        catalog, publication = rebind(args, code, out)
        path = out / "registration.json"
        prior = None if args.prior is None else {"path": str(Path(args.prior[0]).resolve()), "sha256": args.prior[1]}
        value = real_plan(catalog, publication, code, args.devices, prior=prior, factorial=args.factorial,
                          edge_weights=args.edge_weights, messages=args.messages)
        if args.edge_weights or args.messages:
            group = value["group"]
            for arm, condition in EDGE_POINT_GROUPS[group].items():
                _, factory, _ = edge_weight_adapter(condition)
                model, _ = factory.initialize(value["target"], value["method"], condition,
                    model_seed=SEEDS[0], head_seed=SEEDS[0]+1000)
                require(sum(p.numel() for p in model.parameters()) == value["parameters_by_arm"][arm],
                        "Pre-dispatch edge weight parameter count differs")
        if args.factorial:
            for arm, (mode, fusion) in ARMS.items():
                model, _ = initialize_position(value["target"], value["method"], fusion, mode=mode,
                    model_seed=SEEDS[0], head_seed=SEEDS[0]+1000, exact_capacity=mode == "capacity_control")
                require(sum(p.numel() for p in model.parameters()) == FACTORIAL_PARAMS[arm],
                        "Pre-dispatch residual/fusion capacity differs")
        if prior:
            prior_directions(value, code)
            for case in value["cases"]:
                if case["seed"] != SEEDS[0]:
                    continue
                condition = case["condition"]
                operator = local_control if condition in local_control.LOCAL_MATCHES else interaction
                model, _ = operator.initialize(value["target"], value["method"], condition,
                                               model_seed=SEEDS[0], head_seed=SEEDS[0]+1000)
                require(sum(p.numel() for p in model.parameters()) == DIRECTION_PARAMS[case["arm"]],
                        "Pre-dispatch capacity calculation differs")
        write_json(path, value)
        print(json.dumps(pinned(path)))
    elif args.action == "dispatch-real-pairs":
        dispatch_real(args, code, registration(args, code), out)
    else:
        value = registration(args, code)
        require(args.arm is None or args.seed is not None, "An arm requires its model seed")
        require(value["scope"] in REAL_SCOPES or (args.seed is None) == (args.arm is None),
                "Specify a synthetic seed and arm together")
        try:
            (stage if args.action == "stage" else run)(args, code, value, out)
        except p4.ResourceUnavailable as exc:
            print(json.dumps({"status": "waiting_for_resources", "reason": str(exc)}), file=sys.stderr, flush=True)
            raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
