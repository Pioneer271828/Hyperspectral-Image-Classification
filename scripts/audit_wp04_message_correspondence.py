"""Replay trained WP04 messages and the existing joint edge baseline without updates."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import g4_edge_target as edge
from model.region_pixel_alignment import RegionPixelAlignment, broadcast_regions, pool_regions
from model.research_config import ResearchConfig
from research_runtime import code_snapshot, external_directory, require_frozen_code, tensor_hash, write_json
from scripts.run_wp04_factorial import (
    EDGE_WEIGHT_SCOPE, HEADS, MESSAGE_GROUP, MESSAGE_SCOPE, SEEDS, bound, model_snapshot,
    move_items, pinned, read, require, terminal, verify,
)
from target_initialization import state_id
from target_training import observation
from transfer import p4_runtime_device as p4
from transfer.shared_gpu_lease import physical_lease


PRIOR_JOINT = "c4c166aa0c2b7c9ea41eaa1dde888d8e8da131feb87e117ca55b2bbbbb007159"
SHUFFLE_BASE = 23011


class ShuffledCorrespondence:
    def __init__(self, order):
        self.order = order
        self.inverse = order.argsort()

    def to_pixels(self, regions, assignment):
        return broadcast_regions(assignment, regions[self.order])

    def to_regions(self, pixels, assignment):
        return pool_regions(assignment, pixels)[self.inverse]


def scores(probabilities, labels):
    truth = labels.to(probabilities.device)
    actual = probabilities.gather(1, truth[:, None]).squeeze(1)
    other = probabilities.clone()
    other.scatter_(1, truth[:, None], float("-inf"))
    prediction = probabilities.argmax(1)
    margins = actual - other.amax(1)
    points = [
        {"class_id": c + 1, "point": ordinal, "predicted": int(prediction[i]) + 1,
         "true_probability": float(actual[i]), "margin": float(margins[i])}
        for c in range(9)
        for ordinal, i in enumerate((truth == c).nonzero(as_tuple=True)[0].tolist(), 1)
    ]
    return {"probability_sha256": tensor_hash(probabilities), "points": points}


def checked_endpoints(reg, pin, device):
    path = verify(pin)
    row = read(path)
    target = reg["target"]
    tree = path.parent / "split"
    opts, items, _, _, runtime, guard = bound(reg, row["case"], tree, device)
    input_hashes = {key: tensor_hash(items[target][key]) for key in row["inputs"]}
    require(input_hashes == row["inputs"], "Original train/development observation binding differs")
    inputs = move_items(items, device)
    steps = row["steps"]
    check_paths = (tree / "primary.pt", tree / "primary.pt.json", tree / "progress.json",
                   tree / f"raw-validation-{steps['selected']:06d}.pt", tree / "state-000012.pt")
    check_pins = [pinned(item) for item in check_paths]
    initial = runtime.build_model(ResearchConfig.from_dict(opts["config"]), {target: inputs[target]["class_num"]})
    require(model_snapshot(initial, reg["group"]) == row["init"], "Exact original initialization differs")
    selected = runtime.load_selected(check_paths[0], inputs, check_paths[3], device=device)
    initial.load_state_dict(terminal(tree)["model"], strict=True)
    last, _ = edge.calibrate(initial, observation(initial, inputs[target]), expected_binding=runtime.edge_binding,
                             source_state_sha256=state_id(initial))

    def evaluate(model):
        _, by_scene = runtime.validation_snapshot(model, inputs)
        return {head: scores(values, inputs[target]["val_labels"])
                for head, values in by_scene[target]["val"].items()}

    endpoints = {endpoint: evaluate(model) for endpoint, model in (("selected", selected), ("last", last))}
    for endpoint, heads in endpoints.items():
        for head in HEADS:
            saved = row["endpoints"][endpoint][head]
            require(heads[head]["probability_sha256"] == saved["probability_sha256"]
                    and heads[head]["points"] == saved["points"], "Archived endpoint probabilities differed")

    def verify_files():
        for file_pin in check_pins:
            verify(file_pin)
        verify(pin)
        guard()

    return row, selected, inputs[target], endpoints, evaluate, verify_files


def intervention(model, item, alignment, evaluate):
    backbone = model.backbone
    messages = backbone.interaction
    original = messages.forward_aligned
    calls = {"message": 0, "edge": 0}
    support = []
    pixel_features = []

    def exchange(pixels, regions, assignment, _alignment, **kwargs):
        calls["message"] += 1
        require(tensor_hash(assignment) == tensor_hash(item["Q"]), "Message assignment changed")
        result = original(pixels, regions, assignment, alignment, **kwargs)
        pixel_features.append(result[0])
        return result

    def edge_call(_module, args, output):
        calls["edge"] += 1
        _, adjacency, cnn = args
        require(tensor_hash(adjacency) == tensor_hash(item["A"]) and pixel_features
                and torch.equal(cnn, pool_regions(item["Q"], pixel_features.pop())),
                "Frozen intervention changed the candidate graph or edge alignment operator")
        support.append(tensor_hash(output[0].coalesce().indices()))

    hook = backbone.edge_updater.register_forward_hook(edge_call)
    try:
        with patch.object(messages, "forward_aligned", side_effect=exchange):
            result = evaluate(model)
    finally:
        hook.remove()
    require(calls["message"] == calls["edge"] > 0 and not pixel_features and len(set(support)) == 1,
            "Incomplete intervention observation")
    return {"heads": result, "calls": calls["edge"], "support_sha256": support[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--message", nargs=2, required=True)
    parser.add_argument("--prior-edge", nargs=2, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    out = external_directory(args.out)
    current_pin = dict(zip(("path", "sha256"), args.message, strict=True))
    prior_pin = dict(zip(("path", "sha256"), args.prior_edge, strict=True))
    require(not out.exists() and code["commit"] == args.expected_commit and prior_pin["sha256"] == PRIOR_JOINT,
            "Different diagnostic revision, prior or output")
    current, prior = (read(verify(pin)) for pin in (current_pin, prior_pin))
    current_reg, prior_reg = (read(verify(evidence["registration"])) for evidence in (current, prior))
    require(current_reg["group"] == MESSAGE_GROUP and current_reg["scope"] == current["scope"] == MESSAGE_SCOPE
            and prior_reg["scope"] == prior["scope"] == EDGE_WEIGHT_SCOPE
            and current["code"] == current_reg["code"] and prior["code"] == prior_reg["code"]
            and current["code"]["commit"] != code["commit"],
            "Wrong frozen development evidence")
    for key in ("published_catalog", "target", "support_seed", "per_class", "method", "phase_updates",
                "selection", "role_scope", "devices", "arithmetic"):
        require(current_reg[key] == prior_reg[key], f"Prior joint run differs in {key}")

    def case_pin(evidence, arm):
        matches = [pin for pin in evidence["case_pins"]
                   if Path(pin["path"]).parent.name == f"{args.seed}-{arm}"]
        require(len(matches) == 1, "Missing registered seed/arm")
        return matches[0]

    gpu = current_reg["devices"][str(args.seed)]
    device = "cuda:0"
    torch.set_num_threads(1)
    with physical_lease(gpu), p4.arithmetic(current_reg["arithmetic"], gpu, enforce_resource_thresholds=False), torch.no_grad():
        row, model, item, original, evaluate, guard = checked_endpoints(current_reg, case_pin(current, "I8"), device)
        before = state_id(model)
        generator = torch.Generator().manual_seed(SHUFFLE_BASE + args.seed)
        order = torch.randperm(item["Q"].shape[1], generator=generator).to(device)
        fixed_points = int((order == torch.arange(order.numel(), device=device)).sum())
        require(fixed_points < order.numel(), "The shuffle did not change correspondence")
        shuffled = intervention(model, item, ShuffledCorrespondence(order), evaluate)
        global_mean = intervention(model, item, RegionPixelAlignment("global"), evaluate)
        require(state_id(model) == before and evaluate(model) == original["selected"]
                and all(report["support_sha256"] == row["diagnostics"]["selected"]["support_sha256"]
                        for report in (shuffled, global_mean)),
                "Frozen model, support or baseline changed")
        guard()
        compatibility = {"I8": original}
        for arm in ("I9-message", "Wg65"):
            _, _, _, endpoints, _, check = checked_endpoints(current_reg, case_pin(current, arm), device)
            compatibility[arm] = endpoints
            check()
        prior_row, baseline, _, prior_endpoints, _, prior_guard = checked_endpoints(
            prior_reg, case_pin(prior, "Wj"), device)
        matched = read(verify(case_pin(current, "Wg65")))
        require(prior_row["init"]["parameters"] == matched["init"]["parameters"]
                and prior_row["init"]["shared"] == matched["init"]["shared"] and prior_row["inputs"] == matched["inputs"]
                and all(prior_row["diagnostics"][endpoint][key] == matched["diagnostics"][endpoint][key]
                        for endpoint in ("initial", "selected", "last") for key in ("support_sha256", "prior_sha256")),
                "The historical Wj and new Wg65 cannot form a matched pair")
        prior_guard()
        updater = baseline.backbone.edge_updater
        nodes = prior_row["diagnostics"]["selected"]["nodes"]
        width = updater.embed_dim
        features = updater.query.in_features
        pairs = prior_row["diagnostics"]["selected"]["actual_edges"]
    write_json(out, {"schema": "wp04_frozen_message_and_prior_joint_v1", "source": current_pin,
                     "prior_edge": prior_pin, "producer": code["commit"], "seed": args.seed, "device": gpu,
                     "updates": 0, "query_access": False, "selected_step": row["steps"]["selected"],
                     "shuffle_seed": SHUFFLE_BASE + args.seed, "permutation_sha256": tensor_hash(order),
                     "fixed_points": fixed_points, "regions": nodes, "original": original["selected"],
                     "shuffled": shuffled, "global_mean": global_mean, "adapter_compatibility": compatibility,
                     "prior_joint": {"endpoints": prior_endpoints, "case": case_pin(prior, "Wj"),
                                     "input_sha256": prior_row["inputs"], "parameters": prior_row["init"]["parameters"],
                                     "feature_channels": features, "embedding_channels": width,
                                     "projection_dense_macs": 2 * nodes * features * width, "edge_dot_macs": pairs * width}})


if __name__ == "__main__":
    main()
