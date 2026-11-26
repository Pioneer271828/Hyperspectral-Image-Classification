"""Read-only stage-two edge deletion on the certified PaviaU development release."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

UPSTREAM = "aba0525e4ef870bb594c77411224c684cd043674"
NATIVE = "395caef57fba8ab6a09223538f748179dab552f2"
APPROVED = "67e0a881412da6a86e10ca80617f023ac709ff2dc06fad969dbf154a17d34927"
SEEDS = (64, 512, 2048)
HEADS = ("fused", "graph", "cnn")
ARMS = ("none", "important", "low_weight", *(f"random:{seed}" for seed in SEEDS))


def clean(path, expected):
    commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain"], text=True)
    if commit != expected or dirty:
        raise ValueError(f"Expected a clean checkout at {expected}: {path}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distances(indices, grid, shape):
    """Exact squared seed-center distances with denominator (grid_h * grid_w)**2."""
    gh, gw = grid
    h, w = shape
    a, b = indices
    return (((a // gw - b // gw) * h * gw) ** 2 + ((a % gw - b % gw) * w * gh) ** 2)


def matching(graph, grid, shape, tile):
    from evaluation.matched_edge_controls import freeze, verify
    from evaluation.matched_edge_review import coordinates

    started = perf_counter()
    graph = graph.coalesce()
    n = graph.shape[0]
    ix = graph.indices().numpy()
    weights = graph.values().numpy()
    require(n == grid[0] * grid[1], "Native graph does not match its fixed seed grid")
    exact = distances(ix, grid, shape)
    values = sorted({int(x) for x in exact if x > 0})
    require(values, "A nonself support is required by this deletion protocol")
    lengths = [float(np.sqrt(v)) / (grid[0] * grid[1]) for v in values]
    bins = [(a + b) / 2 for a, b in zip(lengths, lengths[1:])] or [lengths[0] * 1.5]
    xy = coordinates({"grid_shape": grid, "shape": shape, "node_order": "row_major_seed_grid_v1",
                      "node_position": "grid_seed_centers"})
    strict = freeze(graph, xy, per_row=1, distance_bins=bins, seeds=SEEDS, time_limit=5.0)
    verify(strict, graph, xy, expected_hash=strict["plan_hash"])
    encoded = {int(r * n + c): j for j, (r, c) in enumerate(zip(*ix))}
    important = strict["baseline"]["removed"]
    rows = sorted({e // n for e in important})
    choices = {row: np.flatnonzero((ix[0] == row) & (ix[1] != row)) for row in rows}
    support = sorted(encoded)

    def summary(edges):
        use = [encoded[e] for e in edges]
        return {
            "count": len(edges),
            "receiver_degree": np.bincount(ix[0, use], minlength=n).tolist(),
            "sender_degree": np.bincount(ix[1, use], minlength=n).tolist(),
            "distance_histogram": {str(k): v for k, v in sorted(Counter(exact[use].tolist()).items())},
            "deleted_weight": float(weights[use].astype(np.float64).sum()),
        }

    target = summary(important)
    controls = {}
    for arm in ARMS:
        started_arm = perf_counter()
        if arm in ("none", "important"):
            mask = [] if arm == "none" else important
            match_status, solver_status = "reference", "not_run"
            reason = None
        else:
            row = strict["controls"][arm]
            solver_status = row["status"]
            reason = row.get("reason")
            if row["status"] == "optimal":
                mask, match_status = row["removed"], "strict"
            else:
                if arm == "low_weight":
                    pick = [min(choices[r], key=lambda j: (weights[j], ix[1, j])) for r in rows]
                else:
                    seed = int(arm.split(":")[1])
                    rng = np.random.default_rng(np.random.SeedSequence([seed, tile]))
                    pick = [int(rng.choice(choices[r])) for r in rows]
                mask = sorted(int(ix[0, j] * n + ix[1, j]) for j in pick)
                match_status = "unmatched_same_mask" if mask == important else "unmatched_diagnostic"
        actual = summary(mask)
        receiving = sum(abs(a - b) for a, b in zip(actual["receiver_degree"], target["receiver_degree"], strict=True))
        sending = sum(abs(a - b) for a, b in zip(actual["sender_degree"], target["sender_degree"], strict=True))
        a, b = actual["distance_histogram"], target["distance_histogram"]
        difference = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in a.keys() | b.keys())
        bound = dict(count_delta=actual["count"] - target["count"], receiver_l1=receiving,
                     sender_l1=sending, distance_exact_l1=difference)
        if arm != "none":
            require(bound["count_delta"] == receiving == 0, "Deletion cardinality or receiving degree differs")
        if match_status == "strict":
            require(sending == difference == 0 and mask != important, "Alternative fails exact distances or degrees")
        controls[arm] = dict(removed=mask, status=match_status, solver=solver_status, reason=reason,
                             geometry=actual, difference=bound, prepare_seconds=perf_counter() - started_arm,
                             graph_after=[edge for edge in support if edge not in set(mask)])
    return {"controls": controls, "support": support, "weights": weights.tolist(), "n": n,
            "grid": grid, "shape": shape, "strict_plan_hash": strict["plan_hash"],
            "matching_seconds": perf_counter() - started, "solver_seconds": strict["elapsed_seconds"]}


def forward(model, block, mask=None, expected=None):
    from evaluation.graph_edge_intervention import delete_support
    from loadData.spatial_protocol import forward_partition
    from research_runtime import tensor_hash

    back = model.backbone
    builder = back.learnable_graph
    build, stage, mm = builder.build_graph, back.graph_stage2.forward, torch.sparse.mm
    capture, execution = {}, []

    def build_graph(*args, **kwargs):
        start = perf_counter()
        state = build(*args, **kwargs)
        capture["build_seconds"] = perf_counter() - start
        capture["state"] = state
        return state

    def sparse_mm(lhs, rhs, *args, **kwargs):
        start = perf_counter()
        result = mm(lhs, rhs, *args, **kwargs)
        execution.append({"seconds": perf_counter() - start, "nnz": lhs._nnz(),
                          "shape": list(lhs.shape), "rhs_width": rhs.shape[1]})
        return result

    def stage2(features, adjacency, dataset_name=None):
        require("adjacency" not in capture and adjacency.layout == torch.sparse_coo,
                "Expected exactly one native sparse graph-stage-two call")
        capture["adjacency"] = adjacency.coalesce()
        start = perf_counter()
        actual = delete_support(capture["adjacency"], mask or [])
        capture["deletion_build_seconds"] = perf_counter() - start
        capture["actual"] = actual
        with patch.object(torch.sparse, "mm", side_effect=sparse_mm):
            return stage(features, actual, dataset_name)

    start = perf_counter()
    with (torch.inference_mode(), patch.object(builder, "build_graph", side_effect=build_graph),
          patch.object(back.graph_stage2, "forward", side_effect=stage2)):
        outputs = forward_partition(model, "paviaU", block["observation"], block["bands"], 50,
                                    wavelength_unit="nm")[:3]
    elapsed = perf_counter() - start
    graph = capture["state"]
    q_hash, a_hash = tensor_hash(graph.Q), tensor_hash(capture["adjacency"])
    if expected:
        require((q_hash, a_hash) == expected, "Original Q or stage-two graph changed")
    actual = capture["actual"]
    require(execution and all(row["nnz"] == actual._nnz() and row["shape"] == list(actual.shape)
                              for row in execution), "Actual square sparse multiply did not use the deletion graph")
    require(all(v.shape == (28 * 28, 9) and torch.isfinite(v).all() for v in outputs),
            "Unexpected three-head output")
    return outputs, graph, capture["adjacency"], actual, {
        "q_sha256": q_hash, "a_before_sha256": a_hash, "a_after_sha256": tensor_hash(actual),
        "forward_seconds": elapsed, "native_build_seconds": capture["build_seconds"],
        "deletion_build_seconds": capture["deletion_build_seconds"], "sparse_mm": execution,
        "actual_nnz": actual._nnz(),
    }


def execute(args, upstream):
    from model.VCASAHGCN import MultiDatasetVCASAHGCN
    from transactions import model_hash, sha

    approved = upstream.read(upstream.bind(args.registration, APPROVED))
    live_registration, live, capsule = upstream.material(args.base, args.analysis, UPSTREAM)
    require(live_registration == approved, "Released PaviaU geometry, points or checkpoint changed")
    by_unit = defaultdict(list)
    for point in approved["points"]:
        by_unit[point["unit"]].append(point)
    start = perf_counter()
    records = []
    for index, seed in enumerate(SEEDS):
        checkpoint = approved["checkpoints"][str(seed)]
        state = torch.load(upstream.bind(capsule / "objects" / checkpoint["sha256"], checkpoint["sha256"]),
                           weights_only=True, map_location="cpu")
        with torch.random.fork_rng(devices=[]):
            model = MultiDatasetVCASAHGCN({"paviaU": 9}, **upstream.MODEL).eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)
        frozen, rng = model_hash(model), torch.get_rng_state().clone()
        for unit, block in live.items():
            original, native_graph, baseline, _, original_cost = forward(model, block)
            require(np.array_equal(upstream.raster(native_graph.Q, 28, 28), block["maps"][index]),
                    "Original region map differs from certified geometry")
            grid, shape = list(native_graph.grid_shape), list(block["observation"].spectral.shape[:2])
            plan = matching(baseline, grid, shape, int(unit[1:]))
            reference = (original_cost["q_sha256"], original_cost["a_before_sha256"])
            arm_data = {}
            points = by_unit[unit]
            for arm, control in plan["controls"].items():
                if arm == "none":
                    out, actual, cost = original, baseline, original_cost
                else:
                    out, _, _, actual, cost = forward(model, block, control["removed"], reference)
                idx = actual.indices()
                ids = (idx[0] * plan["n"] + idx[1]).tolist()
                require(ids == control["graph_after"] and torch.equal(out[2], original[2]),
                        "Deletion changed CNN or used a different topology")
                rows = []
                for point in points:
                    offset = point["y"] * shape[1] + point["x"]
                    probs = [v[offset].tolist() for v in out]
                    if arm == "none":
                        previous = point["prior"][str(seed)]
                        observed = [int(np.argmax(v)) + 1 for v in probs]
                        require(observed == [previous[k] for k in ("left", "native_graph_pred", "native_cnn_pred")],
                                "Original released prediction changed")
                    rows.append({"point": point["point"], "label": point["label"], "probabilities": probs})
                arm_data[arm] = {**control, "points": rows, **cost}
            records.append({"seed": seed, "unit": unit, "step": checkpoint["step"],
                            "checkpoint_sha256": checkpoint["sha256"],
                            **{k: v for k, v in plan.items() if k != "controls"}, "arms": arm_data})
        require(frozen == model_hash(model) and torch.equal(rng, torch.get_rng_state()),
                "Frozen model or Torch random state changed")
    require(len(records) == 51 and Counter(r["seed"] for r in records) == dict.fromkeys(SEEDS, 17),
            "Incomplete released validation population")
    return {"policy": "wp06_wp08_pavia_stage2_matched_deletion_v1", "scene": "paviaU", "role": "devval",
            "commit": args.expected_commit, "upstream": UPSTREAM, "native": NATIVE,
            "registration_sha256": sha(args.registration), "query_accessed": False, "optimizer_updates": 0,
            "heads": HEADS, "arms": ARMS, "records": records, "wall_seconds": perf_counter() - start,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "analysis", "native", "upstream", "registration", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    clean(root, args.expected_commit)
    clean(args.native, NATIVE)
    clean(args.upstream, UPSTREAM)
    require(not args.out.resolve().is_relative_to(root) and not args.out.exists(),
            "Choose a fresh private output outside the checkout")
    require("model.VCASAHGCN" not in sys.modules and not torch.cuda.is_initialized(),
            "Isolated CPU native import required")
    sys.path.insert(0, str(args.upstream.resolve() / "research" / "paper_c_recovery"))
    sys.path.insert(0, str(args.native.resolve()))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import region_intervention as upstream

    with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Frozen inference only")):
        output = execute(args, upstream)
    clean(root, args.expected_commit)
    clean(args.native, NATIVE)
    clean(args.upstream, UPSTREAM)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
