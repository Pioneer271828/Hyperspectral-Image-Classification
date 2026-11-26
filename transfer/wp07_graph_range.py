"""Private WP07 geometry admission and frozen-Q, equal-capacity range controls."""
from contextlib import contextmanager
import math
import sys

import torch

from research_runtime import tensor_hash
from transfer import multiscale_region_graph as ranges
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_scale_observation import forward

ARMS = ("near", "wide", "mean", "content")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_checks():
    # Also guards imported historical audit utilities containing assertions.
    if sys.flags.optimize:
        raise RuntimeError("WP07 acceptance refuses optimized Python (-O/-OO)")


def initialize(baseline, seed):
    model = MultiscaleTrial(cnn_multiscale="fixed", cnn_dilation=1,
                            graph_multiscale="content").initialize(baseline, seed).eval()
    model.backbone.adapter.cache_attention = False
    return model


def identity(actual):
    r = actual.receipt
    return {k: r[k] for k in ("source", "role", "factor", "spectral", "coords", "grid_shape", "footprints")}


def sparse_digest(graph):
    graph = graph.detach().coalesce()
    return {"shape": list(graph.shape), "indices": tensor_hash(graph.indices()),
            "values": tensor_hash(graph.values()), "nnz": graph._nnz()}


def support_report(graph, grid, image_shape, factor):
    graph = graph.detach().cpu().coalesce()
    indices, values = graph.indices(), graph.values()
    require(torch.isfinite(values).all() and bool((values >= 0).all()), "Invalid graph weights")
    active = values > 0
    indices, values = indices[:, active], values[active]
    require(bool(active.any()), "Graph has no positive edges")
    gh, gw = grid
    dy = (indices[0] // gw - indices[1] // gw).abs()
    dx = (indices[0] % gw - indices[1] % gw).abs()
    hops = torch.maximum(dx, dy)
    distance = ((dy.double()*image_shape[0]*factor/gh).square()
                + (dx.double()*image_shape[1]*factor/gw).square()).sqrt()
    mass = torch.zeros(graph.shape[0]).scatter_add_(0, indices[0], values)
    require(bool(torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=0)), "Graph row mass differs")
    return {**sparse_digest(graph), "positive_edges": indices.tolist(), "positive_weights": values.tolist(),
            "row_mass_max_error": float((mass-1).abs().max()),
            "distance_kind": "nominal_grid_seed_centers_in_original_pixels_not_physical_GSD",
            "distance_min": float(distance.min()), "distance_max": float(distance.max()),
            "distance_mean": float(distance.mean()),
            "chebyshev_hop_counts": {str(h): int((hops == h).sum()) for h in torch.unique(hops).tolist()}}


def compare_supports(near, wide, *, require_distinct=False):
    a, b = near.detach().coalesce(), wide.detach().coalesce()
    require(a.shape == b.shape and a.shape[0] == a.shape[1], "Candidate dimensions differ")
    x = set(map(tuple, a.indices()[:, a.values() > 0].T.cpu().tolist()))
    y = set(map(tuple, b.indices()[:, b.values() > 0].T.cpu().tolist()))
    equivalent = x == y
    result = {"nodes": a.shape[0], "near_positive_nnz": len(x), "wide_positive_nnz": len(y),
              "near_subset_of_wide": x <= y, "added_edges": len(y-x), "removed_edges": len(x-y),
              "support_equivalent": equivalent,
              "weighted_equivalent": equivalent and sparse_digest(a) == sparse_digest(b),
              "identifiable": x < y}
    if require_distinct:
        require(result["identifiable"], "Equivalent graph supports cannot count as an effective range ablation")
    return result


class _Captured(Exception):
    pass


@torch.no_grad()
def capture_geometry(model, actual, producer, role, scale, *, condition=None):
    """Stop immediately after graph construction, before branch/head prediction."""
    require_checks()
    require(not model.training, "Geometry requires an initial evaluation model")
    builder = model.backbone.learnable_graph
    require("build_graph" not in vars(builder), "Graph builder already has an observer")
    captured = []

    # native forward's standard recorder expects to own build_graph, so bind
    # the observer at the _pair boundary instead of replacing that recorder.
    pair = builder._pair

    def capture_pair(build, tokens, grid):
        narrow = pair(build, tokens, grid)
        current = ranges._graph_forward.get()
        captured.append((narrow, current.wider_graph, tokens, grid))
        return narrow

    builder._pair = capture_pair
    # A forward hook on the graph builder's method is not an nn.Module hook;
    # stop at forward_stages, after Q exists and before any head is evaluated.
    stages = model.backbone.forward_stages

    def stop(cnn_input, region_input, Q, A, dataset_name=None, *args, **kwargs):
        require(len(captured) == 1, "Expected one Q and candidate pair")
        narrow, wide, _, grid = captured[0]
        require(sparse_digest(narrow) == sparse_digest(A), "Captured near adjacency differs")
        captured[0] = (Q, narrow, wide, grid)
        raise _Captured()

    model.backbone.forward_stages = stop
    try:
        try:
            forward(model, actual, producer, role, graph_scale=scale, condition=condition)
        except _Captured:
            pass
    finally:
        del builder._pair
        del model.backbone.forward_stages
    require(len(captured) == 1 and len(captured[0]) == 4, "Geometry capture did not stop before prediction")
    q, near, wide, grid = captured[0]
    require(q.shape[0] == math.prod(actual.lattice.shape), "Frozen Q has wrong native lattice")
    return {"q": q.detach().cpu().coalesce(), "near": near.detach().cpu().coalesce(),
            "wide": wide.detach().cpu().coalesce(), "grid": tuple(grid),
            "image_shape": actual.lattice.shape, "identity": identity(actual)}


@contextmanager
def frozen_support(model, entry, arm):
    """Keep both original message paths and parameter capacity for every arm."""
    require_checks()
    require(arm in ARMS, "Unknown graph range arm")
    builder = model.backbone.learnable_graph
    require(not any(name in vars(builder) for name in ("_build_assignment", "_build_graph")),
            "Frozen graph adapter already attached")
    target = next(model.parameters())
    q, a, b = (entry[k].to(device=target.device, dtype=target.dtype).coalesce() for k in ("q", "near", "wide"))
    grid = entry["grid"]

    def assignment(features, scale):
        require(tuple(features.shape[:2]) == tuple(entry["image_shape"]), "Frozen Q input shape differs")
        flat = builder.feature_norm(features.reshape(-1, features.shape[-1]))
        tokens, _ = builder._sparse_pool(q, flat)
        zero = features.new_zeros(())
        return q, tokens, {"grid_shape": grid, "assignment_nnz": q._nnz(), "compactness_loss": zero,
                           "assignment_entropy_loss": zero, "assignment_balance_loss": zero}

    def adjacency(tokens, actual_grid):
        require(tuple(actual_grid) == tuple(grid), "Frozen grid differs")
        current = ranges._graph_forward.get()
        require(current is not None and current.owner is model.backbone, "Range forward context differs")
        current.wider_graph = b
        learned = model.backbone.range_weights(tokens, current.observation)
        current.weights = (learned if arm == "content" else
                           tokens.new_tensor({"near": (1., 0.), "wide": (0., 1.), "mean": (.5, .5)}[arm]))
        return a

    builder._build_assignment, builder._build_graph = assignment, adjacency
    try:
        yield
    finally:
        del builder._build_assignment
        del builder._build_graph
