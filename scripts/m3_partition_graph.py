"""Build layout-only Q candidates for the frozen native M3 counterfactual."""
import math

import numpy as np
import torch


def grid_labels(shape, grid):
    h, w = shape
    gh, gw = grid
    return ((np.arange(h, dtype=np.int64)[:, None] * gh // h) * gw
            + np.arange(w, dtype=np.int64)[None, :] * gw // w)


def contact_edges(labels, count):
    """Sorted directed eight-contact region edges, including every self-loop."""
    pairs = [np.arange(count, dtype=np.int64) * (count + 1)]
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[1:, :-1], labels[:-1, 1:])):
        pairs.extend(((a * count + b).ravel(), (b * count + a).ravel()))
    encoded = np.unique(np.concatenate(pairs))
    return np.stack((encoded // count, encoded % count))


def slic_graph(builder, features, labels, nominal_grid):
    """Use native learned projections with tile-local contact support for irregular regions."""
    from model.graph_state import GraphState, graph_diagnostics

    h, w = labels.shape
    count = int(labels.max()) + 1
    flat = torch.as_tensor(labels.ravel(), dtype=torch.long, device=features.device)
    ids = torch.arange(h * w, device=features.device)
    indices = torch.stack((ids, flat))
    q = torch.sparse_coo_tensor(indices, features.new_ones(h * w), (h * w, count)).coalesce()
    tokens, mass = builder._sparse_pool(q, builder.feature_norm(features.reshape(h * w, -1)))
    if torch.any(mass == 0):
        raise ValueError("SLIC assignment contains empty regions")
    centers = torch.stack(((ids // w + 0.5) / h, (ids % w + 0.5) / w), dim=1).to(features)
    positions = features.new_zeros((count, 2)).index_add_(0, flat, centers) / mass[:, None]
    edges = torch.as_tensor(contact_edges(labels, count), dtype=torch.long, device=features.device)
    receivers, senders = edges
    delta = (positions[receivers] - positions[senders]) * features.new_tensor(nominal_grid)
    logits = (builder.graph_query(tokens)[receivers] * builder.graph_key(tokens)[senders]).sum(1)
    logits = logits / math.sqrt(builder.assignment_dim) - delta.square().sum(1) / (2 * builder.graph_spatial_sigma**2)
    offset = logits.new_full((count,), -torch.inf).scatter_reduce_(0, receivers, logits, reduce="amax")
    scores = (logits - offset[receivers]).exp()
    weights = scores / scores.new_zeros(count).scatter_add_(0, receivers, scores)[receivers]
    a = torch.sparse_coo_tensor(edges, weights, (count, count)).coalesce()
    diagnostics = graph_diagnostics(q, a, assignment_mode="slic", adjacency_mode="learned_eight_contact",
                                    position_kind="slic_pixel_centroid", nominal_grid=nominal_grid)
    return GraphState(q, a, positions, (h, w), diagnostics=diagnostics).validate(features)
