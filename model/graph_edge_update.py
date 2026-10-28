"""Feature-conditioned sparse edge reweighting on an existing candidate graph."""

import math

import torch
from torch import nn


class GraphEdgeUpdater(nn.Module):
    """graph uses region features only; joint additionally uses aligned CNN features.

    Q and candidate connectivity stay unchanged during this forward pass.
    Isolated rows receive a self-loop so every row remains a probability vector.
    No dense region-by-region similarity matrix is formed.
    """

    def __init__(self, channels=128, mode="joint", embed_dim=32, mix=0.5):
        super().__init__()
        if mode not in ("graph", "joint"):
            raise ValueError("graph_update_mode must be graph or joint")
        if channels <= 0 or embed_dim <= 0 or not 0 <= mix <= 1:
            raise ValueError("Invalid graph update dimensions or mix")
        self.mode, self.mix, self.embed_dim = mode, float(mix), embed_dim
        input_dim = channels * (2 if mode == "joint" else 1)
        self.norm = nn.LayerNorm(input_dim)
        self.query = nn.Linear(input_dim, embed_dim, bias=False)
        self.key = nn.Linear(input_dim, embed_dim, bias=False)

    def forward(self, regions, adjacency, aligned_pixels=None, collect_stats=False,
                return_diagnostics=False):
        count = regions.shape[0]
        if tuple(adjacency.shape) != (count, count):
            raise ValueError("Graph node count must match region features")
        if self.mode == "joint":
            if aligned_pixels is None or aligned_pixels.shape != regions.shape:
                raise ValueError("joint edge update requires aligned CNN region features")
            features = torch.cat((regions, aligned_pixels), dim=-1)
        else:
            features = regions
        graph = adjacency.coalesce() if adjacency.is_sparse else adjacency.to_sparse().coalesce()
        values = graph.values()
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Edge update expects finite, nonnegative adjacency weights")
        positive = values > 0
        indices, values = graph.indices()[:, positive], values[positive]
        mass = values.new_zeros(count).scatter_add(0, indices[0], values)
        isolated = (mass == 0).nonzero(as_tuple=True)[0]
        if isolated.numel():
            indices = torch.cat((indices, torch.stack((isolated, isolated))), dim=1)
            values = torch.cat((values, values.new_ones(isolated.numel())))
        prior = torch.sparse_coo_tensor(indices, values, (count, count)).coalesce()
        indices, values = prior.indices(), prior.values()
        rows, cols = indices
        mass = values.new_zeros(count).scatter_add(0, rows, values)
        prior_weights = values / mass[rows].clamp_min(1e-12)

        features = self.norm(features)
        query, key = self.query(features), self.key(features)
        logits = (query[rows] * key[cols]).sum(-1) / math.sqrt(self.embed_dim)
        logits = logits + prior_weights.clamp_min(1e-12).log()
        row_max = logits.new_full((count,), -float("inf"))
        row_max.scatter_reduce_(0, rows, logits.detach(), reduce="amax", include_self=True)
        exp = (logits - row_max[rows]).exp()
        denominator = exp.new_zeros(count).scatter_add(0, rows, exp)
        proposal = exp / denominator[rows].clamp_min(1e-12)
        updated = (1.0 - self.mix) * prior_weights + self.mix * proposal
        result = torch.sparse_coo_tensor(indices, updated, (count, count)).coalesce()
        info = {}
        if collect_stats:
            info["edge_update_l1"] = (updated - prior_weights).detach().abs().mean()
            info["edge_update_max"] = (updated - prior_weights).detach().abs().max()
            info["edge_count"] = int(updated.numel())
        if return_diagnostics:
            info.update(edge_indices=indices.detach(), edge_weights_before=prior_weights.detach(),
                        edge_weights_after=updated.detach())
        return result, info
