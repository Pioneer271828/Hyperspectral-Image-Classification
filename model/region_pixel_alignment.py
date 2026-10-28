"""Sparse pixel/region alignment, separate from message selection and graph updates."""

import torch
from torch import nn


def pool_regions(assignment, pixels):
    """Column-normalized Q.T @ pixels; empty regions produce zeros, not NaNs."""
    if assignment.ndim != 2 or pixels.ndim != 2 or assignment.shape[0] != pixels.shape[0]:
        raise ValueError("Q must be [pixels, regions] and features must be [pixels, channels]")
    if assignment.is_sparse:
        assignment = assignment.coalesce()
        rows, cols = assignment.indices()
        values = assignment.values()
        mass = values.new_zeros(assignment.shape[1]).scatter_add(0, cols, values)
        normalized = torch.sparse_coo_tensor(
            assignment.indices(), values / mass[cols].clamp_min(1e-6),
            assignment.shape, device=values.device, dtype=values.dtype,
        ).coalesce()
        return torch.sparse.mm(normalized.transpose(0, 1), pixels)
    normalized = assignment / assignment.sum(0, keepdim=True).clamp_min(1e-6)
    return normalized.t().matmul(pixels)


def broadcast_regions(assignment, regions):
    """Use the original row assignment weights to map regions back to pixels."""
    if assignment.ndim != 2 or regions.ndim != 2 or assignment.shape[1] != regions.shape[0]:
        raise ValueError("Q must be [pixels, regions] and features must be [regions, channels]")
    if assignment.is_sparse:
        return torch.sparse.mm(assignment.coalesce(), regions)
    return assignment.matmul(regions)


class RegionPixelAlignment(nn.Module):
    """region: Q-aware correspondence; global: correspondence-free ablation.

    The global control replaces ONLY cross-branch messages by global means.
    Pooling/broadcasting inside the graph backbone always uses its original Q.
    """

    def __init__(self, mode="region"):
        super().__init__()
        if mode not in ("region", "global"):
            raise ValueError("alignment_mode must be region or global")
        self.mode = mode

    def to_regions(self, pixels, assignment):
        if self.mode == "global":
            return pixels.mean(0, keepdim=True).expand(assignment.shape[1], -1)
        return pool_regions(assignment, pixels)

    def to_pixels(self, regions, assignment):
        if self.mode == "global":
            return regions.mean(0, keepdim=True).expand(assignment.shape[0], -1)
        return broadcast_regions(assignment, regions)

    def forward(self, pixels, regions, assignment):
        return self.to_pixels(regions, assignment), self.to_regions(pixels, assignment)
