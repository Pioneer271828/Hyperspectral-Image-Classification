"""Optional regional cross-branch messages inspired by MiTA v6, section 3.2."""

from contextvars import ContextVar
import math

import torch
from torch import nn

from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.region_pixel_alignment import pool_regions


_block = ContextVar("region_message_block", default=None)
ATTENTION_MODES = ("local", "dense", "compress", "route", "combined")


class RegionMessageAttention(nn.Module):
    """Build shared compressed values and one routed set per query."""

    def __init__(self, channels, width, mode, landmark_grid, top_k, query_chunk):
        super().__init__()
        if mode not in ATTENTION_MODES or len(landmark_grid) != 2:
            raise ValueError("Expected a region attention mode and a 2D landmark grid")
        if any(type(value) is not int or value < 1 for value in (*landmark_grid, top_k, query_chunk, width)):
            raise ValueError("Landmark dimensions, top-k, query chunk and projection width must be positive integers")
        self.mode, self.landmark_grid = mode, tuple(landmark_grid)
        self.top_k, self.query_chunk = top_k, query_chunk
        self.query = nn.Linear(channels, width, bias=False)
        self.key = nn.Linear(channels, width, bias=False)
        self.value = nn.Linear(channels, width, bias=False)
        self.output = nn.Linear(width, channels, bias=False)
        self.scale = 1 / math.sqrt(width)

    def _landmarks(self, queries, positions):
        rows, cols = self.landmark_grid
        grid = positions.new_tensor((rows, cols))
        cells = (positions * grid).long().minimum(grid.long() - 1)
        occupied, members = torch.unique(cells[:, 0] * cols + cells[:, 1], sorted=True, return_inverse=True)
        totals = queries.new_zeros((len(occupied), queries.shape[1])).index_add(0, members, queries)
        counts = torch.bincount(members, minlength=len(occupied)).to(queries.dtype)
        return totals / counts[:, None]

    def _routed_attention(self, queries, keys, values, landmarks, expert_ids, compressed):
        messages = []
        for chunk in queries.split(self.query_chunk):
            landmark_logits = (chunk @ landmarks.T) * self.scale
            selected = expert_ids[landmark_logits.argmax(dim=1)]
            logits = (chunk[:, None] * keys[selected]).sum(dim=-1) * self.scale
            routed_values = values[selected]
            if compressed is not None:
                logits = torch.cat((landmark_logits, logits), dim=1)
                routed_values = torch.cat((compressed.expand(len(chunk), -1, -1), routed_values), dim=1)
            messages.append((logits.softmax(dim=1)[:, :, None] * routed_values).sum(dim=1))
        return torch.cat(messages)

    def forward(self, target, source, positions, valid):
        if target.ndim != 2 or target.shape != source.shape:
            raise ValueError("Expected matched region features")
        size = target.shape[0]
        if positions.shape != (size, 2) or valid.shape != (size,):
            raise ValueError("Expected matched region features, y/x positions and a region mask")
        if valid.dtype != torch.bool or not torch.isfinite(positions).all() or ((positions < 0) | (positions > 1)).any():
            raise ValueError("Expected a boolean region mask and finite normalized positions")
        ids = valid.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return target.new_zeros(target.shape)
        queries, keys, values = self.query(target[ids]), self.key(source[ids]), self.value(source[ids])
        if self.mode == "local":
            messages = (queries + keys).tanh() * values
        elif self.mode == "dense":
            messages = torch.cat([((q @ keys.T) * self.scale).softmax(dim=1) @ values
                                  for q in queries.split(self.query_chunk)])
        else:
            landmarks = self._landmarks(queries, positions[ids])
            probe = (landmarks @ keys.T) * self.scale
            compressed = probe.softmax(dim=1) @ values if self.mode != "route" else None
            if self.mode == "compress":
                messages = torch.cat([((q @ landmarks.T) * self.scale).softmax(dim=1) @ compressed
                                      for q in queries.split(self.query_chunk)])
            else:
                # Equal scores retain the original ascending source-region order.
                experts = probe.argsort(dim=1, descending=True, stable=True)[:, :min(self.top_k, ids.numel())]
                messages = self._routed_attention(queries, keys, values, landmarks, experts, compressed)
        return target.new_zeros(target.shape).index_copy(0, ids, self.output(messages).to(target.dtype))


def _region_geometry(assignment, shape):
    height, width = shape
    if assignment.shape[0] != height * width:
        raise ValueError("Assignment must belong to the current image block")
    with torch.no_grad():
        ids = torch.arange(height * width, device=assignment.device)
        positions = torch.stack(((ids // width + .5) / height, (ids % width + .5) / width), dim=1).to(assignment.dtype)
        occupied = pool_regions(assignment, torch.cat((positions, positions.new_ones((len(ids), 1))), dim=1))
    return occupied[:, :2], occupied[:, 2] > 0


class _RegionAttentionExchange(nn.Module):
    """Replace aligned source messages while keeping the original residual gates."""

    def __init__(self, exchange, **attention):
        super().__init__()
        self.exchange = exchange
        self.to_pixels = RegionMessageAttention(**attention) if exchange.graph_to_cnn is not None else None
        self.to_regions = RegionMessageAttention(**attention) if exchange.cnn_to_graph is not None else None

    def forward_aligned(self, pixels, regions, assignment, alignment, collect_stats=False, return_diagnostics=False):
        bound = _block.get()
        if bound is None or bound[0] is not self or alignment.mode != "region":
            raise ValueError("Region messages require the current image block and Q alignment")
        positions, valid = _region_geometry(assignment, bound[1])
        pixels_pooled = alignment.to_regions(pixels, assignment)
        to_pixels = None
        if self.to_pixels is not None:
            to_pixels = alignment.to_pixels(self.to_pixels(pixels_pooled, regions, positions, valid), assignment)
        to_regions = None
        if self.to_regions is not None:
            to_regions = self.to_regions(regions, pixels_pooled, positions, valid)
        updated_pixels, updated_regions, info = self.exchange(
            pixels, regions, to_pixels, to_regions, collect_stats=collect_stats, return_diagnostics=return_diagnostics,
        )
        return updated_pixels, torch.where(valid[:, None], updated_regions, regions), info


class RegionMessageVCASAHGCN(MultiDatasetVCASAHGCN):
    """Run region-message controls at the existing middle exchange point."""

    def __init__(self, class_counts, *, region_attention=None, **backbone_kwargs):
        super().__init__(class_counts, **backbone_kwargs)
        self.region_attention = None
        if region_attention is None:
            return
        if self.backbone.interaction is None or self.backbone.alignment.mode != "region":
            raise ValueError("Region attention requires an enabled exchange with region alignment")
        if self.backbone.edge_updater is not None or self.backbone.graph_router is not None:
            raise ValueError("Test region messages separately from graph support changes")
        if set(region_attention) != {"mode", "width", "landmark_grid", "top_k", "query_chunk"}:
            raise ValueError("Explicit region attention settings required")
        self.region_attention = {**region_attention, "landmark_grid": tuple(region_attention["landmark_grid"])}
        with torch.random.fork_rng(devices=[]):
            self.backbone.interaction = _RegionAttentionExchange(self.backbone.interaction, channels=128,
                                                                  **self.region_attention)

    def export_config(self):
        config = super().export_config()
        if self.region_attention is not None:
            config["backbone_kwargs"]["region_attention"] = dict(self.region_attention)
        return config

    def forward(self, dataset_name, x, *args, **kwargs):
        if self.region_attention is None:
            return super().forward(dataset_name, x, *args, **kwargs)
        token = _block.set((self.backbone.interaction, tuple(x.shape[:2])))
        try:
            return super().forward(dataset_name, x, *args, **kwargs)
        finally:
            _block.reset(token)
