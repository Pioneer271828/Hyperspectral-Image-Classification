"""Two neighborhood ranges over one learned pixel-to-region assignment."""

from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass

import torch
from torch import nn

from model.VCASAHGCN import VCASAHGCNBackbone
from model.learnable_superpixel_graph import LearnableSuperpixelGraph
from model.region_pixel_alignment import broadcast_regions


GRAPH_RADII = (1, 2)
_graph_forward = ContextVar("multiscale_graph_forward", default=None)


@dataclass
class _GraphForward:
    owner: object
    observation: tuple
    wider_graph: torch.Tensor | None = None
    weights: torch.Tensor | None = None


class _TwoRadiusGraph(LearnableSuperpixelGraph):
    """Build both supports from the same region tokens and edge scorer."""

    def __init__(self, source):
        nn.Module.__init__(self)
        self.feature_dim = source.feature_dim
        self.assignment_dim = source.assignment_dim
        self.assignment_temperature = source.assignment_temperature
        self.assignment_spatial_sigma = source.assignment_spatial_sigma
        self.graph_spatial_sigma = source.graph_spatial_sigma
        self.feature_norm = source.feature_norm
        self.pixel_query = source.pixel_query
        self.seed_key = source.seed_key
        self.graph_query = source.graph_query
        self.graph_key = source.graph_key
        self.register_buffer("assignment_offsets", source.assignment_offsets, persistent=False)
        self.register_buffer("graph_offsets", self._make_offsets(1), persistent=False)
        self.register_buffer("wider_offsets", self._make_offsets(2), persistent=False)

    def _pair(self, build, tokens, grid_shape):
        current = _graph_forward.get()
        if current is None or current.owner.learnable_graph is not self:
            raise RuntimeError("Run the graph through the multiscale model forward")
        narrow = build(self, tokens, grid_shape)
        # Reuse the original sparse scoring and normalization with another
        # candidate list. Only the local view's buffer mapping is replaced.
        wider_view = copy(self)
        wider_view._buffers = {**self._buffers, "graph_offsets": self.wider_offsets}
        current.wider_graph = build(wider_view, tokens, grid_shape)
        current.weights = current.owner.range_weights(tokens, current.observation)
        return narrow

    def _build_graph(self, tokens, grid_shape):
        return self._pair(LearnableSuperpixelGraph._build_graph, tokens, grid_shape)

    def _fixed_graph(self, tokens, grid_shape):
        return self._pair(LearnableSuperpixelGraph._fixed_graph, tokens, grid_shape)


class MultiRangeBackbone(VCASAHGCNBackbone):
    """Run the shared graph stages on each support and mix region outputs."""

    def __init__(self, *, range_mode, condition_encoder, **backbone_kwargs):
        super().__init__(**backbone_kwargs)
        if self.learnable_graph is None or any(module is not None for module in (
            self.interaction, self.edge_updater, self.graph_router,
        )):
            raise ValueError("Graph range comparisons require a learned graph without interaction or edge updates")
        self.learnable_graph = _TwoRadiusGraph(self.learnable_graph)
        self.range_content = nn.Linear(64, 2, bias=False) if range_mode in ("content", "joint") else None
        self.range_encoder = condition_encoder() if range_mode in ("metadata", "joint") else None
        self.range_metadata = nn.Linear(32, 2, bias=False) if self.range_encoder is not None else None

    def range_weights(self, tokens, observation):
        logits = tokens.new_zeros(2)
        if self.range_content is not None:
            logits = self.range_content(tokens.mean(dim=0))
        if self.range_encoder is not None:
            encoded = self.range_encoder(*observation, reference=tokens)
            if encoded is not None:
                logits = logits + self.range_metadata(encoded)
        return logits.softmax(dim=0)

    def forward_stages(self, cnn_input, region_input, Q, A, dataset_name=None,
                       collect_stats=False, return_diagnostics=False):
        current = _graph_forward.get()
        if current is None or current.owner is not self or current.wider_graph is None:
            raise RuntimeError("Both graph ranges must be built on the current assignment")
        wider = current.wider_graph
        count = region_input.shape[0]
        # Block diagonal support runs the two paths with shared parameters and a
        # single normalization update. No edge can cross between the paths.
        support = torch.sparse_coo_tensor(
            torch.cat((A.indices(), wider.indices() + count), dim=1),
            torch.cat((A.values(), wider.values())), (2 * count, 2 * count),
            is_coalesced=True,
        )
        regions = self.graph_stage1(torch.cat((region_input, region_input)), support, dataset_name)
        narrow_regions, wider_regions = self.graph_stage2(regions, support, dataset_name).split(count)
        narrow_weight, wider_weight = current.weights.unbind()
        pixels = self.cnn_stage1(cnn_input, dataset_name)
        cnn = self.cnn_stage2(pixels, cnn_input.shape[:2], dataset_name)
        return cnn, broadcast_regions(Q, narrow_weight * narrow_regions + wider_weight * wider_regions), {}

    def forward_observed(self, observation, forward, *args, **kwargs):
        current = _GraphForward(self, observation)
        token = _graph_forward.set(current)
        try:
            output = forward(*args, **kwargs)
        finally:
            _graph_forward.reset(token)
        if kwargs.get("return_graph_info"):
            output[-1]["multiscale_graph"] = {
                "radii": GRAPH_RADII,
                "weights": current.weights.detach().clone(),
                "nnz": (output[-1]["graph_nnz"], current.wider_graph._nnz()),
            }
        return output
