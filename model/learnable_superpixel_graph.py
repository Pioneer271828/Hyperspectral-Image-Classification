import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSuperpixelGraph(nn.Module):
    """Build a differentiable sparse pixel assignment and dynamic graph.

    Pixels are connected only to nearby regular-grid seeds. The assignment
    weights and graph edge weights are learned from the current feature map,
    while the local candidate pattern keeps memory linear in the image size.
    """

    def __init__(
        self,
        feature_dim,
        assignment_dim=32,
        assignment_radius=1,
        graph_radius=1,
        assignment_temperature=1.0,
        assignment_spatial_sigma=1.0,
        graph_spatial_sigma=1.5,
    ):
        super(LearnableSuperpixelGraph, self).__init__()
        if feature_dim <= 0 or assignment_dim <= 0:
            raise ValueError("feature_dim and assignment_dim must be positive")
        if assignment_radius < 0 or graph_radius < 0:
            raise ValueError("assignment_radius and graph_radius must be non-negative")
        if assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive")
        if assignment_spatial_sigma <= 0 or graph_spatial_sigma <= 0:
            raise ValueError("spatial sigmas must be positive")

        self.feature_dim = int(feature_dim)
        self.assignment_dim = int(assignment_dim)
        self.assignment_temperature = float(assignment_temperature)
        self.assignment_spatial_sigma = float(assignment_spatial_sigma)
        self.graph_spatial_sigma = float(graph_spatial_sigma)

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.pixel_query = nn.Linear(feature_dim, assignment_dim, bias=False)
        self.seed_key = nn.Linear(feature_dim, assignment_dim, bias=False)
        self.graph_query = nn.Linear(feature_dim, assignment_dim, bias=False)
        self.graph_key = nn.Linear(feature_dim, assignment_dim, bias=False)

        self.register_buffer(
            "assignment_offsets",
            self._make_offsets(assignment_radius),
            persistent=False,
        )
        self.register_buffer(
            "graph_offsets",
            self._make_offsets(graph_radius),
            persistent=False,
        )

    @staticmethod
    def _make_offsets(radius):
        values = torch.arange(-radius, radius + 1, dtype=torch.long)
        offset_y, offset_x = torch.meshgrid(values, values, indexing="ij")
        return torch.stack((offset_y.reshape(-1), offset_x.reshape(-1)), dim=1)

    @staticmethod
    def _grid_shape(height, width, superpixel_scale):
        if superpixel_scale <= 0:
            raise ValueError("superpixel_scale must be positive")
        step = math.sqrt(float(superpixel_scale))
        grid_height = max(1, int(math.ceil(height / step)))
        grid_width = max(1, int(math.ceil(width / step)))
        return grid_height, grid_width

    @staticmethod
    def _sparse_pool(assignment, flat_features):
        assignment = assignment.coalesce()
        rows, cols = assignment.indices()
        values = assignment.values()
        mass = torch.zeros(
            assignment.shape[1],
            device=values.device,
            dtype=values.dtype,
        )
        mass.scatter_add_(0, cols, values)
        normalized_values = values / mass[cols].clamp_min(1e-6)
        normalized = torch.sparse_coo_tensor(
            assignment.indices(),
            normalized_values,
            assignment.shape,
            device=assignment.device,
            dtype=assignment.dtype,
        ).coalesce()
        return torch.sparse.mm(normalized.transpose(0, 1), flat_features), mass

    def _pixel_candidates(self, height, width, grid_height, grid_width, device):
        pixel_y, pixel_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        pixel_y = pixel_y.reshape(-1)
        pixel_x = pixel_x.reshape(-1)
        base_y = torch.div(pixel_y * grid_height, height, rounding_mode="floor")
        base_x = torch.div(pixel_x * grid_width, width, rounding_mode="floor")

        offsets = self.assignment_offsets.to(device=device)
        candidate_y = base_y.unsqueeze(1) + offsets[:, 0].unsqueeze(0)
        candidate_x = base_x.unsqueeze(1) + offsets[:, 1].unsqueeze(0)
        valid = (
            (candidate_y >= 0)
            & (candidate_y < grid_height)
            & (candidate_x >= 0)
            & (candidate_x < grid_width)
        )
        safe_y = candidate_y.clamp(0, grid_height - 1)
        safe_x = candidate_x.clamp(0, grid_width - 1)
        candidate_ids = safe_y * grid_width + safe_x

        pixel_y_norm = (pixel_y.to(torch.float32) + 0.5) / float(height)
        pixel_x_norm = (pixel_x.to(torch.float32) + 0.5) / float(width)
        center_y = (safe_y.to(torch.float32) + 0.5) / float(grid_height)
        center_x = (safe_x.to(torch.float32) + 0.5) / float(grid_width)
        distance = (
            ((pixel_y_norm.unsqueeze(1) - center_y) * grid_height).pow(2)
            + ((pixel_x_norm.unsqueeze(1) - center_x) * grid_width).pow(2)
        )
        return candidate_ids, valid, distance

    def _build_assignment(self, features, superpixel_scale):
        height, width, channels = features.shape
        grid_height, grid_width = self._grid_shape(height, width, superpixel_scale)
        token_count = grid_height * grid_width
        flat_features = self.feature_norm(features.reshape(height * width, channels))

        pooled = F.adaptive_avg_pool2d(
            features.permute(2, 0, 1).unsqueeze(0),
            (grid_height, grid_width),
        )
        seed_features = pooled.squeeze(0).permute(1, 2, 0).reshape(token_count, channels)
        seed_features = self.feature_norm(seed_features)

        candidate_ids, valid, spatial_distance = self._pixel_candidates(
            height,
            width,
            grid_height,
            grid_width,
            features.device,
        )
        query = self.pixel_query(flat_features)
        key = self.seed_key(seed_features)
        candidate_key = key[candidate_ids]
        logits = torch.sum(query.unsqueeze(1) * candidate_key, dim=-1)
        logits = logits / math.sqrt(self.assignment_dim)
        logits = logits / self.assignment_temperature
        spatial_distance = spatial_distance.to(dtype=logits.dtype)
        logits = logits - spatial_distance / (2.0 * self.assignment_spatial_sigma ** 2)
        logits = logits.masked_fill(~valid, -1e4)
        weights = F.softmax(logits, dim=1)

        pixel_ids = torch.arange(height * width, device=features.device)
        pixel_ids = pixel_ids.unsqueeze(1).expand_as(candidate_ids)
        indices = torch.stack((pixel_ids[valid], candidate_ids[valid]), dim=0)
        values = weights[valid]
        assignment = torch.sparse_coo_tensor(
            indices,
            values,
            (height * width, token_count),
            device=features.device,
            dtype=features.dtype,
        ).coalesce()

        compactness = torch.sum(weights * spatial_distance * valid.to(weights.dtype), dim=1).mean()
        entropy = -torch.sum(
            weights * weights.clamp_min(1e-8).log() * valid.to(weights.dtype),
            dim=1,
        ).mean()
        token_features, mass = self._sparse_pool(assignment, flat_features)
        expected_mass = float(height * width) / float(token_count)
        balance = torch.mean((mass / max(expected_mass, 1e-6) - 1.0).pow(2))

        auxiliary = {
            "compactness_loss": compactness,
            "assignment_entropy_loss": entropy,
            "assignment_balance_loss": balance,
            "token_count": token_count,
            "assignment_nnz": assignment._nnz(),
            "grid_shape": (grid_height, grid_width),
        }
        return assignment, token_features, auxiliary

    def _build_graph(self, token_features, grid_shape):
        grid_height, grid_width = grid_shape
        token_count = grid_height * grid_width
        node_ids = torch.arange(token_count, device=token_features.device)
        node_y = torch.div(node_ids, grid_width, rounding_mode="floor")
        node_x = node_ids % grid_width

        offsets = self.graph_offsets.to(device=token_features.device)
        neighbor_y = node_y.unsqueeze(1) + offsets[:, 0].unsqueeze(0)
        neighbor_x = node_x.unsqueeze(1) + offsets[:, 1].unsqueeze(0)
        valid = (
            (neighbor_y >= 0)
            & (neighbor_y < grid_height)
            & (neighbor_x >= 0)
            & (neighbor_x < grid_width)
        )
        safe_y = neighbor_y.clamp(0, grid_height - 1)
        safe_x = neighbor_x.clamp(0, grid_width - 1)
        neighbor_ids = safe_y * grid_width + safe_x

        query = self.graph_query(token_features)
        key = self.graph_key(token_features)
        neighbor_key = key[neighbor_ids]
        logits = torch.sum(query.unsqueeze(1) * neighbor_key, dim=-1)
        logits = logits / math.sqrt(self.assignment_dim)
        spatial_distance = (
            (neighbor_y - node_y.unsqueeze(1)).to(logits.dtype).pow(2)
            + (neighbor_x - node_x.unsqueeze(1)).to(logits.dtype).pow(2)
        )
        logits = logits - spatial_distance / (2.0 * self.graph_spatial_sigma ** 2)
        logits = logits.masked_fill(~valid, -1e4)
        weights = F.softmax(logits, dim=1)

        source_ids = node_ids.unsqueeze(1).expand_as(neighbor_ids)
        indices = torch.stack((source_ids[valid], neighbor_ids[valid]), dim=0)
        graph = torch.sparse_coo_tensor(
            indices,
            weights[valid],
            (token_count, token_count),
            device=token_features.device,
            dtype=token_features.dtype,
        ).coalesce()
        return graph

    def forward(self, features, superpixel_scale):
        if features.dim() != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                "features must have shape [H, W, {}], got {}".format(
                    self.feature_dim,
                    tuple(features.shape),
                )
            )
        assignment, token_features, auxiliary = self._build_assignment(
            features,
            superpixel_scale,
        )
        graph = self._build_graph(token_features, auxiliary["grid_shape"])
        auxiliary["graph_nnz"] = graph._nnz()
        return assignment, graph, auxiliary
