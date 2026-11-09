"""E3 directed candidate-only hard support selection after the complete M7 mix.

Independent project implementation. DropEdge (ICLR 2020, section 4.1) is a
random-edge control reference, not this per-row inference-time algorithm.
Budget includes one mandatory self-loop. No reciprocal edges are added.
Selection is detached, with no straight-through estimator. Retained adjacency
values remain differentiable, including the upstream M7 softmax dependencies.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class BudgetedGraphRouter(nn.Module):
    MODES = ("weight", "random", "graph", "joint")

    def __init__(self, budget=4, mode="weight", seed=64):
        super().__init__()
        if type(budget) is not int or budget < 1:
            raise ValueError("budget must be a positive integer including self")
        if mode not in self.MODES or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("invalid routing mode or seed")
        self.budget, self.mode, self.seed = budget, mode, seed
        self.step = 0
        self.fixed = None

    def set_step(self, step):
        """Explicit caller-owned step makes eval/replay independent of call count."""
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        self.step = step

    def get_extra_state(self):
        return copy.deepcopy(dict(version=1, budget=self.budget, mode=self.mode,
                                  seed=self.seed, step=self.step, fixed=self.fixed))

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or set(state) !=
                {"version", "budget", "mode", "seed", "step", "fixed"}
                or (state["version"], state["budget"], state["mode"], state["seed"]) !=
                (1, self.budget, self.mode, self.seed)):
            raise ValueError("routing checkpoint policy mismatch")
        self.set_step(state["step"])
        self.fixed = copy.deepcopy(state["fixed"])

    @staticmethod
    def _graph(adjacency):
        if (adjacency.layout not in (torch.strided, torch.sparse_coo)
                or adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
                or adjacency.shape[0] < 1 or not adjacency.is_floating_point()):
            raise ValueError("adjacency must be nonempty square float dense/COO")
        if adjacency.is_sparse:
            raw_idx = adjacency._indices()
            if (raw_idx < 0).any() or (raw_idx >= adjacency.shape[0]).any():
                raise ValueError("adjacency index outside node range")
        graph = adjacency.to_sparse().coalesce()
        # Inspect raw COO before coalesce so negative duplicate entries cannot hide.
        raw = adjacency._values() if adjacency.is_sparse else adjacency
        if not torch.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("adjacency must be finite and nonnegative")
        idx, val = graph.indices(), graph.values()
        if not torch.isfinite(val).all():
            raise ValueError("duplicate weights overflowed")
        n = graph.shape[0]
        if (idx < 0).any() or (idx >= n).any():
            raise ValueError("adjacency index outside node range")
        positive = val > 0
        idx, val = idx[:, positive], val[positive]
        original_nnz = int(val.numel())
        has_self = torch.zeros(n, device=val.device, dtype=torch.bool)
        has_self[idx[0, idx[0] == idx[1]]] = True
        missing = (~has_self).nonzero(as_tuple=True)[0]
        idx = torch.cat((idx, torch.stack((missing, missing))), dim=1)
        val = torch.cat((val, val.new_ones(missing.numel())))
        graph = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        return graph, original_nnz, int(missing.numel())

    def freeze_support(self, adjacency, region_features=None, aligned_pixels=None):
        """Freeze selected indices, not weights, for fixed-support retraining.

        A subsequent call must present the identical candidate support/node order.
        Upstream callers must additionally bind the scene/graph identity in their
        experiment manifest. Numerically identical graphs alone do not identify a scene.
        """
        if self.fixed is not None:
            raise ValueError("support already frozen")
        result, _ = self(adjacency, region_features, aligned_pixels)
        graph, _, _ = self._graph(adjacency)
        n = graph.shape[0]
        keys = lambda g: (g.indices()[0] * n + g.indices()[1]).cpu().tolist()
        self.fixed = dict(nodes=n, candidates=keys(graph), selected=keys(result))

    def forward(self, adjacency, region_features=None, aligned_pixels=None):
        graph, original_nnz, inserted = self._graph(adjacency)
        idx, val = graph.indices(), graph.values()
        rows, cols = idx
        n = graph.shape[0]
        keys = rows * n + cols
        with torch.no_grad():
            if self.fixed is not None:
                if (not isinstance(self.fixed, dict) or set(self.fixed) !=
                        {"nodes", "candidates", "selected"}
                        or self.fixed["nodes"] != n
                        or self.fixed["candidates"] != keys.cpu().tolist()):
                    raise ValueError("frozen candidate support changed")
                saved = self.fixed["selected"]
                if (not isinstance(saved, list) or any(type(k) is not int for k in saved)
                        or saved != sorted(set(saved))):
                    raise ValueError("invalid frozen selection")
                selected_keys = torch.tensor(saved, device=keys.device, dtype=torch.long)
                take = torch.searchsorted(keys, selected_keys)
                if (take >= keys.numel()).any() or not torch.equal(keys[take], selected_keys):
                    raise ValueError("frozen selection outside candidates")
            else:
                if self.mode == "weight":
                    scores = val.detach()
                elif self.mode == "random":
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed((self.seed + self.step) % (2**63))
                    scores = torch.rand(val.numel(), generator=generator).to(val.device)
                else:
                    features = region_features
                    if (not torch.is_tensor(features) or features.ndim != 2
                            or features.shape[0] != n or features.shape[1] < 1
                            or features.device != val.device or features.dtype != val.dtype
                            or not torch.isfinite(features).all()):
                        raise ValueError("finite region features must match graph")
                    if self.mode == "joint":
                        if (not torch.is_tensor(aligned_pixels)
                                or aligned_pixels.shape != features.shape
                                or aligned_pixels.device != val.device
                                or aligned_pixels.dtype != val.dtype
                                or not torch.isfinite(aligned_pixels).all()):
                            raise ValueError("joint routing requires aligned CNN regions")
                        # Equal branch contribution independent of feature magnitude.
                        features = torch.cat((F.normalize(features, dim=-1),
                                              F.normalize(aligned_pixels, dim=-1)), -1)
                    features = F.normalize(features, dim=-1)
                    scores = (features[rows] * features[cols]).sum(-1)
                # COO canonical order breaks equal scores by ascending column.
                # Stable global score sort followed by row sort avoids N*N storage.
                nonself = (rows != cols).nonzero(as_tuple=True)[0]
                order = nonself[torch.argsort(scores[nonself], descending=True, stable=True)]
                order = order[torch.argsort(rows[order], stable=True)]
                counts = torch.bincount(rows[order], minlength=n)
                starts = counts.cumsum(0) - counts
                rank = torch.arange(order.numel(), device=val.device) - starts[rows[order]]
                take = torch.cat(((rows == cols).nonzero(as_tuple=True)[0],
                                  order[rank < self.budget - 1])).sort().values
            out_rows, out_cols = idx[:, take]
            degree = torch.bincount(out_rows, minlength=n)
            if (degree < 1).any() or (degree > self.budget).any():
                raise ValueError("frozen selection violates budget")
            if int((out_rows == out_cols).sum()) != n:
                raise ValueError("frozen selection must retain every self-loop")
        values = val[take]
        # Scale first to avoid overflow for finite extreme input weights.
        maxima = values.new_zeros(n).scatter_reduce(0, out_rows, values.detach(), reduce="amax")
        scaled = values / maxima[out_rows]
        mass = scaled.new_zeros(n).scatter_add(0, out_rows, scaled)
        normalized = scaled / mass[out_rows]
        if not torch.isfinite(normalized).all() or (normalized <= 0).any():
            raise ValueError("normalization lost a selected positive weight")
        result = torch.sparse_coo_tensor(idx[:, take], normalized, graph.shape).coalesce()
        info = dict(route_input_nnz=original_nnz, route_output_nnz=result._nnz(),
                    route_added_self=inserted, route_max_degree=int(degree.max()),
                    route_min_degree=int(degree.min()), route_budget=self.budget,
                    route_mode=self.mode, route_step=self.step, route_fixed=self.fixed is not None,
                    route_indices=result.indices().detach().clone(),
                    route_weights=result.values().detach().clone())
        return result, info
