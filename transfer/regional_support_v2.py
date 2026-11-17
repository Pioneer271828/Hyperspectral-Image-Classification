"""Versioned support contracts for the native second graph stage."""

from collections import Counter
from copy import deepcopy
from time import perf_counter

import torch

from model.research_config import digest
from research_runtime import tensor_hash
from transfer.regional_support_evolution import SupportEvolution, _cosine, _features, _graph, _tensor_pairs


class RegionalSupport(SupportEvolution):
    """Frozen descriptors select edges; current features and adjacency score them."""

    VERSION = 2
    IDENTITY = frozenset(("scene", "grid", "bands", "source", "assignment", "block_input", "admission"))

    def __init__(self, adjacency, spectra, graph_features, *, identity, **kwargs):
        if set(identity) != self.IDENTITY or not isinstance(identity["scene"], str) or not identity["scene"]:
            raise ValueError("Explicit native context identity required")
        grid = identity["grid"]
        if (not isinstance(grid, list) or len(grid) != 2
                or any(type(v) is not int or v < 1 for v in grid)
                or grid[0] * grid[1] != len(kwargs["context"]["node_ids"])):
            raise ValueError("Native grid and stable IDs differ")
        for key in self.IDENTITY - {"scene", "grid"}:
            value = identity[key]
            if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise ValueError("Invalid native context fingerprint")
        super().__init__(adjacency, spectra, graph_features, **kwargs)
        self.base = self.base.detach().clone()
        self.identity = deepcopy(identity)
        self.contract = digest(dict(selection=self.contract, identity=self.identity))
        self.selected = self._select(self.schedule.budget(0), [])

    def _fresh(self):
        return RegionalSupport(self.base, self.spectra, self.features, context=self.context, identity=self.identity,
                               mandatory=[list(p) for p in sorted(self.mandatory)], schedule=self.schedule,
                               mode=self.mode, seed=self.seed)

    def advance(self, step):
        """Replay each missed global optimizer step in the registered context."""
        if type(step) is not int or not self.step <= step <= self.schedule.updates:
            raise ValueError("Global support step outside registered trajectory")
        for cursor in range(self.step + 1, step + 1):
            super().advance(cursor)

    def snapshot(self):
        return self.receipt()

    def authenticate(self, snapshot):
        if (not isinstance(snapshot, dict) or snapshot.get("contract") != self.contract
                or type(snapshot.get("step")) is not int):
            raise ValueError("Support snapshot context differs")
        shadow = self._fresh()
        shadow.advance(snapshot["step"])
        if shadow.snapshot() != snapshot:
            raise ValueError("Support snapshot differs from deterministic replay")
        return shadow

    def restore(self, receipt):
        if self.step:
            raise ValueError("Support restore requires a fresh context")
        shadow = self.authenticate(receipt)
        self.step, self.pool, self.selected = shadow.step, shadow.pool, shadow.selected
        self.refresh_step, self._rng = shadow.refresh_step, shadow._rng
        self.last_stats = shadow.last_stats

    def render(self, **kwargs):
        raise ValueError("Use the live or authenticated snapshot render contract")

    def render_live(self, adjacency, features, *, support=None):
        pairs = self.selected if support is None else _tensor_pairs(support, self.n)
        if pairs != self.selected:
            raise ValueError("Live rendering requires the active selected support and budget")
        return self._render(adjacency, features, pairs, self.schedule.budget(self.step))

    def render_snapshot(self, adjacency, features, snapshot):
        """Replay the source context before rendering the authenticated origin budget."""
        origin = self.authenticate(snapshot)
        return self._render(adjacency, features, origin.selected, origin.schedule.budget(origin.step))

    def _render(self, adjacency, features, pairs, budget):
        row_counts = Counter(i for i, _ in pairs)
        if not self.mandatory <= set(pairs) or max(row_counts.values()) > budget:
            raise ValueError("Rendered support exceeds its active or authenticated budget")
        started = perf_counter()
        current = _graph(adjacency, self.context["partitions"])
        features = _features(features, self.n)
        indices = torch.tensor(pairs, dtype=torch.long).T.contiguous()
        row, col = indices
        prior = current.to_dense()[row, col]
        values = (prior + (_cosine(features, row, col) * .5).exp()) * .5
        self.cost["scoring_seconds"] += perf_counter() - started
        started = perf_counter()
        totals = values.new_zeros(self.n).scatter_add(0, row, values)
        values = values / totals[row]
        result = torch.sparse_coo_tensor(indices, values, (self.n, self.n)).coalesce()
        if result._nnz() != len(pairs) or not torch.isfinite(result.values()).all() or (result.values() <= 0).any():
            raise ValueError("Selected support must retain positive finite COO values")
        self.cost["sparse_build_seconds"] += perf_counter() - started
        self.last_stats["score_pairs"] = len(pairs)
        return result

    def inputs(self):
        return dict(adjacency=self.base, spectra=self.spectra, graph_features=self.features, identity=self.identity,
                    context=self.context, mandatory=[list(p) for p in sorted(self.mandatory)],
                    schedule=self.schedule, mode=self.mode, seed=self.seed)
