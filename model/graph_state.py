"""Graph exchange contract; differentiable graph/losses and detached diagnostics."""
from dataclasses import dataclass, field
import math

import torch


def _detach(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach(item) for item in value)
    return value


@dataclass
class GraphState:
    """Q[N,M] and A0[M,M] are coalesced COO, nonnegative and row stochastic.

    Region positions are normalized (y,x), with explicitly recorded provenance.
    Empty Q columns are allowed; every Q/A0 row must have unit mass.
    Validation does not detach Q/A0 or regularizers. No labels are stored here.
    """
    Q: torch.Tensor
    A0: torch.Tensor
    region_positions: torch.Tensor
    image_shape: tuple
    grid_shape: tuple | None = None
    regularizers: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.diagnostics = _detach(self.diagnostics)

    @property
    def region_count(self):
        return self.Q.shape[1]

    def validate(self, features=None, atol=1e-5):
        if len(self.image_shape) != 2 or any(type(n) is not int or n <= 0 for n in self.image_shape):
            raise ValueError('image_shape must contain positive integer H/W')
        if self.Q.ndim != 2 or self.Q.shape[0] != math.prod(self.image_shape) or self.Q.shape[1] <= 0:
            raise ValueError('Q shape differs from image/region count')
        if self.A0.shape != (self.region_count, self.region_count):
            raise ValueError('A0 must be square and match Q regions')
        if self.grid_shape is not None and (len(self.grid_shape) != 2
                or any(type(n) is not int or n <= 0 for n in self.grid_shape)
                or math.prod(self.grid_shape) != self.region_count):
            raise ValueError('grid_shape differs from region count')
        if self.region_positions.shape != (self.region_count, 2):
            raise ValueError('region_positions must be [regions,2]')
        for name, tensor in (('Q', self.Q), ('A0', self.A0), ('region_positions', self.region_positions)):
            if not tensor.is_floating_point() or tensor.device != self.Q.device or tensor.dtype != self.Q.dtype:
                raise ValueError(name + ' dtype/device must match floating point Q')
        with torch.no_grad():
            for name, matrix in (('Q', self.Q), ('A0', self.A0)):
                if matrix.layout != torch.sparse_coo or not matrix.is_coalesced():
                    raise ValueError(name + ' must be coalesced sparse COO')
                indices, values = matrix.indices(), matrix.values()
                if not torch.isfinite(values).all() or (values < 0).any():
                    raise ValueError(name + ' values must be finite and nonnegative')
                if (indices < 0).any() or any((indices[axis] >= matrix.shape[axis]).any() for axis in (0, 1)):
                    raise ValueError(name + ' indices out of bounds')
                sums = values.new_zeros(matrix.shape[0]).scatter_add_(0, indices[0], values)
                if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0):
                    raise ValueError(name + ' rows must sum to one')
            if (not torch.isfinite(self.region_positions).all()
                    or (self.region_positions < 0).any() or (self.region_positions > 1).any()):
                raise ValueError('Region positions must be finite normalized coordinates')
        for name, loss in self.regularizers.items():
            if (not torch.is_tensor(loss) or loss.ndim != 0 or loss.device != self.Q.device
                    or not torch.isfinite(loss.detach())):
                raise ValueError('Invalid scalar regularizer: ' + name)
        if features is not None and (tuple(features.shape[:2]) != tuple(self.image_shape)
                                     or features.device != self.Q.device or features.dtype != self.Q.dtype):
            raise ValueError('GraphState image shape/device/dtype differs from features')
        return self

    def to_legacy(self):
        """Return the existing Q,A,auxiliary interface without severing losses."""
        return self.Q, self.A0, {**self.diagnostics, **self.regularizers}

    def detached_diagnostics(self):
        """Explicit opt-in summaries; never use these detached losses to train."""
        return _detach({**self.diagnostics, 'regularizer_values': self.regularizers})


def graph_diagnostics(Q, A0, **metadata):
    """O(nnz) diagnostics, with no dense N*M or M*M materialization."""
    with torch.no_grad():
        q_values, a_values = Q.values(), A0.values()
        q_rows, q_cols = Q.indices()
        a_rows = A0.indices()[0]
        mass = q_values.new_zeros(Q.shape[1]).scatter_add_(0, q_cols, q_values)
        q_sum = q_values.new_zeros(Q.shape[0]).scatter_add_(0, q_rows, q_values)
        a_sum = a_values.new_zeros(A0.shape[0]).scatter_add_(0, a_rows, a_values)
        q_degree = torch.bincount(q_rows, minlength=Q.shape[0])
        a_degree = torch.bincount(a_rows, minlength=A0.shape[0])
        return dict(metadata, token_count=Q.shape[1], assignment_nnz=Q._nnz(), graph_nnz=A0._nnz(),
                    assignment_max_degree=int(q_degree.max()), graph_max_degree=int(a_degree.max()),
                    empty_region_count=int((mass == 0).sum()), region_mass_min=float(mass.min()),
                    region_mass_max=float(mass.max()), assignment_row_error=float((q_sum - 1).abs().max()),
                    graph_row_error=float((a_sum - 1).abs().max()))
