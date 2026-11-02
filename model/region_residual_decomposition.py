"""Standalone M4 region/pixel decomposition; not wired into the backbone yet."""
from dataclasses import dataclass, field

import torch
from torch import nn

from model.region_pixel_alignment import broadcast_regions, pool_regions


@dataclass
class RegionResidualOutput:
    cnn_input: torch.Tensor
    region_features: torch.Tensor
    diagnostics: dict = field(default_factory=dict)
    region_at_pixels: torch.Tensor | None = None
    residual: torch.Tensor | None = None


class _IdentityLinear(nn.Linear):
    """Identity on the first C inputs, without consuming a random-number draw."""

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :self.out_features].copy_(
                torch.eye(self.out_features, device=self.weight.device, dtype=self.weight.dtype))
            if self.bias is not None:
                self.bias.zero_()


class RegionResidualDecomposition(nn.Module):
    """Z=pool(Q,X), Xr=QZ, R=X-Xr using the caller's same row-stochastic Q.

    Dense and sparse COO Q are supported (float32/float64, same dtype/device
    as X). Existing pooling clamps region mass at 1e-6; tiny positive regions
    therefore have attenuated means. Empty columns yield zero region vectors.
    No Q generation, input cache, labels, normalization layers or extra loss.

    `none` returns the identical X object. Optional Xr/R exports retain autograd;
    diagnostics are detached scalars. Soft-Q residuals need not pool to zero.
    Save export_config() together with state_dict(); standalone checkpoints do
    not imply integration with the existing backbone/checkpoint CLI.
    """

    MODES = ('none', 'capacity_control', 'x_residual', 'residual_only', 'x_region')
    MASS_FLOOR = 1e-6

    def __init__(self, channels=64, mode='none'):
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError('channels must be a positive integer')
        if mode not in self.MODES:
            raise ValueError('unsupported M4 mode: ' + str(mode))
        self.channels, self.mode = channels, mode
        self.projection = None
        self.control = None
        if mode in ('x_residual', 'x_region'):
            self.projection = _IdentityLinear(2 * channels, channels)
        elif mode == 'capacity_control':
            self.control = nn.Sequential(
                _IdentityLinear(channels, channels), nn.LeakyReLU(0.2),
                _IdentityLinear(channels, channels))
            with torch.no_grad():
                self.control[2].weight.zero_()

    def export_config(self):
        return {'channels': self.channels, 'mode': self.mode}

    def _validate(self, pixels, assignment):
        if (not isinstance(pixels, torch.Tensor) or pixels.layout != torch.strided
                or pixels.ndim != 2 or pixels.shape[0] <= 0
                or pixels.shape[1] != self.channels):
            raise ValueError('X must be nonempty dense [pixels, channels]')
        if (not isinstance(assignment, torch.Tensor) or assignment.ndim != 2
                or assignment.shape[0] != pixels.shape[0] or assignment.shape[1] <= 0
                or assignment.layout not in (torch.strided, torch.sparse_coo)):
            raise ValueError('Q must be dense or COO [pixels, regions]')
        if (pixels.dtype not in (torch.float32, torch.float64)
                or assignment.dtype != pixels.dtype or assignment.device != pixels.device):
            raise ValueError('X/Q must share float32/float64 dtype and device')
        for parameter in self.parameters():
            if parameter.dtype != pixels.dtype or parameter.device != pixels.device:
                raise ValueError('move the M4 module to the X dtype/device before use')
        # Reject invalid COO indices before coalesce can use them inside a kernel.
        # Raw value checks are validation-only; pooling still uses differentiable Q.
        if assignment.is_sparse:
            with torch.no_grad():
                indices, values = assignment._indices(), assignment._values()
                if (indices < 0).any() or any(
                        (indices[axis] >= assignment.shape[axis]).any() for axis in (0, 1)):
                    raise ValueError('Q indices out of bounds')
                if not torch.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Q must be finite and nonnegative')
        q = assignment.coalesce() if assignment.is_sparse else assignment
        with torch.no_grad():
            if not torch.isfinite(pixels).all():
                raise ValueError('X must be finite')
            if q.is_sparse:
                indices, values = q.indices(), q.values()
                rows = values.new_zeros(q.shape[0]).scatter_add_(0, indices[0], values)
            else:
                values, rows = q, q.sum(1)
            if not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError('Q must be finite and nonnegative')
            if not torch.allclose(rows, torch.ones_like(rows), atol=1e-5, rtol=0):
                raise ValueError('Q rows must sum to one')
        return q

    def forward(self, pixel_features, assignment, *, return_intermediates=False,
                return_diagnostics=False):
        x = pixel_features
        q = self._validate(x, assignment)
        regions = pool_regions(q, x)
        reconstructed = residual = None
        if (self.mode in ('x_residual', 'residual_only', 'x_region')
                or return_intermediates or return_diagnostics):
            reconstructed = broadcast_regions(q, regions)
            residual = x - reconstructed
        if self.mode == 'none':
            cnn_input = x
        elif self.mode == 'capacity_control':
            cnn_input = x + self.control(x)
        elif self.mode == 'residual_only':
            cnn_input = residual
        else:
            hint = residual if self.mode == 'x_residual' else reconstructed
            cnn_input = self.projection(torch.cat((x, hint), dim=1))
        diagnostics = {}
        if return_diagnostics:
            with torch.no_grad():
                if q.is_sparse:
                    mass = q.values().new_zeros(q.shape[1]).scatter_add_(
                        0, q.indices()[1], q.values())
                else:
                    mass = q.sum(0)
                # Float64 summaries avoid squaring overflow for finite fp32 inputs.
                energy = x.detach().double().square().sum()
                residual_energy = residual.detach().double().square().sum()
                diagnostics = {
                    'mode': self.mode, 'pixel_count': x.shape[0],
                    'region_count': q.shape[1], 'nonempty_region_count': int((mass > 0).sum()),
                    'empty_region_count': int((mass == 0).sum()),
                    'clamped_positive_region_count': int(((mass > 0) & (mass < self.MASS_FLOOR)).sum()),
                    'region_mass_min': float(mass.min()), 'region_mass_max': float(mass.max()),
                    'residual_relative_energy': float(residual_energy / energy.clamp_min(1e-30)),
                }
        return RegionResidualOutput(
            cnn_input=cnn_input, region_features=regions, diagnostics=diagnostics,
            region_at_pixels=reconstructed if return_intermediates else None,
            residual=residual if return_intermediates else None)
