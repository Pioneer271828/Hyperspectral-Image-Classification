"""E1-A: label-blind spectral support and standalone conditional modulation.

Coverage geometry is the project's proposed descriptor. The residual affine
operator independently implements the FiLM mathematical idea (Perez et al.,
arXiv:1709.07871v2, sec.2.1); no upstream source is incorporated. See
docs/references/e1-reference-review.json. Production opt-in wiring is E1-B.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from loadData.observation_sampling import BandMetadata


FEATURES = ('anchor_position', 'nearest_distance', 'soft_support', 'log_local_count',
            'local_gap_fraction', 'physical_available', 'verified_centers',
            'approximate_centers', 'fwhm_present')


def validate_binding(metadata, wavelengths, fwhm, bands, device, dtype, wavelength_unit, fwhm_unit):
    """Bind declared quality/identity to exactly the values consumed by VCA.

    Model inputs already contain retained valid columns. Auto units are checked
    against VCA's existing rule, never silently reinterpreted using E1 metadata.
    """
    if metadata is None:return
    require(isinstance(metadata,BandMetadata),'coverage_metadata must be BandMetadata or None')
    require(len(metadata.band_ids)==bands and all(metadata.valid),'pass only retained valid columns to the model')
    require(wavelengths is not None,'declared coverage metadata requires actual VCA wavelengths')
    wave=torch.as_tensor(wavelengths,device=device,dtype=dtype).reshape(-1)
    expected=torch.as_tensor(metadata.wavelengths,device=device,dtype=dtype)
    require(wave.shape==expected.shape and torch.equal(wave,expected),'coverage/VCA wavelength values differ')
    resolved = ('normalized' if wave.max()<=2 and wave.min()>=-.5 else 'nm') if wavelength_unit=='auto' else wavelength_unit
    require(resolved==metadata.wavelength_unit,'coverage/VCA wavelength units differ; supply explicit units')
    require((fwhm is None)==(metadata.fwhm is None),'coverage/VCA FWHM availability differs')
    if fwhm is not None:
        widths=torch.as_tensor(fwhm,device=device,dtype=dtype).reshape(-1)
        expected=torch.as_tensor(metadata.fwhm,device=device,dtype=dtype)
        require(widths.shape==expected.shape and torch.equal(widths,expected),'coverage/VCA FWHM values differ')
        resolved=('normalized' if widths.max()<=2 else 'nm') if fwhm_unit=='auto' else fwhm_unit
        require(resolved==metadata.fwhm_unit,'coverage/VCA FWHM units differ; supply explicit units')


def require(condition, message):
    if not condition:
        raise ValueError('E1 coverage: ' + message)


@dataclass(frozen=True)
class CoverageConfig:
    wavelength_min: float = 400.
    wavelength_max: float = 2500.
    sigma: float = .08
    window: float = .16

    def __post_init__(self):
        for name, value in asdict(self).items():
            require(type(value) in (int, float) and math.isfinite(value), name+' must be finite')
        require(0 < self.wavelength_min < self.wavelength_max, 'ordered positive GLOBAL nm reference required')
        require(self.sigma > 0 and 1e-12 <= self.window <= 1, 'positive sigma and numerically resolvable window required')

    def export_config(self):
        return asdict(self)


@dataclass(frozen=True)
class CoverageDescription:
    features: torch.Tensor
    usable: torch.Tensor
    local_count: torch.Tensor
    max_gap: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def describe(metadata, anchors, config=CoverageConfig(), *, band_mask=None,
             normalized_reference_nm=None):
    """Return detached [K,9] features in the caller's VCA anchor order.

    Anchors are GLOBAL normalized [0,1], not local subset coordinates. Counts
    include duplicate observed columns; gaps use unique centers plus clipped
    window endpoints. Out-of-reference centers are excluded, never clamped into
    endpoint observations. No FWHM values enter the design kernel. Raw physical
    centers may be approximate; index-only/unresolved metadata bypass E1.
    """
    require(isinstance(config, CoverageConfig), 'CoverageConfig required')
    require(isinstance(anchors, torch.Tensor) and anchors.layout == torch.strided
            and anchors.ndim == 1 and anchors.numel() > 0
            and anchors.dtype in (torch.float32, torch.float64), 'dense float32/64 anchor vector required')
    require(bool(torch.isfinite(anchors).all() and ((anchors >= 0) & (anchors <= 1)).all()),
            'finite GLOBAL normalized anchors in [0,1] required')
    a = anchors.detach().to(torch.float64).clone(); k = a.numel()
    features = torch.zeros((k, len(FEATURES)), dtype=torch.float64, device=a.device)
    features[:, 0] = a; features[:, 1] = features[:, 4] = 1.
    count = torch.zeros(k, dtype=torch.long, device=a.device)
    left, right = (a-config.window).clamp_min(0), (a+config.window).clamp_max(1)
    max_gap = right-left
    diagnostics = {'schema_version': 1, 'feature_names': list(FEATURES), 'config': config.export_config(),
                   'scope': 'per-observation metadata support; not pixel reliability or reconstructed spectrum',
                   'normalization': 'fixed global reference; no subset min/max fit',
                   'valid_band_count': 0, 'in_reference_band_count': 0, 'out_of_reference_count': 0,
                   'wavelength_quality': 'missing', 'fwhm_used_in_geometry': False}

    def result(reason):
        return CoverageDescription(features.to(anchors.dtype), count > 0, count,
            max_gap.to(anchors.dtype), {**diagnostics, 'fallback_reason': reason})

    if metadata is None:
        require(band_mask is None, 'cannot bind a mask without metadata')
        return result('missing_metadata')
    require(isinstance(metadata, BandMetadata), 'BandMetadata or explicit None required')
    diagnostics.update(metadata_hash=metadata.fingerprint, wavelength_quality=metadata.wavelength_quality)
    valid = torch.tensor(metadata.valid, device=a.device, dtype=torch.bool)
    if band_mask is not None:
        mask = torch.as_tensor(band_mask, device=a.device)
        require(mask.dtype == torch.bool and mask.shape == valid.shape, 'boolean mask must match original B')
        valid = valid & mask
    diagnostics['valid_band_count'] = int(valid.sum())
    if normalized_reference_nm is not None:
        require(isinstance(normalized_reference_nm, (tuple, list))
                and tuple(normalized_reference_nm) == (config.wavelength_min, config.wavelength_max),
                'normalized physical reference differs from anchor reference')
    if metadata.wavelength_quality == 'index_only':
        return result('index_only_metadata')
    if metadata.wavelength_unit == 'normalized' and normalized_reference_nm is None:
        return result('unresolved_normalized_reference')
    if not valid.any():
        return result('no_valid_bands')
    positions = torch.tensor(metadata.wavelengths, device=a.device, dtype=torch.float64)[valid]
    if metadata.wavelength_unit == 'nm':
        positions = (positions-config.wavelength_min)/(config.wavelength_max-config.wavelength_min)
    inside = (positions >= 0) & (positions <= 1)
    diagnostics['out_of_reference_count'] = int((~inside).sum())
    positions = positions[inside]
    diagnostics['in_reference_band_count'] = positions.numel()
    if positions.numel() == 0:
        return result('no_in_reference_bands')
    distance = (a[:, None]-positions).abs().amin(1)
    count = ((positions[None, :] >= left[:, None]) & (positions[None, :] <= right[:, None])).sum(1)
    unique = positions.unique(sorted=True).expand(k, -1)
    clipped = torch.maximum(torch.minimum(unique, right[:, None]), left[:, None])
    points = torch.cat((left[:, None], clipped, right[:, None]), dim=1)
    max_gap = (points[:, 1:]-points[:, :-1]).amax(1)
    features[:, 1] = distance.clamp_max(1)
    features[:, 2] = torch.exp(-.5*(distance/config.sigma).square())
    features[:, 3] = count.to(torch.float64).log1p()
    features[:, 4] = (max_gap/(right-left)).clamp(0, 1)
    features[:, 5] = 1.
    features[:, 6] = float(metadata.wavelength_quality == 'verified')
    features[:, 7] = float(metadata.wavelength_quality == 'approximate')
    features[:, 8] = float(metadata.fwhm is not None)
    return result(None)


class _LocalLinear(nn.Linear):
    def __init__(self, inputs, outputs, seed):
        self.init_seed = seed
        super().__init__(inputs, outputs)

    def reset_parameters(self):
        generator = torch.Generator(device=self.weight.device).manual_seed(self.init_seed)
        nn.init.xavier_uniform_(self.weight, generator=generator)
        nn.init.zeros_(self.bias)


class _ZeroLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight); nn.init.zeros_(self.bias)


class CoverageModulation(nn.Module):
    """Shared anchor-wise MLP, then X+strength*(tanh(gamma)*X+tanh(beta)).

    Default none creates no parameters and returns the same object. Affine has
    exact identity initialization, with first-layer gradients beginning after
    the zero output layer learns. True unsupported anchors always bypass,
    independent of affine biases. Missing descriptions also bypass. Save both
    export_config() and state_dict(); production checkpoint integration is E1-B.
    """
    def __init__(self, latent_channels, value_dim=1, mode='none', hidden=16,
                 strength=.1, init_seed=0, descriptor_version=1):
        super().__init__()
        require(all(type(v) is int and v > 0 for v in (latent_channels, value_dim, hidden)),
                'positive integer K/value_dim/hidden required')
        require(mode in ('none', 'affine'), 'mode must be none or affine')
        require(type(strength) in (int, float) and math.isfinite(strength) and 0 <= strength <= 1,
                'strength must be finite in [0,1]')
        require(type(init_seed) is int and 0 <= init_seed < 2**32, 'uint32 local initialization seed required')
        require(type(descriptor_version) is int and descriptor_version == 1, 'unsupported descriptor version')
        self.latent_channels, self.value_dim, self.mode = latent_channels, value_dim, mode
        self.hidden, self.strength, self.init_seed = hidden, float(strength), init_seed
        self.descriptor_version = descriptor_version
        self.generator = (nn.Sequential(_LocalLinear(len(FEATURES), hidden, init_seed), nn.GELU(),
                                       _ZeroLinear(hidden, 2*value_dim)) if mode == 'affine' else None)

    def export_config(self):
        return {k: getattr(self, k) for k in ('latent_channels', 'value_dim', 'mode', 'hidden', 'strength', 'init_seed', 'descriptor_version')}

    def forward(self, latent, coverage=None, *, channel_dim=-1):
        if self.mode == 'none' or self.strength == 0 or coverage is None:
            return latent
        require(isinstance(latent, torch.Tensor) and latent.layout == torch.strided and latent.ndim >= 2 and latent.numel() > 0
                and latent.dtype in (torch.float32, torch.float64), 'dense float32/64 latent with spatial/pixel axes required')
        require(type(channel_dim) is int and -latent.ndim <= channel_dim < latent.ndim, 'invalid channel axis')
        require(latent.shape[channel_dim] == self.latent_channels*self.value_dim, 'latent channel count differs')
        require(bool(torch.isfinite(latent).all()), 'nonfinite latent')
        require(isinstance(coverage, CoverageDescription), 'CoverageDescription required')
        require(coverage.diagnostics.get('schema_version') == self.descriptor_version
                and coverage.diagnostics.get('feature_names') == list(FEATURES), 'descriptor schema/feature order differs')
        f, mask = coverage.features, coverage.usable
        require(isinstance(f, torch.Tensor) and isinstance(mask, torch.Tensor)
                and f.layout == mask.layout == torch.strided, 'dense descriptor tensors required')
        require(f.shape == (self.latent_channels, len(FEATURES)) and f.dtype == latent.dtype
                and f.device == latent.device and bool(torch.isfinite(f).all()), 'descriptor shape/dtype/device/values differ')
        require(mask.shape == (self.latent_channels,) and mask.dtype == torch.bool
                and mask.device == latent.device, 'descriptor support mask differs')
        for p in self.parameters():
            require(p.dtype == latent.dtype and p.device == latent.device, 'move module to latent dtype/device first')
        if not mask.any():
            return latent
        coefficients = self.generator(f.detach()).reshape(self.latent_channels, 2, self.value_dim).tanh()
        x = latent.movedim(channel_dim, -1).unflatten(-1, (self.latent_channels, self.value_dim))
        adjusted = x+self.strength*(coefficients[:, 0]*x+coefficients[:, 1])
        output = torch.where(mask[:, None], adjusted, x).flatten(-2)
        return output.movedim(-1, channel_dim)
