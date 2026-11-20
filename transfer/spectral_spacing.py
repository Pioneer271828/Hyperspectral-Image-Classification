"""Opt-in quadrature correction to the existing wavelength adapter."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

MAX_CELL = .04
QUALITY_BLEND = {"verified": 1., "approximate": .5}


def normalize_attention(attention, native_valid, anchors, positions, locality, inside, blend):
    """Bound each retained center's quadrature cell and split repeated centers."""
    centers, inverse, counts = torch.unique(positions[inside], sorted=True, return_inverse=True, return_counts=True)
    if centers.numel() < 2:
        return attention, native_valid
    gaps = (centers[1:] - centers[:-1]).clamp(max=MAX_CELL)
    widths = (torch.cat((gaps[:1], gaps)) + torch.cat((gaps, gaps[-1:]))) / 2
    mass = torch.zeros_like(positions)
    relative = widths[inverse] / counts[inverse] / widths.mean()
    mass[inside] = (1 - blend) + blend * relative
    eligible = inside[None, :]
    if locality is not None and locality > 0:
        eligible = eligible & ((anchors[:, None] - positions[None, :]).abs() <= locality)
    reweighted = attention * mass[None, :] * eligible
    total = reweighted.sum(dim=1, keepdim=True)
    usable = (total > 0) & native_valid[:, None].bool()
    normalized = reweighted / total.clamp_min(torch.finfo(total.dtype).tiny)
    return torch.where(usable, normalized, attention), native_valid


@contextmanager
def spectral_spacing(model, metadata, *, enabled=True):
    """Use the original query/key/value projections and serialize their existing state."""
    adapter = model.backbone.adapter
    blend = QUALITY_BLEND.get(getattr(metadata, "wavelength_quality", None))
    if not enabled or blend is None or metadata.wavelength_unit != "nm":
        yield
        return
    if (adapter.attention_mode != "wavelength" or adapter.attention_dropout.p
            or not all(metadata.valid)):
        raise ValueError("Spacing requires wavelength attention, retained valid columns and zero attention dropout")
    native = adapter._attention

    def corrected(positions, widths):
        wave = positions.new_tensor(metadata.wavelengths)
        expected = adapter._normalize_wavelengths(wave, "nm")
        if not torch.equal(expected, positions):
            raise ValueError("Spacing metadata does not match the bands consumed by VCA")
        attention, valid, anchors = native(positions, widths)
        inside = (wave >= adapter.wavelength_min) & (wave <= adapter.wavelength_max)
        weights, valid = normalize_attention(
            attention, valid, anchors, positions, adapter.locality_window, inside, blend
        )
        return weights, valid, anchors

    # Disable the transient native attention cache while this operator is active.
    adapter.clear_attention_cache()
    try:
        with patch.object(adapter, "_attention", corrected), patch.object(adapter, "cache_attention", False):
            yield
    finally:
        adapter.clear_attention_cache()
