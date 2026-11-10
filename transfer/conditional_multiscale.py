"""Opt-in WP07 CNN receptive-field controls on the existing VCA backbone."""

from contextvars import ContextVar
import math

import torch
from torch import nn
from torch.nn import functional as F

from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN


_observation = ContextVar("multiscale_observation", default=None)
DILATIONS = (1, 2, 3)


class ObservationConditionEncoder(nn.Module):
    """Encode verified physical observations without dataset or sensor IDs."""

    def __init__(self):
        super().__init__()
        self.band_encoder = nn.Sequential(nn.Linear(3, 16), nn.GELU())
        self.scene_encoder = nn.Sequential(nn.Linear(23, 32), nn.GELU())

    def forward(self, bands, coverage_fraction, gsd_m, reference):
        if bands is not None and not isinstance(bands, BandMetadata):
            raise ValueError("BandMetadata required for physical conditioning")
        if coverage_fraction is not None and (type(coverage_fraction) not in (int, float) or not 0 <= coverage_fraction <= 1):
            raise ValueError("Coverage fraction must be in [0, 1]")
        if gsd_m is not None and (type(gsd_m) not in (int, float) or not math.isfinite(gsd_m) or gsd_m <= 0):
            raise ValueError("Verified GSD must be positive meters per pixel")
        physical = bands is not None and bands.wavelength_unit == "nm" and bands.wavelength_quality == "verified"
        if not physical and coverage_fraction is None and gsd_m is None:
            return None
        if physical:
            # Stable ordering also makes the reduction reproducible under a band permutation.
            rows = sorted((wave / 2500, (bands.fwhm[i] / 2500) if bands.fwhm is not None else 0.,
                           float(bands.fwhm is not None)) for i, wave in enumerate(bands.wavelengths) if bands.valid[i])
            table = reference.new_tensor(rows)
            pooled = self.band_encoder(table).mean(dim=0)
            limits = table.new_tensor((rows[0][0], rows[-1][0]))
        else:
            pooled = reference.new_zeros(16)
            limits = reference.new_zeros(2)
        other = reference.new_tensor((float(physical), coverage_fraction if coverage_fraction is not None else 0.,
                                      float(coverage_fraction is not None), math.log(gsd_m) if gsd_m is not None else 0.,
                                      float(gsd_m is not None)))
        return self.scene_encoder(torch.cat((pooled, limits, other)))


class ConditionalSpatialConv(nn.Module):
    """Share the existing depthwise kernel across dilations and mix per scene."""

    def __init__(self, source, mode, dilation):
        super().__init__()
        if mode not in ("fixed", "mean", "content", "metadata", "joint") or type(dilation) is not int or dilation not in DILATIONS:
            raise ValueError("Explicit multiscale mode and dilation 1, 2 or 3 required")
        self.mode, self.dilation = mode, dilation
        self.norm, self.point_conv, self.depth_conv = source.norm, source.point_conv, source.depth_conv
        self.act1, self.act2 = source.act1, source.act2
        self.content_selector = nn.Linear(self.point_conv.out_channels, 3, bias=False) if mode in ("content", "joint") else None
        self.condition_encoder = ObservationConditionEncoder() if mode in ("metadata", "joint") else None
        self.condition_selector = nn.Linear(32, 3, bias=False) if self.condition_encoder is not None else None

    def _filter(self, x, dilation):
        if dilation == 1:
            return self.depth_conv(x)
        kernel = self.depth_conv
        padding = tuple(dilation * size for size in kernel.padding)
        return F.conv2d(x, kernel.weight, kernel.bias, padding=padding, dilation=dilation, groups=kernel.groups)

    def forward(self, x, dataset_name=None):
        features = self.act1(self.point_conv(self.norm(x, dataset_name)))
        if self.mode == "fixed":
            return self.act2(self._filter(features, self.dilation))
        logits = features.new_zeros((features.shape[0], 3))
        if self.content_selector is not None:
            logits = self.content_selector(features.mean(dim=(2, 3)))
        observation = _observation.get()
        if self.condition_encoder is not None and observation is not None and observation[0] is self:
            encoded = self.condition_encoder(*observation[1:], reference=features)
            if encoded is not None:
                logits = logits + self.condition_selector(encoded)
        weights = logits.softmax(dim=-1).unbind(dim=-1)
        combined = sum(self._filter(features, dilation) * weight[:, None, None, None]
                       for dilation, weight in zip(DILATIONS, weights, strict=True))
        return self.act2(combined)


class ConditionalMultiscaleVCASAHGCN(MultiDatasetVCASAHGCN):
    """Use multiscale filtering at the first CNN stage through the existing forward."""

    def __init__(self, class_counts, *, cnn_multiscale, cnn_dilation=1, **backbone_kwargs):
        super().__init__(class_counts, **backbone_kwargs)
        self.backbone.CNN_Branch[0] = ConditionalSpatialConv(self.backbone.CNN_Branch[0], cnn_multiscale, cnn_dilation)

    def export_config(self):
        config = super().export_config()
        first = self.backbone.CNN_Branch[0]
        config["backbone_kwargs"].update(cnn_multiscale=first.mode, cnn_dilation=first.dilation)
        return config

    def forward(self, dataset_name, x, *args, band_metadata=None, coverage_fraction=None, gsd_m=None, **kwargs):
        first = self.backbone.CNN_Branch[0]
        token = _observation.set((first, band_metadata, coverage_fraction, gsd_m))
        try:
            return super().forward(dataset_name, x, *args, **kwargs)
        finally:
            _observation.reset(token)
