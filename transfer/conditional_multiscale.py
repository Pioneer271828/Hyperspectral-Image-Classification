"""Opt-in WP07 receptive-field controls on the existing VCA backbone."""

from contextlib import contextmanager
from contextvars import ContextVar
import math

import torch
from torch import nn
from torch.nn import functional as F

from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from transfer.multiscale_region_graph import GRAPH_RADII, MultiRangeBackbone


_observation = ContextVar("multiscale_observation", default=None)
_partition_bands = ContextVar("multiscale_partition_bands", default=None)
DILATIONS = (1, 2, 3)
MODES = ("fixed", "mean", "content", "metadata", "joint")


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
        if mode not in MODES or type(dilation) is not int or dilation not in DILATIONS:
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
    """Select CNN and graph scales independently through the three-head model."""

    def __init__(self, class_counts, *, cnn_multiscale="fixed", cnn_dilation=1,
                 graph_multiscale="off", graph_radius=1, **backbone_kwargs):
        if graph_multiscale not in ("off", *MODES) or type(graph_radius) is not int or graph_radius not in GRAPH_RADII:
            raise ValueError("Explicit graph multiscale mode and radius 1 or 2 required")
        if graph_multiscale != "fixed" and graph_radius != 1:
            raise ValueError("A single graph radius can only be chosen in fixed mode")
        if graph_multiscale != "off":
            if backbone_kwargs.get("graph_mode") != "learned":
                raise ValueError("Graph range comparisons require graph_mode=learned")
            backbone_kwargs["learned_graph_radius"] = graph_radius if graph_multiscale == "fixed" else 1
        super().__init__(class_counts, **backbone_kwargs)
        self.graph_multiscale, self.graph_radius = graph_multiscale, graph_radius
        with torch.random.fork_rng(devices=[]):
            if graph_multiscale not in ("off", "fixed"):
                backbone = MultiRangeBackbone(range_mode=graph_multiscale, condition_encoder=ObservationConditionEncoder,
                                              **self.backbone_kwargs)
                missing, unexpected = backbone.load_state_dict(self.backbone.state_dict(), strict=False)
                if unexpected or any(not key.startswith(("range_content.", "range_encoder.", "range_metadata."))
                                     for key in missing):
                    raise RuntimeError("The graph comparison must share the full baseline backbone state")
                self.backbone = backbone
        with torch.random.fork_rng(devices=[]):
            self.backbone.CNN_Branch[0] = ConditionalSpatialConv(self.backbone.CNN_Branch[0], cnn_multiscale, cnn_dilation)

    def export_config(self):
        config = super().export_config()
        first = self.backbone.CNN_Branch[0]
        config["backbone_kwargs"].update(cnn_multiscale=first.mode, cnn_dilation=first.dilation)
        if self.graph_multiscale != "off":
            config["backbone_kwargs"].update(graph_multiscale=self.graph_multiscale, graph_radius=self.graph_radius)
        return config

    @contextmanager
    def partition_bands(self, bands):
        """Bind the metadata of the exact spectral view passed to the backbone."""
        token = _partition_bands.set((self, bands))
        try:
            yield
        finally:
            _partition_bands.reset(token)

    def forward(self, dataset_name, x, Q=None, A=None, wavelengths=None, fwhm=None, superpixel_scale=100,
                return_graph_info=False, return_diagnostics=False, graph_state=None, return_logits=False,
                wavelength_unit="auto", fwhm_unit="auto", coverage_metadata=None, coverage_reference_nm=None,
                *, band_metadata=None, coverage_fraction=None, gsd_m=None):
        if self.graph_multiscale != "off" and any(value is not None for value in (Q, A, graph_state)):
            raise ValueError("Graph range comparisons build their supports internally")
        first = self.backbone.CNN_Branch[0]
        bound = _partition_bands.get()
        if bound is not None and bound[0] is self and band_metadata is None:
            band_metadata = bound[1]
        observation = (band_metadata, coverage_fraction, gsd_m)
        token = _observation.set((first, *observation))
        args = (dataset_name, x, Q, A, wavelengths, fwhm, superpixel_scale)
        kwargs = dict(return_graph_info=return_graph_info, return_diagnostics=return_diagnostics,
                      graph_state=graph_state, return_logits=return_logits, wavelength_unit=wavelength_unit,
                      fwhm_unit=fwhm_unit, coverage_metadata=coverage_metadata, coverage_reference_nm=coverage_reference_nm)
        try:
            if isinstance(self.backbone, MultiRangeBackbone):
                return self.backbone.forward_observed(observation, super().forward, *args, **kwargs)
            output = super().forward(*args, **kwargs)
            if self.graph_multiscale == "fixed" and return_graph_info:
                output[-1]["multiscale_graph"] = {"radii": (self.graph_radius,), "weights": x.new_ones(1),
                                                 "nnz": (output[-1]["graph_nnz"],)}
            return output
        finally:
            _observation.reset(token)
