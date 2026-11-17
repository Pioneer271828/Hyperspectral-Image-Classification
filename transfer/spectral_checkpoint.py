"""Opt-in checkpoints that only accept the registered spatial observation."""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import io
import json
import math
from pathlib import Path

import torch

from loadData.observation_sampling import sample
from loadData.spatial_protocol import forward_partition
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from transfer.spectral_registration import restore, verify_registration


FORMAT = "registered-spectral-observation-checkpoint-v1"


def _diagnostics(bands):
    return dict(metadata_hash=bands.fingerprint, source_id=bands.source_id,
                original_band_ids=bands.band_ids, wavelength_quality=bands.wavelength_quality,
                fwhm_available=bands.fwhm is not None)


def adapt_spectrum(adapter, values, bands):
    """Apply the metadata's validity mask using the original two-argument attention path."""
    latent, observed = adapter(values, wavelengths=bands.wavelengths, fwhm=bands.fwhm,
                              wavelength_unit=bands.wavelength_unit, fwhm_unit=bands.fwhm_unit or "auto",
                              band_mask=None if all(bands.valid) else bands.valid, return_attention=True)
    bands = bands.subset(observed["band_indices"].detach().cpu().tolist())
    return latent, {**observed, **_diagnostics(bands)}


def _forward_view(model, dataset, view, superpixel_scale, coverage_reference_nm):
    bands = view.bands
    heads = forward_partition(model, dataset, view.observation, wavelengths=bands.wavelengths,
                              superpixel_scale=superpixel_scale, fwhm=bands.fwhm,
                              wavelength_unit=bands.wavelength_unit, fwhm_unit=bands.fwhm_unit,
                              coverage_metadata=bands, coverage_reference_nm=coverage_reference_nm)
    return (*heads[:3], {**heads[3], "spectral_metadata": _diagnostics(bands)})


def forward_spectral_partition(model, dataset, observation, bands, *, superpixel_scale=100,
                               coverage_reference_nm=None):
    """Forward an explicitly calibrated view without creating a registered checkpoint."""
    view, _ = sample(observation, bands, {"mode": "none"})
    return _forward_view(model, dataset, view, superpixel_scale, coverage_reference_nm)


def _protocol(superpixel_scale, coverage_reference_nm):
    if type(superpixel_scale) not in (int, float) or not math.isfinite(superpixel_scale) or superpixel_scale <= 0:
        raise ValueError("Registered superpixel scale must be positive and finite")
    if coverage_reference_nm is not None:
        if (not isinstance(coverage_reference_nm, (tuple, list)) or len(coverage_reference_nm) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in coverage_reference_nm)
                or not 0 < coverage_reference_nm[0] < coverage_reference_nm[1]):
            raise ValueError("Registered coverage reference must give ordered physical bounds")
        coverage_reference_nm = list(coverage_reference_nm)
    return dict(superpixel_scale=superpixel_scale, coverage_reference_nm=coverage_reference_nm)


def _check_weights(weights):
    if (not isinstance(weights, Mapping) or not weights
            or any(not isinstance(t, torch.Tensor) or t.is_complex()
                   or (t.is_floating_point() and t.dtype != torch.float32) for t in weights.values())):
        raise ValueError("Registered spectral checkpoint requires FP32 weights and floating buffers")


def save_registered_checkpoint(model, path, *, registration, superpixel_scale=100, coverage_reference_nm=None):
    if type(model) is not MultiDatasetVCASAHGCN:
        raise ValueError("Registered spectral checkpoints require the original MultiDatasetVCASAHGCN model type")
    metadata = model.export_config()
    verify_registration(registration, tuple(metadata["class_counts"]))
    if model.backbone.graph_mode != "learned":
        raise ValueError("Registered spectral checkpoints require a label-free learned graph")
    protocol = _protocol(superpixel_scale, coverage_reference_nm)
    weights = model.state_dict()
    _check_weights(weights)
    payload = dict(format=FORMAT, metadata=json.dumps(metadata, allow_nan=False), protocol=json.dumps(protocol),
                   registration=json.dumps(registration, allow_nan=False), weights=weights)
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


class RegisteredSpectralCheckpoint:
    """Expose a spatial entry that sees pixel IDs, plan, partition and model input together."""

    def __init__(self, model, registration, protocol):
        self._model = model.eval()
        self._registration = deepcopy(registration)
        self._protocol = deepcopy(protocol)

    @torch.no_grad()
    def forward_partition(self, dataset, observation, bands):
        view = restore(self._registration, dataset, observation, bands)
        return _forward_view(self._model, dataset, view, **self._protocol)


def load_registered_checkpoint(path, *, expected_sha256, registration, device="cpu"):
    """Require the separately pinned package bytes and observation before returning a reader."""
    verify_registration(registration, (registration.get("dataset"),) if isinstance(registration, dict) else ())
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Registered spectral checkpoint bytes differ from the expected digest")
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (not isinstance(value, dict) or set(value) != {"format", "metadata", "registration", "weights", "protocol"}
            or value["format"] != FORMAT):
        raise ValueError("Expected a registered spectral checkpoint, not a legacy state dictionary")
    metadata, recorded = (json.loads(value[key]) for key in ("metadata", "registration"))
    protocol = _protocol(**json.loads(value["protocol"]))
    verify_registration(recorded, tuple(metadata["class_counts"]))
    if digest(recorded) != digest(registration):
        raise ValueError("Checkpoint spectral registration differs")
    _check_weights(value["weights"])
    model = MultiDatasetVCASAHGCN(metadata["class_counts"], **metadata["backbone_kwargs"])
    model.to(device=device, dtype=torch.float32)
    if model.backbone.graph_mode != "learned":
        raise ValueError("Registered spectral checkpoints require a label-free learned graph")
    expected = model.state_dict()
    if (set(expected) != set(value["weights"])
            or any(expected[k].dtype != t.dtype for k, t in value["weights"].items())):
        raise ValueError("Registered checkpoint parameter or buffer dtype differs from the original model")
    model.load_state_dict(value["weights"], strict=True)
    return RegisteredSpectralCheckpoint(model, recorded, protocol)
