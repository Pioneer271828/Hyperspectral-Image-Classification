import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class VariableChannelAdapter(nn.Module):
    """
    Map an arbitrary number of spectral bands to a fixed latent spectral grid.

    The default adapter uses wavelength-aware attention:
        attention(anchor_k, band_b) -> latent channel k

    attention_mode controls the ablation:
        wavelength: use physical wavelength/fwhm metadata.
        index: use normalized band index only, without wavelength metadata.
        uniform: average all bands for every latent anchor.

    Input shapes:
        [H, W, B] or [N, B, H, W]
    Output shapes:
        [H, W, K * value_dim] or [N, K * value_dim, H, W]

    Optional wavelength_unit/fwhm_unit: auto (legacy heuristic), nm, normalized.
    Missing wavelengths use normalized indices and are marked in diagnostics.
    band_mask is a boolean vector over the original B bands; removed bands and
    their metadata are excluded before attention/value projection.
    Inference attention caching is bounded and bypassed whenever autograd or
    autocast is enabled. Use clear_attention_cache() after unsupported .data
    writes or external changes that bypass PyTorch tensor version counters.
    """

    def __init__(
        self,
        latent_channels=128,
        embed_dim=64,
        value_dim=1,
        num_frequencies=6,
        wavelength_min=400.0,
        wavelength_max=2500.0,
        learnable_anchor_offsets=True,
        max_anchor_offset=0.025,
        locality_sigma=0.08,
        locality_window=0.16,
        attention_dropout=0.0,
        attention_mode="wavelength",
        chunk_pixels=0,
        checkpoint_chunks=False,
        cache_attention=True,
    ):
        super(VariableChannelAdapter, self).__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if wavelength_max <= wavelength_min:
            raise ValueError("wavelength_max must be greater than wavelength_min")
        if attention_mode not in ["wavelength", "index", "uniform"]:
            raise ValueError("attention_mode must be one of: wavelength, index, uniform")

        self.latent_channels = latent_channels
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.out_channels = latent_channels * value_dim
        self.num_frequencies = num_frequencies
        self.wavelength_min = float(wavelength_min)
        self.wavelength_max = float(wavelength_max)
        self.max_anchor_offset = float(max_anchor_offset)
        self.locality_sigma = locality_sigma
        self.locality_window = locality_window
        self.attention_mode = attention_mode
        self.chunk_pixels = int(chunk_pixels)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.cache_attention = bool(cache_attention)
        if self.chunk_pixels < 0:
            raise ValueError("chunk_pixels must be non-negative")

        self.register_buffer("anchor_base", torch.linspace(0.0, 1.0, steps=latent_channels))
        self.register_buffer("frequencies", torch.pow(2.0, torch.arange(num_frequencies)) * math.pi)

        if learnable_anchor_offsets:
            self.anchor_offsets = nn.Parameter(torch.zeros(latent_channels))
        else:
            self.register_buffer("anchor_offsets", torch.zeros(latent_channels))

        metadata_dim = 2 + 2 * num_frequencies
        self.query_encoder = nn.Sequential(
            nn.Linear(metadata_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(metadata_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(1, max(embed_dim // 2, value_dim)),
            nn.GELU(),
            nn.Linear(max(embed_dim // 2, value_dim), value_dim),
        )
        self.attention_dropout = nn.Dropout(attention_dropout)
        self._eval_attention_cache = {}

    def _project_to_latent(self, flat_x, attention):
        values = self.value_proj(flat_x.unsqueeze(-1))
        return torch.einsum("kb,pbd->pkd", attention, values)

    def _project_to_latent_chunked(self, flat_x, attention):
        if self.chunk_pixels <= 0 or flat_x.shape[0] <= self.chunk_pixels:
            return self._project_to_latent(flat_x, attention)

        chunks = []
        for start in range(0, flat_x.shape[0], self.chunk_pixels):
            flat_chunk = flat_x[start:start + self.chunk_pixels]
            use_checkpoint = (
                self.training
                and self.checkpoint_chunks
                and torch.is_grad_enabled()
                and (flat_chunk.requires_grad or attention.requires_grad)
            )
            if use_checkpoint:
                latent_chunk = checkpoint(
                    self._project_to_latent,
                    flat_chunk,
                    attention,
                    use_reentrant=False,
                )
            else:
                latent_chunk = self._project_to_latent(flat_chunk, attention)
            chunks.append(latent_chunk)
        return torch.cat(chunks, dim=0)

    def clear_attention_cache(self):
        """Drop transient inference state; it is never part of state_dict."""
        self._eval_attention_cache = {}

    def train(self, mode=True):
        self.clear_attention_cache()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.clear_attention_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        # This hook also runs when a parent module loads its checkpoint.
        self.clear_attention_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def _to_1d_tensor(self, values, length, device, dtype, name):
        if values is None:
            return torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype)
        if torch.is_tensor(values):
            values = values.to(device=device, dtype=dtype)
        else:
            values = torch.as_tensor(values, device=device, dtype=dtype)
        values = values.reshape(-1)
        if values.numel() != length:
            raise ValueError("{} length must match spectral bands: {} vs {}".format(name, values.numel(), length))
        return values

    def _normalize_wavelengths(self, wavelengths, unit="auto"):
        # Treat [0, 1]-like metadata as already normalized. Known nanometer
        # metadata is mapped onto the global anchor range.
        if wavelengths.numel() == 0:
            raise ValueError("wavelengths cannot be empty")
        if unit == "normalized" or (unit == "auto" and wavelengths.max() <= 2.0 and wavelengths.min() >= -0.5):
            return wavelengths.clamp(0.0, 1.0)
        scale = self.wavelength_max - self.wavelength_min
        return ((wavelengths - self.wavelength_min) / scale).clamp(0.0, 1.0)

    def _normalize_fwhm(self, fwhm, wavelengths, unit="auto"):
        if fwhm is None:
            return torch.zeros_like(wavelengths)
        if unit == "normalized" or (unit == "auto" and fwhm.max() <= 2.0 and fwhm.min() >= 0.0):
            return fwhm.clamp(min=0.0)
        scale = self.wavelength_max - self.wavelength_min
        return (fwhm / scale).clamp(min=0.0)

    def _metadata_features(self, wavelength_norm, fwhm_norm):
        phase = wavelength_norm.unsqueeze(1) * self.frequencies.to(
            device=wavelength_norm.device,
            dtype=wavelength_norm.dtype,
        ).unsqueeze(0)
        return torch.cat(
            [
                wavelength_norm.unsqueeze(1),
                torch.sin(phase),
                torch.cos(phase),
                fwhm_norm.unsqueeze(1),
            ],
            dim=1,
        )

    def _anchors(self, device, dtype):
        offsets = torch.tanh(self.anchor_offsets.to(device=device, dtype=dtype)) * self.max_anchor_offset
        return (self.anchor_base.to(device=device, dtype=dtype) + offsets).clamp(0.0, 1.0)

    def _attention(self, band_positions, fwhm):
        anchors = self._anchors(band_positions.device, band_positions.dtype)
        if self.attention_mode == "uniform":
            band_count = band_positions.numel()
            attention = torch.full(
                (self.latent_channels, band_count),
                1.0 / max(int(band_count), 1),
                device=band_positions.device,
                dtype=band_positions.dtype,
            )
            anchor_valid = torch.ones(self.latent_channels, device=band_positions.device, dtype=band_positions.dtype)
            return attention, anchor_valid, anchors

        anchor_fwhm = torch.zeros_like(anchors)

        anchor_meta = self._metadata_features(anchors, anchor_fwhm)
        band_meta = self._metadata_features(band_positions, fwhm)
        query = self.query_encoder(anchor_meta)
        key = self.key_encoder(band_meta)

        logits = query.matmul(key.t()) / math.sqrt(self.embed_dim)
        distance = torch.abs(anchors.unsqueeze(1) - band_positions.unsqueeze(0))

        if self.locality_sigma is not None and self.locality_sigma > 0:
            sigma = torch.as_tensor(self.locality_sigma, device=logits.device, dtype=logits.dtype)
            logits = logits - distance.pow(2) / (2.0 * sigma.pow(2))

        if self.locality_window is not None and self.locality_window > 0:
            local_mask = distance <= self.locality_window
            has_local_band = local_mask.any(dim=1, keepdim=True)
            logits = logits.masked_fill(has_local_band.logical_and(local_mask.logical_not()), -1e4)
            anchor_valid = has_local_band.squeeze(1)
        else:
            anchor_valid = torch.ones(self.latent_channels, device=logits.device, dtype=torch.bool)

        attention = F.softmax(logits, dim=1)
        attention = self.attention_dropout(attention)
        return attention, anchor_valid.to(dtype=attention.dtype), anchors

    def _attention_cache_key(self, band_positions, fwhm):
        if (not self.cache_attention or self.training or torch.is_grad_enabled()
                or self.attention_dropout.p > 0
                or torch.is_autocast_enabled(band_positions.device.type)):
            return None
        try:
            # Optimizers and normal in-place updates increment tensor versions.
            # Identity also catches replacement via load_state_dict(assign=True).
            state = tuple((name, id(tensor), tensor._version)
                          for name, tensor in list(self.named_parameters()) + list(self.named_buffers()))
        except RuntimeError:
            # Tensors constructed in inference_mode have no version counter.
            return None
        # Exact effective metadata: rounding could alias distinct observations.
        wave_key = tuple(band_positions.detach().cpu().tolist())
        fwhm_key = tuple(fwhm.detach().cpu().tolist())
        return (
            self.attention_mode,
            self.max_anchor_offset, self.locality_sigma, self.locality_window,
            self.embed_dim, self.latent_channels, state,
            str(band_positions.device),
            str(band_positions.dtype),
            wave_key,
            fwhm_key,
        )

    def _get_attention(self, band_positions, fwhm):
        cache_key = self._attention_cache_key(band_positions, fwhm)
        if cache_key is not None and cache_key in self._eval_attention_cache:
            # Diagnostics returned to callers must not mutate the stored entry.
            return tuple(tensor.clone() for tensor in self._eval_attention_cache[cache_key])

        self.clear_attention_cache()
        attention, anchor_valid, anchors = self._attention(band_positions, fwhm)
        if cache_key is not None:
            # Keep only the most recent observation, rather than every subset.
            self._eval_attention_cache[cache_key] = (
                attention.detach().clone(),
                anchor_valid.detach().clone(),
                anchors.detach().clone(),
            )
        return attention, anchor_valid, anchors

    def _adapt_flat(self, flat_x, wavelengths=None, fwhm=None, return_attention=False,
                    wavelength_unit="auto", fwhm_unit="auto", band_mask=None):
        num_bands = flat_x.shape[1]
        if num_bands == 0:
            raise ValueError("at least one spectral band is required")
        for name, unit in (("wavelength_unit", wavelength_unit), ("fwhm_unit", fwhm_unit)):
            if unit not in ("auto", "nm", "normalized"):
                raise ValueError(name + " must be auto, nm or normalized")
        has_wavelengths = wavelengths is not None
        if not has_wavelengths and wavelength_unit == "nm":
            raise ValueError("wavelength_unit=nm requires wavelengths")
        wavelengths = self._to_1d_tensor(wavelengths, num_bands, flat_x.device, flat_x.dtype, "wavelengths")
        fwhm = self._to_1d_tensor(fwhm, num_bands, flat_x.device, flat_x.dtype, "fwhm") if fwhm is not None else None
        band_indices = torch.arange(num_bands, device=flat_x.device)
        if band_mask is not None:
            band_mask = torch.as_tensor(band_mask, device=flat_x.device)
            if band_mask.dtype != torch.bool or band_mask.ndim != 1 or band_mask.numel() != num_bands:
                raise ValueError("band_mask must be a boolean vector matching spectral bands")
            if not band_mask.any():
                raise ValueError("band_mask must retain at least one spectral band")
            flat_x = flat_x[:, band_mask]
            wavelengths = wavelengths[band_mask]
            fwhm = fwhm[band_mask] if fwhm is not None else None
            band_indices = band_indices[band_mask]
            num_bands = flat_x.shape[1]
        if not torch.isfinite(wavelengths).all():
            raise ValueError("retained wavelengths must be finite")
        if fwhm is not None and (not torch.isfinite(fwhm).all() or (fwhm < 0).any()):
            raise ValueError("retained fwhm must be finite and nonnegative")
        resolved_wave_unit = wavelength_unit
        if wavelength_unit == "auto":
            resolved_wave_unit = "normalized" if wavelengths.max() <= 2.0 and wavelengths.min() >= -0.5 else "nm"
        resolved_fwhm_unit = fwhm_unit
        if fwhm is None:
            resolved_fwhm_unit = "missing"
        elif fwhm_unit == "auto":
            resolved_fwhm_unit = "normalized" if fwhm.max() <= 2.0 else "nm"
        wavelengths_norm = self._normalize_wavelengths(wavelengths, resolved_wave_unit)
        if self.attention_mode == "wavelength":
            band_positions = wavelengths_norm
            fwhm_norm = self._normalize_fwhm(fwhm, wavelengths, resolved_fwhm_unit)
        elif self.attention_mode == "index":
            band_positions = torch.linspace(0.0, 1.0, steps=num_bands, device=flat_x.device, dtype=flat_x.dtype)
            fwhm_norm = torch.zeros_like(band_positions)
        elif self.attention_mode == "uniform":
            band_positions = torch.linspace(0.0, 1.0, steps=num_bands, device=flat_x.device, dtype=flat_x.dtype)
            fwhm_norm = torch.zeros_like(band_positions)
        else:
            raise RuntimeError("unexpected attention_mode: {}".format(self.attention_mode))

        attention, anchor_valid, anchors = self._get_attention(band_positions, fwhm_norm)
        latent = self._project_to_latent_chunked(flat_x, attention)
        latent = latent * anchor_valid.view(1, self.latent_channels, 1)
        latent = latent.reshape(flat_x.shape[0], self.out_channels)

        if return_attention:
            info = {
                "attention": attention,
                "anchor_valid": anchor_valid,
                "anchors": anchors,
                "wavelengths": wavelengths_norm,
                "band_positions": band_positions,
                "attention_mode": self.attention_mode,
                "band_indices": band_indices,
                "fwhm_normalized": fwhm_norm,
                "wavelength_unit": resolved_wave_unit if has_wavelengths else "normalized_index",
                "wavelength_source": "provided" if has_wavelengths else "index_fallback",
                "position_source": ("wavelength_metadata" if self.attention_mode == "wavelength" and has_wavelengths
                                    else "normalized_index"),
                "fwhm_unit": resolved_fwhm_unit,
            }
            return latent, info
        return latent

    def forward(self, x, wavelengths=None, fwhm=None, return_attention=False, *,
                wavelength_unit="auto", fwhm_unit="auto", band_mask=None):
        metadata_options = dict(wavelength_unit=wavelength_unit, fwhm_unit=fwhm_unit, band_mask=band_mask)
        if x.dim() == 3:
            height, width, bands = x.shape
            flat_x = x.reshape(height * width, bands)
            adapted = self._adapt_flat(flat_x, wavelengths=wavelengths, fwhm=fwhm, return_attention=return_attention, **metadata_options)
            if return_attention:
                adapted, info = adapted
                return adapted.reshape(height, width, self.out_channels), info
            return adapted.reshape(height, width, self.out_channels)

        if x.dim() == 4:
            batch, bands, height, width = x.shape
            flat_x = x.permute(0, 2, 3, 1).reshape(batch * height * width, bands)
            adapted = self._adapt_flat(flat_x, wavelengths=wavelengths, fwhm=fwhm, return_attention=return_attention, **metadata_options)
            if return_attention:
                adapted, info = adapted
                adapted = adapted.reshape(batch, height, width, self.out_channels).permute(0, 3, 1, 2)
                return adapted, info
            return adapted.reshape(batch, height, width, self.out_channels).permute(0, 3, 1, 2)

        raise ValueError("VariableChannelAdapter expects [H, W, B] or [N, B, H, W] input")
