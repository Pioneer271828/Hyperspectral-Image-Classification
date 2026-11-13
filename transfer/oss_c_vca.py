"""Compare out-of-reference band handling without moving global VCA anchors."""

import torch

from joint_adapter_modes import AdapterModeRuntime, adapter_settings
from model.variable_channel_adapter import VariableChannelAdapter


ARMS = ("native", "exclude_out_of_reference", "matched_boundary_dropout")


class GlobalReferenceAdapter(VariableChannelAdapter):
    """Keep the original parameters and gate only inputs to the spectral adapter."""

    def forward(self, x, wavelengths=None, fwhm=None, return_attention=False, *,
                wavelength_unit="auto", fwhm_unit="auto", band_mask=None):
        if wavelengths is not None and self.study_arm != "native":
            wave = torch.as_tensor(wavelengths, device=x.device)
            is_nm = wavelength_unit == "nm" or (wavelength_unit == "auto" and bool(
                torch.isfinite(wave).all() and (wave.max() > 2 or wave.min() < -.5)))
            width = x.shape[-1] if x.ndim == 3 else x.shape[1]
            if is_nm and wave.ndim == 1 and wave.numel() == width:
                if band_mask is None:
                    valid = torch.ones_like(wave, dtype=torch.bool)
                else:
                    valid = torch.as_tensor(band_mask, device=x.device)
                if valid.shape == wave.shape and valid.dtype == torch.bool and torch.isfinite(wave[valid]).all():
                    covered = valid & (wave >= self.wavelength_min) & (wave <= self.wavelength_max)
                    outside = valid & ~covered
                    count = int(outside.sum())
                    if count and count < int(covered.sum()):
                        if self.study_arm == "exclude_out_of_reference":
                            band_mask = covered
                        else:
                            # Remove exactly as many in-reference columns, starting at the
                            # physical boundary, with wavelength as the deterministic tie-break.
                            distance = torch.minimum(wave-self.wavelength_min, self.wavelength_max-wave)
                            order = torch.argsort(wave, stable=True)
                            order = order[torch.argsort(distance[order], stable=True)]
                            eligible = order[covered[order]]
                            chosen = eligible[:count]
                            # An indistinguishable tie would make the control depend on
                            # column order. Use the original path for such observations.
                            if not bool((wave[chosen,None]==wave[eligible[count:]]).any()):
                                band_mask = valid.clone()
                                band_mask[chosen] = False
        return super().forward(x, wavelengths, fwhm, return_attention,
                               wavelength_unit=wavelength_unit, fwhm_unit=fwhm_unit, band_mask=band_mask)


def attach(model, arm):
    if arm not in ARMS:
        raise ValueError("Unknown fixed-coordinate comparison arm")
    old = model.backbone.adapter
    if type(old) is not VariableChannelAdapter or old.attention_mode != "wavelength":
        raise ValueError("Original wavelength VCA required")
    if arm == "native":
        return model
    keys = ("latent_channels", "embed_dim", "value_dim", "num_frequencies", "wavelength_min",
            "wavelength_max", "max_anchor_offset", "locality_sigma", "locality_window",
            "attention_mode", "chunk_pixels", "checkpoint_chunks", "cache_attention")
    opts = {key:getattr(old, key) for key in keys}
    opts.update(learnable_anchor_offsets=isinstance(old.anchor_offsets, torch.nn.Parameter),
                attention_dropout=old.attention_dropout.p)
    with torch.random.fork_rng(devices=[]):
        new = GlobalReferenceAdapter(**opts).to(old.anchor_base)
    new.load_state_dict(old.state_dict(), strict=True)
    for name, parameter in new.named_parameters():
        parameter.requires_grad_(old.get_parameter(name).requires_grad)
    new.study_arm = arm
    new.train(old.training)
    model.backbone.adapter = new
    return model


def settings(base, arm):
    if arm not in ARMS:
        raise ValueError("Unknown fixed-coordinate comparison arm")
    value = adapter_settings(base, "wavelength")
    value["global_reference_arm"] = arm
    return value


class GlobalReferenceRuntime(AdapterModeRuntime):
    def __init__(self, arm):
        super().__init__("wavelength")
        if arm not in ARMS:
            raise ValueError("Unknown fixed-coordinate comparison arm")
        self.arm = arm

    def validate(self, options):
        if options.get("global_reference_arm") != self.arm or options["optimizer_steps"] != 12:
            raise ValueError("Fixed-coordinate arm or budget differs")
        return super().validate(options)

    def build_model(self, config, counts):
        return attach(super().build_model(config, counts), self.arm)

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        original = super().load_selected(path, items, raw_source_path, device=device)
        return attach(original, self.arm)
