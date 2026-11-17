"""Read-only diagnostics for units and anchor coverage in the released VCA."""

from copy import deepcopy
import math

import torch

from loadData.data_read import read_band_metadata


SCENES = {"paviaU":103, "xuzhou":436, "whu-honghu":270}


def diagnostic(adapter, scene, *, dofa_position_encoder=None):
    meta = read_band_metadata(scene,SCENES[scene])
    wave = torch.tensor(meta.wavelengths,dtype=torch.float32)
    bounds = adapter.wavelength_min,adapter.wavelength_max
    rows = []
    for band,positions in (("full",torch.arange(wave.numel())),("even",torch.arange(0,wave.numel(),2))):
        w = wave[positions]
        for mode in ("wavelength","index"):
            candidate = deepcopy(adapter).cpu().eval()
            candidate.attention_mode = mode
            # A deterministic synthetic observation exercises the real metadata without loading labels.
            x = torch.linspace(0.,1.,steps=6*w.numel()).reshape(2,3,-1).requires_grad_()
            y,trace = candidate(x,w,return_attention=True,wavelength_unit="nm")
            valid = trace["anchor_valid"].bool()
            att = trace["attention"]
            entropy = -(att * att.clamp_min(torch.finfo(att.dtype).tiny).log()).sum(1)
            grad = torch.autograd.grad(y.square().mean(),tuple(candidate.parameters()),allow_unused=True)
            entries = {name:None if item is None else float(item.detach().double().norm())
                       for (name,_),item in zip(candidate.named_parameters(),grad,strict=True)}
            raw = (w-bounds[0])/(bounds[1]-bounds[0])
            rows.append(dict(scene=scene,mode=mode,band=band,kept=w.numel(),
                wavelength_quality=meta.wavelength_quality,wavelength_min_nm=float(w.min()),wavelength_max_nm=float(w.max()),
                outside_global_nm=int(((raw<0)|(raw>1)).sum()),wavelength_unit=trace["wavelength_unit"],
                fwhm_unit=trace["fwhm_unit"],active_anchors=int(valid.sum()),total_anchors=valid.numel(),
                entropy_mean_valid=(float(entropy[valid].mean().detach())/math.log(w.numel()) if valid.any() else None),
                gradient_l2=entries,optimizer_updates=0))
            if dofa_position_encoder is not None and mode=="wavelength":
                correct = dofa_position_encoder(16,(w/1000.)*1000.)
                direct = dofa_position_encoder(16,w)
                wrong = dofa_position_encoder(16,w*1000.)
                rows[-1]["dofa_encoding"] = dict(nm_vs_micrometre_max_abs=float((direct-correct).abs().max()),
                                                  nm_supplied_as_micrometre_max_abs=float((direct-wrong).abs().max()))
    return rows
