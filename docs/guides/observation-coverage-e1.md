# E1-A: standalone observation coverage and modulation

[Issue #85](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/85) follows the negative M1-C development screen. This module supplies an independently testable description and optional affine operator. Existing VCA/backbone/configuration/CLI/checkpoint defaults are unchanged. Production integration is **E1-B**; a registered accuracy comparison comes after integration acceptance.

## Evidence and reuse

Read the [reference review](../references/e1-reference-review.json) before extending this implementation. The project plans propose the coverage geometry. [FiLM v2 §2.1](https://arxiv.org/html/1709.07871v2) motivates conditional affine feature modulation; its author code at `fe43ddf8a22b339dcca2efa07091ce9d498955cf` was inspected along with the actual CC BY-NC 4.0 root license. [DOFA v1](https://arxiv.org/html/2403.15356v1) and author source at `0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9` (root MIT license) provide related wavelength-conditioning context. Neither supplies this coverage mechanism or establishes its HSI accuracy. This code independently implements the published affine idea and project-specific geometry; no external source/trainer/weights are copied or executed, no new dependency is installed.

## Description contract

`describe(BandMetadata | None, anchors, CoverageConfig, *, band_mask=None, normalized_reference_nm=None)` returns detached `CoverageDescription` tensors. Anchors are a nonempty float32/64 vector in the VCA's **fixed global normalized** order. Defaults match the existing 400–2500 nm reference, design sigma .08 and window half-width .16. Window must be at least1e-12 to retain numerical resolution. Do not fit a new min/max to each crop: the same number of columns at a different spectral range must produce different support.

For every anchor the nine features are:

| Feature | Definition |
|---|---|
| anchor_position | Global normalized anchor coordinate |
| nearest_distance | Distance to closest retained in-reference center, capped at1; unavailable=1 |
| soft_support | exp(−.5(distance/design_sigma)²); unavailable=0 |
| log_local_count | log1p(number of retained columns in the closed, clipped anchor window) |
| local_gap_fraction | Largest interval between unique local positions and window endpoints, divided by window width; unavailable=1 |
| physical_available | At least one usable physical center within the fixed reference exists |
| verified_centers | Metadata explicitly marked verified |
| approximate_centers | Metadata explicitly marked approximate |
| fwhm_present | FWHM supplied in the metadata; no width enters this geometry |

Raw integer counts and maximum gaps in global normalized units are also returned. Duplicate centers increase observation counts but not gap density. Values outside the fixed reference are counted in diagnostics and excluded from support, rather than clamped into fictitious endpoint observations. Approximate centers remain approximate. Kernel/window are design scales, never physical FWHM or sensor response functions. `usable` requires a retained center in that anchor's window; it is center support, not per-pixel confidence or a claim to reproduce every VCA mode's mask.

Missing metadata, index-only positions, unresolved normalized physical coordinates, no valid columns and no in-reference columns produce explicit finite fallback descriptions and `usable=False`. Normalized physical coordinates require `normalized_reference_nm=(config.wavelength_min,config.wavelength_max)`; an explicitly different reference is rejected. Bad shapes/units/masks/anchors/configurations raise errors rather than being disguised as missing metadata. A descriptor for zero effective columns does not authorize passing an empty-band input to VCA. No labels, image values, sensor ID or statistics fitting enter the description; metadata identity remains diagnostic only. Anchors and metadata are detached; geometry is not an auxiliary gradient path for learned VCA offsets. Nothing is cached or randomized.

## Modulation contract

`CoverageModulation(K, value_dim=1, mode='none', hidden=16, strength=.1, init_seed=0, descriptor_version=1)`:

- `none` has no parameters and returns the identical latent object. Zero strength and absent descriptions also return it unchanged.
- `affine` uses a shared nine-input MLP over each anchor; default194 parameters for value_dim1. The output is bounded residual modulation `X + strength*(tanh(gamma)*X + tanh(beta))`.
- Initialization uses a private seeded generator, leaving global CPU/CUDA RNG states unchanged. A zero final layer gives exact initial identity. Its hidden layer starts receiving gradients after the output layer learns; zero hidden gradient on the first update is expected.
- The true `usable` mask gates the output. Unsupported channels retain their original values even after learning or nonzero beta; zero VCA channels cannot be revived there.
- Support `[P,K*V]`/`[H,W,K*V]` with the default last channel axis, or `[N,K*V,H,W]` with explicit `channel_dim=1`. A batch shares one observation description; different sensor metadata require separate descriptions/forwards.
- Move the module and description to the same device/dtype as latent. Feature schema/order, shapes, devices and finite values are checked. Caller must bind descriptors to the same observation and anchor order; this standalone operator has no knowledge of a dataset/task identity.
- Save `export_config()` plus `state_dict()` and use strict loading. This is a standalone E1 checkpoint, not an integrated classifier checkpoint or optimizer resume.

Example composition in the **remote training container**:

```python
latent, info = adapter(x, wavelengths=bands.wavelengths, fwhm=bands.fwhm,
    wavelength_unit=bands.wavelength_unit, fwhm_unit=bands.fwhm_unit or 'auto',
    band_mask=bands.valid, return_attention=True)
config = CoverageConfig(wavelength_min=adapter.wavelength_min,
    wavelength_max=adapter.wavelength_max, sigma=.08, window=.16)
coverage = describe(bands, info['anchors'], config)
layer = CoverageModulation(adapter.latent_channels, adapter.value_dim,
    mode='affine').to(device=latent.device, dtype=latent.dtype)
updated = layer(latent, coverage)
```

## Acceptance and next boundary

`scripts/validate_e1a.py` requires a clean exact pushed SHA in the authorized Linux container, exclusive external output, full Git history and the unchanged base ancestry. Run focused CPU/CUDA and full CPU tests, existing M1 CUDA regressions and the established VCA CUDA cache lifecycle. Preserve failures and source fingerprints. Geometry tests include hand calculations, unit/provenance/mask/permutation/crop/duplicate/fallback cases; operator tests cover exact bypass/RNG, analytic output/gradients, real updates, unsupported protection, layouts, strict roundtrip and actual VCA composition.

```bash
python scripts/validate_e1a.py --expected-commit FULL_SHA --out-dir NEW_EXTERNAL_DIR \
  --development-root FROZEN_DEV_ROOT --protocol-root FROZEN_PROTOCOL_ROOT \
  --data-root DATA_ROOT --legacy-checkpoint READ_ONLY_M1C_CHECKPOINT
```

The optional real probe is **three optimizer updates** on one released devtrain tile: full, lower-half range crop and middle-gap. A temporary adapter-output hook composes E1 before denoising; E1 parameters are explicitly enrolled in the existing optimizer. The hook is removed, model/trainer source unchanged, outputs external. Record actual B/metadata/support, losses, gradients/updates and frozen-feature interventions. Save/reload the E1-only state and read-only load an existing baseline classifier checkpoint. No model accuracy, devval/test scoring or deployable E1 classifier checkpoint is produced. Native CUDA classifier equality uses a3e-6 bound with same-model repeat noise recorded; standalone identity checks remain exact.

E1-B must later connect explicit configuration/checkpoint/forward contracts, default-off equivalence and observation identity through the real trainer. Only then register matched no-modulation/same-capacity/binary/continuous/shuffled controls under the same augmentation and held-out coverage-range conditions. Preserve full accuracy and per-class/branch analysis. M4 remains disabled; E1-A engineering passing cannot substitute for those effects or actual sensor transfer evidence.
