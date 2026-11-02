# E1-B: observation coverage in the production training path

Implements [Issue #87](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/87), depending on accepted E1-A / PR #86. This module integrates the existing descriptor and bounded affine operator; it does not add another research mechanism. The [E1 paper/code review](../references/e1-reference-review.json) retains original FiLM/DOFA evidence, pinned source and license scope. No external source, trainer, dependency or data protocol is imported. See [E1-A mathematics](observation-coverage-e1.md) and the [implementation/validation report](../../research/reports/E1B_覆盖度骨干训练与恢复接入验收_20261102.md).

## Configuration and placement

`ResearchConfig`, `SpatialConfig`, the multidataset CLI and backbone constructor accept:

| Constructor/config | CLI | Default |
| --- | --- | --- |
| `coverage_mode` | `--coverage-mode` | `none` (`affine` opts in) |
| `coverage_hidden` | `--coverage-hidden` | 16 |
| `coverage_strength` | `--coverage-strength` | 0.1 |
| `coverage_sigma` | `--coverage-sigma` | 0.08 |
| `coverage_window` | `--coverage-window` | 0.16 |
| `coverage_init_seed` | `--coverage-init-seed` | 0 |
| `coverage_descriptor_version` | `--coverage-descriptor-version` | 1 |

Enabled placement is spectral input → VCA → coverage modulation → append unchanged coordinates → denoising → graph construction → both branches. The common backbone covers static, learned and injected graphs, interaction off/on, and M4 off/on without choosing those switches. At hidden16/value_dim1 the modulator adds **194 parameters**, independent of K. It is registered inside `backbone`, so the existing backbone optimizer group includes it. Head-only transfer freezes it with the backbone. No optimizer group, LR, loss, sampling, budget or selection default changes.

`none` creates no coverage parameters/buffers and executes the original adapter call. All-default coverage settings are omitted from canonical config serialization and checkpoint backbone exports, preserving old default config hashes and state keys. Explicit nondefault settings remain recorded even if the mode is off; the legacy CLI's raw argument record also retains its new flags. Affine initialization preserves existing backbone/head initialization and global RNG; its zero output layer initially implements identity. Earlier-layer gradients become nonzero after the output layer learns.

## Observation identity and fallback

The multidataset/single wrapper and backbone forwards accept `coverage_metadata: BandMetadata | None` and optional `coverage_reference_nm`. Enabled calls check retained B, valid columns, wavelength/FWHM availability, exact values **after input-dtype conversion**, and units against the actual VCA inputs. Use the real subset API first; full-scene metadata cannot be reused for a shorter tensor. This binds declared metadata to consumed values, but cannot establish that externally supplied physical centers are truthful.

Use explicit metadata units, especially for physical values at or below the legacy auto-unit threshold. The integration does not silently reinterpret VCA inputs. Metadata quality never follows a numerical heuristic: `read_band_metadata` marks known loader centers `approximate`, unknown index positions `index_only`; `readWavelengths` remains unchanged.

Descriptors use actual VCA anchor order and fixed global nm reference. The usable mask is intersected with **VCA's actual anchor validity**, including when the coverage window is wider than VCA's locality window. Unsupported anchors cannot be revived by a learned offset. Missing metadata, index-only quality, unresolved normalized physical reference and index/uniform VCA modes bypass physical conditioning, even with trained nonzero biases. Normalized physical centers require an explicit reference exactly matching the configured global range. There is no last-observation cache or automatic subset min/max normalization.

`return_graph_info=True` on an enabled model adds `coverage` with settings, metadata fingerprint, fallback reason and usable-anchor count; `return_diagnostics=True` also adds detached descriptor features/mask. Default-off graph-info keys remain unchanged. Diagnostics are not persistent checkpoint state. Fallback means conditioning is bypassed, not that the classifier output is guaranteed sensor invariant.

## Training and restoration contracts

- Common G3 source/target/development forwards carry the actual observation-view metadata, including shortened M1 tensors. Development validation replays the frozen policy. The legacy support-zero control still describes the full observed column set.
- Spatial custom loaders may return the original `(cube, labels, identity, wave)` or add a fifth `BandMetadata`. Four values with E1 enabled intentionally use missing-metadata fallback. The builtin loader supplies conservative metadata. Spatial evaluation restores this saved context when an explicit replacement is absent.
- Legacy multidataset training, validation and final evaluation pass prepared metadata and explicit units. Existing label-local zero/mixup augmentation retains the original wavelengths/column set; it is not relabeled as actual sensor removal.
- Enabled spatial/legacy verified resume includes metadata identity. Changed quality/source/widths are rejected even when image and wavelength values are unchanged. A completed run is skipped only after artifact verification; this is not exact optimizer-state resumption.
- Existing classifier checkpoint sidecars export all enabled E1 settings and strict weights. Old default sidecars remain readable without rewriting. A raw enabled state dictionary needs explicit architecture settings. Strict tensors cannot detect every same-shape semantic mismatch in an arbitrary raw config: callers must retain the correct config. G3 transfer explicitly rejects mismatched E1 mode, strength, geometry, seed/schema, global reference or adapter mode before loading a source backbone.
- G3 provenance records actual settings, base metadata/hash, registered/optimized parameter counts and optimizer enrollment. Trace hashes identify actual full/subset observations; verification cross-checks policy, sidecar, runtime settings and artifact hashes. These are consistency checks, not protection against adversarial recomputation of every artifact.

Existing prediction/export utilities outside these covered paths do not automatically gain physical metadata. Direct inference must supply it or intentionally accept fallback. E1-C must use the same explicit wavelength/FWHM inputs on both arms; do not compare E1-on physical widths with a legacy path that ignored them. Current Xuzhou metadata has no FWHM, so this difference does not enter the paired preflight.

## Remote acceptance

Run only in an authorized training container, using a clean pushed full SHA/full-history checkout and a new external result directory:

```bash
python scripts/validate_e1b.py --expected-commit FULL_SHA --out-dir NEW_EXTERNAL_DIR \
  --development-root FROZEN_DEV_ROOT --protocol-root FROZEN_PROTOCOL_ROOT \
  --data-root DATA_ROOT --legacy-checkpoint READ_ONLY_M1C_CHECKPOINT
```

The validator runs E1-B focused CPU/CUDA tests, full CPU regression, E1-A/M1/M4 CUDA regressions, VCA CUDA cache lifecycle, static parsing and frozen-source checks. Compatibility tests load the actual pre-E1-B model/config source from Git. Native FP32 CUDA forward comparisons use the established bounded tolerance, with FP64 for paired gradient checks.

The unchanged M1 native CUDA9 suite passed once but twice exceeded its old/new NLL bound by a small amount (3.05175781e-6 versus3e-6). Current-code repeats varied by up to2.88221571e-6. Strict deterministic execution is unavailable for the existing adaptive-pool CUDA backward. The final acceptance separates six native GPU functional tests from three numerical old/new, repeat and full-policy controls: those three run in explicitly typed FP64 with **additional1e-10 NLL/selected-weight bounds** in the E1-B suite. The original CPU9 tests and production precision/selection remain unchanged. Failed attempts, failed diagnostic setups and the adjustment are recorded in Issue #87/report; do not describe the original native FP32 suite as unconditionally stable.

The optional real preflight preregisters **two arms × three updates**, the same model64/head66/sampler68, full/lower-half-range/middle-gap schedule, released devtrain90/devval45, and the existing minimum-full-validation-NLL selection. It verifies registered gradients/updates, actual metadata, unchanged common initialization/sampling, selected integrated checkpoint restoration and a temporary strength-zero intervention. It reads no final test labels, performs no seed/budget/strength search and leaves original inputs/legacy model files unchanged. Three updates do not test convergence or accuracy improvement. Every attempted acceptance retains its own logs and status.

The next boundary is **E1-C**, a separately registered efficacy comparison with no modulation, same-capacity unconditional, binary, continuous and shuffled descriptions under matched observations/budgets/supports/selection. Protect full-observation accuracy and report per-class/branch/coverage results. Real transfer/generalization remains a separate evaluation; M1-C did not justify enabling strong subset training by default.
