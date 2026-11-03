# M1: real spectral subsets and synchronized metadata

`loadData.observation_sampling` is an independent, label-blind data API. It changes the actual spectral dimension B for the entire spatial block, while retaining the observation's coordinates, global pixel IDs, partition and plan identity. It does not modify VCA, old band-drop, default training, losses or sampling budgets. Smaller B is an observation change, not network pruning or changing VCA latent K.

```python
from loadData.observation_sampling import BandMetadata, sample, replay
import numpy as np

bands = BandMetadata(
    source_id=image_sha256, band_ids=tuple(range(B)),
    wavelengths=tuple(wavelengths), wavelength_unit='nm',
    wavelength_quality='approximate', valid=(True,)*B)
view, record = sample(observation, bands,
                      {'mode':'random_subset', 'keep_fraction':0.5},
                      np.random.default_rng(2026))
# view.observation.spectral has fewer columns for ALL pixels.
# view.bands contains matching wavelengths, IDs, validity and any provided FWHM.
# Save record as JSON outside the checkout. Fixed evaluation consumes no RNG:
restored_view = replay(observation, bands, record)
```

The API signature refines the planned interface to `sample(observation, bands, config, rng=None) -> (ObservationView, record)`: metadata is explicit; labels are not an argument. The wrapper has `.observation` and `.bands`. Integrate it in a new explicit training task before calling the existing forward function, supplying the selected wavelengths; do not continue passing the original full-length vector. No training CLI option is added by this module.

Modes and exact fields:

| Mode | Configuration | Selection |
| --- | --- | --- |
| none | mode only | All valid bands; all-valid input returns the same observation |
| random_subset | keep_fraction in (0,1], explicit NumPy Generator | ceil(valid B × fraction), minimum one, without replacement; preserve source order |
| uniform_stride | positive integer stride, integer offset in [0,stride) | Original band_id modulo stride equals offset, then intersect validity |
| contiguous_gap | lower, upper, unit | Remove valid wavelengths in the inclusive interval |
| range_crop | lower, upper, unit | Keep valid wavelengths in the inclusive interval |

Unknown/missing/extra config fields, empty results, duplicate IDs, mismatched lengths, nonboolean validity, nonfinite metadata and invalid units fail explicitly. IDs are source observation column identities, not automatically reconstructed hardware channel numbers. Selections always retain the source order, even if wavelengths are not sorted. `valid=False` bands are filtered before sampling.

Metadata is immutable tuple-based. Units must be nm or normalized, quality verified/approximate/index_only. Index-only positions must be normalized, cannot be relabelled nm, and cannot drive a claimed physical range/gap operation. Preserve original normalized positions when subsetting; do not renormalize each selected range. Approximate center vectors remain approximate after sampling. Provided FWHM must be finite, positive and use the wavelength unit; absent FWHM remains None with no invented bandwidth. This API conservatively requires finite wavelength placeholders even for invalid bands. It does not model spectral response integration, noise, spatial resolution or a real sensor's FWHM.

Records bind source metadata fingerprint, source B, spatial plan, partition, configuration, selected positions/IDs/metadata hash and optional explicit RNG state fingerprints. Replay checks the record digest and identities without generating indices or consuming RNG. The digest detects accidental changes, not authenticity against a caller recomputing it. The caller must provide truthful source identity and retain code/data provenance. Same scene/partition/plan records can be replayed on other spatial blocks; do not mix train/test records or share their RNG streams. Test perturbation lists must be frozen before final scoring.

`subset_statistics(stats, bands, record)` checks original train-fitted min/max metadata and returns a derived slice with parent hash; no new fitting is performed. Applying selection after the existing per-band normalization is equivalent to slicing those original statistics before normalization. The derived record is not an ordinary full-plan statistics object and must not be fed back into `observe_tile`. Coordinates and label geometry are untouched. Source arrays are not modified; newly selected spectral arrays are read-only.

VCA integration equivalence is tested in wavelength mode: physically sliced data/metadata matches full input with band_mask for outputs and parameter gradients, including one visible band. No classifier accuracy benefit is claimed. Existing unit handling elsewhere is unchanged; callers must preserve explicit metadata quality and units rather than infer physical wavelengths from band count.

Remote commands:

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p 'test_*.py' -v
M1_DEVICE=cuda OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p test_observation_sampling.py -v
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python scripts/validate_m1.py \
  --expected-commit FULL_COMMIT_SHA \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/UNUSED_M1_VALIDATION_DIRECTORY
```

The real-data validator checks three datasets and eight selections each on a training block. It uses existing dataset identity inspection (which reads label files for integrity) but passes no labels to sampling or VCA, performs no classifier scoring, and writes records externally. It uses existing runtime center estimates marked approximate; it does not validate their physical accuracy. Strict cross-sensor tasks, class maps, label budgets, training integration and E1 coverage modulation remain separate acceptance steps. Pruning/quantization tools remain frozen while this cross-sensor data/protocol work is prioritized.
