# BASE-97A: original low-label joint benchmark preparation

[Issue #97](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/97) implements a bounded engineering gate for [ACC-1P](data-protocol-matrix.md), not a full accuracy experiment. Existing `train_vca_multidataset.py`, backbone, losses and checkpoint API remain unchanged. The accepted experiment strategy keeps joint seen-dataset accuracy separate from source-to-few-shot-target transfer. Two-source/one-target transfer follows later; zero-label closed-set evaluation first needs verified sensor metadata and shared class semantics.

## Frozen identities and deliberate differences

- `configs/experiments/base97_joint.json` resolves the current implementation's HongHu historical-main wavelength-attention K128/embed64, DomainBN, static train-label-guided LDA-SLIC, fixed fusion, balanced three-head loss, no M4/interaction/coverage or spectral augmentation. The manifest saves the full resolved configuration and executable/config source fingerprints.
- Per-class training1% and validation0.5%, ceil rounding, background excluded; remaining foreground indices are test identities, with no test scoring in this module. `RandomState(seed+dataset_offset)` reproduces the legacy splitter's exact shuffle/order for a given label image/seed. The new frozen root seed is64 (dataset64/65/66); historical0/2026/3407 records are not relabelled as this new repetition.
- The user-selected current datasets are PaviaU/Xuzhou/**complete HongHu** (`whu-honghu`,22 classes). Counts are432/219/42125,692/348/67837 and3881/1947/380865. The historical LongKou records remain a separate recovered example; they are not this current dataset combination. Do not use the legacy `WHU` alias, which crops/relabels HongHu. Raw MAT hashes/keys/shapes/class counts, normalization min/max/dtype and actual prepared input/graph/label fingerprints are saved. Full-scene global min/max and spatial coordinates reuse the original preparation, with full-scene transductive context explicitly declared.
- The existing static hypergraph builder uses both NumPy and Python random walks. The new wrapper scopes Python's seed to the dataset seed and restores its caller state; NumPy follows the original seeded split-then-walk sequence. This makes new graph preparation traceable without modifying the original builder. Historical Python RNG state is unknown, so these graphs are not claimed identical to old recorded graphs.
- VCA projection uses pixel chunks of16384 with checkpointing to fit existing GPUs; FP64 output/gradient equivalence is tested separately. This changes memory execution, not intended model capacity. It does not promise bitwise equality to historical unchunked runs.
- Legacy config budget500 inclusive epochs ×3 gives1503 updates, but **this runner executes only one or two round-robin cycles**, default6 total updates. It deliberately guarantees every dataset participates for engineering inspection. This is not the original random-choice-with-replacement schedule and does not measure convergence or reproduce97% accuracy. Runtime metadata and checkpoint sidecars record the actual short-loop budget separately.
- Original source/data/index provenance gaps and correctness fixes are retained. Historical scores are evidence from old records until a separately registered full batch validates the current model.

## Prepare once in the authorized training container

Keep data, prepared graphs, manifests, logs and checkpoints outside all checkouts. Use a new explicit three-dataset snapshot; do not change the existing `datasets/current` symlink or running checkouts. No machine-specific loader fallback is relied on: the wrapper validates registered MAT files, temporarily selects the external root for the original preparation, then restores the loader setting.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /absolute/existing/python \
  scripts/validate_joint_baseline.py prepare \
  --expected-commit FULL_PUBLISHED_SHA \
  --data-root /absolute/immutable/three-dataset-snapshot/dataset \
  --out-dir /absolute/external/NEW_PREPARED_DIRECTORY
```

The manifest binds ordered split indices, raw identities, prepared tensors/sparse assignment/static adjacency, all resolved settings, source fingerprints and preparation time. Preparation reads labels to construct train/validation/test identities; LDA uses only training labels. Cached training items remove all `test_*` supervision fields. This is a declared legacy benchmark with full-scene input context, not strict spatial isolation.

## Bounded joint training and classifier restoration

Inspect current GPU use first. Select an available device without terminating or changing other jobs. Pin the printed SHA256 of the prepared manifest, not just its directory name:

```bash
CUDA_VISIBLE_DEVICES=AVAILABLE_GPU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /absolute/existing/python scripts/validate_joint_baseline.py short-loop \
  --expected-commit FULL_PUBLISHED_SHA \
  --data-root /absolute/immutable/three-dataset-snapshot/dataset \
  --prepared-dir /absolute/external/PREPARED_DIRECTORY \
  --prepared-manifest-hash MANIFEST_SHA256 \
  --cycles 2 --out-dir /absolute/external/NEW_SHORT_LOOP_DIRECTORY
```

The runner verifies source/config/raw/artifact identities before training, checks optimizer registration, and records finite losses, shared-adapter/backbone gradients and real parameter changes plus all three active task-head changes. Inactive task heads must remain unchanged. After each cycle it evaluates **train and validation only**, selects by mean validation fused OA, saves primary/last classifiers, then reloads the selected classifier and compares all train/validation branch probabilities and predicted labels. Classifier restoration is not exact optimizer-state continuation. Fresh output directories are mandatory; failures remain as external records and no implicit resume overwrites results.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /absolute/existing/python \
  -m unittest discover -s tests -p test_joint_baseline.py -v
BASE97_DEVICE=cuda CUDA_VISIBLE_DEVICES=AVAILABLE_GPU OMP_NUM_THREADS=1 \
  /absolute/existing/python -m unittest discover -s tests -p test_joint_baseline.py -v
```

The second command exercises the synthetic three-task short loop on GPU; ratio, identity, selection and FP64 chunk-equivalence tests remain CPU checks. Tests cover exact legacy ratio parity/RNG preservation, malformed populations, budget/config/index/artifact tampering, train/validation-only access, unequal class metrics, selection, actual task participation and checkpoint restoration. Run the existing relevant regressions too; exact commands/results belong in the acceptance report.

## Next gate

After engineering acceptance, stop and report each dataset's actual participation, loss/branch readouts and restoration evidence. Do not interpret six updates as accuracy evidence. Register the full random-schedule batch separately with fixed split/graph/initialization identities, seed repetitions, source/target access, update budget and validation selection. Later cross-sensor evaluation requires a source checkpoint that excluded the target dataset; the three-dataset joint checkpoint cannot serve as an unseen-target source model.
