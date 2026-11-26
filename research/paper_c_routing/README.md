# Frozen cross-device development routing

Issue[#515](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/515).
Own branch`codex/paper-c-routing-20261126`, base`6b076052da3797c367f262c1ec96bd8c7f4a7328`.
Only this directory is changed. No shared model, default config, CLI, dataset
registry or status ledger is edited. All real executions use authorized Linux
CPU, existing frozen weights and released development inputs; **zero new
optimizer updates**. This is a Draft development delivery, not confirmation.

## Scope and entry points

* `catalog.py`: all30 ordered off-diagonal directions, exact original data gate,
  original-source evidence and additional role requirements. Six are data-ready
  for development, but only PaviaU→Xuzhou currently has admitted original
  single-source model lineage for this lazy-graph experiment. No direction has
  independent calibration/confirmation permission.
* `run.py`/`policy.py`: six arms over unchanged WP11 runtime/graph execution:
  light, full, content, content+coverage, within-parent permuted coverage, and
  matched-call random. Weight/input hashes are checked, evaluation is frozen,
  probabilities and stage records are committed without overwrite, and explicit
  exact resume is required. Completed reentry verifies inputs and outputs before
  model deserialization or neural execution.
* `g3-profiles.json`: exact original model/metadata pins for three G3 run seeds,
  three original methods each. Target scratch is a target-local control, outside
  the30-direction transfer roster. The cheap model copies the existing trained
  CNN branch; there is no new independently fitted light head.
* `audit.py`: independent NumPy reconstruction of whole-tile confidence,
  descriptor permutation, all routes, endpoint probabilities, recalls, confusion
  matrices, real heavy/graph calls and original-file hashes. It imports no neural
  network or producer policy code.
* `check_lineage.py`, `check_reentry.py`, `test_policy.py`, `test_runtime.py`:
  original ancestry, exposed-context checks, actual trapped CLI reentry and
  synthetic fault/negative tests. Synthetic faults abort rather than silently
  spend extra heavy-call budget.
* `analyse.py`, `explain_routing.py`, `plot_results.py`: paired errors and
  original-method effects, a label-free post-hoc score-scale certificate,
  CSV-backed editable SVG figures. No replacement policy is selected.
* `prepare_pair.py`/`dsnet-paired-draft.json`: input/code-bound18-case native/DSNet
  proposal. Proposed30480 updates, authorized/executed0. Training executor,
  actual optimizer recovery and native backward cost profile remain pending;
  this file cannot launch training.

Read[device evidence](device-evidence.md),[independent-role requirements](confirmation-readiness.md)
and the[delivery review](delivery-review.md) before interpreting any score.

## Reproduction identities

The dependency is a separate clean checkout at
`395caef57fba8ab6a09223538f748179dab552f2`.
Primary production for all nine original models, with their original frozen
normalizer, uses`afe4eaed2b51e70a37ce49a48a869227ef598014` (case suffix`span05`).
Earlier full-finetuning production uses`e5cca01ef684a2acb7db8f2634aa50cd9f8cca1d`;
earlier scratch/head-only production uses`a9101c1357ab50470ae247b99cb5d6e88401a1bf`.
Those earlier runs applied the WP11 development normalizer, whose statistics hash
differs from the original G3 checkpoint. They are retained **renormalization
stress tests**, not faithful original-normalizer replay. Later commits add audit,
analysis, documentation or delivery material; they do not change those recorded
producer identities. Resume an old output from its exact frozen producer checkout.

Example on the authorized LXC, with`PROJECT` and the complete expected SHA
resolved and verified before invocation:

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python research/paper_c_routing/run.py \
  --expected-commit afe4eaed2b51e70a37ce49a48a869227ef598014 \
  --dependency "$PROJECT/checkouts/paper-c-wp11-dependency-20261126" \
  --runs "$PROJECT/runs" --out "$PROJECT/runs/NEW-EXPLICIT-OUTPUT" \
  --profile g3_pavia_xuzhou \
  --g3-root "$PROJECT/references/paper-c-routing-20261126/g3-p2-s2036" \
  --g3-seed 2036 --g3-method source_pretrained_head_only \
  --normalization checkpoint_original \
  --g3-statistics "$PROJECT/references/paper-c-routing-20261126/g3-target-statistics.json" \
  --range-view --descriptor approximate_span
```

`--stop-after 2` intentionally stops after two newly committed stages. Repeat the
exact contract with`--resume` to finish only the remainder. Existing orphan NPZ,
changed contracts, original input drift, altered outputs or dirty/wrong code
fail; no output is reset or silently overwritten. Raw probabilities stay remote.

For read-only audit:

```bash
python research/paper_c_routing/audit.py \
  --run "$PROJECT/runs/paper-c-routing-20261126/CASE" \
  --development "$PROJECT/runs/wp11-label-custody-validation01/development-package/development.json" \
  --out "$PROJECT/runs/NEW-AUDIT.json"
```

Tests insert the dependency first in`sys.path`; the task's older repository tree
contains an`inference`package that otherwise shadows the pinned WP11 dependency.
Pytest8.3.5 is isolated in external`references/.../test-tools`, not installed over
the project's environment. The preserved initial missing-pytest/import failures
explain the corrected launch method in the delivery logs.

## Measurement contract

There are five28×28 development tiles and four views each:436 delivered columns,
stride2=218, stride4=109, and122 columns selected by approximate centers≤1000nm.
All four score the **same45 released points**, five per numeric class. Stride
views primarily reduce density; the approximate cut is not a measured VNIR
camera response. Compare the retained-column fraction and approximate-span
descriptors as separate metadata assumptions.

The batch quota is10 full calls out of20 requests, fixed without labels.
Content uses`1 - min_pixel(top1_probability - top2_probability)`; coverage adds
`1 - descriptor`with unit weight. This is a batch allocation rule, not a fitted
online threshold or a risk-calibrated guarantee. Random control reuses WP11's
actual view/shape/coverage strata. Permutation rotates descriptor values within
the same parent while leaving spectra and wavelength inputs unchanged.

Each routed batch pays20 actual light passes and only10 actual full passes/
graph builds. Static full pays20 full/graph calls and no light calls. Graph reuse
is disabled; selected full calls recompute their CNN and graph work. Six rounds
rotate arm order, while weights/inputs remain frozen. Timing covers resident
request execution, real graph building and policy planning, on a shared CPU.
GPU, isolated deployment speed and peak memory were not measured.

Original-normalizer replay does not fit new statistics: it applies the pinned
original`statistics.json`to the already released raw FP32 blocks. Its SHA and
internal statistics hash must match the checkpoint metadata. The independent
audit rebuilds the normalized request bytes and coordinates exactly with NumPy.
These original statistics historically saw the old training geography, including
current development contexts; this correction does not establish independence.

The legacy contract field`warmups:1`means **one setup content+coverage reference
pass** (20 light +10 full calls), not one warmup for every arm. Setup reference
and initial model/input loading are recorded separately; they must not disappear
from a cold-use cost account. Six timing rounds and three original training seeds
do not multiply the45 distinct released labels or create new sensor tasks.

The results include an important failure of identifiability: coverage selects
complete view cohorts at the original unit score weight, so the conditionally
matched random plan is identical. Scores remain valid development observations,
but this control cannot identify a coverage-information advantage. Native-view
gains occur where that policy simply keeps every native tile on its light head;
they cannot be presented as a general gain from real sensor coverage metadata.
