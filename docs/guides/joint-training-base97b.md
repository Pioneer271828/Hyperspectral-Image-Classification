# BASE-97B: random joint training, budgets and continuation

[Issue #99](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/99) adds a separate entry without changing the [BASE-97A producer](joint-baseline-base97.md), legacy trainer, model, losses or defaults. It uses the accepted PaviaU/Xuzhou/complete HongHu preparation and K128/embed64 wavelength/DomainBN configuration.

## Actual budgets and selection

`configs/experiments/base97_training.json` declares two required explicit profiles:

| Profile | Inclusive epoch indices | Updates/epoch | Total updates | Validation epochs |
|---|---|---:|---:|---|
| `engineering` | 0..3 | 3 | 12 | 0,1,2,3 |
| `full` | 0..500 | 3 | 1503 | 0,10,...500 (51 events) |

Each epoch samples three dataset names **with replacement** using `np.random.default_rng(schedule_seed).choice`; it does not force every dataset to appear every epoch. The engineering schedule is the corresponding full schedule's first12 draws, but its validation schedule and total budget are different. It is not a partial completed full experiment.

The original fused validation NLL arithmetic drives `ReduceLROnPlateau(mode=min,factor=.5,patience=20,threshold=1e-4,threshold_mode=rel)`. Strictly greater validation macro-OA selects the primary classifier; ties retain the first selection. Training readouts and test labels never select a checkpoint. There is no early stopping, automatic test scoring or hidden budget extension.

Model and schedule seeds are paired64/512/2048. The accepted split/graph root remains64 (dataset64/65/66), shared across all three training repetitions. This isolates combined training-initialization/sampling randomness on one fixed data/graph realization; it is not a support/split repetition or a reconstruction of historical0/2026/3407 splits. A separately frozen experiment is required to vary supports/graphs.

## Reuse prepared data across consumer versions

The accepted manifest is SHA256`414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b`, produced by`592e24153bdd2e2ea4dcde498da5b1a2f50ad9a1`. The new entry authenticates that producer and verifies every producer executable/config fingerprint remains unchanged in the consumer; new runner/config/test files may be added. It then uses the producer's original strict validator to recheck raw MAT identities, ordered splits and prepared tensors/graphs.

Both producer and consumer provenance are recorded. This does not relabel old data as newly prepared or regenerate graphs for every new script. If a producer file changes, import fails: audit the change and version the import policy rather than silently accepting new preparation behavior. The original validator/API is unchanged.

## Run only from an authorized remote checkout

Inspect the container, checkout and GPU first. Keep outputs outside all checkouts and away from existing projects. Use full history, the published full SHA and a clean worktree. The CLI requires an explicit profile/seed; the full budget has no implicit default.

```bash
CUDA_VISIBLE_DEVICES=AVAILABLE_GPU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /absolute/existing/python scripts/train_joint_baseline.py \
 --expected-commit FULL_PUBLISHED_SHA --profile engineering --seed 64 \
 --prepared-dir /absolute/frozen/BASE97A_PREPARATION \
 --data-root /absolute/frozen/PAVIAU_XUZHOU_HONGHU_SNAPSHOT \
 --out-dir /absolute/external/NEW_RUN --stop-after 5
```

Repeat the same command with `--resume` and omit `--stop-after` to execute the remaining7 updates. Repeating `--resume` after completion verifies the finished experiment and returns `operation=verified_completed, updates_this_call=0`. `result.json` retains `status=completed`; classifier, metric and summary files remain unchanged. Without `--resume`, an existing output directory is rejected.

The `full` command uses the same entry with `--profile full`, a separately registered output/batch and explicit seed. This guide's existence does not launch the1503-update campaign. Current module acceptance uses engineering only.

## Training state and recovery boundaries

- Identity contains resolved actual settings, profile, model/schedule/split seeds, producer identity, consumer executable fingerprint and execution device. Source and prepared-artifact mismatches reject resume before updating models.
- Validation checkpoints and intentional interruptions save model parameters/BN buffers, Adam state, scheduler, CPU/NumPy/Python/device RNG, next update cursor, trace/history and selected model/probabilities. State files have step-specific names and are published before the atomic progress pointer. Older validation snapshots are retained; account for their storage in long campaigns.
- Mid-epoch interruption adds no validation or scheduler event. Resumption checks exact model/optimizer/scheduler fingerprints before restoring RNG and taking the next scheduled update. CPU uninterrupted-versus-split trajectories are tested exactly. GPU restored state is checked exactly, while future bitwise arithmetic across devices/kernels is not promised.
- A process crash resumes from the last published checkpoint; unsaved updates after it are lost and re-executed. The exclusive `.lock` prevents concurrent writers. A hard crash may leave a lock: inspect its recorded PID/ownership and actual process before any manual removal; never automatically remove someone else's lock.
- Primary/last classifier sidecars record actual total budget and checkpoint update separately. The selected classifier is freshly loaded through the existing checkpoint loader and its train/validation probabilities/predictions checked. This is distinct from optimizer continuation.
- Completed resume hashes all recorded artifacts and checks the frozen trace/exposure. It separates the invocation operation from the experiment completion status and never overwrites a completed summary with a skip status.

## Tests and evidence

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /absolute/existing/python \
 -m unittest discover -s tests -p test_joint_training.py -v
BASE97B_DEVICE=cuda CUDA_VISIBLE_DEVICES=AVAILABLE_GPU OMP_NUM_THREADS=1 \
 /absolute/existing/python -m unittest discover -s tests -p test_joint_training.py -v
```

Tests cover full schedule/budget against the old loop, original validation arithmetic and Adam update, producer/config rejection, CPU continuation after steps2/3/5, scheduler decay across interruption, GPU continuation, completed-skip immutability, state corruption and concurrent ownership. Early scheduler decay is a test-only setting; the actual CLI remains patience20.

Each update records losses, active-head/adapter/graph gradient norms, actual HGNN changed elements and learning rates. Finite zero gradients are recorded for diagnosis, not automatically classified as dead neurons or treated as a reason to prune. After every actual run, inspect participation, curves, branch/class metrics, selection and intervention evidence before deciding the next accuracy change. Engineering-profile accuracy does not establish final classification performance.
