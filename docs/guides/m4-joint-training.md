# M4-J1: calibrated joint integration

[Issue #113](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/113) adds an isolated entry, `scripts/train_joint_m4.py`, for the existing [M4 implementation](region-residual-m4.md). No new external mechanism/code is introduced; the accepted M4-A/B1 mathematics, original-reference review and reuse boundary remain applicable.

The entry always uses calibrated validation/selection/inference. `--mode` defaults to `none`; other choices are `x_residual`, `capacity_control`, `residual_only`, `x_region`. The original three-head loss, task schedule, frozen preparation and engineering/full budgets remain those of the calibrated BASE-97 runner. Settings include the M4 mode; mismatched modes cannot resume the same directory. This is separate from the primary-only supervision candidate in BG-ABL2.

The entry stores Q/A tensor fingerprints in provenance. For selected inference use `M4JointRuntime().load_selected(path, items, raw_source_path, device=...)`, which checks both the existing scene/raw-source calibration binding and Q/A identity. The general checkpoint loader does not enforce this caller-side graph check. As with existing loaders, callers must not replace inputs or graphs after verification. These hashes protect accidental mismatches, not malicious rewriting of the checkpoint and its metadata together.

Run on a clean published commit in an authorized remote container, with outputs outside checkout:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
 python scripts/train_joint_m4.py \
 --expected-commit 51bf4e0b8ef6fc86134837e4c906ef7dc92d9605 \
 --profile engineering --seed 64 --mode x_residual \
 --prepared-dir /home/lbx/hsi-pioneer271828/runs/base97a-honghu-issue97-20261103-prepared01 \
 --data-root /home/lbx/hsi-pioneer271828/datasets/snapshots/joint-honghu-20261103-issue97/dataset \
 --out-dir /absolute/external/new-run
```

Engineering is12 updates. `--stop-after 5`, followed by the identical command with `--resume`, exercises continuation; a completed resume performs zero updates and verifies existing artifacts. Full is1503 updates and requires a separately registered research campaign. Do not reuse an engineering directory for full training.

[Acceptance and limitations](../../research/reports/M4J1_校准联合训练接入验收_20261103.md) distinguish engineering behavior from accuracy. Next research step is a frozen baseline/capacity/residual paired campaign; this integration is not evidence of improved accuracy.
