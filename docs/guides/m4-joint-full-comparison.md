# M4-J2 paired joint comparison

The [preregistered experiment](../../research/experiments/M4J2_区域残差联合三种子完整配对预登记_20261104.md) freezes nine runs before training. Implementation is inherited from M4-J1; this document does not introduce a new trainer. Status: **pending remote connectivity and capacity preflight**. No new result or test pass is claimed.

## Remote-only execution

Use an isolated clean checkout of the published `codex/issue-115-m4-j2-full` branch. Record its full SHA from `git rev-parse HEAD` and compare it with GitHub before running. Verify that `scripts/train_joint_m4.py`, the model and trainer match the accepted M4-J1 implementation. Do not edit or update the checkout while any run is active. Outputs must be outside the checkout.

The following Bash template is for the authorized LXC, after source/environment/GPU verification. `RUN_ROOT` must be a new, dedicated directory. The template runs a single case and does not silently enqueue all experiments. Replace the explicit mode/seed/profile per the frozen matrix. For preflight use `capacity_control`, seed64, `engineering` and a separate `preflight` output. For the full matrix use `none`, `capacity_control`, `x_residual`, each with seeds64/512/2048 and profile `full`.

```bash
PY=/home/lbx/hsi-pioneer271828/envs/hsi/bin/python
PREP=/home/lbx/hsi-pioneer271828/runs/base97a-honghu-issue97-20261103-prepared01
DATA=/home/lbx/hsi-pioneer271828/datasets/snapshots/joint-honghu-20261103-issue97/dataset
EXPECTED_COMMIT=<verified-published-full-SHA>
GPU=<checked-available-GPU-index>
RUN_ROOT=<new-absolute-directory-outside-checkout>
MODE=capacity_control
SEED=64
PROFILE=engineering

CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" scripts/train_joint_m4.py \
  --expected-commit "$EXPECTED_COMMIT" \
  --mode "$MODE" --profile "$PROFILE" --seed "$SEED" \
  --prepared-dir "$PREP" --data-root "$DATA" \
  --out-dir "$RUN_ROOT/preflight-capacity-64"
```

Record the exact expanded command, source SHA, GPU, start/finish times, exit code and external log path. Full case outputs use `cases/<mode>/seed-<seed>` under a fresh campaign root. Existing results are never overwritten. `--resume` requires exactly the same arguments and source identity; verify zero extra updates and unchanged files for a completed case.

Run relevant engineering checks remotely before the campaign:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 M4J1_DEVICE=cuda \
"$PY" -m unittest discover -s tests -p test_m4_joint.py -v
```

This test command alone does not cover the real capacity preflight. That acceptance must additionally inspect the control parameters, their finite gradients and actual updates, calibrated selected restoration and immutable completed resume. Do not promote historical residual-only test evidence to capacity acceptance.

## Completion gate

All nine runs must finish and receive train/validation reviews before freezing test evaluation. Save input/checkpoint hashes; use `M4JointRuntime.load_selected` for calibrated scene and Q/A binding. Independently recompute metrics, report all seeds and apply both preregistered contrasts and protection gates. Frozen disabling of a residual is diagnostic and cannot replace retraining controls. Shared-GPU elapsed time is not a controlled inference speed comparison.

Deliver results in a separate report with tested/training SHA, commands, failures, limitations and an explicit keep/change/disable decision. Under the 2026-11-04 user instruction, continue the next planned module after that review; do not bypass this gate merely to advance faster.
