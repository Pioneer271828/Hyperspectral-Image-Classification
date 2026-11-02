# M4-B2: registered paired development screen

[Issue #77](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/77) follows the accepted M4-B1 training integration. The first screen tests the region/pixel mechanism on the frozen Xuzhou development task. It uses target-scratch training, with no source pretraining or final-test evaluation, and does not establish cross-sensor transfer gains.

The versioned [specification](../../configs/experiments/m4_b2_target_screen.json) fixes four modes (`none`, `capacity_control`, `x_residual`, `residual_only`), roots64/512/2048,5000 updates/run and validation every100. Model/head/sampler seeds are root/root+2/root+4. All arms use the existing class_first candidate, learning rates5e-5/5e-4, weight decay1e-3, original losses/fusion/graph and earliest minimum validation-NLL selection. This is an experiment-local sampling choice, not a change to default training. Total12 target runs/60000 updates. Keep the original90 training/45 development labels and split/support identities.

Primary endpoint is fused development OA. Report AA/Kappa/macro-F1/per-class and three-head readouts, sample SD(ddof1), all paired differences, learning curves and costs. The registered candidate rule requires x_residual meanOA above BOTH none and capacity_control, positive OA differences in at least2/3 seeds for each control, and meanAA at least none. Otherwise retain the baseline and report the limited/negative evidence. No within-batch tuning, outcome-based seed selection or automatic default switch. At45 balanced labels each class has5 observations: changes are coarse, AA equals OA in this split, and a candidate still needs independent confirmation. Boundary is N/A; unlabeled pixels are not background. Capacity control has8320 parameters versus8256 for the residual projection and different compute.

## Frozen plan and execution

All commands run remotely in the authorized training container at the exact published full SHA, clean full-history checkout and existing environment. Outputs must be unique and outside checkout. `prepare` binds code/config/image/protocol hashes before training; `run` assigns a complete paired seed group to each explicit GPU UUID, with at most one worker on each selected GPU. Other projects are not changed or signaled. Each selected GPU must have at least8GiB free; shared-load timings are not controlled speed benchmarks.

```bash
python scripts/run_m4_ablation.py prepare --expected-commit FULL_SHA \
  --development-root FROZEN_DEVELOPMENT_DIR --protocol-root FROZEN_PROTOCOL_DIR \
  --data-root DATASET_DIR --out-dir EXTERNAL_NEW_CAMPAIGN
python scripts/run_m4_ablation.py run --expected-commit FULL_SHA \
  --plan EXTERNAL_NEW_CAMPAIGN/plan.json --gpus GPU_UUID_1 GPU_UUID_2 GPU_UUID_3
```

Use `prepare --smoke` for a clearly labeled four-arm/two-update engineering preflight, never as accuracy evidence. A frozen real plan accepts no budget/seed/model overrides on workers. Every completed run is independently verified and reviewed before that worker starts its next arm. A worker failure stops pending seed-group launches; other already running registered groups finish without being killed. Preserve logs and investigate before any retry.

`run --resume` verifies completed cases, their original checkpoints/configuration, plan/data/code hashes and result summaries; it skips training and preserves `case.json` status/artifact location byte-for-byte. Operation=`verified_existing` is written to separate worker-operation records. In-progress, failed, missing or corrupt cases are rejected, with no overwrite or implicit optimizer resume. Use a new explicit run after investigation rather than silently replacing evidence.

## Per-run analysis and comparison

Each case saves a review of the selected checkpoint: all three heads' confusion/metrics and per-class both-branches-wrong rates, learning curves, fixed module/hint interventions, module calls and bounded gradient/feature probes, actual parameter updates, parameter counts and peak training allocations. The original model is unchanged; probes do not change loss, optimizer, tensors or RNG. Frozen interventions show dependency and cannot replace the separately trained four-arm ablation. Auxiliary heads are not independent CNN-only/Graph-only models. Review decisions remain pending until the registered paired batch finishes.

Case receipts hash all result/weight/analysis artifacts. The final summary checks that each seed's four arms share common-backbone initialization, fresh-head initialization, sampled tile/anchor sequence and normalization statistics. It aggregates all registered repeats, retains failures and distinguishes development candidate status from final accuracy claims. Original checkpoints/logs are external; publish only a compact report in research. Source→target experiments with matching M4 modes and independent support/split repetitions remain separate work.
