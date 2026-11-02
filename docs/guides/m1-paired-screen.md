# M1-C: paired full, historical-zero and real-subset screen

[Issue #83](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/83) follows the accepted M1-B integration and reference/code-first workflow. Reuse the in-repository `ObservationPolicy`, `development_training.adapt`, artifact verifier and paired M4 controller. The historical reference is `train_vca_multidataset.make_train_input` at base68c163784b0823ce8fe36bb1cba0c0ec323cbaa0; tests fix the same indices and compare actual tensors. No external model code or new paper algorithm is incorporated.

The [specification](../../configs/experiments/m1_c_paired_screen.json) preregisters roots64/512/2048,5000 updates each, validation every100 and three arms: full, legacy_support_zero, real_subset. Total9 runs/45000 target-scratch updates. Use90 train/45 development labels, original devtrain normalization, class_first candidate, unchanged LR/loss/graph/fusion/architecture and earliest minimum FULL development NLL. Model/head/tile seeds=root/root+2/root+4; observation seed=root+6; validation seed3407 common to all roots/arms. M4, interaction, edge updates and coverage modulation remain disabled. No source pretraining, raw target GT or final-test scoring.

For real subsets, cycle full/random75/random50/random25/middle wavelength-span25% gap/stride2. Freeze all nine per-update policies before any training. The legacy control shares the exact retained columns but zeroes complementary bands only at released train-support pixels, retaining full B/metadata/context. It reproduces the historical mechanism, not its original GPU random sequence. Real removal and legacy masking also differ in spatial extent and metadata availability: this is a method control, not a pure deletion-versus-zero attribution. An all-pixel zero control is separate follow-up if needed. Defaults/old policy hashes remain unchanged; the opt-in `training_application='support_zero'` is serialized and reconstructed by the existing verifier.

Each selected checkpoint is reloaded and scored on the same fixed full/random75/random50/random25/gap/stride development bank, with three heads, per-class/both-wrong errors, curves, actual adapter widths/gradient/update probes, parameter count, peak allocations and timing. Primary robustness endpoint is equally weighted fused OA over the five non-full conditions. A development candidate requires:

- mean full OA/AA at least full training, paired full OA nonnegative in at least2/3 roots;
- mean perturbed OA strictly higher than BOTH controls, positive paired robustness differences in at least2/3 roots for each control.

Otherwise retain full baseline and disclose any accuracy/robustness tradeoff. Report sampleSD(ddof1), every paired difference and worst conditions. At45 balanced validation labels AA=OA; these repeats do not establish statistical noninferiority, independent-support/test performance or actual unseen-sensor gains. Boundary remains N/A. Do not tune the registered batch or pick seeds based on intermediate results. Full-budget historical M4 results are not pooled into this baseline.

```bash
python scripts/run_m1_ablation.py prepare --expected-commit FULL_SHA \
  --development-root FROZEN_DEVELOPMENT_ROOT --protocol-root FROZEN_PROTOCOL_ROOT \
  --data-root DATASET_ROOT --out-dir NEW_EXTERNAL_CAMPAIGN
python scripts/run_m1_ablation.py run --expected-commit FULL_SHA \
  --plan NEW_EXTERNAL_CAMPAIGN/plan.json --gpus GPU_UUID_1 GPU_UUID_2 GPU_UUID_3
```

Use `prepare --smoke` for a labeled three-arm/two-update engineering preflight. One complete root group runs on each inspected low-load GPU. Each run is verified/reviewed before its worker starts the next arm. Other projects are not signaled or modified. Code stays pinned/clean until every worker completes; results/data/logs remain external. Controller callbacks reuse the existing scheduler without changing M4 defaults.

`run --resume` verifies completed cases/policy/input/model/summary identities and skips training, preserving original completed status/artifact path and all case files; operation status is separate. Failed/in-progress/corrupt cases are not overwritten and optimizer resume is not implied. Retain failed attempts; investigate before an explicit new run. Engineering tests cover actual legacy zeroing/context protection, frozen policies/default hash compatibility, actual three-arm short training/readout/restore, immutable resume, tamper/identity/budget rejection, paired aggregation/candidate/SD rules and the reused controller scheduling/failure behavior.
