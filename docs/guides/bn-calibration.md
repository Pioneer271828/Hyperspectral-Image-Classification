# BN-ALIGN1: opt-in scene-population calibration

[Issue #103](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/103) follows the [BASE-97C frozen diagnosis](../../research/reports/BASE97C_三种子完整联合训练结果与双支路诊断_20261103.md). The new helper does not change the original model, trainer, checkpoint loader or defaults.

`model.bn_calibration.calibrate_denoise` defaults to `mode='none'`, returning the untouched source object. Explicit `scene_population_denoise_bn_v1` with context `acc1p_full_scene_transductive` returns a separate eval clone. Its only changed state is selected domains' two front-end denoise running means/variances. Parameters, affine weights, eps, counters, other domains/layers and source model remain unchanged. The old trained checkpoint remains authoritative; calibrated artifacts are separate evaluation models and cannot substitute for exact optimizer continuation.

The calibrator accepts only `net_input/wavelengths/fwhm` observations, never label-bearing items. Each whole scene (including unlabeled pixels) is the explicitly allowed population. Means and variance with correction0 are computed in layer order: the second layer sees the first layer's calibrated normalization. Ordinary eval inference subsequently uses fixed stored buffers; no live training mode, hooks or implicit recalibration are needed. This full-scene policy cannot be silently transferred to strict target-sensor holdout or arbitrary deployment scenes.

Source checkpoint SHA256, exact observation/metadata identities, context, changed buffers, moment convention and calibrated state digest are persisted in `training_config.bn_calibration` through the existing checkpoint format. After `load_model_checkpoint`, call `verify_calibration_binding` with the same scene/context before use. Unknown scenes, contexts, changed state or missing metadata are rejected. The old general-purpose loader remains unchanged and does not enforce this new policy by itself.

```python
calibrated, receipt = calibrate_denoise(model, observations,
    mode=POLICY, context=CONTEXT, source_checkpoint_sha256=original_sha256)
save_model_checkpoint(calibrated, separate_path)
restored = load_model_checkpoint(separate_path, device="cuda")
verify_calibration_binding(restored, observations,
    context=CONTEXT, source_checkpoint_sha256=original_sha256)
```

The production checkpoint gate uses FP32. FP64 is an additional numerical formula control; the existing loader constructs an FP32 model, so FP64 debug state is not exported as an ordinary checkpoint. For this control only, dense adjacency is represented sparsely with exactly identical coefficients, bypassing the legacy dense converter's fixed FP32 cast. Production graph representation/math remains unchanged.

The initial real FP32 live-reference tolerance failed at4.72e-5. A native replay found up to7.09e-5 with identical supervised predictions, while FP64 agreed within9.27e-15. Final acceptance adds per-seed strict FP64 checks(rtol1e-9/atol1e-10), explicitly records native FP32 differences(rtol1e-4/atol1e-4) and requires exact train/val hard predictions. Saved/reloaded calibrated FP32 state is exact and probabilities retain rtol1e-5/atol1e-6. These contracts do not promise native FP32 bitwise equality. Failure logs and the initial numerical/debug setup failures remain in the report/evidence.

Reference/code/license inspection is recorded in [the reuse review](../references/bn-calibration-review.json). This is a small independent adaptation, not a claim of reproducing an external benchmark.

The bounded gate compares all three existing BASE-97C primary models without reselection or gradient updates: original buffers, calibrated saved/reloaded buffers, live denoise batch-stat reference and permuted-domain statistic negative control. Only train/val readouts are evaluated; previous test results are preserved. The reused validation data supports engineering and mechanism checks, not independent confirmation of final accuracy. See the module acceptance report/current plan for actual outcomes.

The delivered [BN-ALIGN2 follow-up](calibrated-selection.md) adds explicit training-time calibrated validation/selection and a complete historical checkpoint replay. It preserves raw optimizer continuation. Current formal-evaluation work is tracked in the shared checklist; BN-ALIGN1's fixed-primary experiment above remains a separate historical scope.
