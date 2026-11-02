# E4a-2: independent G3 source pretraining and target adaptation

`train_cross_sensor.py` implements the frozen G3 task with three modes: `preflight` prepares actual inputs/statistics without training, `smoke` runs a synthetic two-source-step / two-target-step engineering fixture, and `run` executes the declared10000-source /1000-target-update pilot. No existing model, checkpoint format or training entry point is changed. This module does not start a full pilot automatically.

The stage implementation is `transfer/g3_training.py`. Source training accepts only the source scene/labels, fits only source-train normalization, and selects the checkpoint by source validation NLL every100 updates. It never evaluates source test. Source selection finishes before the real campaign loads target images or target normalization. Protocol-file identities may be read earlier; file hashing is not target feature training.

Target training receives only its scene and frozen support manifest, not a complete target label image. Each method uses the same fresh target three-head initialization and the same target tile sampler seed. Only backbone state transfers, even when both datasets happen to have nine classes. Target-scratch starts from the same seeded backbone initialization as source pretraining; both pretrained methods start from the selected source backbone. Assertions check these initializations. Head-only freezes backbone parameters and switches the backbone to eval while the heads train; full fine-tuning has the ordinary training behavior. Record differing trainable parameter counts and dropout behavior rather than claiming the conditions are identical in every respect.

The baseline uses the existing screen_m6_none model/loss/hyperparameter configuration: GroupNorm, learned label-free graph, no interaction,64 latent/32 adapter embedding, alpha0.8; Adam backbone LR5e-5, heads5e-4, weight decay1e-3 and the existing auxiliary/graph-loss weights. Source ReduceLROnPlateau uses source validation NLL only. Target LR remains constant: no target-val-loss scheduling or early stopping. The source initialization and target-sampler seeds come from the frozen task. A target epoch/step budget in resolved configurations records the actual updates, not the longer source budget.

M1 full-delivered-band views keep exact data/metadata alignment. Source/target per-band statistics are fitted independently using only each train region; target train unlabeled context is allowed by the G3 protocol. Sampling target updates only among tiles containing support labels does not read other target label values. Absolute coordinates and existing superpixel scale50(PaviaU)/100(Xuzhou) remain unchanged.

After all three target final-step checkpoints are saved, the campaign writes target-checkpoints-frozen.json, then loads final target label values and evaluates each method once. Both weights and sidecar SHA256 must match, along with task/method/support/plan/statistics identity. Metrics are reported for all prespecified methods, not used to choose target steps or tune the run. These are few-shot results, not zero-shot or G2 claims.

## Commands (remote container only)

```bash
# Engineering fixture; no real-data accuracy claim.
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python train_cross_sensor.py smoke \
  --expected-commit FULL_COMMIT_SHA --out-dir /home/train/hsi-pioneer271828/runs/NEW_G3_SMOKE
# Validate actual frozen inputs/statistics, without optimization or scoring.
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python train_cross_sensor.py preflight \
  --expected-commit FULL_COMMIT_SHA \
  --protocol-root /home/train/hsi-pioneer271828/runs/transfer-protocol-20261031-02 \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/NEW_G3_PREFLIGHT
# Full declared pilot; execute only as a separately scheduled experiment.
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python train_cross_sensor.py run \
  --expected-commit FULL_COMMIT_SHA \
  --protocol-root /home/train/hsi-pioneer271828/runs/transfer-protocol-20261031-02 \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/NEW_G3_PILOT
```

The CLI requires a clean exact SHA and training container; smoke/run require CUDA with no fallback. It verifies frozen task/plan/support identities and raw input-file hashes, preserves frozen inputs, rejects an existing output directory and writes running/passed/failed manifests and progress outside the checkout. Failed runs are retained; there is no skip/resume or optimizer-continuation support in this first entry point. A resumed full campaign is not implied by the independent MASK-2 checkpoint module.

Real preflight does not decode target label values; target label-file bytes are hashed against the protocol for integrity. Synthetic smoke deliberately constructs all fixture arrays up front and is explicitly labelled synthetic, whereas the real stage ordering is tested using dependency guards. No target validation evaluation exists. Python stage functions are a correctness API, not protection against malicious callers or repeated manual test evaluation.

## Actual stage configuration and provenance (Issue #68)

`resolved_config` remains loadable by `SpatialConfig`: it records the actual model seed and a single inclusive epoch (`epochs=0`, `steps_per_epoch=actual_updates`). Source and development stages also record the actual validation interval in update units and validation-loss selection. These metadata corrections do not change initialization, loss, optimizer settings, sampling or selection. The base template is copied rather than edited.

Both `result.json` and checkpoint `training_config` contain identical `resolved_config`, `runtime_config`, and `runtime_config_hash`. Runtime metadata binds the resolved-config hash, stage/method, model/head/sampler/training-Torch seeds, update budget, validation/selection policy, eligible sampling tiles, initial Adam groups and scheduler settings. Source heads initialize jointly with the model seed, so their separate head seed is null. Target heads are reset with their explicitly recorded seed. Source scheduler settings are initial policy, not an optimizer-state checkpoint.

The fixed-final target stage has no validation interval or scheduler. Its runtime policy identifies unused template checkpoint/validation fields explicitly; scene tile size/gap/split come from the supplied plan, not the template. Development sampling records include the complete support-only sampler metadata. A component config alone is therefore insufficient to replay the G3 runner: use the matching stage runtime record, scene/plan identities and released support manifest.

Each stage verifies its artifacts after completion. New-record final target evaluation verifies them again before evaluating test labels. For a read-only audit in the remote container:

```python
from transfer.run_provenance import verify_run_artifacts
verified = verify_run_artifacts('/absolute/external/run/stage')
```

The verifier checks record equality/hashes, effective budgets/seeds, optimizer settings, sampler-seed replay against recorded training steps/anchors, validation schedule, earliest-minimum or fixed-final selected step, and checkpoint/sidecar hashes. It never rewrites artifacts. Conflicting records fail with `ValueError`. Historical stage records without provenance are rejected by this strict audit with an explicit legacy message; preserve their original files and use a separate evidence-backed correction record. The old final-evaluation path for historical sidecars remains available and is not retroactively labelled provenance-verified.

This is model checkpoint recovery and execution provenance, not exact optimizer/RNG continuation. No long training or new test-based tuning is implied. Current remaining tasks and research priorities are maintained in [the shared audit checklist](../../research/当前搭建状态与审查整改清单_20261102.md); old pilot-stage descriptions above retain their original protocol scope.
