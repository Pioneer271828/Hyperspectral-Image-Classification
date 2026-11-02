# M1-B: opt-in frozen real-band training

[Issue #79](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/79) integrates the existing label-blind M1 sampler into `transfer.development_training.adapt`. Existing calls remain disabled, with the same model, losses, optimizer, tile sampler, budget and full-development NLL selection. M4 remains disabled. Source/target G3 campaign CLI integration and full-budget accuracy experiments are separate tasks.

```python
from transfer.observation_policy import ObservationPolicy, evaluate_conditions
from transfer.development_training import adapt
from model.checkpoint_io import load_model_checkpoint

policy = ObservationPolicy.freeze(
    scene, released_development_record, updates=5000,
    training_configs=[{'mode': 'none'},
                      {'mode': 'random_subset', 'keep_fraction': 0.75},
                      {'mode': 'random_subset', 'keep_fraction': 0.5},
                      {'mode': 'random_subset', 'keep_fraction': 0.25}],
    train_seed=64, validation_seed=512)
# Save policy.to_dict() outside checkout BEFORE a registered run.
result = adapt(scene, released_development_record, base_config, external_new_directory,
               steps=5000, val_every=100, seed=64, device='cuda',
               observation_policy=policy)
selected = load_model_checkpoint(external_new_directory / 'model.pt', 'cuda')
review = evaluate_conditions(selected, scene, released_development_record, policy)
```

This is an API example, not an instruction to launch a5000-update study. Register budgets/paired roots/conditions first. `training_configs` cycles in explicit order; random subsets are sampled once per update during freezing using a private generator. Model and spatial-sampler RNGs are untouched. Named validation conditions have independent, name-derived RNGs, unchanged when the training budget/configs or other conditions change. Replay consumes no RNG. Default development conditions are full/random75/50/25%, middle25% wavelength-span gap and stride2. These are observation corruptions, not measured alternative sensor responses. Invalid/empty selections fail before creating the training output directory; callers with very small B must explicitly specify a feasible condition bank.

The policy binds metadata/source identity, full spatial plan, released development roles/supports, original devtrain statistics, budget and records. Augmentation can only access devtrain tiles; evaluation can only access devval. Statistics are fitted once on original devtrain pixels before selection; derived min/max slices retain the parent hash. Actual B changes for all pixels in a tile; coordinates/global IDs and spatial context stay unchanged. Wavelengths/FWHM/validity/source IDs are selected together. If FWHM exists it is passed to VCA through the optional forward argument; absent bandwidth stays absent. Centers must be explicit nm, with approximate quality retained. Index-only fallback and coverage modulation are outside this first integration. No final-test or raw target GT access is introduced.

`adapt(..., observation_policy=None)` retains legacy behavior, including its historical FWHM omission. An explicit full policy is numerically equivalent for the current delivered scenes without FWHM; it deliberately passes provided FWHM, so do not claim equivalence to a legacy path that ignored real supplied bandwidth. `g3_training._forward/_update` accept an optional preselected view, and `forward_partition` accepts optional FWHM; unchanged callers take the original path.

Each augmented run saves `observation-policy.json`; checkpoint sidecar/result bind its hash and full selection condition. Trace rows bind the selected record hash/B at every update. `verify_run_artifacts` rejects missing/corrupt/mismatched policies or traces. Reload the matching policy explicitly when evaluating an augmented checkpoint; it is not hidden global state. `evaluate_conditions` reports three heads' confusion/OA/AA/Kappa/F1/NLL and per-class both-wrong rates on the fixed45 released validation labels, preserving weights. Extra conditions do not select checkpoints. The policy is not optimizer-resume state. Failed/existing run directories are never overwritten by adapt.

Remote acceptance (existing authorized container/environment; external outputs):

```bash
M1_DEVICE=cpu python -m unittest discover -s tests -p test_observation_training.py -v
M1_DEVICE=cuda python -m unittest discover -s tests -p test_observation_training.py -v
python scripts/validate_m1_training.py --expected-commit FULL_SHA \
  --out-dir NEW_EXTERNAL_DIRECTORY \
  --development-root ACCEPTED_DEVELOPMENT_ROOT --protocol-root ACCEPTED_PROTOCOL_ROOT \
  --data-root DATASET_ROOT
```

The validator runs focused CPU/full CPU/GPU/existing M1 GPU regressions and an optional real full/subset four-updates-per-arm preflight. It freezes both policies before training, matches initializations/tile samples, verifies actual B/centers, adapter gradients/updates, selected checkpoint reload, fixed-condition three-head/per-class/curve reviews and unchanged input hashes. Four updates establish engineering participation only; formal full-versus-subset accuracy, historical zero-fill control, multiple seeds/supports and cross-sensor directions remain uncompleted research work.

CPU disabled/full-policy training comparisons require exact numerical/state equality against the actual parent implementation. NativeFP32 CUDA keeps initialization, sampling, selected steps and discrete validation outputs exact, with absolute3e-6 bounds on loss/NLL/selected weights and a same-version repeat control. The first acceptance attempt retained two exact-GPU-equality failures. Independent old/old and new/new repeats also differed: maximum state differences8.20e-7/3.95e-7 and validation NLL differences6.78e-7/8.48e-7; old/new state difference9.91e-7. These measurements justified the bounded native-GPU comparison; production arithmetic, defaults and training budgets were not changed.
