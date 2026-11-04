# M1-J1: real-band views in calibrated joint training

[Issue #125](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/125) reuses the existing M1 sampler in the recovered PaviaU/Xuzhou/HongHu joint benchmark. The independent entry is `scripts/train_joint_observation.py`; existing entry points and model defaults stay unchanged. This is an engineering integration, not an accuracy result.

## Fixed scope

Use the authenticated BASE-97 prepared snapshot, original per-class1% train/.5% validation, original three-head supervision, M4/E1 disabled, and existing full-observation DomainBN calibration/selection. Keep the source full-scene normalization and fixed Q/A. Select already normalized spectral columns without refitting statistics; the last two spatial coordinate channels are retained exactly. The whole scene's input bands change together. Only training labels enter the supervised loss.

These graphs were derived from full-spectrum observations. Retaining them gives the model additional spectral context even during a subset forward. This protocol is a controlled observation-augmentation benchmark, **not strict unseen-sensor validation**. A later real sensor protocol must construct graph/context from the observations it actually permits. It must also address approximate wavelength centers and absent measured response widths.

The CLI permits only the existing12-update engineering profile. `full` retains all bands; `alternate75` alternates full and75% random subsets by global update index, using the existing ceil-count rule. Sampling seed is explicit and independent of model/task scheduling seeds. No strong25% augmentation, M4 residual or primary-only supervision is implicitly enabled. A full-budget experiment requires a separate registered entry/recipe and comparison gates.

## Identity, replay and units

`freeze_settings` calls existing `BandMetadata`/`sample` on a1×1 label-free stub. It writes all selected indices and metadata hashes into the runtime settings before any optimizer update. The policy includes the original settings, input/graph/supervision fingerprints and prepared-plan hash. Labels are hashed for identity but never inspected to select bands. The private NumPy generator does not advance training RNGs. Policies are embedded in `identity.json`, training states, and classifier metadata; no unattached mutable sidecar is required.

The optional joint runtime receives the persisted **zero-based global update index**, making5+7 continuation replay exactly the same views as continuous12. Each trace row records actual/source band counts and sampling-record hash. Changed inputs, settings, metadata, recipe, budget or checkpoint policy fail validation. Already completed resume verifies existing artifacts and performs0updates.

Views synchronize spectral tensor, wavelength vector, optional FWHM and original band IDs. Validity is prefiltered by the delivered preparation; this integration rejects invalid source entries instead of silently altering the baseline. Metadata quality remains `approximate` or `index_only` as supplied. Explicit units reach training forward. Full-observation calibration retains its old auto-unit path, so metadata ambiguous for that path (for example2nm FWHM) is explicitly rejected. Current real datasets have no measured FWHM. Synthetic20nm synchronized widths are covered; this does not validate a sensor response.

## Shared interfaces and reuse

Only two optional hooks were added to `joint_training.run_training`: `validate_inputs(items)` before creating outputs, and `train_step_at(index, ...)` instead of the legacy step when provided. Other adapters retain their existing behavior. `train_step(..., forward_fn=None)` preserves the original forward by default. Architecture, optimizer, loss, budget and selection code are not replaced.

Reuse is project-local `loadData/observation_sampling.py` and `read_band_metadata` at parent84b7b6c. This task adds no new published mechanism and copies no external source. The existing [primary-paper/author-code review](../references/e1-reference-review.json) records related wavelength-conditioning literature, including pinned DOFA source/license; that is context, not the implementation of this sampler and not a claim of upstream reproduction. The G3 tile/statistics protocol is deliberately not transplanted into this joint benchmark.

## Remote verification

Actual code/test SHA: `27f14f1c101ce76c4b2d4d8f4ca76eb57fbd02f2`, branch `codex/issue-125-m1j1`. Primary container checkout `/home/rmvision/m1j1-issue125-20261104`; LXC checkout `/home/lbx/hsi-pioneer271828/checkouts/m1j1-issue125`. Code was pushed to GitHub then synchronized by Git bundle; both checkouts are clean and match the full SHA. This is not a claim of direct authenticated GitHub fetch in the container.

```bash
M1J1_DEVICE=cpu /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p test_joint_observation.py -v
M1J1_DEVICE=cuda /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p test_joint_observation.py -v
/home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -v
```

CPU5/GPU5/full372 passed. The two synthetic training comparisons require exact fingerprints of model, optimizer, scheduler, history, probabilities and RNG, including native GPU runs. Source views/coordinates/Q/A stay unchanged; subset gradients and final weights differ from full controls as expected. Corruption, changed-policy/graph and ambiguous-unit negative tests pass. Logs are `/home/rmvision/m1j1-{cpu01,gpu01,full01}.log`. These do not prove native equivalence for all shapes or long training.

Real preflight uses two arms×12updates, sampling seed8128/model seed64, on an available LXC GPU. Its external script and results are recorded in the acceptance report; do not substitute this short run for accuracy evidence. No formal test scoring or compression is included.
