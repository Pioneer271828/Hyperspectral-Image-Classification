# E1-C: matched conditioning controls and registered efficacy screen

[Issue #89](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/89) / [PR #90](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/90) depends on [E1-B](coverage-integration-e1b.md). E1-C1 delivers control implementation, an immutable experiment runner and a short engineering preflight. The full accuracy batch is a separate execution/acceptance step. Reuse the accepted in-repository E1 operator, [pinned FiLM/DOFA review](../references/e1-reference-review.json), `ObservationPolicy`, development trainer, spectral observer and existing GPU controller. No external code/dependency, source pretraining, main merge or automatic compression change is involved.

E1-C2 is complete under [Issue #91](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/91) / [PR #92](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/92): all15 cases/75000 updates used frozen source21f72ad073737144896c4c3d8937b1ba43b8fe00 on GPU2/5/6. None versus continuous full OA was71.85±15.13% versus71.85±15.77%; primary range OA56.30±2.22% versus54.81±3.23%. The registered joint rule failed; keep E1 off. The [results/per-run review](../../research/reports/E1C2_五组覆盖调制三种子完整配对结果_20261102.md) records all-case artifact/pair/arithmetic checks and representative full-size resume with120 unchanged case files and no added updates. These are reused45-label development results, not independent real-sensor confirmation. Next: MON-2 monitoring, without automatic retraining or M4/E1 combination.

## Five paired arms

All enabled arms use the same194-parameter MLP (hidden16/value_dim1), zero output initialization, strength0.1, geometry sigma0.08/window0.16 and local initialization seed0. Four controls transform only the MLP input; the original physical `CoverageDescription` and the true support mask are retained.

| Arm | MLP input | Meaning |
| --- | --- | --- |
| `none` | no E1 module | Matched-observation baseline; VCA still supplies its existing support handling |
| `anchor_only` | `[p,sin(pi*p),cos(pi*p),sin(2*pi*p),cos(2*pi*p),sin(4*pi*p),cos(4*pi*p),1,1]` | Ordinary static anchor-wise modulation, independent of observation geometry/quality |
| `binary` | actual VCA `anchor_valid` repeated in the four geometry slots; anchor/quality slots retained | Binary support information without continuous distance/count/gap |
| `continuous` | original9 descriptor inputs | Unchanged accepted E1-B hypothesis |
| `shuffled` | jointly permute geometry columns1:5 across anchor rows; anchor/quality slots retained | Misaligned geometry negative control |

`anchor_only` is independent of observation **values**, but still shares the mandatory actual support/fallback gate. This is a parameter-count control, not a claim of identical effective expressivity for all input bases. Binary and continuous can coincide in information when all anchors have identical support; that is an experimental observation to report. True unobserved anchors remain protected even after nonzero affine biases.

Shuffling uses a private CPU generator seeded by SHA256 of recorded init seed, metadata fingerprint and K. It is fixed for a given observation identity, restored without extra state, and does not consume the model/global RNG. If K>1 an identity permutation is replaced by a one-position roll. Some rows may still be fixed points; this is not a guarantee of a full derangement. Metadata identity affects only this deliberate negative control, never the physical continuous descriptor. All arms keep the original input bands/coordinates and observation policy.

Enable through `coverage_mode='affine'` and `coverage_condition=anchor_only|binary|continuous|shuffled` (`--coverage-condition` in the legacy CLI). Default `continuous` is omitted from canonical model/config exports and runtime signatures, **including affine models**, preserving E1-B hashes and sidecar compatibility. All-default none behavior remains unchanged. Binary direct standalone calls require actual boolean `anchor_valid`; integrated backbone calls supply it. Strict transfer verifies condition semantics. Raw state dictionaries alone cannot detect same-shape semantic mismatches; retain the sidecar/config.

## Frozen protocol and interpretation

The versioned [specification](../../configs/experiments/e1_c_paired_screen.json) registers seeds64/512/2048 × five arms ×5000 updates =75000 future target updates. Model seed/root, head+2, spatial sampler+4, observation sampler+6, Adam/LR5e-5/head5e-4/decay0.001, class-first sampling, GroupNorm, M4/interaction off and full-validation-NLL selection every100 updates are fixed. All arms share **one observation schedule per seed**. Their common backbone parameters (excluding E1), three heads, sampler trace, observation sequence, normalization and evaluation bank are checked before aggregation. None and affine backbone hashes are not compared with their different parameter sets mixed in.

Following the negative strong-subset M1-C screen, this is an explicit opt-in moderate training cycle: full, random75%, full, lower75%-range, upper75%-range. It is not a default augmentation change or a claim that this new cycle improves accuracy. Crop bounds are computed from the delivered **full** observation's physical range, never from each selected subset. VCA/E1 normalization continues to use the fixed400–2500nm reference.

Evaluation freezes nine conditions: full; random75/50/25%; middle25% contiguous gap; stride2; lower50%, upper50%, middle50% ranges. The three range masks must differ from every scheduled training mask; low-resolution inputs causing an exact collision are rejected. The held-out distinction is **training-unseen combinations of bands**, not unseen wavelengths, an independent validation population, or real new sensors. All conditions use the existing same45 devval labels; devtrain90 alone supervises training. No final test evaluation is provided by this runner. Historical full-trained M1-C results are descriptive context, not an unpaired causal E1 control.

Register the decision before running: continuous must have mean full OA/AA at least matched none and nonnegative full-OA differences in at least2/3 roots. Its mean OA over the three training-unseen ranges must exceed **each** none/anchor-only/binary/shuffled control, with at least2/3 positive paired range differences for each. Otherwise retain E1 off. Report sample standard deviations, every seed/condition/three-head/per-class record and paired differences; a passing candidate remains exploratory pending independent splits/supports/sensors. Balanced five-label-per-class devval makes full OA and AA numerically dependent; do not present them as two independent confirmations. No threshold, seed, strength, window or budget search is included.

## Entry points, artifacts and recovery

All commands execute in the authorized remote training container, from a clean exact full SHA/full-history checkout. Data and outputs stay outside checkout. Do not update an active/frozen checkout. Use an unused external output and explicit GPU UUIDs:

```bash
python scripts/run_e1_ablation.py prepare --expected-commit FULL_SHA \
  --out-dir NEW_OUTPUT --development-root DEV_ROOT --protocol-root PROTOCOL_ROOT \
  --data-root DATA_ROOT --smoke
python scripts/run_e1_ablation.py run --expected-commit FULL_SHA \
  --plan NEW_OUTPUT/plan.json --gpus GPU-UUID
python scripts/run_e1_ablation.py run --expected-commit FULL_SHA \
  --plan NEW_OUTPUT/plan.json --gpus GPU-UUID --resume
```

Smoke is seed64 × five arms × five updates, validates full NLL each update, and always returns `engineering_preflight_only`. Omitting `--smoke` **prepares** the future fifteen-case plan; training starts only with a subsequent `run`. At most one seed worker uses a GPU, and modes within each seed run sequentially. Reuse the existing exclusive controller lock, process accounting, GPU snapshot and failure handling. No unrelated process is signaled. End-to-end elapsed/peak allocated memory on shared GPUs are recorded, not used as controlled speed evidence.

Each case saves training trace, selected model/sidecar, frozen observation policy, three-head/per-class/condition review, module calls/gradients/actual parameter updates, and a completed receipt with artifact hashes. Completed cases require `--resume` plus full artifact/protocol verification, then return operation `verified_existing` while experiment status remains `completed` and `artifact_dir` still points to the original training. A failed/in-progress/corrupt case is preserved and rejected; create a separately registered retry location after diagnosing the cause, never overwrite it. Resume is not optimizer-state continuation.

Final aggregation checks pairing and reads every case; it does not select a favorable arm/seed to hide failures. A frozen plan is tied to exact source SHA/fingerprint and files. A later documentation commit is not automatically the training SHA. Retain the tested checkout or check out the immutable experiment tag; if implementation/settings change, preregister a new batch instead of mixing old cases.

## Acceptance scope

`scripts/validate_e1c.py` checks control CPU/CUDA, five-arm synthetic training/recovery, fullCPU, E1-A/E1-B CUDA and VCA lifecycle. Exact continuous operator/config compatibility is tested against actual E1-B source. The prior documented native M1 numerical limitation remains: E1-B numerical training controls use explicit FP64 and1e-10 bounds, while native control functionality and real preflight remain FP32. No tolerance is silently widened.

Optional real roots trigger only the five-arm × five-step preflight, byte-preserving verified resume, read-only historical E1-B provenance/checkpoint loading, and freezing the future full-budget plan **without running it**. Actual hashes, commands, failures, analysis and the next execution boundary are in the [acceptance report](../../research/reports/E1C1_公平覆盖对照与精度批次预登记验收_20261102.md). Module engineering and accuracy effectiveness are separate conclusions.
