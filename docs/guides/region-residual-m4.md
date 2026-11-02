# M4-A: standalone region representation and pixel residual

This is an independent API on the verified pre-M4 baseline, tracked by [Issue #73](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/73). Backbone, training CLI/configuration, existing checkpoint wrappers and default model behavior remain unchanged. M4-B will connect the module to training and evaluate accuracy. The dated pre-M4 delivery ledger/verifier continues to describe its pinned historical baseline; it deliberately rejects later model additions such as M4-A.

## Inputs and mathematics

`RegionResidualDecomposition(channels=64, mode='none')` takes dense `X[N,C]` and the caller's existing `Q[N,M]`. Q is finite, nonnegative, row-stochastic (absolute row-sum tolerance 1e-5), and dense or sparse COO; empty columns are allowed. X and Q share a device and float32/float64 dtype. Move the module to that same dtype/device. Half precision/autocast is outside this acceptance scope. No labels, new Q, dense conversion of sparse Q, input cache or new reconstruction loss.

The module reuses existing `pool_regions` and `broadcast_regions`:

```text
mass = Q.sum(0)
Z = (Q / clamp_min(mass, 1e-6)).T @ X
Xr = Q @ Z
R = X - Xr
```

For hard assignments, nonempty region means and per-region zero-mean residuals have their usual interpretation. For soft Q, `QZ` is a smoothing reconstruction; residuals need not pool to zero and are not claimed orthogonal/high-frequency components. Empty regions produce zero Z. Positive mass below 1e-6 is attenuated by the existing clamp, explicitly counted in diagnostics rather than silently claimed an exact mean.

## Modes and controls

| Mode | CNN-side output | Added parameters at C=64 | Initialization |
|---|---|---:|---|
| `none` | original X object | 0 | bypass; region pooling still available |
| `x_residual` | Linear([X,R], C) | 8,256 | [I,0], zero bias |
| `residual_only` | R | 0 | parameter-free |
| `capacity_control` | X + Linear2(LeakyReLU(Linear1(X))) | 8,320 | Linear1=I, Linear2=0, biases=0 |
| `x_region` | Linear([X,Xr], C) | 8,256 | [I,0], zero bias |

Canonical names follow the module implementation plan; earlier research prose names such as `x_plus_residual` are descriptions, not accepted CLI aliases. The capacity control uses only X, has 64 more parameters than the paired projection, and a different compute graph; it is a nearby-capacity control, not an exact FLOP match. Initialization consumes no random numbers, so adding the standalone object does not shift later random draws. At [I,0], the residual/region hint initially has zero influence on the CNN-side output (and zero Q-gradient through that hint); hint weights must learn. Z remains differentiable from X/Q. The control's first-layer gradient starts at zero until its initially-zero second layer changes. These are disclosed initialization properties, not proof of an inactive trained module.

## Output and lifecycle

```python
layer = RegionResidualDecomposition(channels=64, mode='x_residual').to(X)
out = layer(X, Q, return_intermediates=False, return_diagnostics=True)
cnn_features = out.cnn_input           # [N,C], differentiable
region_features = out.region_features # [M,C], from ORIGINAL X, differentiable
```

`return_intermediates=True` additionally returns differentiable `out.region_at_pixels` and `out.residual`. Both are None by default. `out.diagnostics` is empty unless requested; requested summaries are detached Python scalars (mode/counts, min/max mass, empty/tiny-positive regions, residual squared energy divided by X squared energy with denominator floor 1e-30). They are logging values, not losses. Zero input yields relative energy zero. Diagnostics and input validation can synchronize GPU work; this version makes no speed claim.

Standalone save/reload (artifacts belong outside the checkout):

```python
torch.save({'config': layer.export_config(), 'state_dict': layer.state_dict()}, path)
saved = torch.load(path, weights_only=True, map_location=X.device)
restored = RegionResidualDecomposition(**saved['config']).to(X)
restored.load_state_dict(saved['state_dict'], strict=True)
restored.eval()  # caller restores execution mode explicitly
```

This saves module configuration/parameters, not optimizer/RNG or exact training continuation. No change to existing model checkpoint formats is implied. Purely parameter-free modes have empty state_dicts; their mode/channel config is still needed.

## Remote acceptance and next phase

Engineering acceptance passed at `f6e937300b3393e0adf2f685909ce1fc289cf56c`: 16 focused CPU, 16 focused CUDA, all 222 CPU regressions and both original model smokes. [Report, commands and limits](../../research/reports/M4A_区域表示与像素残差独立模块验收_20261102.md); [PR #74](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/74). This records standalone engineering only; the pre-M4 baseline above remains useful for the next integration comparison.

Use a clean, full-history checkout at the exact published SHA in the authorized training container, with an external unique output directory and process-level GPU allocation. Preserve the other projects and existing environment. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/validate_m4a.py --expected-commit FULL_PUBLISHED_SHA \
  --out-dir /absolute/external/NEW_M4A_RUN
```

The entry verifies container/SHA/history/cleanliness, scope and source fingerprints; it runs focused CPU/CUDA checks, the full CPU regression suite and original CPU/GPU model smoke. Logs/commands/environment/failures are retained outside checkout. The independent tests use hand calculations, analytical/numerical gradients, invariances, no-densification/no-cache checks, exact bypass/RNG tests, save/reload and short synthetic optimization showing the hint/control can actually learn. They do not estimate real-data OA, boundary accuracy or cross-sensor gains.

Next, M4-B1 wires both ordinary/interactive paths, original-X regional features and shared Q; preserves default-off behavior and old checkpoint compatibility; verifies invocation/gradient/feature-change diagnostics and a short closed loop, then stops. A separate M4-B2 will run prespecified budget/capacity-matched accuracy ablations. Establish valid boundary-label coverage before claiming boundary gains. Existing 45-label development validation has no valid boundaries under its declared rule.
