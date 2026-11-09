# BN-ALIGN3: freeze calibrated baseline evaluation

[Issue #107](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/107), [PR #108](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/108) follow the accepted [calibrated selection module](calibrated-selection.md). No earlier model, trainer or default is changed. The same pinned BN reference review applies; no new external implementation or dependency is introduced.

The reuse validator runs paired raw/calibrated tiny synthetic training for all three seeds. It requires exact raw model, Adam, RNG and step-trace fingerprints, and directly verifies that calibration consumes no Python/NumPy/CPU/selected-CUDA RNG and leaves the source state unchanged. CPU and CUDA receipts must match the exact clean evaluator source and all output hashes. Scheduler history and best selection intentionally differ; they are not asserted equal.

The evaluator separately reconstructs both schedulers at all153 historical events. It authenticates the saved raw scheduler using the original raw NLL, then requires calibrated-NLL learning rates to match at every event. A divergence rejects reuse before scoring. Accepted source files, input identities and all previous calibration artifacts are authenticated. This is an audited reuse of historical weights, not a claim that a new4509-update calibrated training was executed. Other schedules, budgets or protocols need their own gate.

Before any new test readout, the CLI writes `reuse-gate.json` and `frozen-before-test.json`, locking all three seeds and all three arms:

- Original raw selected checkpoints.
- BN-ALIGN1 calibration of the original selected checkpoints.
- BN-ALIGN2 calibrated reselection, already fixed by validation.

Only then does scoring assemble test labels/indices. Calibration and selection receive no test supervision. The primary metric is mean three-task fused test OA; report each task's OA/AA/kappa/NLL, per-class recall, every seed, paired differences and sampleSD. The preregistered gain gate requires positive mean macroOA change, positive changes in at least2/3 seeds, and no task's meanAA dropping more than0.5 percentage points. The evaluator reports failure of this research gate as a valid result; it never changes the selected model or configuration to pass.

The existing full-scene transductive access policy includes unlabeled pixels and train-label-LDA static graphs. This is a seen-task joint accuracy benchmark. The split's historical test results were already inspected; this new frozen matched comparison is not untouched independent confirmation or strict unseen-sensor generalization. M4/M1/E1/compression are disabled. Auxiliary CNN/Graph heads are not independently trained single-branch models.

Compact records retain the frozen pixel order, hard prediction and true-label probability for all243 metric cases, enabling independent confusion/OA/AA/kappa/recall/NLL recomputation without storing every class probability. All original files stay immutable; outputs belong outside checkout.

Run only in the authorized container, with an unused external directory and the full published SHA:

```bash
python scripts/validate_calibrated_reuse.py --expected-commit <SHA> --device cpu --out-dir <cpu-gate>
CUDA_VISIBLE_DEVICES=2 python scripts/validate_calibrated_reuse.py --expected-commit <SHA> --device cuda --out-dir <cuda-gate>
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_calibrated_baseline.py \
  --expected-commit <SHA> --campaign-dir <base97c-full01> \
  --prepared-dir <frozen-preparation> --data-root <frozen-dataset> \
  --original-analysis-dir <base97c-analysis01> --bn-align1-dir <bn-align1-real02> \
  --bn-align2-dir <bn-align2-replay01> --cpu-reuse-dir <cpu-gate> \
  --cuda-reuse-dir <cuda-gate> --out-dir <new-evaluation> --device cuda
```

After this module, freeze independent CNN-only/HGNN-only and hypergraph/ordinary-graph/identity-propagation comparisons. Any later change remains validation-selected; no follow-up tuning on this test readout.
