# BG-ABL1: independent branch and fixed propagation controls

[Issue #109](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/109) / [PR #110](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/110) follow the restored calibrated joint baseline. This module delivers engineering, default equivalence and short real-data acceptance; the full three-seed accuracy campaign is a separate task.

| Variant | Active feature branches | Training losses | Propagation |
|---|---|---|---|
| `dual_legacy_hypergraph` | CNN + Graph | Original primary + Graph + CNN | Frozen original hypergraph |
| `dual_fused_hypergraph` | CNN + Graph | Primary only, coefficient1 | Frozen original hypergraph |
| `cnn_fused_only` | CNN | Primary only, coefficient1 | Not executed |
| `graph_fused_hypergraph` | Graph | Primary only, coefficient1 | Frozen original hypergraph |
| `dual_legacy_ordinary` | CNN + Graph | Original three heads | Geometry-only centroid4NN union, self loops, symmetric degree normalization |
| `dual_legacy_identity` | CNN + Graph | Original three heads | Identity matrix |

The three primary-only variants form the independent branch comparison. Comparing the two dual hypergraph variants measures auxiliary-supervision differences. The three legacy-loss variants form the propagation comparison. Retained parameters share the original initialization: construct the original architecture first, then actually remove unused branches/auxiliary heads. All variants keep VCA and the two shared denoise blocks. Graph-only removes the spatial `CNN_Branch`; it retains the common denoise front. The primary classifier retains its original `heads` parameters, and its readout key remains `fused` even in a single-branch model. No inactive auxiliary readouts are stored.

Independent branches differ in parameter count; this is not a matched-capacity comparison. Identity still pools through Q, applies learned regional layers, and broadcasts through Q; it is not absence of the regional branch. Ordinary propagation uses the same frozen train-label-LDA/SLIC Q, so only its new edge construction is geometry-only. Hypergraph-versus-ordinary changes the neighborhood/construction/weights and does not isolate only hyperedge order or reproduce the canonical GCN architecture. [Original papers, pinned official source/license review and reuse boundary](../references/branch-graph-ablation-review.json) are recorded; no external trainer/dependency is copied or executed.

`BranchGraphAblation` returns a mapping of actual active heads. CNN-only ignores Q/A in its forward, and Graph-only never executes the spatial CNN branch. `controlled_items` constructs fixed graph controls once without modifying the source items; it hashes source/effective Q/A. The trainer uses an explicit versioned `branch_graph_joint_ablation_v1` identity and a validated `AblationRuntime` adapter. Original trainer/CLI defaults retain the `runtime_adapter=None` path.

The shared trainer still owns the original random schedule, profile budgets, Adam, scheduler, validation events, strict first maximum selection, raw training-state continuation, atomic publication and completed resume. Scene-calibrated validation/selection/inference uses the accepted BN policy; calibration never writes back into the raw optimizer model. Active-head gradient/update counts and parameter manifests expose actual branch activity.

Checkpoints have a persistent version/variant signature and explicit sidecar policy. The ordinary legacy loader rejects these states. Use `load_ablation` for a raw ablation model or `load_ablation_selected` for a selected calibrated model, providing the matching raw validation source and controlled items. The latter verifies source/scene/state and Q/A identity. Reconstruct controls and use the bound loader before inference; changing Q/A after loading bypasses that caller-side contract and is only appropriate for explicitly declared diagnostic interventions. Cross-variant resume, wrong source/scene/graph and signature tampering are rejected.

Run inside the authorized remote container on a clean published full SHA, with outputs outside checkout:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_branch_graph_ablation.py \
  --expected-commit <SHA> --variant cnn_fused_only --profile engineering --seed 64 \
  --prepared-dir <accepted-honghu-preparation> --data-root <immutable-dataset-snapshot> \
  --out-dir <new-owned-run>
# Same command with --resume verifies a completed run without changing its files.
```

The engineering profile executes12 updates, not a convergence experiment. The full profile retains1503 updates, seeds64/512/2048 and51 validation events, but needs a separately frozen campaign. Keep test labels out of this engineering entry. Accuracy conclusions require independently trained branches and complete paired budgets; auxiliary-head scores and short-loop results cannot substitute for them. Current status, failures and next task are in the shared checklist/report.
