# BG-ABL2: complete independent branch and propagation comparison

[Issue #111](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/111) / [PR #112](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/112). The [preregistration](../../research/experiments/BGABL2_六组独立支路图传播三种子完整配对预登记_20261103.md) fixes all six arms, three seeds, budgets and five contrast directions before training. The [result report](../../research/reports/BGABL2_六组独立支路图传播三种子完整配对结果_20261103.md) and [machine-readable summary](../architecture/bg-abl2-results.json) record the completed evidence and limitations. See the [engineering guide](branch-graph-ablation.md) for routing, initialization, head removal, graph controls and source/license boundaries.

Training and frozen evaluation use branch `codex/issue-111-bg-ablation-full`, full commit `7f4688076bf19b814f210676d64d1d08a2e3610c`, tag `experiment-bg-abl2-full-v1`. All221 executable/config fingerprints match the accepted BG-ABL1 snapshot. This experiment changes no production source, optimizer, loss, sampling, normalization default or budget. Later result documentation has its own delivery commit; it is not the training SHA.

The authorized alternate LXC uses `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`, Python3.10.20, PyTorch2.7.1+cu128/CUDA12.8. The primary `ssh rmvision` was checked and available; this campaign reused the alternate's authenticated preparation and dedicated output folders. GPUs2/4/5 execute one independent job each; they are shared with other workloads and this is not a speed benchmark. No other process or dependency environment was modified.

Before running, confirm container, GPU availability, the exact clean checkout, preparation manifest and input hashes. The manifest SHA256 is `414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b`. Code is synchronized from the published GitHub branch through a verified Git bundle because the container lacks repository credentials; this is not a claim that direct authenticated GitHub cloning succeeded. Keep the source checkout unchanged throughout training, freeze, readout and independent QA.

The actual per-case command, run inside the container, is:

```bash
cd /home/lbx/hsi-pioneer271828/checkouts/bg-abl2-issue111
PY=/home/lbx/hsi-pioneer271828/envs/hsi/bin/python
SHA=7f4688076bf19b814f210676d64d1d08a2e3610c
PREP=/home/lbx/hsi-pioneer271828/runs/base97a-honghu-issue97-20261103-prepared01
DATA=/home/lbx/hsi-pioneer271828/datasets/snapshots/joint-honghu-20261103-issue97/dataset
CAMPAIGN=/home/lbx/hsi-pioneer271828/runs/bg-abl2-issue111-20261103-full01
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/train_branch_graph_ablation.py \
 --expected-commit "$SHA" --variant dual_legacy_hypergraph --profile full --seed 64 \
 --prepared-dir "$PREP" --data-root "$DATA" \
 --out-dir "$CAMPAIGN/cases/dual_legacy_hypergraph/seed-64"
```

The detached supervisor repeats this exact CLI for `dual_legacy_hypergraph`, `dual_fused_hypergraph`, `cnn_fused_only`, `graph_fused_hypergraph`, `dual_legacy_ordinary`, `dual_legacy_identity`, with seeds64/512/2048 on GPUs2/4/5 respectively. Each arm is independently trained from scratch for1503 updates; no historical reference weights are reused. There are18 runs and27054 updates. The full eighteen command arrays and assignments are retained in `launch.json`; progress, exit codes and failures are retained in `status.json` and logs. Completed `--resume` must verify zero updates and unchanged artifacts. Never reuse an existing output directory for a fresh experiment.

Every completed case receives a train/validation-only review: actual schedule/exposure, loss, selected epoch, calibration/selection identity, train-validation gap, final50-epoch validation change, finite gradients and actual active-group updates, parameter counts and checkpoint hashes. These reviews do not change any remaining arm. Freeze all18 selected models, raw sources, Q/A bindings, results and reviews before reading test labels for scoring. Historical test inspection is disclosed; this is a matched fixed-split joint benchmark, not unseen-sensor generalization or an untouched-test confirmation.

External orchestration/readout/QA scripts and their hashes are retained with the acceptance archive, outside the repository. After all18 cases complete, the actual evaluation commands are:

```bash
RECORDS=/home/lbx/hsi-pioneer271828/transfer-records/issue111-20261103
ANALYSIS=/home/lbx/hsi-pioneer271828/runs/bg-abl2-issue111-20261103-analysis01
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" "$RECORDS/analyze_campaign.py" \
 --checkout "$PWD" --commit "$SHA" --campaign "$CAMPAIGN" --out "$ANALYSIS"
"$PY" "$RECORDS/plot_curves.py" --campaign "$CAMPAIGN" --out "$ANALYSIS"
"$PY" "$RECORDS/independent_qa.py" --checkout "$PWD" --commit "$SHA" \
 --campaign "$CAMPAIGN" --analysis "$ANALYSIS"
```

The evaluator writes `frozen-before-test.json` before the new readout. It uses bound calibrated loaders, verifies train/validation predictions against saved records, and stores compact labels/predictions/true-class probabilities in split-index order. Independent QA recomputes324 metric cases,81 complementarity cases and all five contrasts, and rechecks budgets, first-maximum selection, code, inputs and training-file immutability. `validation-curves.png` is a standalone scientific figure; the report lists every training review and exposes all failed contrast gates as well as successful ones.

Parameter counts differ across independent branches. Graph-only retains the shared denoise front, identity still includes Q pooling/broadcast and learned region layers, and ordinary graph construction changes neighbors/weights. These controls do not isolate pure hyperedge order, equalize capacity, establish boundary improvement, or measure deployment speed. The previous95.86±0.35% calibrated baseline remains a separate historical result. Do not tune on this test, automatically replace defaults, merge main or launch another module from the result. The shared current checklist defines the next reviewed task.
