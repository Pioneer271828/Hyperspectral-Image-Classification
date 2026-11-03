# Hyperspectral Image Classification

Research code for hyperspectral image classification: the SAHGCN baseline, a Variable Channel Adapter (VCA), a CNN/HGCN dual-branch backbone, end-to-end learnable superpixel graphs, region–pixel alignment, selective interaction, and graph edge-weight updates.

This repository was organized from an existing research workspace. It retains the core models, training, ablation, diagnostic export, and paper visualization code. Datasets, checkpoints, experimental results, and presentation materials must be supplied separately.

## Modular Development

**Use the [current baseline and delivery guide](docs/guides/current-baseline.md) for cloning, reproducibility and the next task.** The tested delivery snapshot is `2938698540b4eb51b4098aa781dd5e984b1613e1`, on `codex/issue-70-delivery-baseline`; its production implementation matches functional reference `6fa2d06747f79e90f95928b099ba2c3c0b18cbd0`. At the 2026-11-02 delivery snapshot, default `main` lacked the later delivered work. [Delivery status](docs/architecture/delivery-status.json) records exact PR dependencies, bridge commits, engineering/effect scope and pending integration. [Draft integration PR #71](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/71) is available for separate review.

Follow the [research master checklist](research/研究内容整合与详细搭建测试计划_2026-10-28.md) and the [module implementation plan](research/模块化搭建与测试消融实施计划_2026-10-28.md). Each module is versioned and validated remotely, followed by a pause for review. Engineering validation has been completed for the baseline, C0, C1, M2, M3, M5, M8, E0a, M6, and M4-A/M4-B1. M7 candidate-edge reweighting remains in the engineering roadmap; formal accuracy experiments are assessed separately. The current research priorities and monitoring plan are linked below.

- [G3 paired three-seed replication](research/reports/G3P2_三种子配对复验结果_20261102.md): scratch OA 63.03±1.76% versus full fine-tuning 62.89±4.74%; stable OA gains were not established. Class-average recall improved. Additional pretraining cost and a filesystem timestamp audit limitation are documented.
- [Earlier G3 exploratory pilot](research/reports/G3P1_跨传感器完整初筛结果与逐类分析_20261101.md): the single-seed gain did not translate into a stable OA gain in the three new repetitions above.
- [Baseline provenance and fingerprints](docs/migration/source-inventory.md), [baseline validation report](research/reports/baseline_2026-10-29.md)
- [Remote validation and synchronization guide](docs/guides/remote-validation.md)
- [Data protocols for the three research axes](docs/guides/data-protocol-matrix.md): preserve the original per-class1% training/0.5% validation joint accuracy benchmark; distinguish few-shot transfer, UDA and DG; inherit the dense baseline split for pruning/quantization and explicitly declare calibration context. Literature examples and proposed protocols do not change existing defaults.
- [BASE-97A joint benchmark engineering gate](docs/guides/joint-baseline-base97.md): opt-in frozen three-dataset1%/0.5% preparation, raw/split/graph/config identities, bounded round-robin updates and validation-selected classifier restoration. The short loop does not reproduce historical97% accuracy; full training is a separately registered task.
- [BASE-97B formal random training and continuation](docs/guides/joint-training-base97b.md): explicit12-update engineering/1503-update full profiles, original validation scheduling and selection, authenticated reuse of frozen graphs, full training-state continuation and immutable completed resume. [Engineering acceptance](research/reports/BASE97B_正式联合训练调度与断点恢复工程验收_20261103.md) covers328 regressions and real5+7+0 updates; the complete accuracy campaign is assessed below.
- [BASE-97C complete joint baseline and branch diagnosis](research/reports/BASE97C_三种子完整联合训练结果与双支路诊断_20261103.md): three seeds x1503 updates completed in8.5min at `8daa65b116ba585b33a84e3af93f3741716b710a`. PaviaU/Xuzhou/HongHu fused test OA96.52±3.05/85.06±12.43/84.40±8.04%, dataset macro88.66±6.10%; not a stable recovery of historical accuracy. Both branches update and affect predictions; fusion is0.56pp above the CNN auxiliary head on average, while HongHu's graph head is stronger than fusion. Validation-only fixed-weight probes localize a large train/eval discrepancy to front-end DomainBN statistics; official test scores remain unchanged. Frozen verification preserved183 training files and13 inputs; independent single-branch retraining remains pending. Next validate normalization-statistic consistency before more budget or new mechanisms.
- [BN-ALIGN1 scene-statistic calibration](docs/guides/bn-calibration.md), [acceptance and paired validation](research/reports/BNALIGN1_DomainBN统计校准与保存恢复验收_20261103.md): opt-in label-free denoise DomainBN calibration produces a separate eval model with unchanged parameters and scene/source/context/state binding. CPU9/GPU9/full337 and frozen three-seed acceptance passed; reused validation macroOA88.44±6.64→94.60±2.12%, with no new test scoring or optimizer updates. Strict FP64 controls distinguish native FP32 reduction differences; saved/reloaded state and supervised predictions agree. Historical183 training files/13 inputs unchanged. Original training defaults remain off; the BN-ALIGN2 follow-up below handles training-time selection. Independent accuracy confirmation is pending.
- [BN-ALIGN2 consistent calibrated selection](docs/guides/calibrated-selection.md), [acceptance and full historical replay](research/reports/BNALIGN2_验证选模统计一致性与历史轨迹复评验收_20261103.md): explicit versioned training flag aligns validation NLL/scheduler, OA selection and saved inference while preserving raw optimizer continuation. CPU7/GPU7/full344, three-seed legacy-default equivalence, all153 frozen validation states and5508 independently recomputed metric cases passed. Reused validation macroOA88.44±6.64→94.60±2.12→96.04±0.52%; all calibrated selections are step1503. No new real-data updates or test readouts; original183 files/13 inputs unchanged. Fixed-trajectory LR replay matches all153 original events. Frozen benchmark evaluation and reuse audit follow in BN-ALIGN3 below; independent generalization evidence remains separate.
- [BN-ALIGN3 frozen calibrated benchmark](docs/guides/calibrated-evaluation.md), [reuse gates, test results and per-class analysis](research/reports/BNALIGN3_校准联合基线冻结测试评价与轨迹复用验收_20261103.md): CPU/GPU paired synthetic training across all3 seeds and all153 scheduler events permit audited historical-weight reuse. Frozen test macroOA raw88.66±6.10%, original-checkpoint calibration94.48±2.03%, calibrated reselection95.86±0.35%; paired gains in3/3 seeds, preregistered AA guard passed. PaviaU/Xuzhou/HongHu OA98.81±0.55/98.18±0.13/90.58±0.78%; HongHu remains the bottleneck and its Graph auxiliary head exceeds fusion. New8/full352 tests,243 independently recomputed metric cases/81 complementarity cases passed;216 historical artifacts/13 inputs unchanged, zero new real-data updates. Previously inspected test split, full-scene transductive context, no unseen-sensor or untouched-test claim. Next independent single-branch/graph comparisons; current checklist is authoritative.
- [BG-ABL1 independent branch/graph training](docs/guides/branch-graph-ablation.md), [engineering acceptance](research/reports/BGABL1_独立支路与图传播对照工程验收_20261103.md): six variants separate primary-only dual/CNN/Graph comparisons from auxiliary supervision and fixed hypergraph/ordinary/identity propagation. Unused branches/heads are removed; retained initialization matches. New checkpoint signatures and bound loaders protect routing and Q/A. CPU9/GPU9/full361,48 parent/current default-state fingerprints and six real12-update loops passed;72 train/val metric cases independently recomputed,183 historical files/13 inputs unchanged. No new test scoring or full accuracy claim. Independent branches differ in capacity; identity retains regional processing, and the ordinary control changes graph construction. Next preregister full three-seed paired training, without adding another normalization/monitoring prerequisite.
- [C0 configuration, run records, and ablation checklist](docs/guides/c0-experiments.md): start with a dry run; verify matching run identity before skipping completed work or retrying failed attempts
- [C0 delivery report and acceptance scope](research/reports/C0_配置与实验记录模块交付_2026-10-29.md)
- [C1b spatial access contract and limitations](docs/guides/spatial-protocol.md), [validation report](research/reports/C1b_空间访问契约_20261029.md)
- [C1c strict multi-block training](docs/guides/spatial-training.md), [two-step end-to-end validation report](research/reports/C1c_严格空间训练闭环_20261029.md)
- [M2 VCA metadata, bad-band handling, and cache contract](docs/guides/vca.md), [GPU validation report](research/reports/M2_VCA接口与缓存回归_20261029.md)
- [M3 regional graph state, label-free controls, and manual injection](docs/guides/graph-state.md), [GPU validation report](research/reports/M3_区域图契约与无标签对照_20261029.md)
- [M1-B real-band training integration](docs/guides/observation-training-m1.md), [acceptance and per-run review](research/reports/M1B_真实波段子集训练接入验收_20261102.md): opt-in frozen per-update spectral subsets, independent fixed development perturbations, explicit metadata units and checkpoint-policy verification. Full CPU251/251, focused CPU/GPU9 each, existing M1 GPU6/6 and real full/subset four-update checks passed. Actual Xuzhou input changed from436 bands to327/218/109/328; these four-update checks established engineering behavior. The full-budget development comparison follows below; actual cross-sensor gains remain unverified. Existing defaults and M4-disabled baseline are retained.
- [M1-C paired full/legacy-zero/real-subset screen](docs/guides/m1-paired-screen.md), [results and per-run analysis](research/reports/M1C_完整波段与真实子集配对精度筛选_20261102.md): nine paired target-scratch runs/45000 updates at `experiment-m1c-dev-v1`. Full development OA was74.07%, legacy support-pixel zeroing74.81%, real subsets72.59%; five-condition robustness OA was73.04/73.48/71.11%. The registered joint-gain rule failed; retain full training and keep strong subset augmentation opt-in. CPU258/258, new CPU/GPU7 each, prior M1 GPU9, M4 controller GPU8 and real preflight/immutable resume passed. This is a45-label development screen, not final-test or actual unseen-sensor evidence. Independent E1 engineering is assessed below.
- [E1-A standalone coverage](docs/guides/observation-coverage-e1.md) and [E1-B production integration](docs/guides/coverage-integration-e1b.md), [E1-B report](research/reports/E1B_覆盖度骨干训练与恢复接入验收_20261102.md), [upstream review](docs/references/e1-reference-review.json): the opt-in194-parameter modulator now reaches backbone, configuration, training and classifier checkpoint restoration. CPU/GPU integration16 each, fullCPU288/288, E1-A GPU14, M1 native GPU functional6, M4 GPU12 and VCA lifecycle passed. Three numerical training comparisons use additional strict FP64 controls; prior native M1 NLL flakiness is documented. Paired three-update real preflight confirmed actual436/218/328-band metadata, parameter updates and selected-model restoration. Defaults remain off; engineering acceptance establishes no accuracy gain. E1-C progress is assessed below.
- [E1-C matched controls and frozen screen](docs/guides/coverage-screen-e1c.md), [E1-C1 acceptance](research/reports/E1C1_公平覆盖对照与精度批次预登记验收_20261102.md): none/anchor-only/binary/continuous/shuffled share observations, budgets and selection; four enabled arms have194 parameters. Control CPU/GPU8 each, campaign CPU/GPU5 each, fullCPU301/301, prior E1-B/E1-A GPU16/14 and a real five-arm x5-update preflight passed. Verified resume preserved40 case files, and fixed-model diagnostics confirmed the intended conditioning behavior. The E1-C2 fifteen-case/75000-update batch is complete: full OA71.85±15.13% for none versus71.85±15.77% for continuous; held-out-range OA56.30±2.22% versus54.81±3.23%. The registered joint-gain rule failed; keep E1 off. [Results and per-run review](research/reports/E1C2_五组覆盖调制三种子完整配对结果_20261102.md) records all cases, artifact/pair/arithmetic verification and zero-update resume with120 unchanged files. This reused45-label development screen establishes no final-test or real unseen-sensor gain.
- [DIAG-2 accuracy and variance diagnosis](research/reports/DIAG2_原高精度基线与当前低精度波动诊断_20261103.md): original joint PaviaU/Xuzhou/LongKou VCA records reach98.37/96.93/98.23% mean OA under different supervision, graph, normalization and evaluation settings. Nine current selected models fit all90 training labels at100% fused OA while development OA remains57.78–88.89%. Optimizer coverage and bounded GPU gradients passed; no new training/test evaluation. Recover a matched joint baseline before further hyperparameter/budget searches. Historical high accuracy is recorded evidence, not a new reproduction or an unseen-sensor result.
- [M4 region representation and pixel residual](docs/guides/region-residual-m4.md), [M4-A standalone report](research/reports/M4A_区域表示与像素残差独立模块验收_20261102.md), [M4-B1 integration report](research/reports/M4B1_骨干配置与训练接入验收_20261102.md): five modes now reach both backbone paths and shared training/checkpoint configuration. Integration checks passed on CPU/GPU (12 each), full CPU regression 234/234, and a two-update real development check confirmed hint participation. Accuracy evidence is assessed separately in M4-B2.
- [M4-B2 paired development screen](docs/guides/m4-paired-screen.md), [registration and results](research/reports/M4B2_四组配对开发筛选预登记与结果_20261102.md): all 12 target-scratch runs / 60000 updates completed at frozen tag `experiment-m4b2-dev-v1`. Baseline development OA was 75.56±3.85%, input+residual 71.85±3.39%; the registered candidate rule failed, so retain the baseline with M4 disabled. Regression 242/242 and real preflight/resume passed. The 45-label development screen establishes no final-test, boundary or cross-sensor transfer gains.
- [M5 two-stage backbone, graph contract, and ablation configurations](docs/guides/backbone-stages.md), [GPU validation report](research/reports/M5_统一两阶段骨干与图契约接入_20261029.md)
- [M8 fixed fusion, optional logits, and loss components](docs/guides/fusion-losses.md), [GPU validation report](research/reports/M8_固定融合与损失契约验收_20261029.md)
- [E0a frozen predictions, boundary evaluation, and branch error diagnostics](docs/guides/evaluation-e0a.md), [validation report](research/reports/E0a_冻结预测评估与诊断验收_20261029.md)
- [Per-class diagnostics for the three heads after full-budget screening](docs/guides/screening-branch-diagnostics.md), [CNN/Graph/fusion errors and gating report](research/reports/M6_冻结三路逐类错误与门控诊断_20261030.md)
- [E0b validation-only message interventions](docs/guides/message-interventions.md), [frozen bidirectional, unidirectional, and disabled-message report](research/reports/E0b_验证区冻结消息干预_20261030.md)
- [M6 alignment and selective interaction contract](docs/guides/interaction-m6.md), [GPU validation report](research/reports/M6_对齐与选择性交互验收_20261029.md)
- [Machine-readable module index](docs/architecture/module-map.json), [contribution workflow](CONTRIBUTING.md), [agent instructions](AGENTS.md)
- [Dynamic-network survey and candidate roadmap](research/动态计算与压缩课程学习_论文评审及搭建测试补充计划_20261102.md), [reference/code inventory](docs/references/research-code-inventory.json), [implementation-before-reuse checks](docs/guides/reference-code-reuse.md): low-rank/ternary compression, protected curriculum, early exits, fallback cascades and routing are planned candidates. Review original papers and author code/license/revision before implementation. Following the completed M1-C/E1 screens above, the next gate is recovery of the original three-dataset joint-training benchmark; MON-2 monitoring supports diagnosis before fixed probes and compression; none of these dynamic/compression candidates is enabled or empirically validated here.
- [Long-term repository architecture RFC](research/长期工程仓库结构与Issue协作方案_RFC草案_2026-10-28.md): migrate toward the proposed layout one module at a time

**Current execution policy: run all code, dependency installation, and tests inside the training container through `ssh rmvision`. Windows is used only for editing, documentation, and Git.** Run the Python, installation, and training commands below remotely. The current container uses its existing framework installation; follow the remote guide for the actual environment and validation commands. Synthetic checks and empirical research conclusions are recorded separately.

## Installation and Quick Checks

For a fresh environment, Python 3.10 or 3.11 is recommended by the original dependency setup, which retains the research environment's `numpy==1.24.3` pin. Install a PyTorch build appropriate for your CUDA environment if GPU training is required. For the existing remote training container, follow the environment notes in the remote guide instead of reinstalling its framework.

```bash
git clone --branch codex/issue-70-delivery-baseline https://github.com/Pioneer271828/Hyperspectral-Image-Classification.git
cd Hyperspectral-Image-Classification
git switch --detach 2938698540b4eb51b4098aa781dd5e984b1613e1
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
python -m unittest discover -s tests -p "test_interaction_modules.py" -v
```

The two checks use synthetic data. They cover different band counts, forward and backward passes with static and learnable graphs, interaction, checkpoint handling, and the training-to-diagnostic-export workflow. No real dataset download is required. Run installation and training commands from the repository root.

The commands above pin the tested delivery snapshot, including the ledger and verifier. Follow the current baseline guide for its verification receipt. Use a full-history checkout; historical source fixtures are part of the regression suite. Keep the existing remote environment when one is already configured.

## Code Entry Points

| Path | Purpose |
| --- | --- |
| `model/variable_channel_adapter.py` | Map variable band counts into a shared latent spectral space |
| `model/VCASAHGCN.py` | Shared VCA and CNN/HGCN backbone with dataset-specific classification heads |
| `model/learnable_superpixel_graph.py` | Differentiable sparse pixel-to-region assignment and dynamic graphs |
| `model/region_pixel_alignment.py` | Bidirectional mapping between pixel and regional features |
| `model/cross_branch_interaction.py` | Direct or gated, unidirectional or bidirectional interaction |
| `model/graph_edge_update.py` | Learnable edge-weight updates over candidate edges |
| `model/checkpoint_io.py` | Save and restore weights and model configurations |
| `model/SAHGCN.py`, `model/GCN.py`, `model/F2HGNN.py` | Original models and reference implementations |
| `createGraph/`, `loadData/` | Graph construction, data loading, label masks, and dataset splits |
| `train_vca_multidataset.py` | Main multi-dataset training entry point |
| `main.py` | Original single-dataset entry point, with optional `--use-vca` |
| `run_paper_ablation.py` | Schedule VCA and interaction ablation experiments |
| `export_interaction_diagnostics.py` | Export interaction gates, graph edge weights, and predictions |
| `export_*classification_maps.py` | Export classification maps |
| `generate_vca_visualizations.py` | Visualize VCA attention |
| `make_paper_figures.py`, `scripts/` | Paper figures, data cubes, and HongHu result processing |

## Dataset Preparation

Arrange data as described in [dataset/README.md](dataset/README.md). Common training names include `paviaU`, `xuzhou`, `whu-honghu`, and `WHUL`. `WHUL` refers to LongKou. The legacy name `WHU` applies historical cropping and label transformations to HongHu; use `whu-honghu` for experiments on the full HongHu dataset.

## Training and Ablations

Display all command-line options:

```bash
python train_vca_multidataset.py --help
```

Single-dataset training with an end-to-end learnable graph:

```bash
python train_vca_multidataset.py --datasets paviaU --graph-mode learned --device cpu --epochs 500
```

Multi-dataset training with a shared model:

```bash
python train_vca_multidataset.py --datasets paviaU,xuzhou,whu-honghu --graph-mode learned --device cpu --epochs 500
```

Interaction and graph edge-weight updates are disabled by default. `--graph-mode static` retains the LDA-SLIC path. Generate the full set of experiment commands for inspection first:

```bash
python run_paper_ablation.py --suite interaction --experiments all --datasets paviaU,xuzhou,WHUL --seeds 0,2026,3407 --device cpu --dry-run --out-dir experiments/interaction_plan
```

Full interaction configuration and diagnostic export:

```bash
python train_vca_multidataset.py --datasets paviaU --graph-mode learned --interaction-mode gated --graph-update-mode joint --device cpu --epochs 500 --save-path weights/interaction_full.pt --metrics-json experiments/interaction_full_metrics.json
python export_interaction_diagnostics.py --checkpoint weights/interaction_full.pt --datasets paviaU --device cpu --out-dir results/interaction_full_diagnostics
```

When CUDA is available, replace `--device cpu` with `--device cuda`. Keep both the checkpoint `.pt` file and its companion `.pt.json` configuration file. See [INTERACTION_ABLATIONS.md](INTERACTION_ABLATIONS.md) for module definitions, the ten interaction presets, and validation scope.

## Scope and Provenance

- `make_paper_figures.py` and the HongHu-specific scripts require existing experiment tables, checkpoints, or images. See their source for arguments and inputs; those artifacts are not included. Some legacy baselines and dataset paths retain historical absolute paths that need adjustment before use with other datasets.
- Some sensor wavelength metadata is approximate; see `loadData/data_read.py`. Functional validation does not establish improved classification accuracy or cross-sensor generalization.
- The original local Git remote was `https://gitee.com/lbxhhh/hyper-spectr.git`. The original README is preserved in [docs/SAHGCN.original.md](docs/SAHGCN.original.md), including its paper description for provenance. This repository does not claim to be that paper's official repository.
- No license file was found in the source directory at import time, so no licensing terms have been added on behalf of the original authors.

## Additional Experiments and Current Research Priorities

- [E0b validation-only coordinate sensitivity diagnostics](docs/guides/coordinate-interventions.md) (Issue #31)
- [E0b coordinate sensitivity validation report](research/reports/E0b_验证区冻结坐标敏感性_20261031.md)
- [B0 coordinate retraining with a matched budget](docs/guides/coordinate-retraining.md) (Issue #33)
- [B0 coordinate retraining launch and validation record](research/reports/E0b_B0坐标重训启动与验收_20261031.md) (see the report for full training results)

The current [three research axes and pruning/quantization plan](research/三主线研究重心与剪枝量化及论文用途_20261031.md) all prioritize improving accuracy:

1. Region–pixel complementarity.
2. Cross-sensor spectral adaptation.
3. Pruning and quantization, followed by speed optimization under a predefined accuracy constraint.

The [unified training monitoring implementation and integration plan](research/统一训练监测搭建与训练接入计划_20261031.md) has an initial opt-in MON-0/1 API (Issue #43); gradient/update monitoring and probe integration remain **planned** (Issue #37).

Linked research documents retain their original filenames; many are currently written in Chinese.

- [MON-0/1 read-only activation and weight measurements](docs/guides/monitoring-readonly.md): opt-in API; later monitoring integration stages remain planned.

- [MASK-1 explicit reversible weight masks](docs/guides/reversible-weight-masks.md): opt-in synthetic training validation; no automatic pruning or speed claim.
- [MASK-2 masked-training checkpoints](docs/guides/masked-training-checkpoints.md): explicit weight/mask/optimizer/RNG recovery for isolated experiments; default training is unchanged.
- [M1 real spectral subsets and synchronized metadata](docs/guides/observation-sampling-m1.md): independent data API; cross-sensor protocols and accuracy experiments remain separate.
- [E4a-1 frozen cross-sensor few-shot protocol](docs/guides/transfer-protocol-e4a.md): PaviaU to Xuzhou, explicit target-label budget and spatial access boundaries; no zero-shot claim.
- [E4a-2 G3 training entry point](docs/guides/g3-training.md): source-only pretraining, fresh target heads and fixed-budget adaptation; full pilot results are assessed separately.
- [G3-P1 full-budget pilot preregistration and run record](research/reports/G3P1_跨传感器完整初筛运行记录_20261101.md): completion and accuracy are reported separately.
