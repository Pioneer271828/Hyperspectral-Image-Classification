# Hyperspectral Image Classification

Research code for hyperspectral image classification: the SAHGCN baseline, a Variable Channel Adapter (VCA), a CNN/HGCN dual-branch backbone, end-to-end learnable superpixel graphs, region–pixel alignment, selective interaction, and graph edge-weight updates.

This repository was organized from an existing research workspace. It retains the core models, training, ablation, diagnostic export, and paper visualization code. Datasets, checkpoints, experimental results, and presentation materials must be supplied separately.

## Modular Development

Follow the [research master checklist](research/研究内容整合与详细搭建测试计划_2026-10-28.md) and the [module implementation plan](research/模块化搭建与测试消融实施计划_2026-10-28.md). Each module is versioned and validated remotely, followed by a pause for review. Engineering validation has been completed for the baseline, C0, C1, M2, M3, M5, M8, E0a, and M6. M7 candidate-edge reweighting remains in the engineering roadmap; formal accuracy experiments are assessed separately. The current research priorities and monitoring plan are linked below.

- [Baseline provenance and fingerprints](docs/migration/source-inventory.md), [baseline validation report](research/reports/baseline_2026-10-29.md)
- [Remote validation and synchronization guide](docs/guides/remote-validation.md)
- [C0 configuration, run records, and ablation checklist](docs/guides/c0-experiments.md): start with a dry run; verify matching run identity before skipping completed work or retrying failed attempts
- [C0 delivery report and acceptance scope](research/reports/C0_配置与实验记录模块交付_2026-10-29.md)
- [C1b spatial access contract and limitations](docs/guides/spatial-protocol.md), [validation report](research/reports/C1b_空间访问契约_20261029.md)
- [C1c strict multi-block training](docs/guides/spatial-training.md), [two-step end-to-end validation report](research/reports/C1c_严格空间训练闭环_20261029.md)
- [M2 VCA metadata, bad-band handling, and cache contract](docs/guides/vca.md), [GPU validation report](research/reports/M2_VCA接口与缓存回归_20261029.md)
- [M3 regional graph state, label-free controls, and manual injection](docs/guides/graph-state.md), [GPU validation report](research/reports/M3_区域图契约与无标签对照_20261029.md)
- [M5 two-stage backbone, graph contract, and ablation configurations](docs/guides/backbone-stages.md), [GPU validation report](research/reports/M5_统一两阶段骨干与图契约接入_20261029.md)
- [M8 fixed fusion, optional logits, and loss components](docs/guides/fusion-losses.md), [GPU validation report](research/reports/M8_固定融合与损失契约验收_20261029.md)
- [E0a frozen predictions, boundary evaluation, and branch error diagnostics](docs/guides/evaluation-e0a.md), [validation report](research/reports/E0a_冻结预测评估与诊断验收_20261029.md)
- [Per-class diagnostics for the three heads after full-budget screening](docs/guides/screening-branch-diagnostics.md), [CNN/Graph/fusion errors and gating report](research/reports/M6_冻结三路逐类错误与门控诊断_20261030.md)
- [E0b validation-only message interventions](docs/guides/message-interventions.md), [frozen bidirectional, unidirectional, and disabled-message report](research/reports/E0b_验证区冻结消息干预_20261030.md)
- [M6 alignment and selective interaction contract](docs/guides/interaction-m6.md), [GPU validation report](research/reports/M6_对齐与选择性交互验收_20261029.md)
- [Machine-readable module index](docs/architecture/module-map.json), [contribution workflow](CONTRIBUTING.md), [agent instructions](AGENTS.md)
- [Long-term repository architecture RFC](research/长期工程仓库结构与Issue协作方案_RFC草案_2026-10-28.md): migrate toward the proposed layout one module at a time

**Current execution policy: run all code, dependency installation, and tests inside the training container through `ssh rmvision`. Windows is used only for editing, documentation, and Git.** Run the Python, installation, and training commands below remotely. The current container uses its existing framework installation; follow the remote guide for the actual environment and validation commands. Synthetic checks and empirical research conclusions are recorded separately.

## Installation and Quick Checks

For a fresh environment, Python 3.10 or 3.11 is recommended by the original dependency setup, which retains the research environment's `numpy==1.24.3` pin. Install a PyTorch build appropriate for your CUDA environment if GPU training is required. For the existing remote training container, follow the environment notes in the remote guide instead of reinstalling its framework.

```bash
git clone https://github.com/Pioneer271828/Hyperspectral-Image-Classification.git
cd Hyperspectral-Image-Classification
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
python -m unittest discover -s tests -p "test_interaction_modules.py" -v
```

The two checks use synthetic data. They cover different band counts, forward and backward passes with static and learnable graphs, interaction, checkpoint handling, and the training-to-diagnostic-export workflow. No real dataset download is required. Run installation and training commands from the repository root.

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
