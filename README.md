# Hyperspectral Image Classification

高光谱图像分类研究代码：SAHGCN 基线、可变通道适配器（VCA）、CNN/HGCN 双分支、端到端可学习超像素图，以及区域与像素对齐、选择性交互和图边权更新。

本仓库整理自现有研究工作目录，保留核心模型、训练、消融、诊断导出和论文制图代码。数据集、权重、实验结果及会议材料需自行准备。

## 模块化搭建入口

以[研究主清单](research/研究内容整合与详细搭建测试计划_2026-10-28.md)为工作依据，按照[模块交付计划](research/模块化搭建与测试消融实施计划_2026-10-28.md)逐项搭建。每完成一个模块，独立版本管理、远程验收并暂停评审。当前先固定工程基线，随后进行 C0 配置与运行记录。

- [基线来源与指纹](docs/migration/source-inventory.md)、[本次模块报告](research/reports/baseline_2026-10-29.md)
- [远程测试与同步指南](docs/guides/remote-validation.md)
- [C0 配置、运行记录与消融清单](docs/guides/c0-experiments.md)：先 dry-run，按相同身份验证跳过或保留失败重试
- [机器可读模块索引](docs/architecture/module-map.json)、[协作流程](CONTRIBUTING.md)、[智能体入口](AGENTS.md)
- [长期结构 RFC](research/长期工程仓库结构与Issue协作方案_RFC草案_2026-10-28.md)：未来目标目录仍按模块渐进实施

**当前执行约定：所有代码运行、依赖安装和测试统一在 `ssh rmvision` 的训练容器中执行。Windows 只编辑、维护文档和执行 Git。**下文 Python/安装/训练命令均在远程运行；当前容器沿用预装框架，使用上述远程指南中的环境和验收入口。合成检查与真实研究结论分别记录。

## 安装与快速验证

建议使用 Python 3.10 或 3.11（依赖文件保留了研究环境的 `numpy==1.24.3` 固定版本）。GPU 训练请按自己的 CUDA 环境安装相应 PyTorch。

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

以上两个检查使用合成数据，覆盖不同波段数、静态/可学习图前后向，以及交互、检查点和训练到诊断导出的流程，无需下载真实数据集。安装与训练命令均从仓库根目录执行。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| `model/variable_channel_adapter.py` | 将不同波段数映射到统一隐光谱空间 |
| `model/VCASAHGCN.py` | 共享 VCA、CNN/HGCN 骨干及数据集专属分类头 |
| `model/learnable_superpixel_graph.py` | 可微稀疏像素到区域分配和动态图 |
| `model/region_pixel_alignment.py` | 像素与区域特征双向映射 |
| `model/cross_branch_interaction.py` | 直接/门控及单向/双向交互 |
| `model/graph_edge_update.py` | 候选边上的可学习边权更新 |
| `model/checkpoint_io.py` | 权重及模型配置保存/恢复 |
| `model/SAHGCN.py`、`model/GCN.py`、`model/F2HGNN.py` | 原始模型与对照实现 |
| `createGraph/`、`loadData/` | 建图、数据读取、标签掩码与数据划分 |
| `train_vca_multidataset.py` | 多数据集训练主入口 |
| `main.py` | 原单数据集训练入口，可用 `--use-vca` |
| `run_paper_ablation.py` | VCA 与交互消融实验调度 |
| `export_interaction_diagnostics.py` | 交互门控、图边权及预测导出 |
| `export_*classification_maps.py` | 分类图导出 |
| `generate_vca_visualizations.py` | VCA 注意力可视化 |
| `make_paper_figures.py`、`scripts/` | 论文图表、数据立方体及 HongHu 结果整理 |

## 数据准备

按 [dataset/README.md](dataset/README.md) 放置数据。常用训练名称包括 `paviaU`、`xuzhou`、`whu-honghu`、`WHUL`。其中 `WHUL` 指 LongKou；旧名称 `WHU` 会对 HongHu 执行历史裁剪和标签变换，完整 HongHu 实验请使用 `whu-honghu`。

## 训练与消融

查看完整参数：

```bash
python train_vca_multidataset.py --help
```

单数据集、端到端学习图示例：

```bash
python train_vca_multidataset.py --datasets paviaU --graph-mode learned --device cpu --epochs 500
```

多数据集共享模型示例：

```bash
python train_vca_multidataset.py --datasets paviaU,xuzhou,whu-honghu --graph-mode learned --device cpu --epochs 500
```

交互和图边权更新默认关闭；`--graph-mode static` 保留 LDA-SLIC 路径。先生成整组实验命令供检查：

```bash
python run_paper_ablation.py --suite interaction --experiments all --datasets paviaU,xuzhou,WHUL --seeds 0,2026,3407 --device cpu --dry-run --out-dir experiments/interaction_plan
```

完整交互模型与诊断导出：

```bash
python train_vca_multidataset.py --datasets paviaU --graph-mode learned --interaction-mode gated --graph-update-mode joint --device cpu --epochs 500 --save-path weights/interaction_full.pt --metrics-json experiments/interaction_full_metrics.json
python export_interaction_diagnostics.py --checkpoint weights/interaction_full.pt --datasets paviaU --device cpu --out-dir results/interaction_full_diagnostics
```

CUDA 可用时可将 `--device cpu` 改成 `--device cuda`。检查点需同时保留 `.pt` 与同名 `.pt.json` 配置文件。模块含义、10 组交互预设和验证边界见 [INTERACTION_ABLATIONS.md](INTERACTION_ABLATIONS.md)。

## 使用范围与来源

- `make_paper_figures.py` 和 HongHu 专用脚本依赖已有实验表格、检查点或图像，具体参数及输入见源码；仓库没有附带这些产物。部分旧基线和数据分支保留历史绝对路径，使用其他数据集前需调整。
- 传感器波长元数据中存在近似值，说明见 `loadData/data_read.py`。功能验证不代表分类精度提升或跨传感器泛化已获验证。
- 本地原 Git 远程地址为 `https://gitee.com/lbxhhh/hyper-spectr.git`；原 README 保存在 [docs/SAHGCN.original.md](docs/SAHGCN.original.md)，其中的论文描述用于保留来源信息。本整理仓库不声明为该论文的官方仓库。
- 导入时源目录没有发现许可证文件，因此没有替原作者添加授权条款。
