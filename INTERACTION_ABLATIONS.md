# VCA 双分支交互与图边权更新：模块及消融指南

推荐从克隆后的仓库根目录运行。三个模块均已实现；默认关闭交互和边权更新，保留原始模型路径。正式精度实验需要自行分组训练。数据集需按 `dataset/README.md` 自行准备。

## 模块边界

| 模块 | 文件 | 输入与输出 | 消融方式 |
|---|---|---|---|
| 区域与像素对齐 | `model/region_pixel_alignment.py` | 使用 Q 将像素特征汇聚到区域、区域特征映射到像素 | `--alignment-mode region/global` |
| 双向选择性交互 | `model/cross_branch_interaction.py` | 两路 128 维中间特征、对齐消息 → 残差更新后的两路特征 | `--interaction-mode none/add/gated`，支持单向交互 |
| 稀疏图边权更新 | `model/graph_edge_update.py` | 区域特征、可选 CNN 区域特征和 A₀ → A₁ | `--graph-update-mode none/graph/joint` |
| 参数与校验 | `model/interaction_config.py` | 所有入口共用参数定义 | 无需修改模型源码即可切换 |
| 保存和加载 | `model/checkpoint_io.py` | state_dict 与同名 `.pt.json` 配置文件 | 自动按保存配置重建模型，严格加载权重 |
| 预设实验 | `interaction_ablation_presets.py` | 10 个实验配置 | `run_paper_ablation.py --suite interaction` |

实际路径：VCA → 共享 64 维特征 → CNN/图网络第一层（128 维）→ 对齐与交互 → 可选边权更新 → 两路第二层（64 维）→ 原始加权融合 → 分类头。

`SharedHGCN.forward_stage1/forward_stage2` 暴露中间层，原有参数名和维度保留。交互和边权更新均关闭时，原始权重可以严格加载。

## 三个开关的含义

### 对齐

- `region`：使用同一个 Q 做列归一化聚合和像素回映，支持稀疏软分配并保留梯度。
- `global`：用整幅图像/所有区域的均值作为跨分支消息，移除区域对应关系，作为对照。图分支自身的区域聚合与输出回映仍使用原 Q。
- 对齐是消息交换的形状桥梁，不能在保留交换的同时简单删除。`global` 是无区域对应关系的有效对照。
- 两个下游模块都关闭时，对齐设置不影响输出。只启用 `graph` 边权更新且关闭交互时，对齐同样不参与计算。

### 交互

- `none`：没有跨分支特征残差。
- `add`：经归一化和投影的消息直接作为残差加入；没有选择门控。
- `gated`：根据目标特征、来源特征及其差异生成 sigmoid 门控，再加入消息。
- `--interaction-direction both/cnn_to_graph/graph_to_cnn`：双向、CNN → 图、图 → CNN。
- 双向消息都从同一份交互前特征计算，避免先后顺序带来的偏置。
- `--interaction-strength 0.1`：固定残差强度。设为 0 可检查分阶段执行与原模型输出的一致性。
- `--interaction-hidden-dim 32`：门控隐藏层宽度。
- `--interaction-chunk-pixels 16384`：分块计算消息，训练时配合梯度检查点减少门控中间激活；设为 0 关闭分块与此处的检查点。
- 门控值是信息选择权重，不直接等价于校准后的置信度。

### 边权更新

- `none`：两层图卷积均使用 A₀。
- `graph`：只用图分支中间特征重新计算边权，不读取 CNN 特征。
- `joint`：使用图特征和对齐后的 CNN 特征；启用交互时使用交互后的两路特征。
- `--graph-update-dim 32`：边权相似度嵌入维度。
- `--graph-update-mix 0.5`：A₁ = (1 − mix) × 行归一化 A₀ + mix × 新边权。
- 更新仅在已有候选边上进行，不生成全连接相似度矩阵。原本没有出边的节点补自环。
- 一次前向中 Q 不变，下一次前向仍由可学习分配模块重新生成 Q。所有相关梯度均保留。
- 默认消融使用 `learned` 初始图，它已行归一化。若改用 `static`，建议额外运行 `graph` 模式加 `--graph-update-mix 0`，区分行归一化与学习边权的收益。彻底关闭模块请使用 `none`。
- 不包含学习增删边、更新 Q、多轮循环；本版是一次中间交互、一次边权更新。

## 10 个预设实验

所有预设共用网络深度、初始图模式、归一化、训练参数和分类头。新增模块初始化不会改变原有主干/分类头的随机初始化。

| 名称 | 对齐 | 交互 | 方向 | 边权更新 | 目的 |
|---|---|---|---|---|---|
| `interaction_baseline` | 未使用 | 关闭 | — | 关闭 | 原始双分支＋同一种初始图 |
| `interaction_add` | 区域 | 直接相加 | 双向 | 关闭 | 普通交互 |
| `interaction_gated` | 区域 | 门控 | 双向 | 关闭 | 门控增益 |
| `interaction_cnn_to_graph` | 区域 | 门控 | CNN → 图 | 关闭 | 方向消融 |
| `interaction_graph_to_cnn` | 区域 | 门控 | 图 → CNN | 关闭 | 方向消融 |
| `interaction_edge_graph` | 未使用 | 关闭 | — | 图特征 | 图更新本身 |
| `interaction_edge_joint` | 区域 | 关闭 | — | 两路特征 | CNN 信息用于边权的价值 |
| `interaction_add_edge` | 区域 | 直接相加 | 双向 | 两路特征 | 完整结构的无门控对照 |
| `interaction_full` | 区域 | 门控 | 双向 | 两路特征 | 完整研究原型 |
| `interaction_global` | 全局汇聚 | 门控 | 双向 | 两路特征 | 区域对应关系的贡献 |

## 分开运行

先只生成命令，不训练：

```powershell
python run_paper_ablation.py --suite interaction --experiments all --datasets paviaU,xuzhou,WHUL --seeds 0,2026,3407 --device cpu --dry-run --out-dir experiments/interaction_plan
```

分别运行基线、门控交互、完整结构。下面每条只执行一个配置的三个种子；先按需要调整数据集、轮数和设备。有 CUDA 版 PyTorch 与可用显卡时可将 `--device cpu` 改为 `--device cuda`。

```powershell
python run_paper_ablation.py --suite interaction --experiments interaction_baseline --datasets paviaU,xuzhou,WHUL --seeds 0,2026,3407 --device cpu --epochs 500 --out-dir experiments/interaction_baseline
```

```powershell
python run_paper_ablation.py --suite interaction --experiments interaction_gated --datasets paviaU,xuzhou,WHUL --seeds 0,2026,3407 --device cpu --epochs 500 --out-dir experiments/interaction_gated
```

```powershell
python run_paper_ablation.py --suite interaction --experiments interaction_full --datasets paviaU,xuzhou,WHUL --seeds 0,2026,3407 --device cpu --epochs 500 --out-dir experiments/interaction_full
```

其余配置替换 `--experiments` 的名称，使用不同输出目录。默认初始图为 `learned`、归一化为 `domain_bn`；整组切换可用 `--interaction-graph-mode static`、`--interaction-norm-type group_norm`，比较时所有组保持一致。

直接使用主训练入口也可以：

```powershell
python train_vca_multidataset.py --datasets paviaU --graph-mode learned --interaction-mode gated --interaction-direction both --alignment-mode region --graph-update-mode joint --epochs 500 --save-path weights/interaction_full.pt --metrics-json experiments/interaction_full_metrics.json
```

`--cnn-loss-weight`、`--graph-branch-loss-weight` 默认均为 1，保留原有辅助监督。主融合仍使用原来的固定加权。第一轮消融建议保持相同损失权重。

单数据集旧入口也支持这些模型开关，参数加 `vca-` 前缀，例如 `main.py --use-vca --vca-graph-mode learned --vca-interaction-mode gated --vca-graph-update-mode joint`。推荐使用多数据集入口做本套消融及自动配置恢复。

## 权重、结果与诊断

多数据集训练保存原格式 `.pt` 权重，以及旁边的 `.pt.json`（结构、类别数、训练配置）。移动模型时应一起移动这两个文件。模型仍返回融合、图分支、CNN 分支三个概率输出；需要诊断时额外返回 `graph_info`。

每次运行的 metrics JSON 包含所有开关、参数量及门控均值/标准差、边权变化。调度器输出 `commands.txt`、`all_runs.csv`、`summary.csv`、`summary.md`。命令文件用于审阅，实际执行使用参数列表。

导出完整模型的门控和边权：

```powershell
python export_interaction_diagnostics.py --checkpoint weights/interaction_full.pt --datasets paviaU --device cpu --out-dir results/interaction_full_diagnostics
```

输出：

- 两个方向的像素门控 PNG，固定 0–1 显示范围；
- `interaction_arrays.npz`：融合/两分支预测、原始门控数组、候选边索引、更新前后边权；
- `interaction_summary.json`：门控及边权变化统计。

关闭某模块时不输出其专属诊断。NPZ 保存全部像素预测，不将测试标签用于前向计算或门控。旧模型缺少侧边配置时可通过 `--metrics-json` 指定原实验配置。

既有分类图和 VCA 可视化脚本已接入配置恢复，并传递正确的建图尺度。旧论文制图脚本本身仍保留其原有数据集/配色选择；新实验的通用诊断入口是上面的脚本。

## 验证与实验边界

运行不依赖真实数据的检查：

```powershell
python -m unittest discover -s tests -p "test_interaction_modules.py" -v
```

覆盖稀疏/密集对齐、空区域、软分配梯度、双向/单向开关、分块梯度、边权归一化与稀疏支持、全部预设的静态/可学习图前后向、多波段多分类头、权重保存加载、合成数据训练到诊断导出。

为公平比较，训练入口现在同时设置 PyTorch 随机种子；这不保证不同 GPU 平台逐位一致。按数量划分数据时已修正验证/测试重叠，并改为使用传入的 `val_num`。历史按数量划分的结果需要重新跑基线。

预设不包含真实传感器留出、空间隔离划分或目标域少样本适配，仍沿用现有多数据集训练协议。通过功能检查不能说明精度提升或跨传感器泛化有效。后续论文实验应报告多种子结果、参数/运行成本，并单独验证泛化主张。
