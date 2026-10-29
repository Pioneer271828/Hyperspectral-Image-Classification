# M5：统一两阶段骨干

`VCASAHGCNBackbone.forward_features` 在 VCA/去噪和构图之后，只调用一套 `forward_stages`。普通、交互、边更新路径共享顺序：CNN/图第一阶段 → 可选一次消息交换 → 可选图边权更新 → CNN/图第二阶段 → 图广播 → 原固定权重融合。分类头、三头监督和训练选模不变。

| 接口 | 输入 | 输出 |
|---|---|---|
| cnn_stage1 | `[H,W,64]` | `[N,128]` |
| graph_stage1 | `[M,64] + A0` | `[M,128]` |
| cnn_stage2 | `[N,128] + (H,W)` | `[N,64]` |
| graph_stage2 | `[M,128] + A1` | `[M,64]` |

这些是现有模块的方法包装，没有新增参数或复制层。`CNN_Branch.0/1`、`HGNN_Branch.hgc1/hgc2` 等权重键、卷积核 3/5、归一化和宽度保留。没有插件时 A1=A0；边更新启用时，第一阶段用 A0、第二阶段用更新后的 A1。

`forward_stages(cnn_input, region_input, Q, A, dataset_name=None, collect_stats=False, return_diagnostics=False)` 返回 `(cnn_pixels, graph_pixels, interaction_info)`。CNN 输入与区域输入是独立参数：区域描述由调用方池化，在统一调度前可以单独变换 CNN 输入。M4 尚未实现，本模块只验证该入口在消息关闭/开启时都生效，不声称完成区域残差分解。没有新诊断分类器或第二轮交互。

## 图契约与对照

learned 路径改用 M3 `build_graph`，再由 Q/A0 和原正则进入统一阶段；原 `graph_info` 损失键保留并增加 M3 诊断。默认数值与旧模型对比回归。新增构造器和 C0 ModelConfig 字段：

- `graph_assignment_mode`: `learned`（默认）或 `fixed_grid`。
- `graph_adjacency_mode`: `learned`（默认）或 `fixed_local`。

非默认对照要求 `graph_mode=learned`。`graph_mode=static` 仍表示历史外部 Q/A 路径；其 Q/A 不强加 GraphState 新的行归一化约束。两者名称与元信息分开。

旧训练 CLI 支持：

```bash
python train_vca_multidataset.py --graph-mode learned \
  --graph-assignment-mode fixed_grid --graph-adjacency-mode learned
```

上面命令的其他训练参数使用原默认预算，仅说明选项；本轮验收使用下文明确的两步配置。C0 JSON、严格空间入口、checkpoint 导出/恢复均携带这两个字段，不同对照具有不同配置指纹。新默认字段会进入配置序列化；旧实验保留原配置记录，不覆盖旧身份或产物。

`forward_features`、多数据集 forward 和单数据集包装器新增可选 `graph_state=`：校验后使用注入图，绕过构图器。多数据集接口不能同时传 Q/A；单数据集包装器在明确提供 GraphState 时覆盖其旧 Q/A 缓冲。注入图使用当前 clean 特征池化区域输入，Q/A/正则的梯度不被 detach。旧三元组/概率输出结构不变，不在 graph_info 中塞入不可序列化的 GraphState 对象。

旧 checkpoint 缺少新字段时，构造器按 learned/learned 填充；没有权重迁移或键重命名。新 checkpoint 通过已有 `.pt.json` 保存图对照。仍使用 state_dict，不直接 pickle GraphState。

## 验收

仅在 `ssh rmvision` 的独立、冻结 checkout：

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python scripts/validate_m5.py \
  --expected-commit <完整SHA> \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/m5-<唯一批次>
```

完整 CPU 测试含旧版骨干对照：3 种 norm × static/learned × 5 种插件配置 × 两个数据集/波段数，比较输出、输入/参数梯度、参数量/键和 BN 缓冲；另测阶段调用顺序/形状、独立 CNN 输入、图控制梯度、手工注入、新旧 checkpoint、配置/CLI 和历史 static 兼容。CPU/CUDA smoke 保留。

真实 GPU 用 `m5_spatial_plain.json / m5_spatial_fixed_q.json / m5_spatial_joint.json`：PaviaU、K64/embed32、明确 group_norm、tile32/gap2/seed2026，各两次优化，原三头监督和图正则、验证选模/恢复后测试。跟踪四阶段前向和反向，成功后恢复必须 skipped，不能重新优化。epochs=0 沿用历史含端点语义，表示这一轮两步工程检查。没有更改默认模型或正式预算。

这三个配置是训练链路检查，不是收敛、三模式精度优劣或跨传感器实验。新 GraphState 校验/摘要会产生 GPU 同步开销，重构不承诺加速。M8 的融合分类契约、M4 的残差实现和正式精度消融分别验收。
