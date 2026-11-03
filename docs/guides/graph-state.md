# M3：区域图契约与独立对照

现有 `LearnableSuperpixelGraph.forward(features, scale)` 仍返回 `(Q,A,auxiliary)`，保留 learned 数学行为、参数和 state_dict 键。新增 `build_graph` 返回 `model.graph_state.GraphState`，用于后续模块独立消费，尚未改变骨干/CLI 默认路径。

```python
builder = LearnableSuperpixelGraph(feature_dim=64)
state = builder.build_graph(latent_hwc, superpixel_scale=16)
regions = pool_regions(state.Q, latent_hwc.reshape(-1, 64))
propagated = torch.sparse.mm(state.A0, regions)
pixels = broadcast_regions(state.Q, propagated)
loss = task_loss(pixels) + weight * state.regularizers["compactness_loss"]
```

## 状态、损失与诊断

- `Q[N,M]`：合并后的稀疏 COO，非负、每行和为 1，列质量用于池化归一化，允许空区域。
- `A0[M,M]`：初始有向行归一化图，非负，含自环；不保证对称。方向与现有稀疏矩阵乘法一致，M7 的后续重加权另行处理。
- `region_positions[M,2]`：归一化 `(y,x)`；默认来自规则网格种子中心，不能解释为学习到的语义中心。手工图须自行提供对应位置。
- `image_shape`、可选 `grid_shape`、`region_count`：实际形状和 M。scale 是目标面积参数；实际 M 为 `ceil(H/sqrt(scale))*ceil(W/sqrt(scale))`，不等于简单的 `H*W/scale`。
- `regularizers` 保留 compactness/entropy/balance 三项可微标量；不要用 detach 后的诊断数值构造训练损失。
- `diagnostics` 为 detach 后的元信息/摘要；包含实际 M、网格、Q/A nnz、最大行度、空区域数、区域质量范围、行和误差及构造模式。`detached_diagnostics()` 另返回含 detach 损失值的独立副本。新增摘要不会自动加入旧 forward 的热路径。

`validate(features)` 检查形状、dtype/device、有限非负值、索引范围、行和、坐标范围与标量正则，不截断 Q/A 或损失的梯度。新 build_graph 默认执行验证与摘要，GPU 上会有同步成本；未据此宣称性能优化。

默认归属半径 1、图半径 1，每行最多 9 个局部候选，边界删除越界候选。候选均局限于本次输入块的局部编号，原图全局位置/分区由 C1 observation 管理。空区域池化为零，A 行仍有效。半径为零或只有一个候选时，softmax 没有可学习的选择自由度，不应要求评分梯度非零。

## 无标签对照与手工注入

| assignment_mode | adjacency_mode | 含义 |
|---|---|---|
| fixed_grid | fixed_local | 像素按所在空间格硬归属；局部区域邻居均匀权重 |
| fixed_grid | learned | 固定归属；保持原有学习边权公式 |
| learned | learned | 既有端到端 learned 基线，默认 |
| learned | fixed_local | 可选的另一因素对照 |

模式仅是新 builder 的显式选项。fixed_grid/fixed_local 的 Q/A 只取决于形状、scale 和候选偏移，不取决于标签或特征值；固定正则是对应几何的数值，不创建额外梯度。fixed Q/learned A 的图评分仍使用输入特征及共享 feature_norm。学习 Q 会同时改变图节点描述，不能将其与边权学习混为一个因素。

这些对照与历史 `graph_mode=static` 分开命名；历史 static 路径可能用标签引导 LDA-SLIC，不作为无标签固定图的替代。正式精度对照的训练预算和正则权重应在实验 Issue 预先固定，不能据本次数值检查声称分类收益。

手工构造 `GraphState(Q,A0,positions,image_shape,...)` 后，使用 `builder.build_graph(features, scale, graph_state=state)` 注入，验证后直接返回该对象，不重建 Q/A。也可让独立消费者直接读取 state。手工注入不能同时指定非默认构图对照。`state.to_legacy()` 提供旧三元组视图，损失保持原计算图。GraphState 是单次输入的临时数据，checkpoint 继续保存模块 state_dict 与构造配置，不直接 pickle GraphState。

M5 再统一骨干消费者和 CLI 配置；M3 不增加全局边、交互、剪枝或分类头。固定图实验不能通过修改现有训练默认模式静默启用。

## 远程验收

仅在 `ssh rmvision` 的干净独立 checkout 执行：

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python scripts/validate_m3.py \
  --expected-commit <完整SHA> --device cuda \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/m3-<唯一批次>
```

运行完整单元测试、CPU/CUDA smoke，与旧图实现对比 Q/A、正则及梯度。真实 PaviaU/Xuzhou/HongHu 使用 C1 固定 tile32/gap2/seed2026 和训练块统计，各取首个 train/val/test 块，经 K64/embed32 VCA 后独立构图。三个主要对照用无标签数值目标检查梯度；只有训练块允许更新图参数，验证/测试块只检查图与分区访问，不计算分类指标。固定 Q/A 没有可更新图参数，记录零优化步。访问其他分区后回访训练块验证无跨块缓存影响。原始产物写 checkout 外；失败保留并用新目录重试。
