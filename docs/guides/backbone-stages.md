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

## 动态新增任务（Issue #66）

`MultiDatasetVCASAHGCN.add_dataset_head(name, class_count)` 保持原接口与三头结构。新头继承对应分类头族已有参数的设备/dtype（空头族回退到 backbone），运行模式继承外层模型。`backbone.add_dataset` 新建的域 BatchNorm 继承该归一化包装器的设备/dtype 和 train/eval 模式；例如外层训练但 backbone 冻结为 eval 时，新归一化层仍保持 eval。新域使用全新默认仿射参数及运行统计，不复制源域已学习状态，也不改动已有任务。

新增参数默认可训练。已经创建的优化器不会自动发现它们；应在注册任务后创建优化器，或显式加入新头和新域归一化参数。例如保留原优化器状态：

```python
known = {id(p) for group in optimizer.param_groups for p in group['params']}
model.add_dataset_head('new_sensor', class_count)
added = [p for p in model.parameters() if p.requires_grad and id(p) not in known]
optimizer.add_param_group({'params': added, 'lr': target_lr})
```

`target_lr`、权重衰减和冻结策略由当前任务明确指定；若需要不同参数组，按头/归一化分别登记。该方法返回值仍为 `None`，不会修改外部优化器，也不会自动继承旧参数的 `requires_grad=False`。DDP 等包装器和编译后的训练图应在任务注册后重新按调用方既有流程构建，本模块未实现运行中结构迁移。

动态新增任务会进入 `export_config()` 的任务列表，现有 checkpoint 接口可重建并恢复三头及域归一化层。checkpoint 加载沿用原有“构建默认 dtype 模型并加载权重、返回 eval”行为；不声称保存运行模式或任意混合 dtype。重复添加分类任务仍报错，重复注册已有归一化域仍保持幂等。

针对性回归：`python -m unittest discover -s tests -p test_dynamic_task_heads.py -v`；设置 `DYNAMIC_HEAD_DEVICE=cuda` 在远程 GPU 验证。非默认 dtype 的验证覆盖分类头及归一化单元，整模型静态/学习图前后向使用现有 float32 路径，不扩展为任意 dtype 支持。

## 验收

稠密邻接转换缓存修复（Issue #64）：`SharedHGCN` 仅在 eval、无梯度且未开启 autocast 时缓存最近一次稠密图转换，并检查输入对象身份与张量版本。原地修改、不同对象／视图、训练模式切换、设备／dtype 迁移和权重加载后不会继续使用旧缓存。无版本计数器的 inference tensor 直接重算；稀疏图沿用 coalesce 路径。缓存不进入 state_dict，保留原稠密图转换为 float32 的行为。

缓存只支持 PyTorch 能追踪版本的更新；通过 `.data` 或外部共享存储修改图后，调用 `clear_graph_cache()`。梯度开启时重新转换，保证预热推理不会截断之后的邻接梯度；稠密图训练可能因此增加转换成本，当前 G3 稀疏图路径不承担该转换成本。本修复不作精度或速度提升主张。

针对性回归：`python -m unittest discover -s tests -p test_graph_cache.py -v`；同一测试加 `GRAPH_CACHE_DEVICE=cuda` 可在远程 GPU 执行。正式验收结果另见模块交付报告。

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
