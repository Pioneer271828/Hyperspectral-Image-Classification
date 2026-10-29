# M6：对齐与选择性交互契约

关联 Issue #21；工程验收与正式精度/机制实验分别记录。

## 接口与边界

`BidirectionalSelectiveInteraction.forward_aligned(pixels[N,C], regions[M,C], Q[N,M], alignment, collect_stats=False, return_diagnostics=False)` 返回两路更新特征和诊断。`RegionPixelAlignment` 由调用者提供，不注册新子模块，不改变旧 checkpoint 参数键。原 `forward` 接受已对齐消息的 API 保留。

region 使用列质量归一化的 `Q.T @ pixels` 聚合（空列为0），使用原 Q 权重回映。Q 为非负浮点、行归一化，骨干由 GraphState 验证；此处不重复做全图值扫描。global 对照只把跨路消息换成均值，图分支池化/传播仍使用原 Q，不表示取消全部区域结构。

两方向始终从同一交互前快照构造消息，不把已更新 CNN 特征送入另一方向。关闭方向不创建对应消息模块，也不做对应对齐；none 返回原对象。strength=0 对有限输入保持数值恒等，仍执行现有门控计算以保留旧梯度/诊断语义。默认 strength=0.1、hidden_dim=32、chunk_size=16384；不改变层宽、损失、图、采样或选模。

add 是现有的归一化＋线性投影残差交换，不是原特征直接相加。gated 在此基础上使用每个像素/区域一个标量门，不能称逐通道门或已校准可靠度。门值诊断 detach，不参与优化。分块 checkpoint 作用于消息计算，不切断稀疏 Q 值梯度。无分块使用 chunk_size=0。空的底层消息输入返回空输出和 [0,1] 形状门；正式骨干仍要求非空图与影像。

## 独立配置与远程检查

`configs/experiments/m6_spatial_{none,add,gated,cnn_to_graph,graph_to_cnn,global}.json` 六组只变交互条件，全部关闭 M7 图重加权；固定 PaviaU、tile32/gap2/split_seed2026、group_norm、模型种子2026及相同两步工程预算。global 组为双向 gated/global，其余为 region。不要将这些工程配置直接当完整精度实验预算。

通过 `ssh rmvision` 在独立 checkout 执行：

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python scripts/validate_m6.py \
  --expected-commit <完整已同步SHA> \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/m6-validation-<唯一批次>
```

入口要求容器、CUDA、完整SHA及干净工作树，执行全量回归和 CPU/CUDA smoke；六组各2步，记录消息/方向对齐实际调用、参数梯度与门范围，严格恢复并验证成功运行跳过优化器；全 test 块冻结预测通过 E0a 导出器核对混淆矩阵和指标，输入与模型指纹不得改变。原始日志/权重/数组在 checkout 外，不进 Git。失败保留，重试新目录。

## 研究验收仍需完成

本模块工程检查不能决定哪种交互更好。正式任务需固定完整训练/选模预算，比较无交互、add、gated、两单向和 global，至少三个配对重复，报告分支错误组合与 E0b 受控干预。分支前后纠错需要可比较读出或干预，不能从最终分支预测倒推中层变化。

当前固定 PaviaU test 分区缺类3/7，按双方有效异类四邻域定义无有效边界，边界指标 N/A。不得修改边界定义或依据 test 精度挑选划分。M4 尚未实现，本轮仅验证其未来可使用的通用阶段特征接口；M7 独立验收留到下一模块。
