# E0b消息路径干预：仅原验证分区

关联Issue #29。`scripts/intervene_screening_messages.py`只使用原验证NLL选定的三个checkpoint，预先固定四条件：`original`保留原消息；`cnn_to_graph`仅保留CNN→Graph；`graph_to_cnn`仅保留Graph→CNN；`none`将两方向残差强度置零。B0无交互，为四条件不变的负对照。所有推理在`ssh rmvision`容器GPU上运行。

这是冻结已训练权重后的路径干预：临时将指定`SelectiveMessage.strength`置零，保持模块/权重、Q/A、门控、同时更新和计算路径；正常或异常退出时恢复属性。零消息仍会计算投影和门控，因此不用于加速或删模块性能宣称。`original`表示各checkpoint原始状态，B0的original并无双向消息。

```bash
cd /workspace/hsi-message-intervention
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -u scripts/intervene_screening_messages.py \
  --expected-commit <实际完整SHA> \
  --training-commit b7d2b30015b27c4f9a80be4baeeefa051185bda2 \
  --campaign-root /home/train/hsi-pioneer271828/runs/m6-screening-20261030 \
  --accepted-diagnostics /home/train/hsi-pioneer271828/runs/m6-screening-branches-20261030-03 \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/message-intervention-<新批次>
```

入口固定分区为val，没有test选项；原始文件读取/指纹可能覆盖整个场景，但模型观察与计分仅用保存plan的验证块，不拟合统计或重新选择checkpoint。核查训练代码与权重身份、#27诊断入口不变、原验证三路概率/混淆矩阵复现，以及每个条件相同像素ID/顺序。CUDA稀疏舍入沿用`atol=2e-5, rtol=1e-6`，等价路径强制预测ID相同，原融合NLL与保存记录差不得大于`1e-6`。

同一阶段的两方向读取交互前快照，因此保留相同入向消息的一路，其读出应不变。入口检查CNN(original)=CNN(graph_to_cnn)、Graph(original)=Graph(cnn_to_graph)，以及CNN(none)=CNN(cnn_to_graph)、Graph(none)=Graph(graph_to_cnn)。B0检查四条件三路全一致。

输出包括manifest、`message-intervention.json`和各组四条件的三路逐类/错误交集/门控报告、像素概率和块级记录。`paired_vs_original`中的corrected/harmed分别表示同一真值像素从原条件错→对/对→错；这与双分支诊断中“两个辅助头都对但融合错”的harm分母不同。全部产物写到新外部目录，已存在目录拒绝覆盖。

干预结果说明当前权重对消息路径的依赖，不等同于从头训练的单向网络，也不能把冻结关掉消息的下降解释为该路径在所有训练方案中必需。单种子和空间相关像素不支持显著性结论。按完整验证指标和逐类变化解释，不只挑第6类最好结果；本轮不更换默认方向/主头/alpha，不做坐标消融或长训练，完成后停下评审。
