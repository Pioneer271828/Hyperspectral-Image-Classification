# 完整初筛的冻结三路逐类诊断

关联Issue #27。`scripts/diagnose_screening_branches.py`读取已完成的B0/Add/Gated campaign，只恢复各组按验证NLL选出的`model.pt`，分别评估原验证和测试块。它不训练、不更新归一化统计、不改alpha或选择其他checkpoint。命令只在`ssh rmvision`训练容器运行。

```bash
cd /workspace/hsi-screening-branches
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -u scripts/diagnose_screening_branches.py \
  --expected-commit <本诊断入口的完整commit> \
  --training-commit b7d2b30015b27c4f9a80be4baeeefa051185bda2 \
  --campaign-root /home/train/hsi-pioneer271828/runs/m6-screening-20261030 \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/m6-screening-branches-<新批次> \
  --focus-class 6 --confused-as 2
```

运行前校验当前完整SHA、干净工作树与CUDA；训练campaign必须完成且无锁。诊断代码允许新增，但模型、数据处理、原训练、既有评价和实验配置相对训练commit必须相同。核查训练manifest内的所有产物hash、共有初始化、数据身份、模型侧边配置与保存协议；结束后再次核对原产物、原始数据、模型参数/缓冲和诊断代码不变。

GPU稀疏归约的浮点次序会导致普通重复推理也有微小概率差异。首块核查诊断开/关时采用`rtol=1e-6, atol=1e-6`，同时强制三路预测类别完全相同；全分区融合混淆矩阵须与原训练结果逐项一致，NLL差须不超过`1e-6`。这不改变选模或评分规则，原始差异写入报告。

输出在checkout外的新目录，包括带命令/身份/失败记录的`manifest.json`、汇总`branch-audit.json`、每组每分区的`*-report.json`、`*-tiles.json`及`*-predictions.npz`。NPZ只含计分像素：truth、global_ids、tile_ids、probabilities，概率轴固定为fused/graph/cnn；还有可用的像素门控值。cnn_to_graph原本是区域门，使用原Q广播到像素，均值按有标签像素加权；它不是区域均值或校准可靠度。Add的门值为常数1，不能当作学习出的门。

报告复用E0a的显式类别ID、混淆矩阵及支持类宏指标，另对每类报告双分支正确性的四组合、融合纠错/伤害，以及指定真值类误入指定预测类的27种三路状态联合计数。缺少真值支持的类使用N/A。focus/confused类是事后解释对象，不能据此自动选超参数。

融合是`0.8*CNN特征 + 0.2*Graph特征`后使用独立任务头；CNN与Graph各有自己的辅助头。它不是概率加权投票，因此即便两个辅助头都正确，融合头仍可能错误；两个辅助头都错误，融合头也可能正确。“branch oracle”只表示至少一个辅助头正确的比例，对这种独立融合头不构成严格上界。

这些是同一次前向的描述性读出：Add/Gated的两条分支已经受中层交互影响，不能称为独立CNN-only或Graph-only模型，不能仅凭“某路错、融合错”认定该路造成错误。判断因果还需要另行设计的受控消息/边干预或重新训练消融。当前任务不完成整个E0b，也不实施MiTA、缩宽或多种子训练。
