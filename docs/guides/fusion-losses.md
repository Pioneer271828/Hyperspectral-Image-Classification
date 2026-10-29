# M8：固定融合、分类与损失契约

默认仍为 `F=alph*CNN+(1-alph)*Graph`、alph=0.8；三头概率输出顺序为 fused/graph/CNN。固定融合提取到无参数 `model/branch_fusion.py`，权重键与模型参数量保留。alph 是有限的 [0,1] 常数，通过 C0 `model.alph` 和旧训练 CLI `--alph` 显式配置，已有侧车自动保存和恢复，缺字段时按 0.8 构造。alpha=0/1 只用于推理端点检查，单分支训练结果另做实验。

## 分类接口

- `backbone.fuse_features(cnn_features, graph_features)` 使用同一个固定融合函数。
- `model.fuse_and_predict(CNN[N,64], Graph[N,64], dataset_name, return_logits=False)` 独立测试融合和分类，无需调用构图器或骨干。
- `model.predict_features(dataset_name, fused, graph_features, cnn_features, return_logits=False)` 消费已经融合的特征。
- 多/单数据集 forward 新增 `return_logits=True` 可选项，返回同样顺序的三路原始 logits；`return_graph_info=True` 时仍附加原 graph_info。默认继续返回概率，不同时叠加新的 tuple 层或字典。

logits 是显式调用接口，没有改变当前训练损失或配置默认输出。原概率调用按每头 linear→softmax 的顺序计算，与旧版对比；权重不因输出模式改变。门控融合、共享语义头和 logits CE 都未在本模块引入，需要对应实验或类别映射后再实施。

## 损失与记录

`model/task_losses.py::probability_loss_components(outputs, onehot, mask, ...)` 使用未修改的 `utils.compute_loss`：概率 clamp 到 [1e-8,1]、原类别平衡和监督 mask 归一化，保持原三头及图正则加法顺序。必须输入概率，不能将 logits 直接传给此函数。

返回 13 项可微标量：`fused_supervised / graph_supervised / cnn_supervised`、`graph_compactness / graph_entropy / graph_balance`；六项对应的 `weighted_*`；以及 `total`。融合监督权重固定为 1，辅助和图正则权重沿用现有配置。

`train_vca_multidataset.supervised_loss` 默认仍返回 total Tensor，新增 `return_components=True` 返回分项。`detached_loss_values` 只用于记录，不用于反向；每次打包为一次 CPU 传输。两个训练入口均在 metrics 的 `loss_component_history` 按 epoch 记录每步平均值，包含未验证 epoch，不改变采样、优化次数或选模依据。

监督 mask 只影响监督项。零辅助权重移除相应辅助监督梯度，共享骨干仍受融合监督和图正则训练；图正则描述当前观测块，可以在无标签像素上产生梯度。关闭辅助权重不意味着冻结分支或移除优化器 weight_decay。空监督/缺失类保持原有限行为；图正则即使在空监督下也可非零。

日志中的原始辅助损失可非零，而其 weighted 值为零。每 epoch 的 total 与六项 weighted 之和允许浮点加法/平均的微小舍入误差，不能用 detach 日志重构训练计算图。没有额外重建或一致性损失。

## 远程验收

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python scripts/validate_m8.py \
  --expected-commit <完整SHA> \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/m8-<唯一批次>
```

所有运行通过 `ssh rmvision`。入口核对容器、SHA、干净工作树和源码指纹，运行完整单元测试、CPU/CUDA smoke，再进行 PaviaU 三组 GPU 两步闭环：默认、关闭辅助监督、alph=.35。K64/embed32、显式 group_norm、原三头/图正则、C1 训练块统计和固定空间划分，验证选模/恢复、分项和 logits/probability 对照、成功后 skipped。产物在 checkout 外，失败保留。

这些配置是接口与记录检查，不能比较精度或收敛。可选门控融合与 M4 分解四因素研究尚未启动，后续 E0 再完善逐类和分支机制诊断。
