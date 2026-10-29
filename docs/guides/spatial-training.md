# C1c：固定多块训练适配器

入口为 `train_spatial.py`，配置示例 `configs/experiments/spatial_smoke.json`。所有执行在 ssh rmvision 的容器，数据根目录显式指定，输出在本项目专用 runs/。它复用 C0 的身份/attempt/日志/失败/完整性核对机制以及现有 VCA 双分支模型，不修改历史 train_vca_multidataset.py 的默认行为。

## 冻结几何与信息边界

`spatial_tiles_v1` 先按 tile_size 建网格，再在每格四周各剔除 gap 个像素；边缘不足 4×4 的内部块丢弃。仅使用几何尺寸及 split_seed 打乱块编号，验证和测试各取 floor(n/5)（至少一块），其余用于训练。保存全部块、分区和 plan_hash；计划不根据标签选择，不自动重抽。

训练统计是所有训练块内部光谱的逐波段 min/max。每块单独归一化、CNN 前向和 learned 图构建；图节点编号是局部的，不跨块共享 Q/A。相邻块之间的缓冲为两侧 gap 之和。几何隔离并不保证消除所有空间自相关，块内上下文也不等同于全景推理。

只有训练块监督参与优化；仅训练块中有前景的块进入训练采样，按 seed 有放回抽取。训练区缺类立即拒绝运行，不调整划分。标签 0 不进入监督。原始文件校验和登记可以读取标签元数据，但优化和选模不使用测试标签；测试语义评价在验证选模及严格恢复之后执行。模型观测没有标签字段。

验证/测试采用 eval，冻结 BN 运行状态；小块若只生成一个图节点，BN 训练会提前报错，需明确增大块或选择 group_norm，不自动替换归一化。图尺度沿用原 dataset 的默认值（PaviaU=50，其余=100），在源码版本中固定；更改尺度时应以独立配置扩展交付。

## 配置与预算

配置只接受 schema/name/dataset/device/model/interaction/loss/training/tile_size/gap/split_seed。模型、交互、损失和训练参数沿用 C0 校验；未知字段拒绝。当前严格入口只支持 learned 图、单数据集、明确的正 steps_per_epoch 和 validation-selected primary；未实现的增强不会通过配置静默生效。

保留 epochs 的包含端点含义，实际更新为 `(epochs+1)*steps_per_epoch`。训练复用原三头监督和图正则，训练块采样/上下文和预处理与历史全景流程不同，不能只按 epoch 数宣称预算一致。配置示例只有两次更新，用于工程检查，不是论文训练配置。

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python train_spatial.py --config configs/experiments/spatial_smoke.json --data-root /home/train/hsi-pioneer271828/datasets/current --out-root /home/train/hsi-pioneer271828/runs/<new-run-id> --dry-run
```

dry-run 不读取数据或构造模型。正式运行去掉 dry-run；相同运行加 resume 会重新核对原始数据、计划、统计及产物后跳过成功任务。失败重试另建 attempt，不恢复优化器中间轨迹。out-root 不得位于代码目录。

## 检查点与评价

每次验证按 primary_checkpoint 选择：macro_oa 在单数据集这里表示验证 OA；val_loss 表示验证前景逐像素 NLL。只由验证值控制选择和学习率调度；测试只在选择完成后评价一次。保存原 `.pt/.pt.json` 格式，并配套 protocol.json/statistics.json。`restore_spatial_checkpoint` 验证三者身份，拒绝混用其他空间计划或统计。

指标包括流式混淆矩阵、前景 OA、逐像素 NLL、Kappa、每类支持数、缺失类别和 `aa_supported`。最后一个指标只平均有支持的类别，不冒充完整类别 AA。缺类不能通过重新抽取同一个测试划分来修补；正式研究应预先确定多折方案并保留全部结果。

当前 PaviaU 示例训练/验证有全部九类，测试缺 3、7 类。仅两步检查不提供可用精度结论。下一步可回到 M2 VCA 封装与缓存回归；正式空间交叉验证和预算实验单独登记，不能据本次测试结果挑有利折。
