# C1b 空间访问契约

`loadData/spatial_protocol.py` 实现 `spatial_stripes_v1`：按事先确定的轴及比例建立三个不重叠矩形，边界两侧各剔除 gap 个像素。默认 axis=0、比例 60/20/20、gap=4。空分区立即报错。划分不使用类别标签优化，plan_hash 核对保存/回读内容。

`fit_train_statistics` 只读取训练矩形的光谱，逐波段保存 min/max 和统计指纹。每个分区独立应用同一统计；不重估目标区统计、不强制截断目标值到 [0,1]。常数波段分母置 1。该统计适用于训练区无标签光谱可见的协议，并不等同于只使用标注像素。

`observe_partition` 只复制当前分区或其内部小块，返回无标签的 PartitionObservation。CNN 边界填充与图候选仅作用于该次局部输入，不读取裁块外像素；坐标来自全景几何尺寸，不依赖影像值或标签。训练监督单独通过 training_labels 读取，拒绝验证/测试 observation。Python API 是访问契约，不是对恶意手工篡改对象的安全沙箱。

`forward_partition` 复用现有 MultiDatasetVCASAHGCN，强制 learned 图模式，不接受全景 Q/A。每次调用仅构造当前输入的区域和图，区域编号局部独立；不能将不同分区图拼起来传回全景网络。验证/测试要求 model.eval()，避免更新 BN/权重；训练区允许 train 模式。GroupNorm 等无运行状态算子仍按当前输入计算。

仅分离监督 mask、在全景拟合归一化、全景构图后再切边，均不属于这个协议。历史训练入口和 C0 的 legacy_random_pixel 保持原语义，本模块没有把它们改名为空间隔离。新的协议 API 和审计入口已完成，严格训练调度接入需独立交付，不能将旧训练 CLI 当作本协议训练。

所有命令在远程容器执行，产物放本项目专用 runs/：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python prepare_spatial_protocol.py --data-root /home/train/hsi-pioneer271828/datasets/current --out-root /home/train/hsi-pioneer271828/runs/<new-run-id> --dataset paviaU --axis 0 --gap 4
```

CLI 保存原始数据指纹、计划、训练区统计、类别覆盖、独立小块检查和失败状态，不训练模型，输出目录不得复用。原始快照以解析后固定路径和哈希追溯。改变目标区光谱不应改变训练统计、训练观测或冻结模型的训练区输出；相关反事实测试已实现。

类别缺失必须如实报告，不能从测试标签反复调整几何边界。默认 PaviaU 例子不是完整九类评估配置。下一步应预先固定多块/交叉验证方案，再接入训练和评价；需对缺失类定义报告规则。gap 仅是几何缓冲，不保证消除所有空间自相关，也不构成真实传感器留出。
