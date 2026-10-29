# E0a：冻结预测的确定性评估基础

本模块最初在容器不可用时实现，当前恢复远程验收。所有代码执行/依赖安装/检查/测试仍只允许在 `ssh rmvision` 训练容器。M8 上游已通过验收，入口继续要求其成功 manifest 与匹配源码；本模块实际结果以交付报告为准。

E0 拆分为 E0a 指标/空间/分支/稀疏图摘要与冻结导出，E0b 再做敏感度、遮挡、区域替换、删边干预、成本 profiling 与成组统计。小目标/细窄结构的预声明定义和干预实验后续单独验收。本轮不修改训练、损失、采样、选模或已有导出器。

## 输入与数值约定

`evaluation.classification_metrics` 接收 hard label ID，必须显式给出正整数 `class_ids`，按该顺序输出混淆矩阵（行真值、列预测）。支持非连续 ID，不把 argmax 的 0-based 下标直接当作标签；0 在真值中仅表示未标注。布尔评估 mask 与标签同形，选中且真值>0 的像素才评分；这些位置的预测/标签必须为声明的有限整数 ID。掩码外和未标注位置的预测不参与评价。

- OA 是有效像素准确率。
- 逐类 precision 在没有该类预测时为 None；recall 在没有该类真值时为 None。该类有真值但全漏检时 F1=0；无真值时 F1=None，即使有误预测也不加入当前支持类宏平均，误预测仍在混淆矩阵和 precision 中记录。
- AA 与 macro-F1 默认只平均真值有支持的类，报告 `macro_class_ids / missing_class_ids`。完整类别覆盖缺失时 `aa_full / macro_f1_full` 为 None，不能将支持类结果解释为完整类别结果。
- Kappa 使用行/列边缘计数；空集合或理论一致率为1时为 None。空集合的 OA/AA/macro-F1 也为 None。JSON 使用 null，不写 NaN/Infinity。

手算 fixture `[[2,2],[1,4]]`：OA=6/9、AA=(1/2+4/5)/2、macro-F1=(4/7+8/11)/2、Kappa=4/13；这些是待测试的期望值，当前没有运行结果。

## 空间和双分支诊断

边界种子只由评估 mask 内四邻域双方有标签且异类的像素对产生，两侧都标记；对角相邻不构成边界。width=1 表示这些 incident 像素，width=2/3 各额外扩张一/两次四邻域；每步都限制在有效标签 mask 内，不跨未标注洞或分区间隙。分别输出每种宽度的边界带和其补集内部指标，空带为 N/A，不把未知位置当背景类。

`branch_diagnostics` 返回 CNN/Graph 正确性的四组合：both_correct、cnn_only_correct、graph_only_correct、both_wrong。每组报告人数、融合正确人数及组内融合正确率；分母为零时 None。融合纠错率的分母明确是“恰有一路正确”的像素；融合伤害率的分母是“两路都正确”的像素；另记录两路都错而融合正确和两路预测不一致人数。分支 oracle 是至少一路正确的比例，只作参照，不宣称模型能实现该上界。

小类只能由调用者传入预声明的 class IDs，例如依据训练区比例预定义；不按本次测试样本数或改善程度自动挑选。连通小目标和细窄结构需要考虑碎片化标注，留给 E0b。

## 稀疏图摘要

输入 `edge_indices[2,E]`、显式 node_count，以及对齐的非负有限 before/可选 after 权重。重复有向边须先 coalesce；不从最大端点猜节点总数，以保留孤立区域。计算正权边出度、非自身邻居隔离数、弱连通分量、行质量误差和权重变化/支持变化；空间复杂度 O(N+E)，没有 N×N 稠密图。这里的连通性是忽略方向后的弱连通性。

没有 after 权重时变化项为 None，不解释为“没有变化”。行质量误差只报告，不强制历史图归一化；存储 nnz 和正权支持分别记录。本模块不据边权尖锐程度认定传播有效，也不执行删边实验。

## 冻结预测导出入口

`export_research_diagnostics.py` 不加载模型或训练数据。NPZ 必须包含 `prediction / cnn_prediction / graph_prediction` 三个 2D 标签 ID 图；其字段名兼容原交互导出器，但不继承原导出器的统计或数据协议。严格空间 checkpoint 的预测必须由其训练保存的 plan/statistics 生成，不能以历史整图 preparation 的输出冒充严格结果。

示例命令仅在容器中运行，路径和 SHA 需替换为实际冻结产物：

```bash
python export_research_diagnostics.py \
  --arrays <冻结预测.npz> --labels <标签.npy> --mask <布尔分区mask.npy> \
  --checkpoint <实际冻结checkpoint.pt> --class-ids 1,2,3 \
  --prediction-commit <预测代码完整SHA> --expected-commit <当前导出代码完整SHA> \
  --dataset <登记名> --protocol-id <固定协议或plan指纹> --partition test \
  --out-dir /home/train/hsi-pioneer271828/runs/<唯一诊断目录>
```

实际 checkpoint 只做 SHA256，不反序列化。导出前后核对预测/标签/mask/checkpoint 输入指纹，记录数据集、协议、分区和两个代码 SHA，输出目录必须在 checkout 外且不存在。结果是 `research_diagnostics.json`，不写回训练配置或自动选模。预测代码/协议来源由预测阶段登记，此导出器不替代其来源核查。CLI 要求当前 SHA 与干净工作树，评估标签只在冻结后的报告阶段访问。

## 容器恢复后的验收顺序

1. 先完成 M8 验收并同步其已接受实现。
2. E0a 独立 checkout 中运行以下计划入口；它要求成功 M8 manifest，并核对模型/训练及 M8 测试源文件与其 tested SHA 一致。

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python scripts/validate_e0a.py \
  --expected-commit <E0a实际完整SHA> \
  --m8-manifest <成功的M8验收manifest.json> \
  --real-data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/e0a-validation-<唯一批次>
```

包含13项针对性测试，预计全量98项（上游85 + E0a13）。入口会跑全量测试、CPU/CUDA smoke 和冻结 fixture CLI；fixture checkpoint 是明确标识的字节文件，不是真实模型，没有分类结论。

指定 `--real-data-root` 时，从成功 M8 manifest 的 default 训练 attempt 读取实际 checkpoint，核查 C0 产物指纹、原始数据、plan/statistics 和保存配置；只在原 test 块推理，冻结三路预测和 eval mask。用独立导出器生成报告，与已验收 M8 的完整混淆矩阵及 OA/AA_supported/Kappa 核对，并用 sklearn 核对支持类 macro-F1。参数/缓冲和输入产物前后须不变，没有新的优化步骤。额外捕获首个 test 块的初始稀疏图单独报告，不能当作整景图摘要。

真实冻结预测的来源代码为本次实际完整 SHA，checkpoint 权重来自 M8 验证选模结果，二者分别记录。输入指纹与保存配置一致才能继续；不重新拟合统计、改测试划分或根据指标选模型。这是评价链路核对，E0b 的模型干预与多次实验配对统计后续单独实施，不把相邻像素当作独立重复。
