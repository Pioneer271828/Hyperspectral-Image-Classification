# G4 M8目标有界训练与严格恢复工程验收

Issue216 / PR217；对应研究主清单§6.2/§7，模块计划§12、§18/§19。实际tested SHA **9b0e76a75f6c8907eda9c3654ccf45b84c3c68de**，分支 `codex/g4-fusion-target-runtime`，工作树 `F:/E2E_VCA_worktrees/g4-fusion-target-runtime`。既有固定融合/辅助监督三条件目标有界训练和恢复工程通过；真实效果尚未测。

## 实现和合同

只新增 `g4_fusion_target_runtime.py`、显式短训CLI和五项测试；复用214接受的初始化/实际alpha和损失绑定、原ordered总体BN，以及原190支持上下文、TargetRuntime概率损失13项、Adam全部参数/weight_decay、scheduler和first-min完整观察fused验证NLL。原331 Python/config没有修改；无新外部源码、依赖或算法，沿用214复用清单。

default(alpha0.8、辅助1/1)、aux_off(alpha0.8、辅助0/0)、alpha035(alpha0.35、辅助1/1)×scratch/完整none源FT，共六角色。唯一预算12，允许新实例2+10恢复，完成恢复0；无full/campaign/query、M4/E1/M6/M7或观测增强入口。源初始化/fresh三头/DomainBN身份和分离的model/head/support种子保持。

实际训练配置与绑定逐步核验；原父类在optimizer.step前检查各分类头和共享分组梯度，保持原训练算术。trace保留13项raw/weighted loss、原梯度范数和学习率，另记三头实际参数变化。aux_off保留非零raw辅助loss，weighted及相应梯度为零；原weight_decay=1e-3仍使辅助头参数变化，不能称其冻结。融合损失与共享学习路径保持，不把组级梯度/变化扩称每个标量均更新。

选中态保存完整原BN回执在声明的M8回执内；训练器规范化metadata后，重载先核其嵌套摘要，再恢复内存原BN记录交给214原验证。绑定实际源/配置/支持上下文/raw文件/校准/三头；不修改共享训练器、原校准门槛或目标标签访问。

## 实际Linux验收

授权LXC `/home/lbx/hsi-pioneer271828`，干净 `checkouts/issue216-fusion-runtime-engineering01`，分支和完整SHA同上；Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090。Windows仅编辑/读文件/Git/SSH/SCP；无rmvision复跑或自动CI声明。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`，下述Python均为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= FUSION_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -p test_g4_fusion_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 FUSION_RUNTIME_DEVICE=cuda python -m unittest discover -s tests -p test_g4_fusion_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu FUSION_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 9b0e76a75f6c8907eda9c3654ccf45b84c3c68de
python scripts/train_g4_fusion_target_short.py --help
```

CPU **5/5，76.129s**；GPU **5/5，89.472s**；完整 **650项，649通过+1既有GPU专用CLI跳过，1093.234s**；checker **402Python/101JSON**、help通过。GPU专项与全量分别检查free≥12288MiB/util≤60%，证据 `runs/issue216-{gpu,full}01-gate.txt`；全回归GPU0保持可见供旧钩子，不称纯CPU。

六角色各完整12、新实例2+10及完成0恢复、18次选中态精确重载/专项通过。CPU状态、optimizer、scheduler、history、RNG、三头概率和trace逐位一致；default两角色对190原none轨迹精确。GPU default完整控制事前使用测试专用FixedGraphReference；其余功能更新/恢复/ordered重载走原GPU路径，不称原生CUDA独立优化轨迹等价。

每步验证三条件合同和13loss项；aux_off raw非零、weighted0、两辅助头梯度0但实际参数改变，fused头正梯度通过。错误alpha/实际配置/源/上下文/预算、query字段、选中raw身份和嵌套BN回执拒绝通过。向零辅助梯度注入NaN时，父类在参数更新前拒绝，所有参数不变、optimizer state为空；失败前向仍可能更新训练BN缓冲，不扩大为全部状态不变。实际CLI验证2+10+0、support2048/K5及禁止full/query/其他机制/额外预算。

## 真实输入零更新和独立配置重建

三景×两角色×三条件 **18配置与初始化**，model64/head1064、support512/K1，逐项绑定214接受初态。原optimizer和验证快照方法身份不变，91保护文件不变，**72.936776s**；读取已释放支持/验证覆盖层，不反序列化家族标签/原GT。禁止Module调用与Adam构造，实际NN/BN/优化/query0，无GPU上下文。

独立干净 `checkouts/issue216-fusion-runtime-independent01`，分支 `codex/issue216-fusion-runtime-independent01`，同tested SHA。不调用216 settings/runtime/factory，使用旧190设置、手工typed alpha/辅助配置及原策略字段重建18配置/初始化绑定，92保护文件不变，**18.704205s**；不重新构造模型。该证据限配置组合与214已接受状态身份，不冒充独立概率或完整训练审计。

工程原件 `runs/g4-fusion-runtime-issue216-engineering01/receipt.json` SHA **3641bac68fd653de19e9cb34fcd6ba9bc11ae05124c1783a07f4877fb1694d26**，同字节仓库副本 `docs/architecture/g4-fusion-runtime-engineering-results.json`。producer `runs/g4-fusion-runtime-issue216-preflight01/receipt.json` SHA **c6cb70a44f39ecd9b0d40c27487b88c8da84def9ac7ca4e6e4d11dab410849e3**；独立 `runs/g4-fusion-runtime-issue216-independent01/receipt.json` SHA **95db98ecad6dc3fe80869ff0762fc1b58922762810de465e9313985140a33258**。日志 `runs/issue216-{cpu,gpu,full,checker,help}01.log`，外部驱动及全部产物在checkout外。

## 失败、限制和下一依赖

本实现版本CPU/GPU/完整验收未出现测试失败。214文档依赖与216源码分叉，首次FF合并被Git拒绝且未修改工作树；随后仅metadata合并910b71c6ff4c9e08cff4ffac55f60e9455653d91，Python/config与tested9b逐项无差异。TLS/连接器失败保留，未强推或重复创建Issue/PR。

这是有界合成工程验收及真实输入零更新身份检查，不是六个真实12步/完整训练，不给出精度、充分收敛或速度结论。新门控、独立单分支、M4×门控、共享语义头映射、支持/K/空间/传感器与全部计划范围仍待完成。184全54接受→独立331审计→186真实拟合/逐类/成本复盘→原331全冻结ordered query→204预登记结论后才登记新的真实实验；不接管198交接。

下一项 **Issue218 / PR219** E0-R同区域支持冻结特征相关/PCA/有效维度诊断已独立登记并验证，当前不执行真实184特征提取。主入口仅同步报告/证据/current-status及远程生成投影，main不合并；按用户要求简报进度后连续推进。
