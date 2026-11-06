# G4 M7完整源目标二因素初始化与BN工程验收

Issue210 / PR211；研究主清单P1§7既有交互/图更新二因素诊断，模块计划§10/§11及§18/§19工程验收。实际tested SHA **0827fb3c64b3e7d473c4443183288c25a3d2fdea**，开发分支 `codex/g4-edge-target`，工作树 `F:/E2E_VCA_worktrees/g4-edge-target`。本项是既有算子目标接入组件，不是完整预算实验或真实精度收益。

## 实现及对应计划范围

仅新增 `g4_edge_target.py`、`scripts/audit_g4_edge_target_initialization.py`、专项测试和 `docs/references/g4-edge-target-project-reuse.json`。直接复用原331的GraphEdgeUpdater、骨干调度、InteractionConfig、完整none源目标初始化和ordered总体BN；原模型/配置/训练代码均未改。复用206的目标绑定设计，新增实际边算子mode/mix/embed_dim、模块树、方向/对齐/强度及全部可训练参数身份保护。没有引入新外部算法、源码、依赖或许可声明；现有源码核查和适用范围记录于复用清单。

六个显式条件组成两套干净的2×2核心设计：

| 条件 | 直接交换 | 候选边更新 | 新增参数 |
| --- | --- | --- | --- |
| none | none | none | 0 |
| gated | gated | none | 58498 |
| graph | none | graph | 8448 |
| gated_graph | gated | graph | 66946 |
| joint | none | joint | 16896 |
| gated_joint | gated | joint | 75394 |

原M6强度0.1、隐藏宽32，M7维数32、mix0.5均固定。调度保留stage1→使用交互前快照同时交换→基于交换后区域及CNN聚合更新边→stage2。none/joint已有CNN进入关系层，不叫完全无交互。GraphEdgeUpdater仅在原正候选支持上归一化/重加权，孤立行补自身；不新增远程边或硬剪枝。static图本身非行归一化时，原M7内的归一化仍是机制的一部分，后续必须保留仅归一化对照。

scratch和完整none源FT两角色。所有原共享/fresh目标三头/目标DomainBN状态逐项复制，新M6/M7张量在目标按原构造器初始化；新构造器额外随机消耗隔离。它们不是M6/M7源预训练权重，也不保证启用初始输出恒等。M4/E1关闭，图Q保持原输入，不改损失/优化预算/选模/标签合同；组件本身没有训练器/full/campaign/query入口。

前向只接受无标签观察字段。ordered总体BN调用原校准器，参数、计数器、非目标BN状态、原模型与观察保护不变；原完整BN回执保留，另增M7身份回执。权重与sidecar先后核哈希，再验证实际源/架构/模块/观察/raw源/校准绑定。FP32以外校准拒绝。

## 实际Linux环境与验收

授权LXC `/home/lbx/hsi-pioneer271828`，Python3.10.20，Torch2.7.1+cu128/CUDA12.8，GPU0 RTX4090。实际干净checkout `checkouts/issue210-edge-engineering03`，分支 `codex/issue210-edge-final03`，完整SHA同上。Windows仅编辑/Git/SSH/SCP；本轮没有rmvision复跑或自动CI声明。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`，命令：

```bash
CUDA_VISIBLE_DEVICES= EDGE_TARGET_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_edge_target.py -v
CUDA_VISIBLE_DEVICES=0 EDGE_TARGET_DEVICE=cuda /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_edge_target.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py --expected-commit 0827fb3c64b3e7d473c4443183288c25a3d2fdea
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/audit_g4_edge_target_initialization.py --help
```

CPU **8/8，37.979s**；GPU **8/8，40.475s**；完整回归633项，632通过+1既有GPU专用CLI跳过，873.291s。checker **393Python/97JSON**、help通过；GPU专项和全量启动门槛见 `runs/issue210-{gpu,full}03-gate.txt`，free≥12288MiB、util≤60%。全量保留GPU0可见供旧CUDA钩子，不称纯CPU回归。

12角色的原共有状态/fresh头域/RNG、唯一全参数登记和新增参数数量通过。none相对原none工厂的前向/BN两角色精确；none/gated的edge-off相对206另做scratch状态/输出精确控制。五启用条件×两角色各实际合成3步，消息/边组件分别有有限正总梯度和实际参数变化；不把总量大于零当每个标量都更新。原A=I退化例精确恒等且边参数梯度为零，作为保留控制；多候选合成图用于参与、总体BN和重载测试。

所有12角色多候选图BN只改5个目标BN的10个mean/var缓冲，参数/计数器/非BN状态与原观察不变，各3次精确重载，共36次/专项。属性mode/mix/embed维度、方向/强度/分块/对齐、参数权限、错误源状态、标签字段、文件对/输出/种子/CLI/full/query保护通过。实际CLI仅CPU零前向初始化，不作真实网络实验。

独立数值参考分两项：4节点小张量的FP64密集边公式、区域/CNN及参数梯度，包含支持集合、逐行和与孤立点；另为一个gated_joint角色的CPU FP64总体矩参考，再转原FP32规范化与三头路径，使用原rtol1e-5/atol1e-6。不是全网FP64、全部角色独立概率或原生CUDA独立优化轨迹等价。graph对额外CNN输入不敏感，joint有实际依赖；调度实测边更新接收交换后的区域/聚合CNN，没有把计划顺序只写在配置里。

## 真实零更新与独立重建

真实三景×两角色×六条件共36初始化，model64/head1064，support512/K1上下文仅用于认证。非新增状态与206接受none基础初态一致；去除M7边张量后与206对应none/gated完整初态一致，同场景条件新参数在scratch/FT间相同。79保护文件不变，**23.103573s**，真实forward/BN/优化/query0，无GPU上下文；读取已释放支持/验证覆盖层，不反序列化家族标签或原GT。

独立干净checkout `checkouts/issue210-edge-independent01`，分支 `codex/issue210-edge-independent01`，同tested SHA。使用旧none工厂6个基础模型和原构造器、手工六条件重建36状态/架构/新增参数数量，不调用210新工厂。80文件不变，**3.760413s**；不读支持/验证覆盖层，不执行模型/BN/优化/query，无GPU上下文。它是状态重建，不冒充独立全事件概率/BN审计。

原始producer `runs/g4-edge-target-issue210-preflight01/receipt.json`、独立 `runs/g4-edge-target-issue210-independent01/receipt.json`、日志 `runs/issue210-{cpu,gpu,full}03.log`。工程汇总及SHA为runs/g4-edge-target-issue210-engineering01/receipt.json（SHA a2b4d0fa6a3c89ab431aa4e1fdd8cbf02d4ba1799cd8d65a25f5d829fb85f576），同字节副本 docs/architecture/g4-edge-target-engineering-results.json；producer SHA e782397fc55b7e74db641dcdc26a4b0ee8d97a7e89adfd48c4f456b33c12adbd，独立 SHA 0ad5b41eb9af74d4ab83d8e9b33fe4630cf0d2e3deaf44a1835ab4c8c1dc7c90，全部在checkout外；仓库仅保留小型工程回执副本。

## 失败、限制与下一依赖

98d8c8c50c3a4eb0ecaa2bd0517de4a591d01948首轮CPU **7通过/1失败，36.439s**：继承的A=I合成上下文每行只有一个候选，边softmax常数，正确零梯度与测试要求正梯度冲突。保留明确的identity恒等/零梯度控制，只为合成参与测试提供两个合法候选；31a0bdd5828d21fd95dff5b1cb867fe314f0fac6 CPU8/8 37.026s、GPU8/8 41.738s通过。最终082进一步把BN/精确重载也改为多候选合成观察；不改变生产算子、真实图/上下文或阈值，重新执行最终完整验收。旧clone/日志均保留。

GitHub TLS/连接器失败保留；创建后不确定时先权威GET确认，恢复后才继续，没有重复Issue/PR。开发分支只合208报告/状态依赖，原Python/config与tested082一致；最终文档提交不替代实际测试SHA。

本项不包含M7有界12训练/恢复接口、M6/M7源预训练、完整配对效果、同容量替代、仅归一化、支持/K/种子/空间/传感器/M4组合。原生CUDA独立优化轨迹等价未接受。工程通过不说明图更合理、精度提高或速度更快；这些仍须真实对照和逐类/成本证据。

验收结论为既有M7完整none源目标二因素初始化/实际算子绑定/ordered总体BN/重载组件工程通过。下一项Issue212 / PR213的既有M7有界12步训练/恢复及消息/边分项参与工程已独立准备，不发起真实数值实验。184完整54接受→独立331审计→186真实拟合/逐类/成本复盘→原331全冻结ordered query→204预登记结论仍优先且不得跳过；本工作未接管198资源交接或改冻结184输入。主入口仅同步报告/证据/current-status/远程投影，main未合并；按用户要求简短进度后连续下一项。

证据整理失败保留：外部assembler01残留旧runtime目录前缀，在输出创建前FileNotFoundError拒绝；assembler02仅更正为已有edge同级目录并重新核保护文件，未更改原receipt/源码/测试/阈值。首版日志及失败SCP读取记录保留。
