# 完整源M6目标初始化与BN组件工程验收

Issue206 / PR207；开发分支 `codex/g4-interaction-target`，工作树 `F:/E2E_VCA_worktrees/g4-interaction-target`。实际测试/生产组件SHA **`4ede0984ba78c9e31c5a176c4f8e54c74da3e97e`**。本项只完成初始化/总体BN组件工程，不是有界目标训练、完整真实实验或效果结论。

## 计划、用途与复用范围

对应研究主清单P1 §7及P4 §10.3、实施计划§10 M6/§18.3/§19。将已验收的M6消息机制接到完整none源的目标初始化及总体BN，不新增研究机制或替换训练协议。前置为178完整none源、180完整源目标接入及190独立支持/K绑定。204分析工程是同分支依赖中的独立交付，未将其真实query输入用于本模块。

新增 `g4_interaction_target.py`、只读CPU入口 `scripts/audit_g4_interaction_target_initialization.py`、专项测试和项目复用核查JSON。原331已有Python/config文件不改，公共模型/配置/训练/校准接口不改。先回读M6原工程报告e3adf597e0606c7a366db88494856361319e5782及两份计划，复用当前331保留的同一实现；没有复制外部代码、运行上游、添加依赖或将此接入追认为某篇注意力论文复现。来源和边界见 `docs/references/g4-interaction-target-project-reuse.json`。

六条件none/add/gated/cnn_to_graph/graph_to_cnn/global，固定原强度0.1、门宽32、chunk16384；M4/M7/E1关闭。新增消息参数分别为0/33280/58498/29249/29249/58498。add是归一化和投影后的残差消息，global只更换跨支路对齐，图本身仍使用Q；当前门是标量，不能称逐通道门或已校准可靠性。

两种角色为scratch与完整none源全量微调。原共享骨干、fresh目标三头和目标DomainBN所有张量逐项复制，不复制源分类头或源域统计；新增M6参数在目标初始化，**不是M6预训练源迁移**。新构造的额外随机消耗被隔离，与同角色原工厂的调用后RNG相同；不声称原工厂自身完全不消耗全局RNG。原M6强度和投影初始化保留，启用后不保证初始输出恒等。

绑定源、角色、种子、架构、参数数量/可训练权限、M6实际模块树与非张量属性，包括方向、对齐、强度、分块和规范化属性，防止仅核state字典时漏掉属性变化。前向只接无标签观察字段。校准直接调用原总体BN函数并置于原OrderedValidation内，只允许5个目标BN的10个mean/var缓冲改变，源模型、观察、所有参数、仿射、计数器及M6权重不变；原BN回执保留，另增M6绑定回执，不替换原校准合同。权重/sidecar预期哈希和完整绑定均核验后才重载。

## 实际环境、命令与结果

授权LXC `/home/lbx/hsi-pioneer271828`，Python3.10.20、Torch2.7.1+cu128/CUDA12.8，GPU0 RTX4090；测试checkout `checkouts/issue206-interaction-engineering01`，分支同开发分支，干净完整SHA同上。本机仅编辑/Git/SSH/SCP；无本轮rmvision复跑或自动CI声明。

公共环境为 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`。实际执行：

```bash
CUDA_VISIBLE_DEVICES= INTERACTION_TARGET_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_interaction_target.py -v
CUDA_VISIBLE_DEVICES=0 INTERACTION_TARGET_DEVICE=cuda /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_interaction_target.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py --expected-commit 4ede0984ba78c9e31c5a176c4f8e54c74da3e97e
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/audit_g4_interaction_target_initialization.py --help
```

CPU **7/7，39.815s**；GPU **7/7，42.574s**。完整 **620项，619通过+1既有GPU专用CLI跳过，750.025s**。checker **387Python/94JSON**、help通过；GPU专项和全量开始门槛均为24079MiB/0%，符合free≥12288MiB、util≤60%。全量为旧CUDA钩子保留GPU0可见，不能称纯CPU回归。

覆盖全部六条件×两角色的共有状态、fresh头/域和RNG，实际参数数量/唯一优化器登记；none对原前向和orderedBN逐位一致；单向只执行对应消息和对齐，global路径调用及观察不变。五启用条件×两角色各合成3步，参数有有限正梯度和实际变化，不把登记数量当作每个标量均变化。12角色合成校准均保持参数/计数器/非BN状态不变，各3次精确重载，共36次/专项。不是有界12步训练接口或完整真实实验。

独立数值参考为一个gated角色的CPU测试：用FP64均值和中心化平方计算总体矩，再转换到原FP32规范化路径，逐BN统计和三头输出按原rtol1e-5/atol1e-6核对；不是全网FP64或所有角色的独立概率复算。GPU专项中的该参考仍在CPU。FP32以外校准被提前拒绝，不改变原图缓存/dtype协议。

方向/强度/分块/对齐属性变更、参数权限、错误源状态、标签字段、坏文件对、重复输出/错误SHA/非法种子/未知条件及full/query入口均拒绝。真实CLI只做CPU初始化，前向/BN/优化0。原生CUDA独立优化轨迹等价没有在此验收，196/202既有限制仍保留。

## 真实零更新与独立重建

真实三场景×scratch/完整none源FT×六条件，共36初始化，仅model64/head1064；support512/K1的观察和已释放支持/验证上下文用于认证，不参与网络前向。所有不属于M6的状态与199已接受6个基础初态逐张量哈希一致；同一场景/条件的M6新参数在scratch/FT之间一致。74个保护文件不变，**22.054069s**；无真实forward/BN/优化/query，无GPU上下文。此侧会反序列化已释放support/validation覆盖层，不读原GT或反序列化家族标签。

producer `runs/g4-interaction-target-issue206-preflight01/receipt.json`，SHA **`c9738ce61ff3aaee28cc7a5f1ec15acc0db7e7f995738fb68df79688fdcb0cd1`**。

独立干净checkout `checkouts/issue206-interaction-independent01`，同4ede。使用旧none工厂6个基础模型和未修改的原模型构造器、手工六条件配置重建36个完整状态/架构/M6参数数量，不调用新初始化器；75文件不变，**3.692787s**。不重新读取支持/验证覆盖层，不执行网络/BN/优化/query，无GPU上下文。它是状态重建核验，不是36个模型的独立前向或真实校准审计。

独立receipt `runs/g4-interaction-target-issue206-independent01/receipt.json`，SHA **`cf2a219b8301b472e18fdfbf4083a7a2411aa8d923820c1576f6d4571432c406`**。原始日志 `runs/issue206-{cpu,gpu,full,real,independent}01.log`，代码和产物全部隔离，旧184输入与运行代码未改。完整工程receipt为 runs/g4-interaction-target-issue206-engineering01/receipt.json，同字节副本 docs/architecture/g4-interaction-target-engineering-results.json，SHA 4d733c56c814572fb5492492a329f008324a8506558e8b7faace4eae42e947e8。

## 失败、限制、验收和下一依赖

当前版本CPU/GPU实现测试失败0；GitHub TLS/连接器失败保留。不确定创建PR后先搜索和权威REST head查询确认未创建，再恢复注册为207；没有盲目重复POST。一次Windows Git自动整理对象不是训练命令，未删除用户untracked或改运行代码。开发分支仅合入204报告/状态依赖，Python/config与实际4ede测试版一致；最终文档提交不替代tested SHA。

本组件不包含目标有界训练/恢复接口，不代表完整M6源预训练、真实适配收益或速度优势。尚未完成相同容量分支内变换控制，不能用gated比add好就排除容量解释。空间/传感器身份、更多模型/支持种子、M4交互与其余计划范围继续保留。

验收结论为 **既有M6完整none源目标初始化、方向/算子绑定、ordered总体BN及重载组件工程通过**。下一项是既有M6有界12步目标训练/恢复/模块参与证据工程，再按依赖补M7/M8等；真实数值实验仍须先完成184全54接受、独立审计、186真实拟合/逐类/成本复盘、原331全冻结query和204预登记结论。本模块没有接管198资源交接或重启184训练。

主入口只同步报告、证据、current-status和远程投影，候选源码/main不合并，用户untracked保留。按用户最新要求以进度汇报衔接下一工程项，不在单模块交付后等待确认。



证据整理失败保留：外部issue206-receipt01.py将独立回执SHA转录为65位，断言在工程receipt目录创建前拒绝。已从原文件重新计算并核实64位SHA，使用独立assembler02修正；原source、producer和独立receipt、全部测试结果及接受门槛未变。这是证据整理错误，不是训练/数值失败。

