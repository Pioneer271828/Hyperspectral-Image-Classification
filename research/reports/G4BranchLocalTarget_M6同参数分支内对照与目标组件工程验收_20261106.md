# G4 M6同参数分支内对照与完整源目标组件工程验收

Issue226 / PR227；对应主清单§7.2、实施计划§10/§18/§19及原项目纪要§8.2的同参数、同深度无中层跨支路消息对照。实际tested SHA **6e78153c50467d67f6de46f8f2d1d564b6a915da**，开发分支 `codex/g4-branch-local-target`，工作树 `F:/E2E_VCA_worktrees/branch-local-target`。本次为初始化/算子/BN/重载组件工程，不是完整容量实验或效果结论。

## 用途、实现及复用范围

新增 `g4_branch_local_target.py`、`scripts/audit_g4_branch_local_initialization.py`、七项测试及内部复用清单。回读两主计划、用户原项目纪要§8.2、技术方案§5.11/§5.12、206组件及222/224已验报告；最小复用原331 `SelectiveMessage`、206 none/gated工厂及222版本标记/loader/BN包装模式。无新外部机制、源码或依赖，不提出新外部许可授权；原公共模型、配置、loader、训练器与331文件不改。

| 条件 | 中层源特征 | 变换参数 | 必需额外状态 |
| --- | --- | ---: | ---: |
| none | 无中层变换 | 0 | 0 |
| gated | 原区域对齐后的另一支路 | 58498 | 0 |
| branch_local | 各支路自身特征 | 58498 | 8字节version buffer |

三条件×scratch/完整none源FT六角色。local复用两套原LayerNorm、projection和标量gate，固定强度0.1、隐藏32、chunk16384；每套原变换把自己的支路特征同时作为target/source，两条支路仍同时读取各自前态。逐个参数与匹配gated初态一致，只多版本buffer；fresh目标三头、DomainBN、源、参数权限和调用后RNG也匹配。不是M6源预训练。历史参数槽位名graph_to_cnn/cnn_to_graph保留以核同一schema，local诊断明确叫pixel_local/region_local。

local算子不读取Q值、不调用对齐pool/broadcast，也不读取另一支路中间特征；**整个网络仍共享上游adapter、原Q/图构造和末端融合**，不能称全网统计独立。相同参数和变换深度也不等于相同计算预算或速度，local少了跨路对齐操作。两套LayerNorm初始相同使差值gate输入初始为零，不保证所有权重元素第一步都有非零梯度。

绑定实际类/forward与forward_aligned/原SelectiveMessage方法、配置/strength/chunk、参数权限/schema、架构和源。旧export kwargs仍表示gated形状，custom binding另声明local语义；必需version使旧通用gated loader拒绝local状态，防止静默执行错误跨路语义。新成对哈希loader安装算子后严格载入并验证原ordered FP32总体BN回执；组件不开放有界12/full/campaign/query。

## 实际验收环境、命令和结果

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue226-branch-local-engineering01`，实际分支 `codex/issue226-final01`，干净完整SHA同上。Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090；Windows仅编辑/读文件/Git/SSH/SCP。未声称rmvision复跑或CI自动执行。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`；Python为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= BRANCH_LOCAL_TARGET_DEVICE=cpu python -m unittest discover -s tests -p test_g4_branch_local_target.py -v
CUDA_VISIBLE_DEVICES=0 BRANCH_LOCAL_TARGET_DEVICE=cuda python -m unittest discover -s tests -p test_g4_branch_local_target.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu FUSION_RUNTIME_DEVICE=cpu NORMALIZATION_TARGET_DEVICE=cpu BRANCH_LOCAL_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 6e78153c50467d67f6de46f8f2d1d564b6a915da
python scripts/audit_g4_branch_local_initialization.py --help
```

CPU **7/7，34.040s**，GPU **7/7，37.633s**；完整 **665项664通过+1既有GPU专用跳过，1148.738s**；checker **408Python110JSON**、help通过。GPU专项/完整回归分别独立核GPU0 free≥12288MiB/util≤60%，`runs/issue226-{gpu,full}01-gate.txt`保留；全量GPU可见供旧钩子，不称纯CPU。

独立CPU FP64手工LayerNorm/projection/GELU/sigmoid/gate公式及输入/全部参数梯度与原变换对应；固定中间输入时跨支路Jacobian未连接（零）、改变另一支路/Q不影响本路、无对齐调用、空输入、强度0、分块/不分块通过。none/gated对206两角色前向及总体BN状态精确一致。六角色各合成3步原三头loss更新，图分支、两变换槽位和三头有限正组梯度及实际变化通过；不声称所有标量每步均改变。

原BN仅允许5层10个总体mean/var缓冲变化，参数/计数器/源/观察保护、六角色各3次精确重载共18/专项通过；local旧通用loader拒绝version。另用一个local角色CPU FP64总体矩参考，再转原FP32规范化，沿用rtol1e-5/atol1e-6，不称全网FP64或所有角色独立概率复算。算子版本/非张量属性/权限/标签、错误接口/外部cross消息、源/文件对、seed/条件/输出/full/query拒绝通过。GPU专项验功能与ordered重载，不接受独立原生CUDA优化轨迹等价。

## 真实输入初始化与独立重建

三景×两角色×三条件 **18初始化**，model64/head1064、support512/K1，继承206 none/gated的12基础身份；local去版本buffer后与匹配gated所有张量、架构和权限一致。78保护文件不变，**21.822792s**。只读取已释放支持/验证覆盖认证，未反序列化家族标签或原GT。Module调用/Adam构造被禁止，真实NN/BN/优化/query0，无GPU上下文。

独立干净checkout `checkouts/issue226-branch-local-independent01`，同tested SHA，分支 `codex/issue226-independent01`。仅用旧206工厂12次、手工version状态及配置/根类型签名，重建18状态/schema/架构/权限；不调用226新factory/operator，未读支持/验证覆盖层。79文件不变，**3.936664s**。这是状态/配置独立重建，不冒充真实前向、BN或训练/概率审计。

工程原件 `runs/g4-branch-local-target-issue226-engineering01/receipt.json`，同字节仓库副本 `docs/architecture/g4-branch-local-target-engineering-results.json`，SHA **c77bfee7449cc81d93dd31e12791ccbae7d5878ea0b55e4ebd5d99373f7d51fc**；producer `runs/g4-branch-local-target-issue226-preflight01/receipt.json` SHA **9909560896151f513bc228b90d3fd8573f972228ece1959a44fab1815c0649db**；独立 `runs/g4-branch-local-target-issue226-independent01/receipt.json` SHA **0a4479a4c2d2643b652f14c58be4ecf2cb72aec371d5b6f40f04e758064be097**。日志 `runs/issue226-{cpu,gpu,full,checker,help}01.log`，driver与产物均在checkout外。

## 保留限制及下一项

本实现版本测试失败0。连接器/TLS注册失败及权威GET后再创建记录保留；传输遵循SCP实际exit0、bundle verify、干净checkout完整SHA后执行。原331和184冻结输入/预算不改，未接管198资源续接，main未合并。

工程通过不代表M6交换优于容量、精度或提速。相同参数不等计算预算；支持/K/空间/传感器、M4组合、计算匹配及完整真实效果仍必测。CLI合成另验证support2048/K5及其他modelseed初始化，不扩称真实多支持/种子复验。

下一 **Issue228 / PR229** 接入有界12、新2+10/完成0恢复和明确pixel_slot/region_slot的梯度/变化证据；真实配置证明须先消费本项完整工程receipt。所有新真实数值实验仍等待184全54接受→独立331审计→186真实复盘→原331全冻结ordered query→204预登记结论。主入口仅接纳报告/证据/参考/current-status和远程投影，候选源码保留独立PR，按用户要求简报后继续搭建。

