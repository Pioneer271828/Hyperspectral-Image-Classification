# G4 M6目标有界训练与恢复工程验收

Issue208 / PR209。用途对应研究主清单P1§7既有交互诊断、P4§10.3合法目标适配，以及模块计划§10 M6、§18/§19验收与证据。实际tested SHA **1a6a7a0613f3833108193ecc2085583d628aea05**；开发分支 `codex/g4-interaction-target-runtime`，独立工作树 `F:/E2E_VCA_worktrees/g4-interaction-target-runtime`。本项声明范围为有界工程验收，真实精度与完整实验仍待原184队列及后续登记。

## 实现与合同

仅新增 `g4_interaction_target_runtime.py`、`scripts/train_g4_interaction_target_short.py` 和专项测试三个文件。原331模型、配置、训练、损失、采样、优化器/调度器与选择合同没有修改。复用206的M6目标初始化/实际算子绑定和ordered总体BN，190的已释放支持/验证覆盖层及上下文保护，原三头selected-row损失、Adam/调度器和first-min full-observation fused NLL。

固定六条件none/add/gated/cnn_to_graph/graph_to_cnn/global，scratch及完整none源全量微调两角色。新增M6参数在目标初始化，不是M6源预训练权重；强度0.1、隐藏宽32、同时更新语义及global仅改变跨路消息均保持。M7/M4/E1/M1/light关闭，工程预算12，CLI只允许先2、续10、完成恢复0；没有full/campaign/query入口。

每步记录各启用方向的参数数、梯度范数和实际变化元素。启用消息在原optimizer.step前安装只读检查钩子，拒绝缺失/非有限梯度并在finally移除；不改原更新函数。记录可以接受有限零梯度，不用工程记录强迫真实数据每步正梯度；合成验收才断言有限正值与实际更新。关闭方向不构造无用模块。

选中模型保留源、上下文、角色/条件、架构、实际非张量属性、参数权限、raw验证源与校准身份。共享训练器会重建标准元数据，因此在唯一声明的M6校准回执内保存原BN完整回执；重载时先核其原始哈希，再仅在内存恢复给未修改的原验证器。没有重算回执或放宽阈值，原rtol1e-5/atol1e-6、分类0不变。

## 实际环境与命令

授权LXC `/home/lbx/hsi-pioneer271828`，Python3.10.20/Torch2.7.1+cu128/CUDA12.8，GPU0 RTX4090。实际干净checkout `checkouts/issue208-runtime-engineering03`，分支 `codex/issue208-runtime-final03`，完整SHA同上。Windows只编辑/Git/SSH/SCP；本轮未复跑rmvision、未声称自动CI。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`：

```bash
CUDA_VISIBLE_DEVICES= INTERACTION_RUNTIME_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_interaction_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 INTERACTION_RUNTIME_DEVICE=cuda /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_interaction_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py --expected-commit 1a6a7a0613f3833108193ecc2085583d628aea05
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/train_g4_interaction_target_short.py --help
```

CPU **5/5，107.638s**；GPU **5/5，131.819s**。全量结果为625项，624通过+1既有GPU专用CLI跳过，833.216s。checker **390Python/96JSON**和help通过。GPU专项及全量的实际门槛保存于 `runs/issue208-{gpu,full}03-gate.txt`，符合free≥12288MiB、util≤60%；全量为旧CUDA钩子保留GPU0可见，不称纯CPU回归。

12角色各实际合成12步及新实例2+10，完成恢复0并核全部产物不变；CPU model/optimizer/scheduler/history/RNG/probability/trace逐位恢复。每角色3次选中态精确重载，共36次/专项。关闭M6与原190完整轨迹控制CPU精确；GPU关闭控制事前使用既有测试专用FixedGraphReference。启用GPU验证真实合成更新、恢复和ordered选中态重载，不声称原生CUDA两条独立优化轨迹等价。

专项还覆盖唯一全参数优化器登记、方向参与、条件/上下文/配置/源/算子/回执篡改拒绝、注入非有限消息梯度时实际更新前拒绝及钩子移除、实际CLI2+10+0和full/query/M4/M1/1500参数拒绝。训练/验证分数只用于合成工程，不作真实精度报告。

## 真实零更新及独立核对

三景×两角色×六条件36组配置和36模型初始化，model64/head1064，support512/K1；全部配置使用已验收合法上下文，初始化及完整绑定与206接受的36状态一致，优化器/验证快照方法仍为原实现。77个保护文件不变，**115.407670s**；真实forward/BN/优化/query0，无GPU上下文。会读取已释放support/validation覆盖层，不反序列化家族标签或原GT。

producer `runs/g4-interaction-runtime-issue208-preflight01/receipt.json`，SHA `412d956028cdcbb6933bb6d14052116cc310f3cdcac1f5a1a0e20f6dbf116aa2`。

独立干净checkout `checkouts/issue208-runtime-independent01`，分支 `codex/issue208-runtime-independent01`，同tested SHA；不调用208新settings/runtime/factory，使用旧190配置和手工六条件/208政策字段重建36配置，与206接受初态绑定和真实上下文逐一核对。78文件不变，**18.880643s**；模型重建0、forward/BN/优化/query0，无GPU上下文。此为配置组合审计，不冒充独立训练、概率或BN复算。

独立回执 `runs/g4-interaction-runtime-issue208-independent01/receipt.json`，SHA `d9652a6d332b77e589017e74ad82e39b60f6cd557bc5c710c1925995a6e64772`。工程汇总及同字节仓库副本为 runs/g4-interaction-runtime-issue208-engineering01/receipt.json，同字节副本 docs/architecture/g4-interaction-runtime-engineering-results.json，SHA 898cecb4c655d52734843348757d8a792186cc8aec810767795db9aa80247f1e。

## 保留失败、范围和下一依赖

首版1c9cdf955d6e30613171162abe5f9c5aa5ce8c20：CPU5项2通过/3错误，28.001s；选中态原BN回执因标准化checkpoint元数据重建缺失。修复b4c34aab5c6104b1a81f7ae9a2106ff3374c1fa1以原哈希绑定完整嵌套回执，未改共享训练器、原校准器或门槛。该版CPU5项4通过/1测试错误，58.343s：测试从返回包装的错误层级读取selected_step；1a6只修正该测试读取。两轮原日志、checkout和版本保留：`runs/issue208-cpu01.log`、`cpu02.log`。最终实际验收使用03新日志，不覆盖失败。

一次增量Git bundle仅fetch到FETCH_HEAD，no-local clone不会带未被分支引用的对象，正式checkout预期SHA检查在测试前拒绝；对新独立clone显式fetch原bundle后成功，未reset/clean或改旧目录。这是同步错误，不是数值失败。

本项只验收有界工程。真实12/1500训练、M6完整源预训练、同容量分支内变换、支持/K/更多种子、空间/传感器、M4组合及真实效果均未测。原生CUDA独立优化轨迹等价仍未接受。184仍必须完整54接受→独立331审计→186真实拟合/逐类/成本复盘→原331全冻结ordered query→204预登记结论，然后才能登记新真实实验；本模块未接管198资源交接或更改184冻结代码/输入。

验收结论为M6目标有界12步训练/严格恢复/参与证据及真实零更新身份工程通过。下一项为计划§11 M7候选边重加权的完整源目标初始化/orderedBN与两套二因素组合工程（Issue210），再补其有界训练/恢复；不新增远程边或硬剪枝。主入口仅同步报告/证据/current-status及远程生成投影，候选源码和main不合并。按用户最新要求用简短进度衔接下一项，不在单模块后等待确认。

证据同步时一次本机手工抄录SHA多一位，被输出写入前断言拒绝；改从SCP原字节计算，并以远程32字节摘要的Base64核对，未更改原回执。
