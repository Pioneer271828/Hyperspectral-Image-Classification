# G4 M7目标有界训练与严格恢复工程验收

Issue212 / PR213，研究主清单P1§7核心二因素诊断和P4目标适配，模块计划§11/§18/§19。实际tested SHA **f7dfcc54a93e348179f3071e77a5e637e7e9611d**，分支 `codex/g4-edge-target-runtime`，工作树 `F:/E2E_VCA_worktrees/g4-edge-target-runtime`。以下验收范围仅为有界工程；真实完整预算及收益不属于本次结论。

## 实现与计划对应

新增 `g4_edge_target_runtime.py`、`scripts/train_g4_edge_target_short.py`、`tests/test_g4_edge_target_runtime.py` 三文件。复用210的实际M6/M7算子绑定与ordered总体BN，208的嵌套原BN回执重载模式，190的合法支持/K上下文和原NoneFullSourceTargetRuntime；原331模型、配置、训练/损失/优化器/调度器及采样/选模文件没有修改。

固定none/gated/graph/gated_graph/joint/gated_joint六条件，scratch与完整none源FT两角色，原消息强度0.1、边mix0.5、维数32及调度次序保持。新增M6/M7参数在目标初始化，不是M6/M7源预训练。仅12步工程预算、先2/续10/完成恢复0；拒绝full/campaign/query/增强/M4/E1/light与任意扩展预算。

逐步记录两个直接消息方向与edge组件各自的参数量、梯度范数和实际变化元素，关闭组件为None且不构造。对启用组件在原Adam.step前通过临时只读钩子核梯度存在/有限，finally移除；允许合法有限零梯度。模块参与测试使用多个候选，identity退化零梯度控制复用210并纳入完整回归，不把参数仅因权重衰减变化当作真实机制有效证据。

选择仍为完整观察的first-min fused验证NLL；训练仍只用原selected-row三头概率损失。raw验证源、角色/条件、真实算子属性（含mix/方向等）、源/上下文/配置/权限、原BN完整回执和校准状态全部绑定。原训练器重建元数据时，唯一声明的M7校准键保存完整嵌套原回执；重载先核其原始哈希，再只在内存恢复给未改的原校准验证器。没有替换原校准或放宽rtol1e-5/atol1e-6、分类0合同。

## Linux环境和实际检查

授权LXC `/home/lbx/hsi-pioneer271828`，Python3.10.20/Torch2.7.1+cu128/CUDA12.8，GPU0 RTX4090；干净checkout `checkouts/issue212-runtime-engineering01`，分支同开发分支，完整SHA同上。Windows仅编辑/Git/SSH/SCP；未复跑rmvision、未声明自动CI。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`：

```bash
CUDA_VISIBLE_DEVICES= EDGE_RUNTIME_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_edge_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 EDGE_RUNTIME_DEVICE=cuda /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_edge_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py --expected-commit f7dfcc54a93e348179f3071e77a5e637e7e9611d
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/train_g4_edge_target_short.py --help
```

CPU **6/6，123.528s**；GPU **6/6，146.841s**；全仓回归639项，638通过+1既有GPU专用CLI跳过，1008.835s；checker **396Python/98JSON**、help通过。实际门槛 `runs/issue212-{gpu,full}01-gate.txt`，free≥12288MiB、util≤60%、GPU0单队列；全量给既有CUDA钩子保留GPU0可见，不称纯CPU回归。

六条件×两角色每组实际合成12与新实例2+10/完成0，CPU model/optimizer/scheduler/history/RNG/best概率/trace逐位一致，完成0核目录全部文件不变；每角色3次选中态精确重载，共36次/专项。各启用消息/边组件分别有正总梯度和实际变化，关闭组件为None；唯一完整参数登记和支持/K/model-head种子分离通过。

none/gated关闭edge的两角色完整轨迹与208控制相同，CPU逐位；GPU该关闭对照事前使用已验收测试专用FixedGraphReference。启用GPU范围是原生合成更新、恢复和ordered选中态重载功能，不称两条原生CUDA独立优化轨迹等价。

新的合成fixture在构建上下文前为临时观察设置两个合法候选，重写对应观察/record/输入清单哈希，再由原prepare_contexts认证；不是训练后篡改绑定，不修改磁盘上真实观察或旧测试源文件。实际CLI在gated_joint条件执行2+10+0，full/query/M4/M1/额外1500等参数拒绝。

包括条件/预算/profile/上下文/query字段、实际消息强度和edge.mix、错误raw/校准源/嵌套原回执等拒绝。NaN直接消息和edge-only query梯度均在实际参数更新前拒绝，optimizer状态未创建、所有参数未改、钩子移除；失败前向可能已更新临时模型的训练BN缓冲，不把这个检查称为整个模型状态未变。真实输入的防护另由只读结束核验保证。

## 真实零更新及独立配置审计

真实三景×两角色×六条件36配置和36初始化，model64/head1064、support512/K1；全部与210已接受状态/完整绑定一致，优化器及验证快照方法仍是原实现。82个保护文件不变，**116.245386s**；真实forward/BN/优化/query0，无GPU上下文。仅反序列化已释放support/validation覆盖层，不反序列化家族标签或原GT。

独立干净checkout `checkouts/issue212-runtime-independent01`、分支 `codex/issue212-runtime-independent01`、同tested SHA；不调用212新settings/runtime/factory，手工重建36政策/配置并对旧190上下文和210接受初态绑定逐一核对。83文件不变，**18.960132s**。模型重建0、forward/BN/优化/query0、无GPU上下文；此为配置组合审计，不冒充独立概率、BN或训练审计。

原始producer `runs/g4-edge-runtime-issue212-preflight01/receipt.json`，独立 `runs/g4-edge-runtime-issue212-independent01/receipt.json`，日志 `runs/issue212-{cpu,gpu,full,real,independent}01.log`，工程回执及SHA为runs/g4-edge-runtime-issue212-engineering01/receipt.json（SHA c65fa559fc69d9ce9cdd7257ef260a5e2fa7c99b7bd36315dd7cd6319fc5cc14），同字节副本 docs/architecture/g4-edge-runtime-engineering-results.json；producer SHA 460df192efb60592093fb12b0c1b52002c317927ebf730e68bbc52496e6771a1，独立 SHA 9b1294dfcb3692bbcf12a2d44811272363f29a0499e1e4f7b53c6765156e943c。产物在checkout外，仓库只同步小型回执和报告。

## 保留失败与未测范围

当前实现版本专项失败0；GitHub TLS/连接器及API传输失败保留。登记不确定时先权威GET查212，短正文POST成功后PATCH完整范围，无重复Issue；发布只重试相同已提交开发分支/标签，不造平行提交或强推。后续只合210报告/证据/状态，不改本次tested源码或冻结184。

工程结论为M7核心二因素目标有界12步/严格恢复/分项参与及真实零更新身份工程通过。真实12/完整预算效果、M6/M7源预训练、仅归一化、同容量替代、更多种子/支持K/空间/传感器/M4组合均未测。原生CUDA独立优化轨迹等价仍未接受。负结果和缺项继续保留，不因工程通过推定精度/成本收益或把有标签少样本传导式适配称为零样本TTA。

下一依赖是模块计划§12 M8原固定融合/三头概率损失的完整源目标初始化与orderedBN组件接入，之后补其有界训练消费者；可选门控/分解×门控与独立单分支完整实验范围保留。184完整54接受→独立331审计→186真实拟合/逐类/成本复盘→原331全冻结ordered query→204预登记结论仍优先，完成前不发起新真实数值实验。本工程未接管198资源交接、未更改冻结184代码/输入，main未合并。按用户要求以进度衔接下一项，不在模块交付后等待确认。
