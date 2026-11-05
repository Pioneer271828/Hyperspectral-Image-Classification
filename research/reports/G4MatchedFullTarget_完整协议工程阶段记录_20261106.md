# G4完整配对目标协议工程验收记录

对应研究主清单P4/§10.3、模块计划§19，Issue182 / Draft PR183。前置172/178完整源和174/180真实短接入均已交付。本项统一核心、完整worker、单卡队列、独立接受/审计和全54冻结评价边界已完成工程验收。实际54条件零更新预检通过；真实81000更新实验及真实query尚未开始，另登记实验Issue，不以工程通过宣称效果。

唯一源码写入工作树`F:/E2E_VCA_worktrees/g4-matched-full-target`、分支`codex/g4-matched-full-target`，基于已验adapter交付f240a7e116f16cb0a431813999e86cc4a04e75d7。范围见[搭建范围](../experiments/G4MatchedFullTarget_完整配对协议搭建范围_20261106.md)。不改原模型、source/target训练、loss、校准器、旧配置/入口及冻结规则。

## 当前实现与已验范围

`g4_matched_target_study.py`复用已验none/活动full1500 runtime，定义3目标×3模型种子64/512/2048×从头/全微调×none/容量/残差共54格。仅剥离声明的架构/身份差异后，其他训练/损失/优化/校准/数值/标签/图/选模合同须逐字段相同。研究manifest在候选分支`research/experiments/g4_matched_target_full_v2.json`并显式绑定文件hash，不触碰冻结configs。规划完整更新none27000+活动54000=81000；当前实际真实完整更新0。

`g4_matched_target_launch.py`先验证已验none6/活动12短回执、全部原artifact和producer文件在consumer保持，再核实际完整源和目标配对上下文；9组target×seed证明三模式共有fresh参数逐元素相同，全部迁移目标头/DomainBN保持对应fresh。工程预检phase独立且`execution_enabled=false`、`optimizer_updates=0`，不能充当训练launch/完整接受。CLI只准备，不提供启动训练选项。

实际测试/预检SHA **3ea557d53ee9c132691dd6f58c14db82e86eba33**。rmvision冻结`/home/rmvision/matched-full-target-core02`，分支`codex/matched-full-target-core02`；解释器`/home/rmvision/venvs/hsi-baseline/bin/python`，GPU0。LXC冻结`/home/lbx/hsi-pioneer271828/checkouts/g4-matched-target-preflight01`，分支`codex/g4-matched-target-preflight01`，解释器`envs/hsi/bin/python`，实际预检只用CPU。

```text
PYTHONPATH=.:tests python -m unittest test_g4_matched_target_study -v
# CPU4/4,25.432s
MATCHED_TARGET_DEVICE=cuda PYTHONPATH=.:tests python -m unittest test_g4_matched_target_study -v
# GPU4/4,69.734s
python -m unittest discover -s tests -v
# 全546/546,202.513s
python scripts/check_repository.py --expected-commit 3ea557d53ee9c132691dd6f58c14db82e86eba33
# 两环境passed,347Python/80JSON；冻结路径不变，CI runner仍未注册
python scripts/prepare_g4_matched_target_full.py --help
# 两环境通过
```

线程环境均为`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。测试覆盖54角色、共同数值合同、fresh状态、来源/标签/图/预算/phase/query负例、短回执artifact与producer保护；并在小型合成目标网络执行**1500实际合成更新**、2+1498恢复、51个验证事件、首次最小NLL选步、逐更新LR重放和完成0不变恢复。该合成1500不是实际研究模型的81000训练预算，也不证明所有真实54模型已完整训练。

最初a6ec24d12d7dab6f6947fd0df048c640c96ee029 CPU3通过1错误：网格fixture缺`dirty`字段，生产严格身份检查拒绝；实际合成1500已过，真实预检当时未跑。仅补齐测试元数据的3ea通过CPU/GPU/全量，未改生产合同。保留`/home/rmvision/matched-full-target-core01`和`matched-full-target-core01-cpu.log`；新日志`matched-full-target-core02-{cpu,gpu,full}.log`均在checkout外。

## 实际54零更新预检

LXC `runs/g4-matched-target-issue182-preflight01`已完成54/54，9组三模式共同fresh证明、完整源角色/种子/模式绑定、配对标签/观察/Q/A与训练数值合同均通过，7923文件/输入不变，37.753422s；优化器更新0/query0、execution_enabled=false。原PID269553和启动shell269550已结束，不能重启既有目录。

- receipt SHA256 `2dacef05240b2d41854c4e70f97e7e3887a3b64bb2e1c3872ffead9bb97d82a5`
- launch SHA256 `f12da737a9bc8c03fd13ea6625198ab50138d0c9ab6b92c61444a22da7d77fe3`
- study文件 SHA256 `1fc4d5588d1a864dcfdcabf9aaa6fcdf0dc50cf12a2ec5c7d2d12d9292a1c04c`
- [54角色及初始化摘要](../../docs/architecture/g4-matched-target-preflight-results.json)，CPU导出自新`runs/g4-matched-target-issue182-review01/summary.json`，SHA256 `a3b51408a84a26647fef8db28f2940d045035fa9d91e850bb0782e73fc3645ca`。原始launch/源权重及所有产物留在checkout外。

实际命令为`python scripts/prepare_g4_matched_target_full.py --expected-commit 3ea557d53ee9c132691dd6f58c14db82e86eba33`，绑定none-source=`runs/g4-none-source-issue178-full01`、active-source=`runs/m4g4bv2-issue172-full01`，两批各自独立audit receipt和6/12短接入根、已验`g4d-issue151-observations02`观察与`g4c-issue149-labels01`标签家族，输出上述全新目录。CLI没有训练分支或query消费者。

## 完整worker、队列、审计与评价工程验收

实际完整测试/后续生产SHA **331d718567b1a9726934f445665999df972392b1**，源码工作分支`codex/g4-matched-full-target`。rmvision冻结`/home/rmvision/matched-target-pipeline03`、分支`codex/matched-target-pipeline03`；LXC冻结`/home/lbx/hsi-pioneer271828/checkouts/g4-matched-target-pipeline03`、分支`codex/g4-matched-target-pipeline03`。两处均干净，checker通过357Python/81JSON。rmvision Python3.12.3/Torch2.7.0a0+ecf3bae40a.nv25.02/CUDA12.8、GPU0 RTX4090/driver610.88；LXC Python3.10.20/Torch2.7.1+cu128/CUDA12.8、GPU1 RTX4090/driver595.71.05。均使用前述单线程环境，CI runner尚未注册。

- `g4_matched_target_execution.py`和full CLI要求单独实验Issue、完整工程receipt、实际生产SHA、study文件hash、81000预算、GPU/估时及事前analysis-plan文件hash；冻结54真实身份/9组共同fresh后才可更新，worker限定full1500、2+1498恢复及完成0。旧3ea生产文件在331保持不变。
- `matched_target_campaign.py`逐子操作资源检查、单卡单队列、逐例完成0不变恢复及独立核验，失败保留并停止。全54严格汇总81000更新、324头指标/4320召回/162精确重载，不能用部分result充当全批次接受。
- `g4_matched_target_audit.py`和`matched_target_full_audit.py`绑定所有源/目标/登记/权重/sidecar/raw状态hash，独立核选中态三头train/val的NumPy指标、小类及三次精确重载，全部51raw验证事件历史/首次最小NLL选步、逐更新LR及权威optimizer/scheduler状态；核实际模块梯度参与与参数变化、拟合和成本。**非选中raw事件核hash/历史/选模/LR，不声称逐事件独立重算全部概率。**独立全批次审计优化器更新0/query0。
- `g4_matched_target_evaluation.py`及评价CLI新增all54冻结边界，不能伪造旧native27模型边界。训练接受和独立审计完整通过后，核全部54模型与输入再允许读取query原GT；每模型核全54可计分权重/sidecar/result及标签/观察身份，开始/结束再核完整保护清单。输出必须全新独立目录，不覆盖源/短测/输入。合成GT正向释放与未冻结拒绝测试均覆盖；**真实query没有读取。**同协议query统一ordered校准/前向；三模型种子样本SD、配对差与逐类差异输出，支持种子仅1次不作独立支持集重复。

```text
# rmvision: CPU专项；12项=11通过+1个GPU专用CLI跳过，28.268s
PYTHONPATH=.:tests python -m unittest test_g4_matched_target_execution test_g4_matched_target_audit test_g4_matched_target_worker_cli -v
# rmvision: GPU16/16，237.292s
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cuda PYTHONPATH=.:tests python -m unittest test_g4_matched_target_study test_g4_matched_target_execution test_g4_matched_target_audit test_g4_matched_target_worker_cli -v
# rmvision: 全量558项=557通过+同一GPU专用项跳过，263.180s
MATCHED_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
# LXC: GPU12/12，231.156s，含该CLI项
CUDA_VISIBLE_DEVICES=1 MATCHED_TARGET_DEVICE=cuda PYTHONPATH=.:tests python -m unittest test_g4_matched_target_execution test_g4_matched_target_audit test_g4_matched_target_worker_cli -v
python scripts/check_repository.py --expected-commit 331d718567b1a9726934f445665999df972392b1
# campaign/audit/evaluation三入口--help均通过；train/prepare帮助也已通过
```

专项包含小型9类目标网络的实际合成1500更新CLI全链路（2+1498+完成0）及独立选中态审计；全54计数、缺模型、权重变更、query早读、逐例数量调换但总数不变、队列失败停止/只读恢复写入、输出隔离等由合成catalog/故障注入验证。不能把一例合成1500称为54真实训练完成。CPU跳过项已单独在两环境GPU运行通过，不写“558/558全通过”。

最终331消费者再次认证原3ea预检/producer文件和7926保护项（原7923加receipt/launch/manifest），115.591629s；没有重跑54初始化或更换预检输入。工程receipt在`runs/g4-matched-target-issue182-engineering01/receipt.json`，摘要见[g4-matched-target-engineering-results.json](../../docs/architecture/g4-matched-target-engineering-results.json)，记录原始命令、环境、日志hash和scope。rmvision日志`matched-target-pipeline03-{cpu,gpu,full}.log`，LXC日志`runs/g4-matched-target-pipeline03-gpu.log`均在checkout外。

归档首次保护清单hash转录错误被断言拦截，在创建receipt目录前退出；重新读取原文件hash后登记，原文件未修改，也未重跑训练或放宽条件。原a6 fixture失败和所有中间工程日志保留。

## 验收结论、限制及下一依赖

**Issue182工程验收通过**。工程receipt明确`full_training_completed=false`、实际研究完整更新0/query0；这里没有新增真实训练指标、三头/小类收益或耗时结论。实际54预检只有身份/初始化和访问边界证据，所有真实效果仍待完整实验。54真实case的队列、独立全批次审计和最终query尚未运行，不能把合成全链路替代实际闭环。

下一项另Issue预登记none18×1500=27000与活动36×1500=54000，总81000目标更新，冻结分析判据、资源/估时与停止规则后依用户既有授权启动。完成全54及独立审计/每例拟合、逐类和成本复盘后才释放新同协议query；不混旧native query。支持种子512/2048、K1/5、空间/传感器、head-only/light-module及其他两主计划必测项继续保留。main未合并。

聊天交付位置：本聊天2026-11-06“G4完整配对目标协议工程模块完成报告”；持久报告为本文件及工程JSON摘要，后续心跳不重复本项报告。
