# G4完整配对目标协议工程阶段记录

对应研究主清单P4/§10.3、模块计划§19，Issue182 / Draft PR183。前置172/178完整源和174/180真实短接入均已交付。本项当前只完成统一核心及实际54条件零更新预检，**整个Issue182仍未完成**；完整worker/队列/独立接受与全冻结评价接口仍需继续搭建，真实长实验尚未启动。

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

## 未完成与下一步

继续显式full worker、预登记/执行phase绑定、资源队列、完整接受/独立审计及全部模型冻结后的评价合同；完整的工程模块验收与报告尚未完成，Issue182保持open、PR183 Draft。已有54零更新通过不能替代这些范围。

工程通过后另Issue冻结真实none18×1500与活动36×1500预算、资源/时长、失败条件及全部身份，再依用户既有授权启动；不把零更新预检复制成完整训练接受，不混旧native query。完整54、独立审计和每例拟合/逐类/成本复盘后才允许新同协议query。其余支持集/K/空间/传感器与必测机制保留，main未合并。
