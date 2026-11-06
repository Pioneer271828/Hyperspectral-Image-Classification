# G4 M8固定融合、损失与BN目标接入组件验收

Issue214 / PR215；对应研究主清单§6.2/§7，模块计划§12、§18/§19。实际tested SHA **045a35ae431b75149367256a476b40b2a3777440**，分支 `codex/g4-fusion-target`，工作树 `F:/E2E_VCA_worktrees/g4-fusion-target`。本项通过既有M8的目标初始化、算子身份、损失及ordered总体BN组件工程验收；完整真实实验和效果仍未测。

## 用途、实现和复用范围

新增 `g4_fusion_target.py`、零前向初始化CLI、六项测试及复用清单 `docs/references/g4-fusion-target-project-reuse.json`。回读已验收M8报告与原4675ed200e5b2f8f3699de750fc5cfac8e7b1f0c，复用331原固定特征融合、三个独立分类头、概率损失13项、完整none源目标初始化及ordered总体BN；不改原331模型、配置、训练器或损失公式，无新增外部源码/依赖或许可声明。

| 显式条件 | alpha | CNN/Graph辅助权重 | 新增参数 |
| --- | --- | --- | --- |
| default | 0.8 | 1/1 | 0 |
| aux_off | 0.8 | 0/0 | 0 |
| alpha035 | 0.35 | 1/1 | 0 |

三条件×scratch/完整none源FT共六角色。共享参数、fresh目标三头、目标DomainBN状态与原none基础逐项一致；alpha改变实际构造属性/导出配置而不改初态张量，新构造随机消耗隔离。不是新的M8源预训练。绑定实际固定融合和分类方法、alpha、损失配置、参数权限、架构及源状态；无标签观察前向入口保留原probability/logits路径。

`loss_terms`仅验证三路概率矩阵/标签/声明配置后调用原公式。辅助权重为零只消除其监督梯度；融合和共享分支继续学习，原Adam weight_decay=1e-3保留，所以辅助头仍可能因衰减变化。alpha=0/1只说明融合头接收图/CNN特征，不等于相应独立辅助头预测，更不能替代单分支独立训练。ordered BN和文件成对重载保护复用原门槛，组件没有训练/full/campaign/query入口。

## 实际环境、命令和结果

授权LXC `/home/lbx/hsi-pioneer271828`，干净checkout `checkouts/issue214-fusion-engineering01`，分支和完整SHA同上。Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090。Windows仅编辑/读文件/Git/SSH/SCP；无rmvision复跑和自动CI声明。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`；Python均为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= FUSION_TARGET_DEVICE=cpu python -m unittest discover -s tests -p test_g4_fusion_target.py -v
CUDA_VISIBLE_DEVICES=0 FUSION_TARGET_DEVICE=cuda python -m unittest discover -s tests -p test_g4_fusion_target.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 045a35ae431b75149367256a476b40b2a3777440
python scripts/audit_g4_fusion_target_initialization.py --help
```

CPU **6/6，32.224s**，GPU **6/6，35.586s**；完整回归 **645项，644通过+1既有GPU专用CLI跳过，1033.119s**；checker **399Python/101JSON**、help通过。GPU专项和完整回归均独立核GPU0 free≥12288MiB/util≤60%，门槛记录在 `runs/issue214-{gpu,full}01-gate.txt`。全回归保留GPU可见供旧CUDA钩子，不称纯CPU运行。

六角色共有初态/fresh头域/RNG/零新增参数、default相对原none前向与BN精确一致；alpha035手工特征/融合头logits及端点、logits-softmax对应、13项原损失精确及独立手算平衡NLL、关闭辅助梯度与保留衰减语义通过。各角色先合成3步更新，再总体BN、参数/计数器/观察保护及各3次精确重载，共18次/专项。一个alpha035角色另用CPU FP64总体矩参考，再转原FP32规范化路径；不称全网FP64或所有角色独立概率复算。

实际alpha/方法替换/源身份/参数权限/标签字段/错误概率形状与logits/配置/文件/种子/输出/full/query拒绝通过。GPU专项验证功能与ordered重载，不扩称原生CUDA独立优化轨迹等价。

## 真实输入零更新和独立重建

真实三景×两角色×三条件共 **18初始化、6基础状态身份**；model64/head1064、support512/K1，88保护文件不变，**20.294837s**。所有共有初态与已接受210的none状态及184相应冻结初始化一致；去除显式alpha差异的导出架构与原基础一致。读取已释放支持/验证覆盖层用于认证，不反序列化家族标签或原GT。Module调用和Adam构造被禁止，实际NN/BN/优化/query均0，无GPU上下文。

独立干净 `checkouts/issue214-fusion-independent01`，同tested SHA。仅用旧none工厂6个基础、原构造器和手工三条件合同，重建18状态/架构/训练参数身份；不调用214新工厂，不读支持/验证覆盖层。89保护文件不变，**3.616360s**；NN/BN/优化/query0。该独立证据仅限初始化/配置重建，不冒充独立BN或真实训练审计。

工程原件 `runs/g4-fusion-target-issue214-engineering01/receipt.json`，同字节仓库副本 `docs/architecture/g4-fusion-target-engineering-results.json`，SHA **d04ca1e0983833a494bb31d2e07e93ef9b9fb660a5fcf8846704a3c398079ceb**。producer `runs/g4-fusion-target-issue214-preflight01/receipt.json` SHA **efc3b1b9cdbb50d53140eabd984f825b3bb602383f180fc9a86b963f2fa258af**；独立 `runs/g4-fusion-target-issue214-independent01/receipt.json` SHA **ffc15ff13d2ff1e05797fb9fa9e2ef5f37fb86c4da96e7dbbb6cc673514ed3a7**。实际日志 `runs/issue214-{cpu,gpu,full,checker,help}01.log`；产物全部在checkout外。

## 保留失败、限制与下一项

首次SCP未观察exit0就消费bundle，Git在新clone拒绝invalid gitfile format，测试未启动，HEAD仍为原干净4a306。已承认并改正：观察SCP真实exit0、bundle verify、检查原checkout干净后重新fetch045a才执行验收；失败clone/日志和 `runs/issue214-transfer01.json` 均保留。这是传输顺序错误，不是数值训练失败。GitHub连接器/TLS失败及权威GET后的创建记录保留，无重复Issue/PR。

工程通过不代表真实精度、成本、收敛或跨传感器收益。新门控融合、M4×门控四组、独立单分支、共享语义头合法映射、支持/K/空间/传感器等必测范围未删除；原生CUDA独立优化轨迹等价仍未接受。

下一项 **Issue216 / PR217** 既有M8有界12步/严格恢复及分头损失、梯度和变化证据工程已独立开始。真实实验仍必须先184全54接受→独立331审计→186真实拟合/逐类/成本复盘→原331全冻结ordered query→204预登记结论；本项不接管198资源交接或修改冻结184。主入口只同步报告/证据/参考清单/current-status和远程投影，main未合并；依用户要求简报进度后连续推进。
