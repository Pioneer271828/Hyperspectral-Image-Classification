# G4 M7仅归一化目标有界训练与严格恢复工程验收

Issue224 / PR225；研究主清单§7.1/§7.2，实施计划§11/§18/§19的仅归一化公平对照。实际tested SHA **5c7ae64e2e0f4a1ecd4ef90c2d225369ebaa2749**；开发分支 `codex/g4-normalized-edge-target-runtime`，工作树 `F:/E2E_VCA_worktrees/normalized-edge-target-runtime`。本项只验收有界工程，未执行真实目标训练或评价，未取得仅归一化/M6/M7精度收益结论。

## 用途和实际实现

新增 `g4_normalized_edge_target_runtime.py`、`scripts/train_g4_normalized_edge_target_short.py`、六项专项测试及 `docs/references/g4-normalized-edge-runtime-project-reuse.json`。沿用222原先验控制、208有界runtime和212恢复保护模式；原331、222组件、配置、损失和公共训练器/loader全部未改，既有源初始化、图/标签及选模协议保留。内部项目代码最小接入，无新外部机制/依赖/许可声明。

none/row_normalized/gated/gated_row_normalized×scratch/完整none源FT共八角色，12步仅工程profile；先2步、全新runtime恢复10步，已完成再次恢复0更新。原三头监督、Adam/LR/采样/first-min fused完整观察验证NLL及ordered总体BN不变，M4/E1关闭，不增加增强/预算，不开放full/campaign/query。仅归一化仍0训练参数、启用version buffer8字节；匹配gated基底仍58498 M6参数，原1e-12极小质量阻尼不改。

配置、实际算子/位置/version/floor、源、支持上下文、架构和全部可训练权限均核。trace单列零参数归一化控制，以及两个M6方向梯度norm、参数量、实际变化元素；继承原头/骨干梯度保护。NaN消息或融合头梯度在Adam参数更新前拒绝，hook清理及optimizer空状态可核；不声称失败前向全部BN缓冲回滚。

检查点声明完整匹配M6绑定，并在专属校准回执嵌套完整原BN回执。选中态自有loader先核权重/sidecar身份、原来源文件/元数据/嵌套回执，按原登记设备重建初始化身份，再迁移到请求设备严格载入、恢复原BN回执并验概率。初始化工厂会读取已认证源权重，所述加载前守卫不扩称阻止所有源初始化张量读取。启用态旧通用loader继续因必需version buffer拒绝；CPU读取GPU登记短测时仍会临时在原训练设备重建，不称跨设备无GPU迁移。

## 实际环境、命令和结果

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue224-normalization-runtime-engineering03`，实际分支 `codex/issue224-final03`，干净完整tested SHA如上。Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090；Windows仅编辑/读文件/Git/SSH/SCP。无本轮rmvision复跑或CI自动执行声明。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`，Python均为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= NORMALIZATION_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -p test_g4_normalized_edge_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 NORMALIZATION_RUNTIME_DEVICE=cuda python -m unittest discover -s tests -p test_g4_normalized_edge_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu FUSION_RUNTIME_DEVICE=cpu NORMALIZATION_TARGET_DEVICE=cpu NORMALIZATION_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 5c7ae64e2e0f4a1ecd4ef90c2d225369ebaa2749
python scripts/train_g4_normalized_edge_target_short.py --help
```

最终CPU **6/6，99.465s**，GPU **6/6，122.531s**；完整 **664项663通过+1既有GPU专用跳过，1215.666s**；checker **408Python110JSON**、help通过。GPU专项及完整回归分别核GPU0 free≥12288MiB/util≤60%，门槛记录 `runs/issue224-{gpu,full}03-gate.txt`；全量GPU可见供旧钩子，不称纯CPU。

八角色各完整12对新实例2+10，CPU模型/optimizer/scheduler/history/RNG/选中概率/trace逐位一致；完成0及输入产物哈希不变。none/gated关闭控制对208两角色精确，GPU该对照使用事前固定 `FixedGraphReference`，不是原生CUDA独立训练轨迹证明。原生启用GPU验功能和ordered选中态各3次精确重载，24次/专项；原rtol1e-5/atol1e-6及分类零差不变。两方向启用时正组梯度/实际参数变化，不声称每个标量每步都改变。

配置/budget/模式/M4/seed/支持上下文/标签/full/query/输出拒绝，实际算子floor/mode/version、M6strength、NaN消息和主头参数更新前保护，以及源/完整M6绑定/嵌套回执拒绝通过。实际合成CLI2/10/0通过，不能称真实场景12步或完整1500训练。

## 真实输入零更新与独立配置复核

真实三景×两角色×四条件 **24配置/初始化组合**，model64/head1064、support512/K1，继承222已接受状态、绑定、支持/源身份。配置、原optimizer与validation_snapshot方法身份逐项核；82文件不变，**87.406606s**。Module调用和Adam构造明确禁止，真实NN/BN/优化/query0，无GPU上下文；已释放支持/验证覆盖读取只用于认证，家族标签和原GT未反序列化。

独立进程用旧support settings、旧206匹配配置及手写新policy字典重建24配置摘要，与222接受状态和绑定比对；不调用224新settings/runtime/factory，不新建模型。83文件不变，**18.967296s**。此为配置组合独立复核，非独立真实概率/BN/训练审计；同一干净tested checkout，不冒称另一个独立checkout。旧c93零前向receipt01及其独立成功材料保留，最终preflight02另确认全部24行与旧01完全一致，修复未改变初态或配置。

工程原件 `runs/g4-normalized-edge-runtime-issue224-engineering01/receipt.json`，同字节仓库副本 `docs/architecture/g4-normalized-edge-runtime-engineering-results.json`，SHA **225a30819aa4e99ac7102363b6996bb979349e873100529299e188e7191f4122**。producer `runs/g4-normalized-edge-runtime-issue224-preflight02/receipt.json` SHA **dcaf3aaa8c4442d201f9438c4485850e5fbc6f684205f8d9ccd76f18e6a7a08e**；独立 `runs/g4-normalized-edge-runtime-issue224-independent02/receipt.json` SHA **87f5974584c993f21d21e225e364a495be9725773febd943b67dfa9190fb0f71**。最终日志 `runs/issue224-{cpu,gpu,full,checker,help}03.log`；所有产物在checkout外。

## 失败记录、限制和下一项

首版 **0fe20056c5b87219f90539c0a3748ab9012c28cc** CPU6项3通过3errors、25.452s：公共训练器重建metadata时漏掉匹配206 M6绑定，222严格身份检查拒绝。仅在224自有checkpoint metadata声明并核对完整原绑定后修复，未改222或守卫。

中间 **c93bf1405f0438f4d3fb6d272c46aa85c0fb2131** CPU6/6、99.259s，但GPU6项3通过3errors、30.602s：新loader强制CPU重建，和原CUDA登记的initial_device/target_initial_config身份不一致。最终先用原登记配置重建再迁移，保留源/上下文/选模/预算/门槛，没有容差放宽。两版失败日志、clone及成功CPU中间记录保留；未在失败版本继续全量回归。一次clone已有同名分支导致checkout拒绝，测试0；随后在同一干净clone用新的独立验证分支显式checkout完整SHA，未reset/clean。连接器/TLS失败与权威GET后的注册恢复记录保留。工程receipt首次SCP认证窗口关闭exit1，未消费产物；重连实际exit0后才同字节核hash并同步。

工程通过不代表真实精度、充分收敛或速度收益。原生CUDA独立优化轨迹等价、仅归一化对学习M7/相近容量、支持/K/空间/传感器/M4组合完整对照仍未测，不删除范围。下一 **Issue226 / PR227** 按主清单§7.2补同参数分支内变换组件，区分M6交换与容量；同参数不等于同计算预算。

真实后续顺序不变：184全54接受→独立331审计→186真实拟合/曲线/逐类/成本复盘→原331全冻结ordered query→204预登记结论后再登记新实验。本项不接管198多卡交接或改184冻结生产，不合并main。主入口只同步报告/证据/当前JSON及远程投影，按用户要求简报后继续搭建。

