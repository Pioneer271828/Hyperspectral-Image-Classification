# WP04：残差/容量与位置融合组合目标接入验收

Issue343，对应主清单§8.4 R0/R1/R2/R4/R5、§22/WP04与实施计划§19.5。分支`codex/wp04-residual-position`；实际tested SHA `94d96ce3e55a1cba2d9a2bd1846efa471afbfaba`，基准main `beffba7e03057d72ab33375f9e586af5749d2b31`。

## 实现与初始化合同

在已有工厂和12步运行器增加显式`mode=none/capacity_control/x_residual`；none默认保持原载荷与行为。复用既有残差、容量控制和位置门控算子，不改backbone、优化器、损失、采样、图、标签权限、校准或选模算术。原zero-forward初始化CLI未扩展；真实执行短测CLI显式支持`--mode`，不接受full/query。

| 模式 | 固定融合 | 位置门控 | 参数说明 |
|---|---|---|---|
| none | R0既有 | R4既有 | 门控另增129参数 |
| capacity_control | R1既有 | 新容量×融合控制 | 容量模块8320，非精确等容量 |
| x_residual | R2既有 | R5新增组合 | 残差模块8256；门控另增129 |

同模式的fixed/position_gate复制相同共享参数，门控以既有logit(.8)初始化、不消耗随机数；fresh目标头、目标DomainBN保持原合同。完整源FT分别使用该模式已接受的源权重（包含残差/容量参数），门控在目标新初始化。**跨模式的FT具有不同源训练权重，不能把该比较解释为纯目标端机制的差中差。** 从头四组可另行登记目标端2×2；本次未运行该完整实验。

新增残差梯度在Adam更新前检查，缺失/非有限梯度拒绝；复用原M4逐步梯度/实际更新证据。零梯度合法，不能为了通过保护而改阈值。启用门控仍必须使用专用配对加载器。项目复用和外部论文边界见[复用记录](../../docs/references/wp04-residual-position-project-reuse.json)；没有新增外部代码/依赖或GMU复现声明。

## 实际验收

授权LXC `lbx@10.69.37.243:1222`，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`；干净checkout `checkouts/wp04-residual-position01`。CPU执行，CUDA_VISIBLE_DEVICES为空。

- 新组合5/5测试通过，101.412秒：8角色源复制/fresh头/域/RNG、模式拒绝；两模式两方法fixed与原active运行器的权重/优化器/调度器/history/RNG/概率/trace逐位一致；8角色连续12与2+10恢复完全一致；24精确重载、完成恢复0；嵌套身份篡改拒绝；残差非有限梯度拒绝发生在Adam之前；实际CLI2+10+0。512/2048及HongHu22类只测试初始化，不称真实训练。
- 原融合/位置融合工厂与运行器20/20通过，149.508秒；旧none条件仍通过恢复、原路径与权限负例。
- 完整源、支持上下文、门控核与原固定融合运行器相关回归17/17通过，149.624秒；共42项CPU测试通过。
- checker554 Python/180 JSON及help通过。没有运行全仓回归；GPU未测。当前八卡free均小于12288MiB，不占用其他任务、不降低启动门槛。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python -m unittest tests.test_g4_residual_position -v
python -m unittest tests.test_g4_fusion_target tests.test_g4_position_fusion_target tests.test_g4_position_fusion_target_runtime -v
python -m unittest tests.test_m4_full_transfer tests.test_g4_support_target tests.test_position_gated_fusion tests.test_g4_fusion_target_runtime -v
python scripts/check_repository.py --expected-commit 94d96ce3e55a1cba2d9a2bd1846efa471afbfaba
python scripts/train_g4_position_fusion_target_short.py --help
```

日志`runs/wp04-residual-position-tests01/active01.log` SHA256 `34185280c16006dbcb79558acde0f7e8bbd7e3a9614ba0992d1aa9407b22854b`，`regression01.log` SHA256 `83454b57af48eae4e026e8945f4d5c1f27e05feeafed25cb5a338961cb8496f8`。依赖日志`dependencies01.log` SHA256 `ef1b439abeff848ddbafd658d5bb3752c66e95eb418a32523ca50fb5f3670693`。本模块上述测试首轮通过，无失败/跳过；CSR beta提示为既有库警告，历史失败保留。

## 独立留存合成短测与复盘

外部脚本`transfer-records/wp04-residual-position-evidence01.py` SHA256 `0fd2b0992d66ca8cbd37deb6a4bfddc008bdeccd3ecd685499bd3cd0394a2de7`；产物`runs/wp04-residual-position-synthetic01`，均在checkout外。八例=容量/残差×固定/门控×scratch/完整源FT合成fixture，各2+10=12步，总96合成更新；训练21.706586秒，总29.865956秒。

NumPy独立复算48头指标、3888混淆格、432逐类召回；复核权威模型/optimizer/scheduler指纹、first-min选步、每步残差和启用门控的正梯度及实际参数变化。24精确重载、完成重入0更新、59输入和227复盘文件前后哈希一致。前向和加载复用生产组件，不称全网络独立oracle。

八例都选第12步。合成验证fused OA：容量scratch两组22.2222%、容量FT两组15.5556%、残差scratch两组22.2222%、残差FT两组6.6667%。所有案例三头仍有零召回类别（每头4—8类），不得宣称真实收益、充分收敛或小类保护通过；短测不用于决定正式候选或修改预算。

回执`receipt.json` SHA256 `8762d71b507b7648d0fc7fc8092d93a563743c42596f42721f45462968c87182`，仓库[同字节副本](../../docs/architecture/wp04-residual-position-engineering-results.json)。真实更新/query/GPU执行均0，未启用长训练定时任务。

## 结论及下一依赖

本模块仅交付R4/R5及容量×融合的CPU有界组合接入。WP04真实多场景、多种子、等预算矩阵与GPU验收仍未完成，R3/R6和学习Q也未由本模块覆盖。报告在本聊天交付一次后继续计划：WP05尚缺的真实重复受GPU资源及协议准入约束，先补WP06同容量逐层线性探针诊断接口，冻结主网络、训练/验证角色分开，不能将线性可分性直接称为语义因果证据。
