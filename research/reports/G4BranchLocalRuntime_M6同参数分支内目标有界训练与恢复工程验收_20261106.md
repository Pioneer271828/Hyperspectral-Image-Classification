# G4 M6同参数分支内目标有界训练与严格恢复工程验收

Issue228 / PR229；对应研究主清单§7.2、实施计划§10/§18/§19的同参数无中层跨支路消息对照。实际tested SHA **03df02b5c886e19d58b12a43f3a6c3c321642bf3**；开发分支 `codex/g4-branch-local-target-runtime`，工作树 `F:/E2E_VCA_worktrees/branch-local-target-runtime`。本项为有界工程验收，不是完整容量对照或真实效果结论。

## 用途和实现

新增 `g4_branch_local_target_runtime.py`、`scripts/train_g4_branch_local_target_short.py`、六项专项测试与内部复用清单。消费已验226组件6e78153c50467d67f6de46f8f2d1d564b6a915da及工程receipt，沿用224严格恢复、208原训练方法。独立核对226的443个Python/config文件全部同hash，新代码仅三个runtime/CLI/test文件；原331/226公共模型、配置、损失、采样、预算和loader不改。

none/gated/branch_local×scratch/完整none源FT六角色，12步工程profile；全新runtime先2再10步，完成恢复0。原三头监督、Adam/LR/采样/first-min fused完整观察验证NLL、ordered总体BN不变，M4/E1关闭，无full/campaign/query。gated/local同58498参数和初态，local多8字节必需版本buffer；相对匹配M6新增可训练参数0。

trace明确pixel_slot/region_slot，分别核历史graph_to_cnn/cnn_to_graph参数槽位的梯度norm、参数数与实际变化；local含义是各分支自身变换，不将槽位名称冒称跨支路消息。继承原骨干/三头保护，NaN槽位或融合头梯度在Adam参数步前拒绝并清理hook；失败前向BN缓冲不承诺全部回滚。

checkpoint含完整匹配M6绑定、自有绑定及嵌套原BN回执。选中态先核文件对/元数据/源/上下文/嵌套回执，再按登记训练设备重建初始化、迁移请求设备并严格载入。初始化仍会读取已认证源权重；不扩称所有源张量加载前拒绝。CPU恢复GPU登记短测仍临时使用登记GPU，不称无GPU迁移。旧通用loader继续拒绝local必需版本，避免误用跨支路语义。

## 实际环境、命令和验收

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue228-branch-local-runtime-engineering01`，实际分支 `codex/issue228-final01`，干净完整tested SHA同上。Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090。Windows仅编辑/文件/Git/SSH/SCP；不声称rmvision复跑或CI自动执行。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`；Python为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= BRANCH_LOCAL_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -p test_g4_branch_local_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 BRANCH_LOCAL_RUNTIME_DEVICE=cuda python -m unittest discover -s tests -p test_g4_branch_local_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu FUSION_RUNTIME_DEVICE=cpu NORMALIZATION_TARGET_DEVICE=cpu BRANCH_LOCAL_TARGET_DEVICE=cpu BRANCH_LOCAL_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 03df02b5c886e19d58b12a43f3a6c3c321642bf3
python scripts/train_g4_branch_local_target_short.py --help
```

CPU **6/6，90.473s**，GPU **6/6，107.810s**；完整 **671项670通过+1既有GPU专用跳过，1170.278s**。checker **411Python113JSON**、help通过。专项及全量分别核GPU0 free≥12288MiB/util≤60%，`runs/issue228-{gpu,full}01-gate.txt`保存；全量GPU可见供旧钩子，不称纯CPU。精确执行环境另存checkout外 `issue228-validation01.sh` 并绑定hash。

六角色合成12/新实例2+10/完成0、每角色三次选中态精确重载共18次/专项通过。CPU模型/optimizer/scheduler/history/RNG/选中概率/trace逐位恢复；none/gated对208两角色CPU精确。GPU该旧控制对照使用事前固定FixedGraphReference；原生启用GPU只接受功能/ordered选中态重载，不接受独立原生CUDA优化轨迹等价。原rtol1e-5/atol1e-6及分类零差不变。

实际local算子forward_aligned方法、版本、配置、权限、源/支持上下文、完整M6绑定与嵌套BN回执拒绝，以及NaN两槽位/融合头、完成恢复输入产物不变、合成CLI2/10/0通过。不把合成网络训练称真实场景训练，也不称所有标量每步均有正梯度或改变。

## 真实输入配置与独立复核

真实三景×两角色×三条件 **18配置/初始化**，model64/head1064、support512/K1；按已接受226状态、绑定、支持和源认证runtime组合与原optimizer/validation方法身份。**81保护文件不变，73.214005s**。Module调用与Adam构造禁止，真实前向/BN/优化/query0，无GPU上下文；已释放支持/验证覆盖仅认证，家族标签/原GT未反序列化。

独立进程用旧support settings、旧206匹配配置和手写policy字典重建 **18配置摘要**，与226已接受状态及绑定比对；不调用228新settings/runtime/factory、不新建模型。**82文件不变，19.252255s**。同干净tested checkout的不同进程，不称另一个独立checkout或概率/BN/训练审计。

工程原件 `runs/g4-branch-local-runtime-issue228-engineering01/receipt.json`，同字节仓库副本 `docs/architecture/g4-branch-local-runtime-engineering-results.json`，SHA **2a2e94bb799dd8909b50e9a3975db329dc84ce0647ce211364f62e2294323382**。
producer `runs/g4-branch-local-runtime-issue228-preflight01/receipt.json` SHA **61ba33e2c0af1c5f2180d2bb0732bd0c0c2a24026233b0aad801ce5a31cd7fd8**；
独立 `runs/g4-branch-local-runtime-issue228-independent01/receipt.json` SHA **843356dadd807267cbba5ac9b8ac97f5f83da7595657b32b43e2cf88162144d0**；
443文件代码身份 `runs/g4-branch-local-runtime-issue228-component-code01/receipt.json` SHA **da01d91d219be241fff2ef6919ab6458fc723f79632c2c651af8720831709bf5**。
日志 `runs/issue228-{cpu,gpu,full,checker,help}01.log`，driver/原件及全部产物在checkout外。

## 失败、限制和下一项

本实现数值测试失败0。首次只读工程receipt汇总因手工预期hash抄录错误在创建输出前被断言拒绝；保留 `issue228-engineering-receipt01.py`，从分别SCP实际exit0副本及本机原生SHA256取得真实hash后，`issue228-engineering-receipt02.py`通过。没有改producer文件、模型、数值或容差；不将登记/转录错误当数值训练失败。

工程通过不代表精度、收敛或速度。仅中层local算子不读另一支路/Q，整个网络仍共享adapter、原Q/图与末端融合；相同参数/变换深度不等计算匹配。初始两LayerNorm相同使差值gate输入0。真实容量/计算匹配、支持/K/空间/传感器/M4组合及原生CUDA独立优化轨迹仍待测，范围保留。

下一按主清单M8和实施§12读原论文及作者固定源码/许可，搭建独立位置门控融合核；后续接入与真实对照分项验收。所有新真实数值实验仍等待184全54接受→独立331审计→186真实曲线/逐类/成本复盘→原331全冻结ordered query→204预登记结论。不接管198资源续接、不改184冻结生产、不合main；主入口仅同步报告/证据/参考/current-status和远程投影。按用户要求简报后继续搭建。
