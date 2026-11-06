# G4独立支持集与K目标训练工程验收

Issue190 / PR191；对应研究主清单P4 §10.3第3、5、6项和模块计划§19“补机制与独立重复”。本项接在188输入绑定工程之后，使支持种子512/2048、K1/5等后续重复能够通过显式入口使用已接受的完整源模型。用户要求正常工作回合连续搭建；本工程独立于正在运行的184，不启动下一真实数值实验。

## 实现及范围

工作树`F:/E2E_VCA_worktrees/issue-190-support-target`，分支`codex/issue-190-support-target`，基于188交付`ac13d95e499178322b875b6a92ed7138da0d43fe`，唯一写入人为本聊天助手。只新增`g4_support_target.py`、`scripts/train_g4_support_target_short.py`、`tests/test_g4_support_target.py`；原331/184的全部Python/config文件不改，无多智能体、无main合并。

复用188的`load_context`和已验收none/活动M4完整源初始化、目标训练、ordered校准与验证。没有新外部机制、源码、依赖或上游协议；不将工程接线写成新算法复现。原三头损失、Adam参数、调度、采样、首次最小fused验证NLL和严格重载门槛保持原协议。

- 显式选择目标、支持种子64/512/2048、K1/5/10和独立模型种子；head seed=model seed+1000。三种模式none/capacity_control/x_residual，各有从头与完整源全微调两种角色。
- 入口只有12步工程配置，支持2+10断点恢复和完成后0更新验证；不提供full/1500、批次循环、研究登记或query子命令。head-only/light-module留作独立后续模块。
- 188目录仍是`engineering_only/execution_enabled=false/optimizer_updates=0`的数据目录；新执行协议单独绑定有限12步工程用途，不修改原manifest开关。真实支持/K实验仍需后续独立登记。
- settings、恢复identity、权重metadata同时绑定manifest/family/阶段/图/实际标签与观察指纹、支持/K/模型种子以及完整源规格摘要；校验输入文件、源权重/sidecar/回执，输出与所有源、观察及覆盖层目录隔离。
- 拒绝错角色、错种子/K、标签/图变化、额外query张量、完整预算、改变ordered阈值、错误预期SHA、错误恢复身份及改变选中态context metadata。逻辑接口不冒充OS沙箱。

## 实际版本与验收

实际tested/工程生产完整SHA：`e7424559e1bc72c655f511c213e551ced9b95be3`。

| 环境 | 独立checkout和分支 | 结果 |
|---|---|---|
| rmvision Linux，Python3.12.3，Torch2.7.0a0+ecf3bae40a.nv25.02，CUDA12.8，GPU0 RTX4090/driver610.88 | `/home/rmvision/issue190-support-engineering01`，`codex/issue-190-support-target` | CPU5/5，41.286s；GPU5/5，65.231s；全量577项576通过+1既有GPU专用跳过，336.113s；checker366Python85JSON、help通过 |
| LXC Linux，Python3.10.20/Torch2.7.1+cu128，CPU且CUDA_VISIBLE_DEVICES为空 | `/home/lbx/hsi-pioneer271828/checkouts/g4-support-target-issue190-engineering01`，同分支/同SHA | 真实1组支持/K ×6角色零更新初始化验收；63文件不变，47.674548s；checker366Python85JSON通过 |

Windows仅编辑/读文件/Git/SSH/SCP。CPU专项禁用可见GPU；GPU专项启动前GPU0 free14407MiB/util1%，完整回归启动前free14407MiB/util0%，均满足free≥12288MiB/util≤60%，单队列。完整回归保持GPU0可见，以覆盖既有默认CUDA的source hooks和M4显存统计测试；配对目标长测显式设CPU，新模块专项默认CPU。本次不干扰LXC GPU1生产训练。CI runner未注册，以上均为显式远程验收。

```text
# rmvision，/home/rmvision/venvs/hsi-baseline/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p test_g4_support_target.py -v
CUDA_VISIBLE_DEVICES=0 SUPPORT_TARGET_DEVICE=cuda python -m unittest discover -s tests -p test_g4_support_target.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py
python scripts/train_g4_support_target_short.py --help
# LXC，同上线程设置，CUDA_VISIBLE_DEVICES为空
python /home/lbx/hsi-pioneer271828/transfer-records/issue190-real-preflight01.py
python scripts/check_repository.py --expected-commit e7424559e1bc72c655f511c213e551ced9b95be3
```

CPU/GPU专项均实际运行六角色合成网络：每角色完整12与2+10两条路径，三次同路径选中态精确概率重载和完成0恢复；另有实际CLI2+10+0（support2048/K5）。每个成功专项中本模块合成更新156次，均非真实数据训练。CPU逐项比较恢复后的模型、optimizer、scheduler、history和选中概率精确一致；GPU只要求各自严格重载和状态恢复，不夸为跨路径bitwise一致。adapter正梯度、graph有限梯度及实际参数变化，以及活动M4正梯度和参数变化均核验。

三个显式轴组合还验证support64/K10/model2048、support512/K1/model64、support2048/K5/model512的标签数、头种子与零更新模型构造。错误预算、输入、metadata、输出目录、恢复身份和CLI选项负例通过。原有GPU专用CLI在CPU完整回归中仍跳过，其既有182/186 GPU验收保留，本轮不冒充重跑该项；本轮新增CLI已在CPU/GPU实际通过。

## 真实输入与初始化证据

复用188真实manifest，SHA256 `57d13699414b1c041c5332d834a392e7d3af499684a56886d75e76c3c87d66c6`。外部独立验收驱动固定Xuzhou/support512/K1/model64：9训练标签+45验证标签=54。通过184冻结launch读取相应完整源规格，构造none/容量/残差 × 从头/全微调六种初始化；每种初始state SHA均与184原冻结对应状态完全相同，证明支持标签变化没有改变初始化或源身份。

驱动在CPU上禁止`Adam`构造与所有模型forward；原始GT、query标签均未读取，优化更新0、forward0、GPU上下文未初始化。63个context/源/launch文件前后不变，生产331源码仍干净；计划的六个训练输出目录均未创建。该真实检查只覆盖一个支持/K与模型种子条件的六种初始化，不是全部27组真实初始化、真实12步训练、概率审计或支持集重复效果。

真实回执`runs/g4-support-target-issue190-preflight01/receipt.json` SHA256 `f930862c303f06ac11cb1a8d58c414469c7a3c3e4787dfb8009d30892f0bd13f`，外部驱动`transfer-records/issue190-real-preflight01.py`及日志保存在checkout外。汇总回执`/home/rmvision/issue190-support-engineering01-receipt.json` SHA256 `487c1063ab44cffd467b017e10cc0f936a8782fe6aaabf34c0e1a3312405cf62`，同字节副本`docs/architecture/g4-support-target-engineering-results.json`。该JSON保存准确日志哈希、环境、代码身份和真实六角色绑定；数据、标签张量、权重、运行日志不提交。

## 保留失败、限制及下一依赖

初版`cdf3558c35072ea8a940be407d49d9d73b387c97`六角色更新/恢复通过，但CLI机器输出测试失败：训练进度与JSON混写stdout，产生JSONDecodeError。e742将进度转stderr，并补适配器类型/架构模式约束，未修改算术、阈值、训练预算或选模。原`issue190-cpu02.log`保留，后续CPU/GPU/完整回归按新版本独立验收。

首次完整回归禁用CUDA，577项中出现1失败+1错误+1跳过（317.947s）；新增5项全通过。两个既有用例分别在默认CUDA的source hooks初始化和M4失败探针的显存统计处失败，尚未到预期断言。已核源码原因，保持同一e742版本，改用GPU0可见、MATCHED_TARGET_DEVICE=cpu的既有环境重跑，未加跳过、放宽断言或修改源码；`issue190-full01.log`保留。

首次bundle只公布HEAD，按分支名fetch失败；cpu01实际0测试，不能算通过。已显式fetch HEAD、核对完整SHA后运行cpu02/03。GitHub PR创建transport失败后先GET核无同head PR，再通过REST创建191，无重复POST。失败记录均进入工程回执，不覆盖或挑选原日志。

汇总证据打包时，外部receipt02驱动因抄录的真实回执hash多一位，在写入任何汇总前被断言拒绝；原驱动和失败保留。已用本机Get-FileHash与远程sha256sum核对同一64位原文件身份，receipt03接收该核验值作为显式参数；原真实回执、模型/输入与测试不改，不重新训练。

**有限12步训练工程验收通过。**生产tag `module-g4-support-target-engineering-v1`指实际e742；PR191独立评审，main不合并。真实支持集重复/K精度、空间/传感器泛化以及head-only/light-module尚未验收，本模块不支持任何精度收益或充分收敛结论。

184仍按全54接受→独立331审计→186真实拟合/逐类/曲线/成本复盘→全54冻结同协议query及预登记结论的顺序推进。交付前最新43/54接受、64500/81000更新，第44例HongHu512容量scratch运行（新增none scratch选751/LR1382，三头无零召回），无全批次接受或query；HongHu64残差FT选721/LR1352和残差scratch验证CNN第11类0召回等限制保留，不据此改配方或预算。之后另登记真实支持/K重复；下一独立工程依赖为完整源head-only/light-module适配对照的明确参数/恢复合同，不能由本次两角色工程替代。

本聊天报告位置：2026-11-06 Issue190独立支持集与K目标训练工程模块完成报告。该报告作为本模块交付的一部分，后续不重复。
