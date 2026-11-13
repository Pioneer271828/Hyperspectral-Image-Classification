# WP05 HongHu 三臂完整拟合执行与 CUDA 准入衔接

承接 Issue #466 和 Draft #468，依据主计划 §9–10 与实施 §19.6。WP07 §19.8 仅作已有严格 CUDA 算术复用。原 P4 v4 执行仍固定 `3a6381cb4b0c089a10123292bc831a3a613e6106`。本候选基于该生产提交，九例模式独立登记，测试源码 `65dd7864ad2d54e658b1efb91e408bb99e687c04`。为与旧 CUDA 草案的代码差异可直接审查，发布分支补入其已存在的文档提交 `f2ae3e4210bffa5610f8e830e699a78f375576cc`，合并前后无代码或配置差异。

## 九例注册和运行

`python -m transfer.p4_three_arm_fit` 提供 `prepare`、`run`、`accept`、`review`。执行复用 P4 campaign、单一 worker、原 Adam、ReduceLROnPlateau、DomainBN、学习图、实际删带、固定支持和曝光序列。真实集合固定 HongHu devtrain/devval、support 2048、K5 scratch、模型种子 64/512/2048，三臂为 `none/continuous/anchor_only`。每种子的骨干和三头初态相同，两个覆盖臂同 194 参数与恒等初态，种子组三臂固定同一物理 UUID。

每例 1,500 次，分进程到 2、30、1,500，50 次验证按 30 步，保存首个最低 `training_stride` 融合 NLL。归一化来自冻结 devtrain 的对应保留列，模型图以当前带子集重新构建，保留原坐标和角色空间。七条件与原 1,500 步预检逐项对照。五个未见缺带条件与完整视图均在已经选中的状态复算。原五臂 20 例短准入和 120 例全预算入口保留原预算、分段和校验规则，旧 prepare 不接受新模式。

真实预检 SHA `0676d13736c0c6be4106171205e873fde9d81e9124245a9d0fe5dbb62244b2ca8`。独立写入目录，硬件引用旧实体资格；前向、更新均 0。运行入口已实测拒绝该预检。在正式运行前，另有审核的执行释放文件绑定完整源码、GPU、该版本独立资格、指定目录、新研究 Issue 与原 2,400 步接受链。两张或三张分配设备各自需实体资格，同时依原门槛准入。13,500 次实际更新为 0，原 180,000 保持暂停。

## 恢复、诊断与指标

旧 worker 每步保存状态。新模式的推进回执记录边界前后 SHA、RNG 和下一块，独立审计对照分进程恢复记录与最终状态，复算 named Adam moments、梯度/变化标志、LR、采样曝光和首个最低选中事件。完成重入复核状态字节，新增 0。未通过任一案例审计或少于九例时，汇总拒绝。原始 GT、query、旧独立确认集均未读取。

独立模型重载重新前向，核对已选状态概率和三头混淆、类别支持、召回及零召回，报告初态、选中、最后状态，含 Q/A 摘要与显存/耗时。训练首末 loss、最后五分之一训练 loss 与验证融合 NLL 的每百步斜率、50 次融合 NLL、三头 OA、末步选中比例与逐类曝光均列入回执。用已有 WP12 计时摘要以及原 P4 同步计时。五个未见条件平均融合 OA 是主要指标，分别与 `none` 和 `anchor_only` 按三个种子配对。两对照各需正均值、至少 2/3 正收益，继承七条件的 OA/AA 0.5pp、训练支持不高于 25 的类别三种子平均召回 2pp 保护。合成结果只验工程。

CUDA 模式沿用 P4 `cuda_deterministic_fp64_sparse_weight_accum_v4`，含已有 `adaptive_bin_mean` 拦截与稀疏边权梯度 FP64 累加。关闭路径保持原生，WP05-META 旧路径的严格池化后向限制和 Issue #459 的端到端 80 个 adapter 梯度/CPU32 超限另保留。此模块未修改池化器、容差、生产默认或 WP07 模型。

## 授权 LXC 实绩

| 测试 | 实际工作量与结果 |
| --- | --- |
| 新工程专项 CPU | 4/4，59.302 秒。含缺例、错臂/种子/UUID、重复输出、未接受汇总、错误续接哈希、旧入口误用、两对照主指标及逐类保护。 |
| 9 类和 22 类 CPU | 各完整 9 例，2+1+3，共 54+54 合成更新，27 路径段各运行独立进程。各 189 个七条件三头读数，逐类召回分别 1,701 与 4,158。全 9 接受后独立汇总，完成重入 0。 |
| 22 类长程 CPU | 1,500 个合成优化步骤，进程边界 2+28+1,470，50 次验证，独立首个最低事件为 30，末步未被选中，完成重入 0。推进耗时 309.44 秒，独立复算 1.99 秒。验收 `b5d7cdbda2aa75500ba5012abc51a5c4c01be7f8ccbc32b28ad273d368614b53`。 |
| RTX 4090 GPU 6 | 新候选精确源码同几何 52 夹具/438 比较，容差占比最大 0.9836203；资格 `18c983bc742ac4c72325d415d21d14168645dd63f582819ccd6c5b1f1102ffbd`。22 类九例同卡严格算术 54 次合成更新，完成 0，推进时长合计 75.03 秒、独审 21.61 秒，训练更新显存峰值分配 80,199,680 字节、预留 98,566,144 字节。仅为有界合成工作量。 |
| 旧 P4 相关回归 | 54 通过、1 个旧的显式双实体 CUDA 测试按 CPU 条件跳过，1066.067 秒。包括旧 1,500 步连续/分段、120 例合成接受和资源门槛；独立比较旧 20/120 例标识及预算在冻结生产与候选中一致。 |
| 真实只读预检 | 独立 NumPy 抽取授权可见数据的 137 个块观测，重算八个训练/验证范围的波段列、归一化、坐标和像素 ID。独审 SHA `7e41acc2d9b0359a2be03aebcb0f297d11239418e6285c7fa3148739513721ba`。 |

迭代 03 和 04 分别核算。03 的新专项 4/4（54.129 秒），相关旧回归 54 通过、1 个 CPU 条件跳过（1061.768 秒）。04 的两项各为 4/4（59.302 秒）与 54 通过、1 个 CPU 条件跳过（1066.067 秒）。两轮各自的 9/22 类 CPU 边界均为 54+54 步，长程 CPU 1,500 步，GPU 6 同卡九例 54 步，另各有新专项内含 108 步。每轮独立进程测例 1,662 步，计入新专项共 1,770 步，两轮合计 3,540 次新三臂合成更新。相关旧回归内的历史协议测例单列，真实优化仍为 0。

较早迭代的 9 类测试夹具初态类别不一致，以及完成态重复进入的边界指纹读取错误，已修正并完整重测。保留两次失败日志。原生产冻结分支未改。

## 原 2,400 步真实准入

沿用原 `a827dec0a6087d1eaf990ca086f1c28dd27e501ba0ae81c3b2c18434672e9ce5`，核对精确冻结提交、两份物理资格与两份同卡恢复；原 20 例、support 2048/model 512、K1/K5、scratch/shared、五臂与两条各 60 步路径均保持。阶段核查中 GPU 1 空闲为 455–1,255 MiB；GPU 6 即使空闲，两张登记卡始终未同时达到 12,288 MiB、利用率不高于 60% 的原门槛。未生成真实执行目录，实际 0/2,400，未产生真实三头或拟合结果。资源达标后原登记预计 20–40 分钟加独立审计。

## 命令与证据

进入精确干净 LXC checkout，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，Python 3.10.20、PyTorch 2.7.1+cu128。设置 `PYTHONPATH=.:tests`、`PYTHONDONTWRITEBYTECODE=1`，BLAS、OpenMP、MKL 各 1 线程，CPU 使用 `CUDA_VISIBLE_DEVICES=''`。合成 CUDA 采用物理 UUID 共享租约及 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

- `python -m unittest -v test_p4_three_arm_fit`
- `python -m test_p4_three_arm_fit fixture --classes <9|22> --variant bounded_synthetic --expected-commit <SHA> --out <external-dir>`，长程选择 `--classes 22 --variant long_synthetic`。
- `python -m unittest -v test_p4_coverage test_p4_exposure test_p4_coverage_batch test_p4_cuda_admission test_p4_stable_cuda test_p4_frozen_diagnostics test_p4_frozen_band_probe test_p4_full_budget_runtime test_repository_delivery`
- `python -m transfer.p4_cuda_qualification --parent <release> --expected-commit <SHA> --gpu <physical-uuid> --protocol cuda_deterministic_fp64_sparse_weight_accum_v4 --out <external-dir>`，复算脚本 `commands/verify_wp05_three_arm_fixtures01.py` 在证据包。
- `python -m transfer.p4_three_arm_fit prepare --parent <release> --prior <prior-receipt> --reference <original-1500-scope> --registration-dir <new-dir> --execution-dir <new-sibling> --gpus <UUIDs> --qualification <receipt-per-card> --expected-commit <SHA>`，仅预检。正式执行使用单独审核的 release 与 SHA。原 2,400 步检查命令收在 `commands/wp05_fit_check_p4_admission01.sh`。
- 仓库检查为 `python scripts/check_repository.py --expected-commit <完整 SHA>`，状态从 JSON 通过 `python scripts/check_repository.py --write-status` 投影。授权 LXC 手工执行。

最终证据 ZIP `/home/lbx/hsi-pioneer271828/runs/wp05-three-arm-fit-evidence02.zip`，SHA-256 `973dd90f81c42328f7ea0de5d2e49e16176b3ef49de2e37046f2c7a41695bcdb`，清单 `60f0625b47f3898a299cb8eb306e6b60321e32f8366b713efb26951d09c34be8`。主审逐项核对 649 个成员及四份候选与主入口 checker 日志，确认 `65dd7864ad2d54e658b1efb91e408bb99e687c04` 至 `eb0206657842922c5a97e75c76b330157fc56558` 仅为文档和投影变动，接受有界拟合交付。PR #469 保持未合并，Issue #470 继续跟踪真实依赖。训练状态保存在容器 checkout 外，ZIP 不含权重和影像。

上一批 #468 的主审已接受候选冻结与两张物理卡同卡合成恢复。补收未入原 ZIP 的两份最终 checker 原日志。候选 `92586890cc3d9efd2e187cb8aab959a3fcd1f4db`，日志 SHA `80ce67006c14f44c577cb4acafce74b0f488abd9d3d9b02b382eac74650706a6`。主入口 `d2c8caef823189d1bc09d430fe646c3ad464042e`，日志 SHA `4076b4d3cc90ae80f13515270c518c95da245430fe850b5080543b57458043eb`。本包真实研究下一依赖是原双卡短准入完成和新九例范围正式审核，开源 21 仓库及 OSS-C-R 结论沿用原记录。
