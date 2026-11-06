# G4完整54目标配对实验：训练、独立验收与冻结query报告

2026-11-07。对应研究主清单P4 §10.3、预算§17.2、记录§19和模块实施计划§19，Issue184 / PR185；预登记、182生产工程、186复盘和204分析工程均复用已验收版本。本报告替换先前运行中快照；旧进度和失败记录保留在Git历史及原运行目录。

**结论：完整实验验收通过，预登记主要与四项次要收益门槛均未通过。** 残差全量微调相对容量控制的等权场景宏fused OA差为 **0.330636 ± 1.613334个百分点**（三个模型种子的样本SD），三个配对种子差为-1.086878 / -0.007431 / 2.086215pp，仅1/3为正。HongHu本地类别ID8平均召回下降3.053561pp，超过2pp保护门槛。保留这些负结果，不换种子、不依query重新选步或追加预算，不能据均值正差宣称M4净收益。

## 范围与冻结合同

54模型＝3目标paviaU/xuzhou/whu-honghu × 模型种子64/512/2048 × scratch/full FT × none/capacity/residual；每例1500次更新，共81000，none27000、活动54000。每例2中断+1498恢复，完成恢复0。源9+18完整模型的27054更新、既有短目标216更新单列，不加入当前81000。

支持种子64、每类K10训练另V5验证，head seed=model seed+1000。三模式9组共同fresh参数已在首更新前固定；迁移只复制共享骨干，目标新头/DomainBN保持fresh。原三头损失、采样、Adam/调度、全图无标签观察/Q/A和首次最小fused验证NLL选步不变。Ordered验证/校准/最终query，原rtol1e-5/atol1e-6、分类变化0和三次同路径精确重载不放宽。容量8320参数与残差8256参数相近，不能称严格等参数或等计算。

这是固定目标标签预算、有无标签整图上下文参与的少样本传导适配（历史代号G4）；不是零样本、纯源无标签TTA、未见传感器泛化。三个模型种子只有一个支持集，不是支持集重复，也不支持显著性声明。预登记原文件/外部副本保持不变：
[预登记](../experiments/G4MatchedFullTarget_完整54目标配对预登记_20261106.md)，analysis-plan SHA256 `0bd71779546fd5ee0cbaab57b7b3639a1c378bcf9d1379f8afb1a80231a92dee`。

## 实际环境、版本与命令

授权LXC `lbx@10.69.37.243:1222`，专属根 `/home/lbx/hsi-pioneer271828`，解释器 `envs/hsi/bin/python`；Python3.10.20、Torch2.7.1+cu128/CUDA12.8、RTX4090。源码/数据/产物分离；Windows仅编辑、读取、Git和SSH/SCP。main未合并，专用CI runner未注册，不声称CI运行。

| 阶段 | 实际tested/执行完整SHA与现场 | 验收 |
|---|---|---|
| 冻结训练/原全批次审计/冻结query | `331d718567b1a9726934f445665999df972392b1`；生产`checkouts/g4-matched-target-issue184-production01` / `codex/g4-matched-target-issue184-production01`；独立`checkouts/g4-matched-target-pipeline03` | 原331干净不改；54/81000接受，审计324头记录/4320召回/162精确重载 |
| 最后三例外部资源控制器 | `5db71e82ad3fc36f3cef7046a22c1162be756097`；`checkouts/g4-resource-issue198-production01` | 只调度，原331 worker/冻结模型合同不改；见198交接报告 |
| 资源终验纯JSON/hash消费者 | `5dcc6964a25858f6793995126ec4606dfb8ad533`；`checkouts/g4-resource-final-audit01` / `codex/g4-resource-final-audit-accept01` | CPU9/9，0.010s；checker322Python118JSON；无GPU/模型推理 |
| 资源兼容真实复盘导出器 | `2df9d2882364137791cef89e92bcb6cc3e29283b`；`checkouts/issue184-resource-review02` / 同名codex分支 | CPU9/9，86.684s；checker360Python83JSON；真实54PNG/54CSV/18组，零更新/query |
| 资源兼容预登记分析器 | `bdb4b3f14cc108e632b986ad7bffec7cd1a70bfc`；`checkouts/issue184-resource-analysis01` / 同名codex分支 | CPU11/11，15.627s；checker384Python93JSON；原七个算术/报告函数对204源码逐段相同 |

原生产331工程验收复用[182报告](G4MatchedFullTarget_完整协议工程阶段记录_20261106.md)：rmvision CPU12项11pass+1GPU-only skip、GPU16/16、full558项557pass+同1skip；LXC GPU12/12含真实合成1500 CLI。它们是此前工程证据，本回合没有重跑该完整工程回归。新增资源/成本元数据修订运行上表完整专项及实际54链路；未声称新增GPU或全仓库回归通过。186旧工程/204旧工程的已交付报告不重复交付。

实际完整训练命令在原生产331已执行一次，绝不重启：
```text
PYTHONPATH=.:tests envs/hsi/bin/python -u research/reproduction/m4g4b/matched_target_campaign.py
 --expected-commit 331d718567b1a9726934f445665999df972392b1
 --registration /home/lbx/hsi-pioneer271828/runs/g4-matched-target-issue184-registration02/registration.json
 --registration-sha256 0957aa3c5177be4faa38d6c68576a1db8e12679ab0d3de0bca67a19db2245017
```

CPU专项分别为 `python -m unittest discover -s tests -p test_g4_resource_final_audit.py -v`、`test_g4_matched_target_review.py`、`test_g4_matched_target_analysis.py`；均在对应干净Linux源码执行，单线程/无bytecode。新资源审计验证缺阶段/重复/预算/零等待/卡身份/51旧行/保护pin/NaN等负例；复盘修订验证资源哈希绑定、未知字段和数值行仍严格相等；分析修订只接受固定2df/5dcc证据和准确成本口径，全部修改和测试在首次真实query释放前完成。

本回合独立原331命令：
```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:tests envs/hsi/bin/python -u research/reproduction/m4g4b/matched_target_full_audit.py
 --expected-commit 331d718567b1a9726934f445665999df972392b1 --run <runs/full01>
 --launch-sha256 0c447087a9679538e0a792aa72cb033197e743bc959cd096ac2e831e46eafdb1
 --acceptance-sha256 451186fc7cab3ebd72bfc8c480aaf165e8095e6c2ce12cb858d76450d5a79681 --gpu 1 --out <runs/independent01>
```

下列为实际后处理命令记录。所有输出已存在，保留原目录，不在原址重跑。统一使用已授权venv、单线程与无bytecode。

```bash
vca_root=/home/lbx/hsi-pioneer271828
r="$vca_root/runs"
p="$vca_root/envs/hsi/bin/python"
run="$r/g4-matched-target-issue184-full01"
launch_sha=0c447087a9679538e0a792aa72cb033197e743bc959cd096ac2e831e46eafdb1
accept_sha=451186fc7cab3ebd72bfc8c480aaf165e8095e6c2ce12cb858d76450d5a79681
audit="$r/g4-matched-target-issue184-independent01/receipt.json"
audit_sha=88e656e4330ca19828e9e2c63682d2c2463717923c684615dfbb07ce605cc81b
amend="$r/g4-matched-target-issue184-resources01/amendment.json"
resource_audit="$r/g4-matched-target-issue184-resource-independent01/receipt.json"
resource_audit_sha=983bcd5c10d9fef815c40ae9b1939c0e6b166042b3bda480398327e2ff1e8609
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
cd "$vca_root/checkouts/g4-resource-final-audit01"
"$p" -u g4_resource_final_audit.py --expected-commit 5dcc6964a25858f6793995126ec4606dfb8ad533 \
 --run "$run" --launch-sha256 "$launch_sha" --acceptance-sha256 "$accept_sha" \
 --audit "$audit" --audit-sha256 "$audit_sha" --amendment "$amend" \
 --amendment-sha256 8e8004aa6b56da7d4b59b0526cbe5248cc588de76138e624313e0c1e5f9be414 \
 --out "$r/g4-matched-target-issue184-resource-independent01"
cd "$vca_root/checkouts/issue184-resource-review02"
CUDA_VISIBLE_DEVICES='' "$p" -u research/reproduction/m4g4b/matched_target_report.py \
 --expected-commit 2df9d2882364137791cef89e92bcb6cc3e29283b \
 --run "$run" --launch-sha256 "$launch_sha" --acceptance-sha256 "$accept_sha" \
 --audit "$audit" --audit-sha256 "$audit_sha" --resource-audit "$resource_audit" \
 --resource-audit-sha256 "$resource_audit_sha" --out "$r/g4-matched-target-issue184-review01"
cd "$vca_root/checkouts/g4-matched-target-pipeline03"
"$p" -u scripts/evaluate_g4_matched_target_full.py --expected-commit 331d718567b1a9726934f445665999df972392b1 freeze \
 --run "$run" --launch-sha256 "$launch_sha" --acceptance-sha256 "$accept_sha" \
 --audit "$audit" --audit-sha256 "$audit_sha" --out "$r/g4-matched-target-issue184-frozen01"
freeze="$r/g4-matched-target-issue184-frozen01/freeze.json"
freeze_sha=7dd7e87334766d74005d14345e994fc2e700e109f6d8141b462004831496b48c
CUDA_VISIBLE_DEVICES=1 "$p" -u scripts/evaluate_g4_matched_target_full.py --expected-commit 331d718567b1a9726934f445665999df972392b1 query \
 --freeze "$freeze" --freeze-sha256 "$freeze_sha" \
 --data-root "$vca_root/datasets/snapshots/joint-honghu-20261103-issue97/dataset" \
 --prepared-manifest "$r/base97a-honghu-issue97-20261103-prepared01/manifest.json" \
 --prepared-sha256 414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b \
 --gpu 1 --out "$r/g4-matched-target-issue184-query01"
cd "$vca_root/checkouts/issue184-resource-analysis01"
CUDA_VISIBLE_DEVICES='' "$p" -u scripts/analyze_g4_matched_target_full.py \
 --expected-commit bdb4b3f14cc108e632b986ad7bffec7cd1a70bfc \
 --query "$r/g4-matched-target-issue184-query01/receipt.json" --query-sha256 33d7d17130ee94834b9ab3bbc21e3e36917f3a638beb4e1dc188c99d64f81b54 \
 --freeze "$freeze" --freeze-sha256 "$freeze_sha" \
 --review "$r/g4-matched-target-issue184-review01/summary.json" --review-sha256 547f7a748a6ed7540366a0f8afe851ae7ad33da37d08891213cc64dc156af7d0 \
 --out "$r/g4-matched-target-issue184-analysis01"
```

query的cohort从固定54 launch中的none FT源launch追溯并核对原保护hash。全部54统一ordered，未混旧native27分数或根据query改变模型/预算。分析只消费已完成的query计数、固定review和freeze；未再反序列化原GT、释放新query或执行模型推理。

## 全批次接受、独立审计及资源交接

- 54/54 ×1500，2754验证事件，324 train/val头记录、4320逐类召回、162同路径精确重载及完成恢复0通过；所有模块要求的正梯度/实际参数变化保留到逐例记录。
- 原331独立审计保护14189个文件/输入，1443.803s，新增更新0/query0。独立选中态NumPy指标及重载通过；非选中事件核hash/history/first-min/LR/权威状态，不扩称全事件概率独立复算。
- 资源终验保护14217项，原51行/5763文件、7930输入不变；最后三例12个成功阶段及1次零更新resource_wait完整保留，分配的4500更新全部完成。receipt的remaining_updates4500是交接时分配量，当前剩余0。
- 旧273458在安全51边界无训练子进程/锁后交接；旧wrapper退出137、77555s是计划内控制器退休，不是数值训练失败。末三例在GPU1/3/7单卡队列完成。事件日志只能证明所记录阶段，不构成独立完整历史并发时间线。
- all54冻结保护14036文件、171个关键边界文件，query54模型/162套头指标通过，优化更新0。后续分析14202文件不变；完整三头、类别支持计数/precision/recall/F1/混淆、每景/seed与样本SD保存在完整分析JSON。

## 真实拟合与曲线复盘

真实导出54PNG/54CSV/18组，14191输入文件不变，耗时215.471s，更新0/query0。三张18例联系图检查全部54布局，并单独查看Pavia64残差FT、Xuzhou2048容量FT、HongHu64残差scratch和HongHu2048-none FT原图；标题、六面板、选步/LR和小类记录齐备。部分柱图图例覆盖柱区域，精确逐类值以JSON/CSV为准，不冒充发表排版验收。

38/54例选中预算末步，42/54例末段149更新窗口的验证NLL仍改善，12例转差；最早选511。不能说全部充分收敛，也不能因此追加步数。HongHu2048 none/capacity FT选571/511，残差FT选1021，保留早选及过拟合迹象。唯一选中验证零召回为HongHu64残差scratch的CNN头本地类11；fused/graph未因此删除或重新选模。每例首末30 loss、三个头/小类、选步、实际模块变化/梯度、LR及成本见结果JSON和外部曲线。

## 冻结query结果及预登记判定

下表为三个模型种子均值±样本SD，单位%。完整三头、每seed、逐类和混淆在[完整分析JSON](../../docs/architecture/g4-matched-target-full-analysis-results.json)，程序直接生成的五项门槛/逐景失败表见[自动分析报告](G4MatchedFullTarget_冻结query预登记自动分析_20261107.md)。本表仅简列fused，不删除其他头证据。

| 目标 | 角色 | 模式 | fused OA | fused AA |
|---|---|---|---:|---:|
| paviaU | scratch | none | 89.1943 ± 0.2606 | 94.4923 ± 0.9922 |
| paviaU | scratch | capacity | 89.6360 ± 0.3858 | 94.8950 ± 0.7842 |
| paviaU | scratch | residual | 89.9666 ± 1.1228 | 94.8167 ± 1.2613 |
| paviaU | full FT | none | 88.6705 ± 0.3834 | 95.4442 ± 0.1357 |
| paviaU | full FT | capacity | 88.2601 ± 2.0115 | 94.8455 ± 2.0439 |
| paviaU | full FT | residual | 88.9879 ± 0.7185 | 95.0220 ± 0.6104 |
| xuzhou | scratch | none | 95.7159 ± 0.0204 | 96.6425 ± 0.1756 |
| xuzhou | scratch | capacity | 95.6999 ± 0.1739 | 96.5750 ± 0.0697 |
| xuzhou | scratch | residual | 95.7852 ± 0.1732 | 96.7062 ± 0.0834 |
| xuzhou | full FT | none | 96.5358 ± 0.0278 | 97.6122 ± 0.1125 |
| xuzhou | full FT | capacity | 96.2657 ± 0.3253 | 97.2948 ± 0.1258 |
| xuzhou | full FT | residual | 96.2716 ± 0.1852 | 97.5554 ± 0.0826 |
| whu-honghu | scratch | none | 81.9273 ± 0.6557 | 80.7155 ± 0.5189 |
| whu-honghu | scratch | capacity | 82.6410 ± 0.8229 | 81.3404 ± 1.2251 |
| whu-honghu | scratch | residual | 83.3210 ± 0.5318 | 81.3696 ± 0.5022 |
| whu-honghu | full FT | none | 80.8258 ± 1.5357 | 83.9917 ± 0.5313 |
| whu-honghu | full FT | capacity | 79.9662 ± 3.7384 | 83.1579 ± 1.3678 |
| whu-honghu | full FT | residual | 80.2245 ± 1.7672 | 84.4131 ± 0.8537 |

主要判据须宏OA均值>0、3/3配对种子>0、每景AA降≤1pp、HongHu OA降≤1pp、各景每类三种子平均召回降≤2pp同时成立。主要比较平均宏OA为正，AA/HongHuOA保护通过，但种子一致性及HongHu类8保护失败。

| 独立命名主张 | 宏OA差均值±样本SD（pp） | seed64 / 512 / 2048差（pp） | 全部门槛 |
|---|---:|---|---|
| 主要 residual FT − capacity FT | 0.330636 ± 1.613334 | -1.086878 / -0.007431 / 2.086215 | 未通过 |
| residualFT_minus_noneFT | -0.182743 ± 0.741944 | -0.870675 / -0.280978 / 0.603425 | 未通过 |
| none_FT_minus_scratch | -0.268416 ± 0.788260 | 0.610375 / -0.913112 / -0.502510 | 未通过 |
| capacity_control_FT_minus_scratch | -1.161601 ± 1.435109 | 0.338749 / -1.302473 / -2.521079 | 未通过 |
| x_residual_FT_minus_scratch | -1.196308 ± 0.848458 | -0.800970 / -2.170290 / -0.617664 | 未通过 |

scratch架构对照、capacity FT−none FT和差中差均按描述性结果保留，不替换主要失败；四项次要也不得互相替代。当前协议不支持“残差净收益”或“源预训练普遍改善”结论，局部正差仍保留。后续模块沿计划独立登记，不用本query追调当前模型或择优重训。

## 成本、失败与限制

原wrapper START_UTC到最终acceptance文件mtime的源文件系统跨度为78954.249s（约21h56m，含最初freeze/资源等待/控制器交接）；它与进程段计时分别报告，不作受控GPU速度结论。旧段77555s、资源receipt1385.002044s、acceptance分段1394.522523s口径不同，不能把最后约23分钟当整批成本，也不把并行case耗时之和47851.743411s当墙钟。

| 本回合阶段 | 记录耗时（s） |
|---|---:|
| 原全批次独立审计 | 1443.802761 |
| 资源终验 | 70.347432 |
| 真实复盘导出 | 215.470532 |
| all54 freeze CLI墙钟 | 265.747602 |
| 原query receipt / driver含启动检查 | 1372.673808 / 1406.824741 |
| 分析receipt | 121.003013 |

保留registration01工程hash多写一位、首更新前拒绝的历史；没有真实模型更新。原186 ad60拒绝资源新增字段的只读探针保留，修订2df要求独立资源审计且仍逐字段比较全部原数值行。预冻结复制summary hash时多一字符（65位）由本机/远程预检拦截，freeze CLI/输出/query均未开始；随后从原文件取同一实际64位hash，原产物/门槛未改。QA01缺系统字体，在任何联系图/归档产生前失败，QA02用现有Matplotlib字体，无安装或环境改变。交付汇总首次换行转录断言在生成完整JSON前拒绝，partial delivery01保留，delivery02从同一已通过原始receipt重新组装；这不是数值失败或择优重跑。TLS重试保留，SCP只在exit0后消费。

支持seeds512/2048、K1/5、独立空间/传感器/波长质量、head-only/light-module、M4+门控四组及两计划其他机制仍必测。当前只完成固定支持64/K10/整景传导协议；真实训练充分性与效果分别判定。未来E0等消费者若写死旧ad60/732身份，须先登记兼容本已验2df/bdb分析身份的修订，不能称旧接口已消费此次真实结果。

## 证据与交付

外部LXC根均为 `/home/lbx/hsi-pioneer271828/runs/`，下表真实hash从原JSON生成，以实际回执和文件hash为准：

| 证据 | 相对runs路径 | SHA256 |
|---|---|---|
| launch | g4-matched-target-issue184-full01/launch.json | 0c447087a9679538e0a792aa72cb033197e743bc959cd096ac2e831e46eafdb1 |
| training acceptance | g4-matched-target-issue184-full01/training-acceptance.json | 451186fc7cab3ebd72bfc8c480aaf165e8095e6c2ce12cb858d76450d5a79681 |
| 原full audit | g4-matched-target-issue184-independent01/receipt.json | 88e656e4330ca19828e9e2c63682d2c2463717923c684615dfbb07ce605cc81b |
| 资源final audit | g4-matched-target-issue184-resource-independent01/receipt.json | 983bcd5c10d9fef815c40ae9b1939c0e6b166042b3bda480398327e2ff1e8609 |
| 真实复盘 | g4-matched-target-issue184-review01/summary.json | 547f7a748a6ed7540366a0f8afe851ae7ad33da37d08891213cc64dc156af7d0 |
| all54 freeze | g4-matched-target-issue184-frozen01/freeze.json | 7dd7e87334766d74005d14345e994fc2e700e109f6d8141b462004831496b48c |
| ordered query | g4-matched-target-issue184-query01/receipt.json | 33d7d17130ee94834b9ab3bbc21e3e36917f3a638beb4e1dc188c99d64f81b54 |
| 分析receipt | g4-matched-target-issue184-analysis01/receipt.json | 9b9d2e1390fa54915cc34bac87e6826cc5cb138cc746ae1c5012bead8fec441b |
| 完整分析 | g4-matched-target-issue184-analysis01/analysis.json | 580b0b5c358d1818af2a97e9f0c7c69cc393d1ac9d54473a535318ef2a80752d |

仓库保留[完整阶段结果](../../docs/architecture/g4-matched-target-full-results.json)、完整分析及[视觉QA记录](../../docs/architecture/g4-matched-target-review-visual-results.json)。本机外部用户副本：`F:/E2E_VCA_worktrees/issue184-review01-export`（全部54图/CSV/summary/audit），`F:/E2E_VCA_worktrees/issue184-evaluation01-export`（冻结/query/分析与交付摘要）。源码候选分别PR235/236/237，原186/204已交付模块不重复宣称新交付；仅报告此次实际全批次消费及必要元数据修订。primary只同步报告/结果/状态，不合候选Python，main不合并。

验收完成范围为54真实训练、独立全批次审计与资源终验、真实拟合/逐类/成本复盘、全部54冻结统一query及原判据分析；**研究主要/次要收益不成立**。vca已在训练完成后先PAUSED并核验，本回合正常工作承接全部后处理，验收/一般测试不重新启用定时器。下一项继续已登记Issue234 / Draft PR238位置门控有界训练/恢复工程（CPU7/GPU7已通过，全回归669通过及1项既有跳过，真实配置核验待完成），不等待用户确认，不发起新的真实长训练。
