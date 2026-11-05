# 完整M4源到目标接入工程与原生复测失败

Issue #174 / Draft PR #175，对应模块计划第19节第三项和研究主计划P4。复用内部已验收M4/目标训练与新域初始化组件（参考 `docs/references/m4-transfer-reuse.json`），无新外部源码、算法或许可主张。源模块Issue #172已独立完成18模型18036更新，源生产固定 `01c29c82e92ce0a529749c53962202e9bf0dc5fe`；本模块仍在验收。

## 完成的接入

`m4_full_transfer.py`要求源launch、全18例acceptance、独立审计、result、weight及sidecar的固定SHA256，匹配留出目标、模型种子、M4模式、图与1002步预算。复用原M4初始化器生成相同新目标结构与新头，保留新目标DomainBN，仅复制共享骨干及M4参数。旧8步源初始化入口保持原接受范围。

`m4_full_target_runtime.py`与独立12步CLI复用目标原生训练/校准、最小验证NLL首个最优选模。输入限G4无标签目标图、支持种子64的K10训练及额外V5验证，query消费者未调用。PaviaU/Xuzhou/HongHu对应90/90/220训练及45/45/110验证像素。源码、目标初始状态、所有12例配置与来源在首个更新前冻结。

## 原生目标协议v1真实预检

源码 `fb3a8b11e887423d50614d8668719759fd257034`。rmvision交付检查、CPU1（4.381秒）/GPU1（9.386秒）、全量516/516（158.699秒）通过；LXC专项复制和GPU路径复测通过。目录 `runs/m4fulltarget-issue174-real01`，主日志同名.log。

预登记3目标×2活动模式×从头/全量微调共12例，每例12更新（2中断+10恢复），拟144更新。实际4例各12，共48更新；3例通过完整独立预检，第4例PaviaU/残差/全微调首次原生收尾有completed结果，但后续三重载中的原门槛复测失败，余8例未启动。因此无全模块receipt，不能把4个result.json当作12例验收通过。

原失败为810元素概率张量中3项超差，最大绝对差4.09782e-6，原 `rtol=1e-5,atol=1e-6` 保留。首三例含容量从头/微调和残差从头，各12步M4正梯度且有实际更新，完成恢复0更新/文件不变。短步数只验收路径和合同，不能作为完整迁移精度结果。

## 零更新诊断

独立外部驱动归档于 `research/reproduction/m4g4b/full_target_failure_audit.py`，提交 `bc7a4b9daaabe25a6185f5b9a86d193223e4f276`；执行时导入冻结fb3a8b1 checkout的实现。核对4例目标文件、18例源训练artifact及源/目标输入共2118项不变，载入模型与权威best_model精确相同，无优化器更新、query访问或模型修改。

固定native、CSR、按列序逐行FP32求和三路径，各3次独立重载。native重复间6次元素超差，CSR重复间6次，最大差均7.33137e-6，分类变化0。固定求和路径重复间逐元素一致；相对旧保存概率仍累计51项超差、最大差7.42078e-6，分类变化0。累计计数跨头/划分/重复，不是独立像素数。

回执 `runs/m4fulltarget-issue174-failure-audit01/receipt.json` SHA256 `cb086fc0f3b693db508a915ba613be61ae75a9f7a14711a4093a1828c1c991ac`。旧目标v1失败及completed结果均保留，没有重写旧概率或反复重试择优。

## 显式目标协议v2与重验

目标v2把 `fp32_sorted_row_segment_sum_v1` 显式用于目标校准和验证，训练仍用原更新路径。原概率/分类门槛保留；数值协议写入settings及checkpoint元数据，读取时验证。关闭开关继续使用原生v1路径。新协议会影响校准缓冲、验证损失与选模，使用全新目录重做12例，而非将旧48步视为新协议完成。

源码 `1d5c66492ce7d4ad33036f43beec22c267ab33a9`，rmvision CPU2（10.159秒）/GPU2（21.875秒）及全量517/517（164.221秒）、交付检查通过。涵盖两活动模式×从头/微调、CPU精确继续、GPU实际更新、目标初始状态、算术元数据篡改、三次精确重载及只读完成恢复；LXC GPU2复测通过。日志 `/home/rmvision/m4-full-target-1d5c664-{cpu,gpu,full}.log` 和LXC `runs/m4-full-target-1d5c664-gpu01.log`。

真实新目录 `runs/m4fulltarget-issue174-ordered02`，仍12例144更新，已提前告知预计5–15分钟，逐例遇失败即停止，原始产物在checkout外。最终回执、指标/召回、文件不变核验与完整复盘完成前保持模块未完成。

下一步在接入工程通过后，先审计G4历史none基线与目标v2的标签、图、初始化、预算、校准、选模及数值兼容性。该兼容性未建立前，不把旧18例none结果与新目标v2直接合称公平完整配对；根据审计决定复用范围或登记必要的新对照。完整目标36例54000更新另行冻结，query只在全批次通过后按冻结协议评价。

## v2真实接入最终验收与复盘

12/12例、144更新全部通过，耗时701.311秒（约11分41秒）。每例2步中断+10步恢复包含在原12步预算内，完成恢复0更新，所有已完成目标文件前后不变。独立复算72项训练/验证头指标和960项逐类召回；每例3次独立重载与选中保存概率逐元素一致，共36次。全部12例每步M4梯度为正且有实际参数变化，全部选中第12步。

冻结源码/实际测试SHA `1d5c66492ce7d4ad33036f43beec22c267ab33a9`，LXC `checkouts/m4-full-target-ordered-v2`、GPU1、Python3.10.20/Torch2.7.1+cu128。2120个受保护源训练、源审计与目标输入文件不变；目标侧query计分0。回执 `runs/m4fulltarget-issue174-ordered02/receipt.json` SHA256 `abc7c5a5c464f4cf976f07b50f5a48ece23f44901b5d29cfc0da88c100fea809`。小型提交汇总 `docs/architecture/m4-full-target-short-results.json`，原始权重/日志在checkout外。

实际工程回归命令（rmvision现有解释器、线程数1）：

```bash
python scripts/check_repository.py --expected-commit 1d5c66492ce7d4ad33036f43beec22c267ab33a9
python -m unittest discover -s tests -p test_m4_full_target_runtime.py -v
M4G4_DEVICE=cuda python -m unittest discover -s tests -p test_m4_full_target_runtime.py -v
python -m unittest discover -s tests -v
```

真实命令使用同SHA、独立外部目录，关键参数为：

```bash
CUDA_VISIBLE_DEVICES=1 python research/reproduction/m4g4b/full_source_target_precheck.py \
 --expected-commit 1d5c66492ce7d4ad33036f43beec22c267ab33a9 \
 --source-root /home/lbx/hsi-pioneer271828/runs/m4g4bv2-issue172-full01 \
 --source-audit /home/lbx/hsi-pioneer271828/runs/m4g4bv2-issue172-independent01/receipt.json \
 --source-audit-sha256 8e0617ec639f75f042d6037702269e96512335d41d4ed2c3ccb2c24921ef7d6f \
 --observation-root /home/lbx/hsi-pioneer271828/runs/g4d-issue151-observations02 \
 --family-root /home/lbx/hsi-pioneer271828/runs/g4c-issue149-labels01 \
 --out-dir /home/lbx/hsi-pioneer271828/runs/m4fulltarget-issue174-ordered02 --ordered-validation-v2
```

四种模式组合均在三种目标完成路径验收。PaviaU训练OA约30.0–36.7%、验证26.7–35.6%；Xuzhou训练10.0–21.1%、验证8.9–20.0%；HongHu训练6.4–11.8%、验证2.7–11.8%，各例损失下降。12步的支持集拟合与新头适配明显不足，全部选最后一步。该模块确认完整源参数正确接入、目标头/域更新、保存与恢复及数值合同；完整迁移精度、容量优势与逐类保护仍须同条件完整预算实验。

原生v1失败现场及零更新诊断继续保留。验收结论为显式目标v2接入工程通过；原生v1重复性限制仍存在。下一模块先登记none基线身份/数值/选模兼容性审计，不能因当前短测通过直接宣布旧none对照可公平复用。

