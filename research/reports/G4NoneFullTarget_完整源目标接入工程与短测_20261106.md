# G4 none完整源目标接入工程与短测

对应研究主清单P4/§10.3、模块计划§19，Issue180 / PR181；前置Issue178完整9源已验收，历史native none复用拒绝结论不变。**本模块工程与约定6例真实短接入完整验收通过**，不含1500步完整目标训练或真实精度收益结论。

## 实现与数值边界

`none_full_transfer.py`绑定源producer52f32978419cbeda216a90bfb0ed76c393ef4f5d的完整9模型launch/acceptance/独立audit/protected manifest/result/weight/sidecar，以及held-target、model-seed、1002预算、选模和源图上下文。只复制共享参数；目标头/DomainBN fresh，scratch不接收源。源文件首尾不变。初始化保留原`g4_target_shared_weights_fresh_domains_v1`校准合同，并增加显式完整源绑定字段，不使用旧native/短源loader。

`none_full_target_runtime.py`复用原TargetRuntime的训练更新、优化器、损失和选模；仅校准/验证用已验M4目标v2同一OrderedValidation。新settings/checkpoint绑定数值合同，原rtol1e-5/atol1e-6、分类变化0不改；短协议12与完整协议1500显式分开，完整入口/完整实验尚未搭建。旧模型/配置/默认入口及冻结路径均不改。

`none_full_target_launch.py`、独立CLI及`none_full_target_precheck.py`在首更新前冻结全部6条件、源spec、标签/观测/图和初始状态；worker再次核验。每例2+10恢复、完成0恢复，独立NumPy三头指标/召回、三次精确重载、权威状态/优化器/调度/first-min选步、graph/adapter正梯度及adapter/CNN/graph实际参数变化，最后全输入保护收尾。

## 实际工程验收

实际源码/生产SHA **9a42b21468442714f780badc2accf1b15e07db7e**，本地分支`codex/g4-none-full-target-adapter`；rmvision冻结`/home/rmvision/g4-none-full-target-cli01`、分支`codex/g4-none-full-target-cli01`，解释器`/home/rmvision/venvs/hsi-baseline/bin/python`，GPU0。LXC冻结`/home/lbx/hsi-pioneer271828/checkouts/g4-none-full-target-cli01`、同名分支，解释器`envs/hsi/bin/python`，GPU1。

线程环境`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests`；实际命令与结果：

```text
python -m unittest test_none_full_transfer test_none_full_target_runtime test_none_full_target_launch -v
# rmvision CPU5/5,5.574s
NONE_TARGET_DEVICE=cuda python -m unittest test_none_full_transfer test_none_full_target_runtime test_none_full_target_launch -v
# rmvision GPU5/5,11.673s；LXC GPU5/5,12.332s
python -m unittest discover -s tests -v
# rmvision 542/542,176.929s
python scripts/check_repository.py --expected-commit 9a42b21468442714f780badc2accf1b15e07db7e
# 两环境passed，343Python/78JSON，冻结路径保持，CI runner仍未注册
python scripts/train_none_full_source_target_short.py --help
python research/reproduction/m4g4b/none_full_target_precheck.py --help
# 帮助通过；LXC三个方向实际完整源CPU绑定核验通过，均seed64/selected1002
```

合成测试覆盖共享参数复制、fresh头/DomainBN、完整源/角色/seed/hash负例、非法数值设置、目标初始状态、6条件覆盖/预算/标签/图保护、12步全程与2+10恢复、完成0、三次精确重载及独立复算。合成数据不是研究结果。日志在rmvision `/home/rmvision/none-full-target-cli01-{cpu,gpu,full,help}.log`，LXC `runs/none-full-target-cli01-gpu.log`。

## 保留失败

首版f6c130c608e31be6a8f497d3300fec4b74fac652的CPU测试2通过1错误：新初始化记录替换policy，原BN校准器按既有合同拒绝。共享复制与角色/hash检查通过，但合成训练在首次验证校准停止；没有真实目标训练。保留`/home/rmvision/g4-none-full-target-core01`及`none-full-target-core01-cpu.log`。修复dee51ec38ae83ff549996f0feb8c9d4f1d0f732d只保留原policy并另绑新源协议，CPU3/GPU3/full540通过（4.611/10.446/176.564s）；后续9a完整5/5/542复核通过。未修改校准器或放宽数值门槛。

## 真实短接入完整验收

[预登记](../experiments/G4NoneFullTarget_真实短接入预登记_20261106.md)在86c383584df428f088a6ab7234eaeea123044f3f提交并推送。目标3×方法2、model seed64/head1064/support64，K10训练另V5验证，6×12=72实际更新；观察来自已验无标签目标图，query0。启动前已在聊天告知5–15分钟、资源等待另计。LXC GPU1单队列，原PID265924已结束，目录`runs/g4-none-full-target-issue180-short01`，日志同名`.log`，状态completed6/6。

全部6条件首步前冻结，逐例2+10中断恢复和完成0不变恢复均通过；36独立NumPy头指标/480逐类召回，18次原严格概率/分类门槛且逐元素精确重载，权威训练/优化器/调度状态、首次最小NLL选步均通过。全部72步graph与adapter正梯度且graph有实际变化；每例CNN/graph参数改变19712/16832元素，adapter改变10385–10465元素。源9例、旧none源/目标、M4完整源及短目标、新none短源、实际输入和launch共**7772**个受保护文件/输入首尾不变（不同审计清单有重叠，不相加）。无真实case失败、补跑或门槛变更。

回执`receipt.json` SHA256 **1a34b9830f8333fe21aab312495b4cc95bbe272d3db50fddfdb690de08a1170c**；launch SHA256 **215be586c7b3529f7e5bea42115e9ce70a9ee169b1e54e5fd9022c246b86b081**。总818.525834s（13分38.526秒，含冻结、资源等待、恢复与独立核验），各case成本见下表，不作为受控硬件速度比较。

| 目标/方法 | loss首→末 | fused训练/验证OA% | 验证fused/graph/CNN OA% | 验证零召回类f/g/c | 含核验耗时s |
| --- | --- | --- | --- | --- | --- |
| PaviaU 从头 | 6.803048→6.357929 | 30.00/26.67 | 26.67/20.00/17.78 | 5/5/7 | 102.749 |
| PaviaU 全微调 | 6.922527→5.988845 | 52.22/44.44 | 44.44/22.22/44.44 | 3/4/3 | 104.629 |
| Xuzhou 从头 | 6.589293→6.168862 | 10.00/8.89 | 8.89/44.44/11.11 | 6/5/6 | 110.701 |
| Xuzhou 全微调 | 6.630092→5.884171 | 28.89/22.22 | 22.22/44.44/33.33 | 6/4/5 | 110.876 |
| HongHu 从头 | 9.370262→9.067858 | 11.82/11.82 | 11.82/8.18/7.27 | 17/18/17 | 180.501 |
| HongHu 全微调 | 9.655044→8.931186 | 5.45/3.64 | 3.64/13.64/13.64 | 20/16/17 | 179.730 |

三头训练/验证OA、AA、NLL、每类支持数、召回及零召回类编号完整保存在[结果摘要](../../docs/architecture/g4-none-full-target-short-results.json)，原始完整混淆计数在receipt/result中。每类等额支持，所以此短验证OA与AA相等（仅浮点舍入差）；最低召回均为0，没有用总体分数掩盖小类未拟合。全部选12，loss均下降但大量类别尚未学会，不能称充分收敛。HongHu全微调fused验证低于从头的负向短读数保留；本阶段非效果实验，不据此调配方或提前取query。只有一个模型/支持种子，不计算或伪造跨种子不确定性。

完成后另作CPU零更新记录复核：再次核验全部7772保护文件和每例result/artifact哈希，代码/预算/覆盖/原回执身份一致，19.895s，不读取query。新输出`runs/g4-none-full-target-issue180-review01/summary.json`，SHA256 **b3d98bb0056f5a6c9d245b80ad349849476cce5f1a65ab21aef4bd7c9d6d7cfde**，不覆写原训练文件。该记录复核不同于新增训练或再次独立GPU重算；GPU重载/NumPy指标已由预登记控制器逐例完成。

## 验收结论与交付边界

接入模块约定工程、6/6真实短测、恢复、独立指标/严格重载、保护收尾及训练复盘全部通过，可以交付。第一版合成校准失败原样保留。完整1500步训练、跨模型种子/支持集重复、query精度和M4容量/残差净收益仍未测，不能以本短测代替。实际回归SHA为9a42b21468442714f780badc2accf1b15e07db7e（CPU5/GPU5/full542）；此后只新增预登记、结果和报告/状态，最终文档版本另做干净交付检查，不把文档SHA写成重新运行全量的SHA。

模块完成报告位置：本聊天2026-11-06的“G4 none完整源目标接入模块完成报告”，同时以本文与结果摘要作为持久证据；后续心跳不重复交付本项。

main不合并；下一项登记none18×1500=27000和活动36×1500=54000完整目标工程/实验，保持同支持/验证/图/校准/数值/选模口径。全模型冻结后才新同协议query，不混旧native分数。独立支持集/K/空间/传感器等计划范围保留。
