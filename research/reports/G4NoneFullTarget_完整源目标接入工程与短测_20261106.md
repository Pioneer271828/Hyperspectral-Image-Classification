# G4 none完整源目标接入工程与短测

对应研究主清单P4/§10.3、模块计划§19，Issue180 / Draft PR181；前置Issue178完整9源已验收，历史native none复用拒绝结论不变。本记录当前为工程通过、真实短接入运行中的阶段证据，**整个模块尚未验收**，不能作完整目标训练或真实精度收益结论。

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

## 真实短接入状态

[预登记](../experiments/G4NoneFullTarget_真实短接入预登记_20261106.md)已经提交。目标3×方法2、model seed64/head1064/support64，K10训练另V5验证，固定6×12=72更新；观察来自已验无标签目标图，query0。启动前已在聊天告知5–15分钟、资源等待另计。LXC GPU1单队列，实际PID265924，目录`runs/g4-none-full-target-issue180-short01`，日志同名`.log`。首步前完成全6冻结；只有6/6 receipt passed、36指标480召回/18精确重载和保护收尾才可验收。当前尚无完成结论，后续以实际receipt补充耗时、三头/类别、拟合和限制。

主分支不合并；下一项只有本短接入通过后才登记none18×1500=27000和活动36×1500=54000完整目标工程/实验。全模型冻结后才新同协议query，不混旧native分数。独立支持集/K/空间/传感器等计划范围保留。
