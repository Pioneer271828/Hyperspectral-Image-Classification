# G4 none同协议源对照：工程与真实短测阶段

对应主清单P4、模块计划§19、Issue178/PR179。本阶段实现显式ordered none源对照的身份冻结、独立short/full入口和完整队列前置门槛。工程与真实短测通过；**完整源模块尚未验收**，后续9例训练及独立审计仍是完成条件。

## 实现与环境

`g4_none_ordered_source.py`复用SourcePretrainRuntime，只在validation_snapshot使用已验OrderedValidation；原none结构、源标签图、损失/采样/训练更新、前端DomainBN校准及scheduler不变。独立研究JSON绑定文件SHA，旧默认与冻结configs不变。

`g4_none_source_launch.py`首更新前冻结完整3/9例群组、源码/研究JSON/launch、另两源context与输入hash、Q/A、fresh初始完整state和共有参数与两活动模式的初始逐元素一致。worker实际加载selective两源并核对；actual build_model在首更新前验证初始state，provenance/checkpoint均绑定launch。两个worker分别只允许8步或1002步，拒绝跨profile或缺失角色、输出路径/上下文/初始身份变化。

短测/工程producer：`a44ce12754dd380e6063aa0a4f1138f1cc5335da`。

- rmvision分支`codex/g4-none-source-cli01`，checkout `/home/rmvision/g4-none-source-cli01`；Python `/home/rmvision/venvs/hsi-baseline/bin/python`。CPU5=4.348s、GPU0 5=9.429s、全量532=171.468s，交付checker329 Python/76 JSON通过。
- LXC分支`codex/g4-none-source-issue178-cli01`，checkout `/home/lbx/hsi-pioneer271828/checkouts/g4-none-source-issue178-cli01`；Python `envs/hsi/bin/python`，GPU1 5=10.232s，checker通过。
- 命令：`python -m unittest test_g4_none_source_launch test_g4_none_ordered_source -v`、`python -m unittest discover -s tests -v`、`python scripts/check_repository.py --expected-commit <上述完整SHA>`。GPU环境设`G4_NONE_DEVICE=cuda`及单卡CUDA_VISIBLE_DEVICES，CPU/GPU均限制数值线程为1。未注册CI，不声称自动CI执行。

完整controller增量producer/tested SHA：`52f32978419cbeda216a90bfb0ed76c393ef4f5d`。只增加campaign与其测试，短测producer全部源码保持。rmvision分支`codex/g4-none-source-campaign01`，CPU门槛2=0.036s、GPU5=10.094s、全量534=174.773s、checker331 Python/76 JSON和CLI帮助通过；LXC同名分支/checkouts下CPU2=0.074s与checker通过。实际完整源码仍冻结52，不随文档更新。

该增量要求真实短测receipt/launch/全部result及artifacts原hash不变、36指标480召回完整覆盖，才允许准备9例full launch；逐操作资源门槛free>=12288MiB/util<=60%，单GPU单队列，不满足等待，不处理其他项目进程。任何失败保留并停止。

## 真实短测与复盘

LXC GPU1，`runs/g4-none-source-issue178-short01`，日志同名`.log`。控制入口`research/reproduction/m4g4b/none_source_precheck.py`，固定已验活动源launch hash取prepared/data_root，保护原none与活动源整个批次。预算3留出方向×seed64×8=24；每例先2步、恢复6步，完成后0步只读恢复。

receipt SHA256：`752301e32fd1585b5a4695adf00c7666be29b10b24fa797db7b9f5ade74e066c`。完整receipt passed，283.576806s；36组三头train/val独立NumPy指标、480逐类召回，9次独立精确重载、3次只读恢复通过；3163个历史文件与输入不变。原rtol1e-5/atol1e-6及分类变化0仍检查，ordered三重载进一步要求逐元素一致；query0。

|留出目标|选中步|两个源的macro验证OA|逐类召回检查数|adapter/CNN/graph变化元素|
|---|---:|---:|---:|---|
|PaviaU|8|17.2518%|186|10465 / 20096 / 17088|
|Xuzhou|2|0.6107%|186|10394 / 20096 / 17088|
|HongHu|4|19.6504%|108|10465 / 20096 / 17088|

每例8步graph正梯度且实际变化；三个共有模块均有真实参数更新。完整逐头OA/AA/NLL/混淆/recall保存在receipt与`docs/architecture/g4-none-source-short-results.json`。全部源验证中存在最小召回0；Xuzhou留出例仅2步选中，短拟合没有稳定提升。8步只验工程与身份，不称源充分拟合、目标精度或M4收益。首/末30窗口在8步短测中重叠，不能拿这两个相同均值宣称loss下降。

## 保留的失败和后续

首核心43候选的JSON位于整个冻结configs/experiments，被checker拒绝，远程source01保留。e8将同内容移到research，原规则不变。新CLI首次fixture把str传给Path helper，第二次缺source_context；远程`none-source-cli01-cpu.log`、`none-source-cli02-cpu.log`保留。均在真实训练前由测试发现，修正后a44重验通过；没有改训练数值或原验收门槛。

完整源预登记见`research/experiments/G4NoneSourceV2_完整源对照预登记_20261105.md`。9模型×1002=9018，先冻结全部角色后执行；每例2+1000、完成0恢复、逐例拟合/小类/梯度/成本复盘及独立全批次audit通过才交付Issue178。已告知预计1–2小时，实际按运行吞吐更新。后续none18目标与活动36目标、全冻结query和支持集/空间等独立重复继续沿两主计划，不混旧native query结果。

独立全批次审计工具版本`6c8a93eb2be0271c37782d254bcc2a8ada55fc20`已部署至rmvision `/home/rmvision/g4-none-source-audit01`和LXC `checkouts/g4-none-source-audit01`，均为独立同名分支、干净checkout。CPU3分别0.032/0.072s，rmvision全量535=172.543s、双方checker333 Python/77 JSON及CLI帮助通过。命令为`python -m unittest test_none_source_full_audit test_none_source_campaign -v`与全量discover。训练producer52所有源码在独立consumer中保持，运行中52不变。

审计入口`research/reproduction/m4g4b/none_source_full_audit.py`固定expected-commit、producer、完整run及launch/acceptance/protected-files三份实际SHA，只接受9例9018和108指标/1440召回的完整接受凭据；从独立checkout重读source context/图、选模/曝光/状态/文件对，复算全部指标与三次精确重载，零更新/query。输出须为runs下全新同级目录。**工具工程通过不代表实际全批次审计通过**；当前训练仍在运行，须结束后读取新receipt。
