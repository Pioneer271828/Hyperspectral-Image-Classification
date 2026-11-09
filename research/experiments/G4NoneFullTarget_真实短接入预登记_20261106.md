# G4 none完整源到目标真实短接入预登记

对应研究主清单P4/§10.3、模块计划§19、Issue180 / Draft PR181。此项是完整源权重到目标的工程接入验收，不能判定完整1500步训练或迁移收益。历史native none复用失败结论保持，源模块178已经独立验收，不重复训练。

## 冻结身份与数据权限

- 实际目标生产/测试代码：`9a42b21468442714f780badc2accf1b15e07db7e`，冻结LXC `checkouts/g4-none-full-target-cli01`；提交后的文档增量不替换运行代码。
- 源producer：`52f32978419cbeda216a90bfb0ed76c393ef4f5d`，源目录`/home/lbx/hsi-pioneer271828/runs/g4-none-source-issue178-full01`。
- 源launch：`6e1b979d4c9375f98e5fbcf4871656bb16e59812140746ee3de06a5b101b1a21`；完整9018更新acceptance：`60f70fe924479c20920b9668dc338a6d4faf1446a108e4637baf794566ebd481`；保护清单：`13b71aa2f5c3418047fb4ccdf75bbe72570cba4ad2453141ea8361181e84b613`。
- 独立源receipt：`runs/g4-none-source-issue178-independent01/receipt.json`，SHA256 `6ff38d1915c53286c1800832234afd3991cc879f029317c6dc01a70e39d4aa0b`。全9源、三个seed64实际checkpoint的结果/权重/sidecar/图/源角色/选模/预算均须由新loader核验。
- 目标观察：`runs/g4d-issue151-observations02/<target>/observation.pt`与`record.json`；标签家族：`runs/g4c-issue149-labels01/<target>-64/curator-family.json`。严格复用已验`load_target_context`的adapt与validation release。每类K10训练、额外V5验证；support seed64，目标全景无标签图，不读取query消费者或标签。只哈希保护历史产物不构成query评分。

## 固定六条件、预算与资源

PaviaU、Xuzhou、HongHu各从头/完整源全量微调，model seed64、head seed1064，共6模型。全部6条件的settings、源spec、目标观察/标签家族/阶段hash、Q/A、CPU初始化及共享权重复制身份须在首更新前写入同一launch并绑定hash。worker重新验证实际初始状态；fresh目标头和DomainBN，源头及源域状态不复制。

每例12更新，固定中断2步、恢复10步、完成恢复0；全批72更新，不因精度提前停止、延长、替换种子或重跑择优。原none架构、损失、采样和训练算术不改；仅校准/验证使用与已验M4目标v2相同OrderedValidation。验证步1/4/7/10/12，以首次最小fused验证NLL选模，原ReduceLROnPlateau不改。短入口不能冒充1500步完整训练。

拟使用LXC GPU1单队列；每个子操作和独立复算前重新检查free>=12288MiB、util<=60%，资源等待不干预其他项目。预计5–15分钟，资源等待另计，实际成本在完成后复盘。唯一新目录`runs/g4-none-full-target-issue180-short01`，日志同名`.log`，禁止覆写旧源/目标产物。启动前在聊天告知预算与时长。

## 预登记验收与停止条件

6/6 completed及状态/优化器/调度/逐验证选步与保存结果一致；36独立NumPy头指标/480逐类召回；18次原rtol1e-5/atol1e-6与分类变化0重载，且ordered路径逐元素精确；6次完成0恢复不得修改产物。逐步graph/adapter有限正梯度、graph实际变化和adapter/CNN/graph参数实际变化均有证据。记录三头train/val完整指标及小类，保留短拟合、末步选中、零召回等限制。

全9新源及其原保护清单、独立源审计、旧none源/目标与活动源/短接入产物和实际输入纳入首尾hash保护，更新0的独立核验只能写本次新目录。任何失败保留日志/已执行预算，写failed停止，独立诊断/版本登记后再决定后续；不静默放宽、不把result文件存在当通过。全6 receipt passed才允许完成Issue180工程验收与报告；尚不含query、完整1500步目标及真实收益。

下一依赖是独立完整入口、none18×1500=27000及活动36×1500=54000配对目标预登记与训练。全比较模型冻结后才作新同协议query；旧native分数不混用。支持种子/K/空间/传感器及其余必测范围不删减，main仍待评审。
