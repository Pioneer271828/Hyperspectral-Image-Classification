# WP09 已知类源模型训练接入

主计划 §13.2 和实施计划 §19.10。承接 Issue #359 的完整已知区域预处理。`scripts/train_known_source.py` 从固定准备回执和源输入重算源区域统计、全局坐标和监督索引，逐项核对准备好的张量。只用源区域训练原生三头模型，非连续的类别 ID 按登记顺序映射到分类列。模型从同一 M8 GroupNorm 结构和明确种子初始化，损失与 Adam 参数沿用登记的空间训练配置。保存固定最终步权重，以便校准与开发评价独立使用各自数据。输出回执、权重、架构 sidecar 和现有开放集生产入口所需的语义声明。

授权 LXC `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，分支 `codex/wp09-known-source-training`，实际测试 `a9255c23e751a2934426fa7df514b7aa45922eb2`。`python -m unittest tests.test_known_source_training tests.test_known_context_preparation tests.test_open_set_producer tests.test_registered_open_set tests.test_open_set` 通过 CPU 32/32，37.502 秒。仓库检查通过，576 Python、198 JSON。合成 CLI 检查覆盖 24 个完整标注源像素、12 个监督位置、三个已知类别和两次实际优化更新。校准与评价输入在优化前从测试临时目录移出，未释放评价标签，成功重载原生三头及绑定的语义声明。首版 `7411438860ed490746a107454e6e24f1ef480ad0` 测试布局将登记放在输出父目录，隔离守卫拒绝。最终测试将登记归入独立输入目录，并增加源归一化范围溢出检查。远端日志 `runs/wp09-source02-tests.log`。

此入口对新模型记录已知类训练来源。真实语义、完整区域标注的真实性和额外预算仍需数据侧凭证。待用真实合格区域训练并接续冻结开放集评价，检查已知类稳定性、未知召回与逐类结果。工程短测没有提供开放集效果数据。
