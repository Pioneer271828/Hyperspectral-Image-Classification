# WP11 同划分轻模型登记、恢复与发布

Issue #379 对应研究主计划 §22 WP11 和实施计划 §19.12，接续 Issue #377 的真实完整模型与开发包。`scripts/run_wp11_light.py` 将既有 CPU 轻头训练和发布器接入固定登记。训练输入只包含隔离开发包和完整模型工件。入口复用完整模型历史检查，固定源码、权重、开发划分、外部审计证据、预算、Adam 选项和原逐步开发验证选模规则。

- `plan` 只构建契约。`run` 依据固定登记和绝对步数执行；恢复需要显式提供检查点哈希，沿用已验证的优化器、随机状态及验证历史。每个分段使用新目录；异常保留分段工件和失败回执。完成后独立回读最终和选中态验证，发布并严格回读选中轻模型。
- 使用 LXC `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`、`codex/wp11-registered-light` 对应的干净提交 `17f08da3d3e97080a9b992034ac3107a8417e5d9`。`python -m unittest tests.test_wp11_registered_light tests.test_light_head_training tests.test_trained_light tests.test_wp11_confirmation tests.test_wp11_label_custody tests.test_wp11_held_aside tests.test_dynamic_cost_pairs tests.test_route_controls tests.test_gate_calibration tests.test_repository_delivery` 通过 51/51，10.913 秒。`python scripts/check_repository.py` 通过，602 Python、198 JSON。
- 合成独立整跑 3 步和分段 1+2 步的状态、历史、选步与轻模型状态一致，完成恢复新增 0 步。越界预算、越界历史、替换审计、覆盖原输出负例均通过。以上验证学习与发布链路的工程行为，真实分类效果、真实早退精度和成本待验。
- 在徐州实际完整模型和 78 块开发包上执行 CLI `plan`。6.900 秒，确认 90/45 支持和固定模型身份。12 步、种子 64、学习率 0.0001、类别平衡用于候选短测登记；实际 optimizer 更新、query 和新标签读取均为 0。仅用已发布的开发标签，未访问独立角色标签。

证据位于 `/home/lbx/hsi-pioneer271828/runs/wp11-registered-light-validation01`。真实零步 `real-zero-step/registration.json` SHA256 为 `b40a6be99c4016d96ce1592f9bf166ec97488b5592e46507a96518c0af71c558`。`synthetic` 留存合成 CLI 分段与发布回执。

下一依赖是轻模型实际训练与复盘、校准和确认标签分别授权及逐类覆盖核查，随后冻结门控与固定路径比较。本轮 GPU 最高空闲显存 6000 MiB，54 例常驻成本登记保持待资源。WP04 的真实短测后补。
