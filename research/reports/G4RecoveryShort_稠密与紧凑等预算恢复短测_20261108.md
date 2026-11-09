# G4 稠密与紧凑等预算恢复短测

对应两份 2026-10-28 计划的监测与结构压缩、8b 和实施计划 §19。接续 PR #258 的真实物理缩宽及 PR #259 的紧凑初始化检查点。使用同一份已审计 G4 初态，稠密组直接使用原结构，紧凑组仅移除前一次冻结几何核验固定的通道 `[2,95,126]`。真实短测使用诊断切片，合成端到端另走带哈希的紧凑文件入口。两组从各自空白 Adam 状态开始，既有优化器状态未继承。

## 执行与验收

- 授权 LXC `/home/lbx/hsi-pioneer271828/checkouts/g4-recovery-short01`，`/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，CPU 单线程，关闭 CUDA。分支 `codex/g4-compact-recovery-short`，干净 tested SHA `d980f2ccf507c679b5bedb6227a7cee9b9a1193d`。`python scripts/check_repository.py --expected-commit d980f2ccf507c679b5bedb6227a7cee9b9a1193d` 通过，331 个 Python 文件、128 个 JSON 文件。
- 远程执行 `python -m unittest discover -s tests -p "<文件模式>" -v`。新增 `test_g4_recovery_short.py` 1/1。相关套件 `test_g4_compact_cnn_stage1.py` 3/3、`test_compact_channel_mapping.py` 3/3、`test_monitoring*.py` 40/40、`test_target_training.py` 5/6、`test_target_bn_calibration.py` 5/5、`test_weight_masks.py` 6/6、`test_masked_checkpoint.py` 4/4。合计 67 通过。旧测试 `test_optional_hooks_preserve_existing_source_and_reject_metadata_collision` 默认调用 CUDA，在当前 CPU 环境报无可用 GPU。同命令在干净父版本 `0332153c44839589a3708207032294e227baaf3f` 复现，保留失败日志。另以合成 MON 已通过组生成紧凑检查点，从固定哈希文件启动配对恢复，12+12 步与精确重载通过。
- 真实短测在上述 checkout 以 `python -u -` 调用 `compression.g4_recovery_short.run_recovery_pair`，固定 PaviaU `paviaU-s64-none-source_pretrained_head_only` 已审计初态、Issue249 登记与逐例回执、先前诊断回执、数据记录和标签族。两组共用固定图和样本，支持集每类 10 点共 90 点，验证每类 5 点共 45 点。运行前后核对 9 个受保护文件及内存数据指纹 `3ab53f35a5aeb63c0db16b6a3ae7bd8ac2360c2227f31d888fe438e441841f0bc`。
- 每组按共同调度各执行 12 次 Adam 更新，优化器步后钩子和 CNN 权重 Adam 状态均计数 12。三头损失和优化器配置一致，骨干/头学习率为 0.00005/0.0005，12 步轨迹相同。各自在第 1、4、7、10、12 步用有序算术校准 BN 并验证。选模用融合头验证 NLL 的首个最小值，两组均选中第 12 步。分别重载两组所选状态，训练和验证三头概率指纹逐项一致。配对 CPU 墙钟 551.753 秒，同容器也执行了相关回归，耗时只用于记录本次执行。

| 真实 PaviaU | 参数 | 验证融合 NLL（步 1→12） | 选中验证 OA（融合/图/CNN） |
| --- | ---: | ---: | --- |
| 稠密 | 61,696 | 0.535560→0.476786 | 73.33% / 84.44% / 73.33% |
| 紧凑 | 61,276 | 0.543461→0.483589 | 73.33% / 82.22% / 77.78% |

验证样本每类等量，表中 OA 与 AA 相同。逐类召回保留于两组 `validation.json`。紧凑组融合 NLL 高 0.006802，图头 OA 低 2.22 个百分点，CNN 头 OA 高 4.44 个百分点。两组仍在末步选中。诊断通道只用于打通短测，真实 MON 候选及其逐类保护、GPU 与全库回归、等预算完整实验、冻结 query 和实际速度仍待验收。本次 query 标签读取 0，长训练暂缓。

## 证据

产物在 `/home/lbx/hsi-pioneer271828/runs/g4-recovery-short-paviau01`，完整步骤及三头逐类数据分别位于 `dense/` 和 `compact/`。独立只读检查重算了步数、首个最小 NLL、学习率与校准输入身份，并重验了选中权重和保护文件。

| 文件 | SHA256 |
| --- | --- |
| `plan.json` | `672e90f3c35364123f30c4ded971a67946c1b131d35a73de69f7125183539568` |
| `receipt.json` | `d54b7d74d0775c3d96cf4039a747b27db67e718f94e422792c9f6ad6eefc6b66` |
| `execution.json` | `cc9423d4885ae83fa5ae540ad27d304296c0b77d8c3d4c269a7dea918919c3e4` |

同级日志 `g4-recovery-short01.log`、`g4-recovery-short-regression01.log`、`g4-recovery-short-regression02.log`、`g4-recovery-short-baseline01.log`、`g4-recovery-short-pinned01.log` 与 `g4-recovery-short-paviau01.log`。本聊天 2026-11-08 等预算恢复短测进度为该阶段交付记录。
