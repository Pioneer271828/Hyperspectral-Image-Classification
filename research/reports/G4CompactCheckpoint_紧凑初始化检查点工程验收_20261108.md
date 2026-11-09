# G4 紧凑初始化检查点工程验收

对应两份 2026-10-28 计划中的 §19 监测与压缩、MON-4 和结构裁剪后的等预算恢复。该步提供独立的紧凑初始化格式，以便后续恢复测试从可追溯的物理缩宽状态开始。文件同时携带 G4 架构、完整新旧通道映射和张量。重载先核对文件及映射 SHA256，再按架构重建，严格核对每个张量及映射的状态指纹。文件以原子方式发布，已有产物拒绝覆盖。保留旧模型检查点入口。该格式专供恢复训练的初始化，恢复期间的优化器、采样器和调度器状态需要单独接入。

## 验收

- 实际环境为授权 LXC `/home/lbx/hsi-pioneer271828/checkouts/g4-compact-checkpoint02`，使用 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。分支 `codex/g4-compact-checkpoint`，干净 tested SHA `809699f79f732d586ae68543b45052fddf930726`。CPU 单线程，关闭 CUDA。
- `python scripts/check_repository.py --expected-commit 809699f79f732d586ae68543b45052fddf930726` 通过，检查 329 个 Python 文件、128 个 JSON 文件。
- 使用 `python -m unittest discover -s tests -p "<套件>" -v`，依次运行 `test_g4_compact_cnn_stage1.py` 3/3、`test_compact_channel_mapping.py` 3/3、`test_monitoring*.py` 40/40、`test_weight_masks.py` 6/6、`test_masked_checkpoint.py` 4/4，合计 56/56。
- 回执驱动的合成数据试验使用完整 G4 骨干，检查 128→127 通道、深度卷积分组、域 BN 和映射，重载后三头输出逐项相同，模型状态指纹相同，CPU RNG 保持一致。已拒绝覆盖、文件字节变化、错误映射身份和改变了的初始化张量。
- 初版 `a2049e5c9126a7d033aa61bfe135b35b3f5e00d2` 中 PyTorch 拒绝用以点开头的临时文件路径写归档。改用已打开的文件句柄写入并刷盘后再原子发布，原失败日志保留。

本阶段是初始化保存和重载工程验收。真实候选及其小类保护、真实紧凑文件往返、GPU、全库回归、等预算恢复、冻结 query 与实测成本待验收。本阶段优化更新 0、query 标签读取 0。下一项接通紧凑模型的等预算恢复短测。

| LXC 证据文件 | SHA256 |
| --- | --- |
| `/home/lbx/hsi-pioneer271828/runs/g4-compact-checkpoint01.log` | `79b293312c557594521010717761d90efe8848998fdbf1fb74d11bcf013d4b1c` |
| `/home/lbx/hsi-pioneer271828/runs/g4-compact-checkpoint02.log` | `a7815aaa1c0377386c1d155e0600282de2c201238921ce66baa73f7d6aa0d650` |

本聊天 2026-11-08 的 G4 紧凑初始化检查点进度为本阶段交付记录。
