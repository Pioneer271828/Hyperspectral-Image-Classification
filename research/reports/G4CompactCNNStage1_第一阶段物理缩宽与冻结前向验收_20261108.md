# G4 CNN 第一阶段物理缩宽与冻结前向验收

## 计划与实现

对应两份 2026-10-28 计划中的 §19 监测与压缩、MON-4 结构通道映射。继通用新旧索引映射后，这一步把依赖闭包落到 G4 的实际 CNN 骨干。Stage 1 点卷积输出、深度卷积权重与偏置及其分组、Stage 2 输入 BN 及点卷积输入一起缩宽。Stage 2 仍输出 64 通道，图分支与三头输出接口保持原有维度。支持全局 BN 和域 BN，显式拒绝中层交互、边更新以及跨通道 GroupNorm 的配置。候选接入口需要固定哈希的、已通过 MON 可逆组试验的回执。当前紧凑模型禁止用旧架构配置导出检查点。

## 验收

- LXC `lbx@10.69.37.243:1222`，`/home/lbx/hsi-pioneer271828/checkouts/g4-compact-cnn-stage102`，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。分支 `codex/g4-compact-cnn-stage1`，tested SHA `0f17c8a9626df57fac471f2b382a737c895193e9`，验收前工作树干净。关闭 CUDA，CPU 单线程运行。
- 执行 `python scripts/check_repository.py --expected-commit 0f17c8a9626df57fac471f2b382a737c895193e9`，通过 328 个 Python 文件与 128 个 JSON 文件检查。对下列文件分别运行 `python -m unittest discover -s tests -p "<文件名>" -v`。

| 套件 | 结果 |
| --- | --- |
| `test_g4_compact_cnn_stage1.py` | 3/3，覆盖带非零统计与偏置的全局/域 BN、完整合成骨干、拒绝不支持的拓扑 |
| `test_compact_channel_mapping.py` | 3/3 |
| `test_monitoring_dependency_groups.py` | 3/3 |
| `test_weight_masks.py` | 6/6 |

- 在同一干净 checkout 用 `python -` 调用 `compression.g4_cnn_stage1.compare_geometry`。输入为既有逐例审计通过的 `paviaU-s64-none-source_pretrained_head_only` 检查点及冻结 PaviaU 验证划分。调用前固定登记、参考 launch、检查点、sidecar、逐例回执与数据身份，调用后重验检查点、上下文与代码。采用有序前向，比较固定验证集 45 点，诊断通道为 `[2, 95, 126]`，屏蔽点为 `backbone.CNN_Branch.1.norm` 的输出。宽度 128→125，模型参数 61,696→61,276。融合、图、CNN 三头的概率最大绝对差均为 0，分类变化均为 0，满足固定阈值 1e-5 和分类完全一致。只读检查用时 17.057 秒，包含模型加载与比较，优化更新 0，query 标签读取 0。

## 首次失败与边界

首次真实对照在 `8ce23e214378b7a41bbcba3e3a7671816744679b` 上拒绝了 BN 前屏蔽。BN 运行均值与偏置让置零通道在 BN 后重新非零，融合头 405 项概率中有 380 项不匹配，最大绝对差为 0.1350755692。修复把屏蔽移动到 BN 后，并补上原合成用例漏掉的非零 BN 统计及偏置。原始失败日志已保留。

本次通过的是受支持 G4 拓扑的物理缩宽、依赖闭包和单个真实检查点的冻结前向验收。测试通道用于诊断，真实 MON 持续候选及其逐类、稀有条件保护仍待验收。GPU、全量回归、紧凑检查点重载、等预算恢复、query 和实测成本均待完成。参数减少不提供延迟、显存或精度收益证据。下一项接通独立紧凑检查点的保存与重载，并验证固定前向一致性。

## 证据

以下位置均相对于 `/home/lbx/hsi-pioneer271828/runs`。

| 文件 | SHA256 |
| --- | --- |
| `g4-compact-cnn-stage102.log` | `8cfbc44a758b91494a744e80377ead596925b9f2240d12dca005d729c012d742` |
| `g4-compact-cnn-stage1-real01.log` | `eaa6d5d1d8473285b2a0283280d08d761ad0920ee1ade1315e87bbf289d1ae57` |
| `g4-compact-cnn-stage1-real02/probe-plan.json` | `c74e5ec21250efb54b8488b6f43519f352288d22fb4d9feb644fe73c958e36f6` |
| `g4-compact-cnn-stage1-real02/receipt.json` | `406987023a98f4571c6ea6c1f080bae8322855fcb46c2fd8a91eb328c194ebe8` |
| `g4-compact-cnn-stage1-real02.log` | `f8e5e9a99b8cae0d4fd0eb7252cd4f1d81cc019fafce65aa141f3125201c31f9` |

本聊天 2026-11-08 的 G4 CNN 第一阶段物理缩宽进度为本阶段交付记录。
