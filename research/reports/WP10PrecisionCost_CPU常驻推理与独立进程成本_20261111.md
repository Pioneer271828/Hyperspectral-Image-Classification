# WP10 CPU 常驻推理与独立进程成本

对应研究主计划 WP10 和实施计划 §19.11，承接 INT8-2（Issue #273）、TQ-2（Issue #277）与逐层回退。本轮按上一阶段的原始逐类保护结果固定三个模型包：FP32、第二层 INT8、第一层打包三值。复用项目已有 INT8 与三值算子及来源核查记录。本轮只增加成本测量，无新校准、优化或标签读取。

`compression/precision_cost.py` 复用已登记的输入块，三臂均执行完整模型前向，VCA 缓存关闭，计时包含图构建、量化边界和三值解码。影像解码、输入张量准备和模型载入位于常驻计时之外。以单 CPU 线程在核 48 执行，每臂预热 3 轮，6 种执行顺序各跑 2 轮。测试通过实际 PyTorch 整数卷积钩子确认 INT8 路径被调用，并与原空间入口核对三头输出。

`compression/precision_startup.py` 在独立进程中测权重加载和第一次完整输入扫描。三臂各 3 个进程，轮换执行位置。加载计时开始前统一校验文件哈希，格式自己的恢复校验包含在加载耗时中。峰值 RSS 是整个工作进程的高水位，包含 Python、依赖和输入张量。文件系统缓存状态未控制。

授权 LXC `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，分支 `codex/wp10-precision-cost`。常驻 tested SHA `71e77687ed4379da8e5173f83c5466cb3d743955`，CPU 33/33，7.959 秒。独立进程 tested SHA `f630f75faaaf52ef6b412a507c8f2c7e1c45b6ed`，CPU 34/34，8.309 秒。命令为 `python -m unittest tests.test_precision_cost tests.test_precision_fallback tests.test_int8_model tests.test_ternary_model tests.test_monitoring_dependency_groups tests.test_repository_delivery` 以及 `python scripts/check_repository.py --expected-commit <tested-sha>`。最终仓库检查 578 Python、198 JSON。现有 PyTorch TypedStorage 弃用提示保留。初次远程同步因状态投影提交不在远端而拒绝，补齐该祖先后正常同步。

真实只读工程对照使用 Xuzhou 原开发短测的 5 个登记观测块，共 3920 像素。三个模型包和原始影像均从上一阶段回执核对。常驻阶段 11 个输入哈希不变，耗时 10.684 秒。独立进程阶段共 9 个独立 PID，13 个输入哈希不变，耗时 49.976 秒。按原始时间记录独立重算中位和尾延迟，9 次恢复验证的三头输出哈希均与对应常驻模型完全相同。

| 模型 | 模型包字节 | 常驻 p50 / p95（毫秒） | 常驻 p50 吞吐（像素/秒） | 加载 / 首扫 p50（毫秒） | 进程峰值 RSS p50（KiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 276217 | 203.65 / 213.36 | 19248 | 30.53 / 217.82 | 620708 |
| 第二层 INT8 | 244856 | 209.38 / 212.45 | 18722 | 44.09 / 220.12 | 621264 |
| 第一层三值 | 234951 | 210.51 / 215.16 | 18621 | 36.22 / 221.30 | 620392 |

模型包分别减少 11.35% 和 14.94%。本次两个压缩方案的常驻和首次扫描中位耗时均略高于 FP32，进程峰值区间重叠。此处只覆盖一个两步开发检查点和一台共享 LXC 的单线程 CPU，暂保留 FP32 默认。正式多种子独立验证、已通过通道剪枝后的量化组合和 GPU 成本仍待。现有真实通道候选为空。已登记 54 例成本批次最高空闲显存 6992 MiB，待 12288 MiB 门槛达标。

证据目录均在 `/home/lbx/hsi-pioneer271828/runs/`。常驻 `wp10-precision-cost-xuzhou01/receipt.json` SHA256 `747b5673781532ff9a6cc244a3ed47c881632e243f0324f39a4b734839a1ffc1`，独立进程 `wp10-precision-startup-xuzhou01/receipt.json` SHA256 `05b1d893b5332e9a11d29d551d45c71210050cca3f119e63d6ff47dfb72d0293`。相应驱动 `wp10-precision-cost-xuzhou01.py` SHA256 `50cdf2c1160cc6614138661742491cafbd0f5c8810a2736725bb924577e6a177`，`wp10-precision-startup-xuzhou01.py` SHA256 `08647bb39cb6726b446304c770c04194119b9ac8334ee2f2f35e61680da4e31ad`。结果不并入已登记的 GPU 队列。
