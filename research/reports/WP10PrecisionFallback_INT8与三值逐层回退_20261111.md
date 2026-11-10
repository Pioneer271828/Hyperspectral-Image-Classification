# WP10 INT8 与三值逐层回退

对应研究主计划 §22 的 WP10、实施计划 §19.11，承接 INT8-2（Issue #273）与 TQ-2（Issue #277）。`compression/precision_fallback.py` 分别实现 CPU INT8 PTQ 和打包三值权重的逐层回退。每种方式比较浮点基准、两种单层配置和双层配置，校准只读登记的 `devtrain` 块，候选使用同一 `devval` 观测和监督位置。三值对照固定已登记的比例且无需激活校准，运行时解码权重后执行浮点算术。沿用项目已有 INT8 和三值实现及其原论文核查记录，无新增外部代码或依赖。

沿用 MON 原融合头保护规则。OA、AA 和任一类召回最多下降 0.5 个百分点，训练支持数不超过 25 的类别要求零新增错误。三个头均记录逐类混淆和保护结果。双层方案要求两个单层与组合均通过。通过的配置按保留低精度层数选取，同层数按预设的层顺序选取；失败层保留原 FP32 权重。结果是固定开发验证候选，独立数据上的验收另行执行。

授权 LXC 分支 `codex/wp10-precision-fallback`。INT8 首轮 `f593753cce38b537108b98362f4711000984e49e` 29/29，三值接入 `717b59ed100397baf51b274a99001953c22f8f97` 30/30。最终回归 `f235fb20d7c0ffb9491b9a4b16e053e060f1b1c2` 通过 31/31，7.355 秒，仓库检查 575 Python、198 JSON。用 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`、CPU 单线程运行 `python -m unittest tests.test_precision_fallback tests.test_int8_model tests.test_ternary_model tests.test_monitoring_dependency_groups tests.test_repository_delivery`，再运行 `python scripts/check_repository.py --expected-commit f235fb20d7c0ffb9491b9a4b16e053e060f1b1c2`。覆盖整体指标改善仍因新错被拒、两个单层均通过但联合失败、全拒回到浮点、验证数据或门槛篡改、低精度模型保存恢复。PyTorch 发出既有 TypedStorage 弃用提示。

用 `717b59ed100397baf51b274a99001953c22f8f97` 只读复用 Xuzhou 原两步工程检查点，45 个原开发验证点，9 类各 5 点。FP32 混淆矩阵与历史选中态一致，融合 OA/AA 均为 20.00%。PTQ 只用原 `devtrain` 块 1、4，三值固定比例 0.05。此次新增优化 0，query 读取 0，10 个输入哈希不变，两个选中混合模型的三头验证预测在恢复后完全一致，两组比较与恢复合计 5.049 秒。

| 配置 | 仅第一层 | 仅第二层 | 两层 |
| --- | --- | --- | --- |
| INT8 PTQ | 17.78%，第 7 类新增 1 错，拒绝 | 20.00%，通过 | 20.00%，第一层单独失败，拒绝 |
| 打包三值 | 24.44%，通过 | 20.00%，第 2、7 类各新增 1 错，拒绝 | 22.22%，第 7 类新增 1 错且第二层单独失败，拒绝 |

数值为融合 OA。第 7 类损失一个验证样本时召回下降 20 个百分点。开发样本少、模型只训练两步，此处用于检验保护机制。正式多种子等预算恢复、独立确认、端到端成本和已合格结构剪枝后的量化组合仍待。现有真实通道筛选名单为空，保留全部通道。54 例成本批次的登记文件哈希仍为 `e373b6648314e3e0514506ecf78fa8dcb582791a24f98fa23d2060f89a84c978`，最高空闲显存 6992 MiB，等待已登记的 12288 MiB 门槛。

证据目录 `/home/lbx/hsi-pioneer271828/runs/wp10-precision-fallback-xuzhou01/`。`receipt.json` SHA256 `ddc1e7b8287f11c2a350aec765d401bc1935455f2ea3432821ff6ab1b00ac057`，`registration.json` SHA256 `d575d5477afb7bc7fa17d2fadbfa854d1d11b17d93301c59b5f2cd7f9b43a3ed`。驱动 `runs/wp10-precision-fallback-xuzhou01.py` SHA256 `d3f73cc78ce12e34cb86fb90dac07dbe740a7d4f167f26e4fd794c3f00b7187f`，回归日志 `runs/wp10-joint01-tests.log`。
