# WP07 条件多尺度 CNN 实现

Issue #386 对应研究主计划 §11 与实施计划 §19.8。近期 WP11 主要补充登记、权限和冻结入口。本项转回架构实现，增加 `transfer/conditional_multiscale.py` 中可选的 `ConditionalMultiscaleVCASAHGCN`。它在完整三头模型的第一层 CNN 中执行实际的膨胀深度卷积。

- 膨胀率 1、2、3 共用原深度卷积权重，保持输出通道和空间尺寸。实现固定尺度、平均、内容、物理元数据和联合选择。内容使用整块池化。物理条件编码使用有效的已核实波长、FWHM、显式覆盖率、GSD 和缺失状态。仅元数据模式缺条件时回到平均，联合模式缺条件时由内容选择。此版本把近似波长按缺失处理。
- 以授权 LXC `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python` 验收分支 `codex/wp07-cnn-multiscale`，完整提交 `f2c043e7333cfd8ac166b4ed1abf4a6db7ffa2ef`。`python -m unittest tests.test_conditional_multiscale tests.test_int8_model tests.test_repository_delivery` 通过 18/18，5.806 秒。`python scripts/check_repository.py` 通过，608 Python、198 JSON。PyTorch 在既有测试中发出 TypedStorage 弃用提示，测试通过。
- 三头实际前向验证膨胀率 1 与原模型逐元素相同。膨胀率 2、3 改变 CNN 输出，图头结果逐元素一致。条件分支、内容分支和深度卷积收到非零梯度。覆盖缺失回退、波段置换和依据架构配置及状态严格恢复。测试数据为合成输入。
- 当前实现覆盖 CNN 分支。真实训练入口、同 Q 的半径 1/2 图分支、真实尺度失效证据和留出条件对照待补。真实 GPU 成本批次仍按已登记条件启动，最近检查最大空闲显存 6996 MiB，低于 12288 MiB 门槛。WP04 真实短测按原安排后补。
