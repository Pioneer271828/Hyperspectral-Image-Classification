# WP08 区域跨支路注意力原型

依据主计划 §12.4、实施计划 §10.1 和 §19.9。`RegionMessageVCASAHGCN` 将独立区域消息接入 CNN/HGCN 的中层交互点。CNN 特征按 Q 汇聚，两个方向读取交互前的区域特征，区域消息仍经现有残差门控。地标由当前块 Q 的区域中心在二维网格中分组。五种模式共用同形投影，覆盖局部同容量、稠密注意力、仅压缩、仅路由和组合。组合使用一个共同 softmax，每个查询只选一路原始 KV。候选不足时取实际可用项，并列按源区域顺序。GPU 稀疏核和图边支持研究另有独立任务。

实现基于 [MiTA v6](https://arxiv.org/html/2602.01219v6) §3.2 公式 5–12。核对 [作者仓库](https://github.com/QishuaiWen/MiTA) 及[根 MIT 许可](https://github.com/QishuaiWen/MiTA/blob/main/LICENSE)。作者入口依赖 FlashAttention，采用原论文公式独立实现区域跨注意力。固定 SHA 和文件级许可核查因连接错误未完成，外部代码引入和运行均为零。当前实现逐查询块构建注意力，专家候选使用稳定排序，完整测量须计入这些成本。

授权 LXC `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，分支 `codex/wp08-region-message`，tested SHA `9996c4c389b3c31a7592e6501fe51bdaf353610e`。`python -m unittest tests.test_region_message_attention tests.test_interaction_modules tests.test_backbone_m5 tests.test_repository_delivery`，CPU 36/36，7.411 秒。仓库检查 608 Python、198 JSON。合成测试验证组合路径数值与梯度、分块一致、空区域和预算不足、全候选路由与稠密一致、零强度回归。两个方向经同一次反向与优化实际改变参数，三头在 `load_region_message_checkpoint` 恢复后逐元素一致。尚待真实等预算训练、三种子逐类对照、GPU 延迟显存和单独的图补边机制。当前最大空闲显存 6992 MiB，WP12 的冻结成本批次等待 12288 MiB 门槛。
