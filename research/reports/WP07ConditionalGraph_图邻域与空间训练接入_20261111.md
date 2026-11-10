# WP07 条件多尺度图邻域与空间训练接入

关联 Issue #386、PR #387，依据研究主计划 §11、实施计划 §19.8 和原方案 §4.6。图分支在一次像素到区域的软分配 Q 上构建半径 1 和 2 的稀疏邻域，两条路径共用边打分与两层 HGCN 参数。图头在两层传播之后混合区域表示，再用 Q 回到像素。平均、内容、物理元数据和联合模式在同一拓扑规则上比较，固定模式只构建一个半径。CNN 与图的开关独立。

双半径在一个块对角稀疏矩阵中传播。若使用 BatchNorm，两种范围的区域共享当次归一化统计，仅更新一次。本次空间训练短测使用既有 GroupNorm 协议。条件缺失时，元数据模式回退到均匀权重，联合模式回退到内容权重。

空间开发训练使用 `transfer.development_training.adapt(..., multiscale=MultiscaleTrial(...))`。模型沿用同一初始共享权重、头、训练与验证划分、采样、损失、更新预算和首次最低验证 NLL 选模。训练从实际输入波段取得条件，当前空间训练入口只自动绑定波段和已有 FWHM。覆盖率与 GSD 可经模型前向传入，真实流程仍需要可信的数据来源与登记。使用 `transfer.multiscale_training.load_multiscale_checkpoint` 从同一 `.pt` 和架构 sidecar 严格恢复。

授权 LXC 环境为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，分支 `codex/wp07-cnn-multiscale`。图原型 `b87dc035e8f81052a1f5607b7730cbeb39d8d460` CPU 19/19，通过共享 Q、实际边数、固定半径 1 与基线逐元素一致、平均结果、条件扰动及梯度检查。训练接入 `7773e135f901be59b0a31aa46c063bc90127e9ac` 执行 `python -m unittest tests.test_conditional_multiscale tests.test_development_training tests.test_run_provenance tests.test_backbone_m5 tests.test_repository_delivery`，CPU 28/28，6.599 秒。仓库检查通过，610 Python、198 JSON。每臂两步的合成测试确认关闭时训练和验证逐项一致、联合选择器权重更新、相同采样以及严格恢复后验证重算一致。真实精度、逐类收益与硬件成本仍待完整配对。

范围候选和融合公式来自项目原方案 §4.6。相邻文献 MixHop 使用邻接幂组合邻域，其结果作为技术背景。核对了 PMLR 论文页与作者仓库身份，本轮作者仓库固定版本及文件许可未确认。实现复用本项目原有稀疏图构造与 HGCN，外部引入文件为零。[MixHop 论文](https://proceedings.mlr.press/v97/abu-el-haija19a.html) · [作者实现](https://github.com/samihaija/mixhop)
