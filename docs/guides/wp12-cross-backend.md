# 冻结跨模型配对入口

在授权 LXC 中用 `scripts/run_wp12_cross_backend.py capture` 读取受权 Bank 和冻结候选，用 `scripts/ingest_wp12_evidence.py --cross-backend-review` 在干净检出独立消费。两条命令均需要 `--runs`、`--out`、`--expected-commit` 和路径与完整 SHA-256。生产使用 `--input PATH SHA256`，消费使用 `--capture PATH SHA256`。输出命名为授权运行根内的全新文件。

## 发布和配对

- 按[独立确认发布](independent-attribution-package.md)提交原 Bank、历史、三角色和持有人批准。持有人可在原清单中显式列出 `observation_views`，可选 `full`、`central_gap`、`matched_dispersed`。批准绑定整个清单的 SHA-256。只批准基础分类用途且未列视图时使用全带。可信空间几何沿用可选附件与评价角色，成本按原权限使用观测。
- 比较清单的 `schema` 为 `wp05_wp08_frozen_comparison_v1`，并包含 `intake`、`authority` 及 `groups`。固定引用只含绝对 `path` 和 `sha256`。每个群组包含 `seed`、`protocol`、`arms` 以及有序的 `pairs`。每对先写基线臂，再写候选臂。每个臂是单例回执引用。
- 群组协议为 `budget`、`data`、`exposure`、`initialization`、`class_axis`、`band_axis`、`arithmetic`、`selection`。其中，`data`、`exposure`、`initialization`、`band_axis`、`arithmetic` 使用生产方固定哈希。类别从 1 到 Bank 头的最后一类。选模为 `first_min_released_point_fused_nll`。
- 单例回执的 `schema` 为 `wp05_wp08_frozen_endpoint_v1`，包含该群组的 `protocol`、`seed`、`phase`、`step`、`calibration` 及 `checkpoint`。校准轨迹提供有序 `step` 和融合 `nll`，包含预算终点。选中阶段使用各臂同一选模规则，其实际步数逐臂记录。终点阶段强制为登记预算。跨阶段、跨校准事件日历、跨原始类别或波段身份的差值拒绝。
- `checkpoint` 固定 `backend`、`weights`、相邻 `.pt.json` 的 `metadata` 及 `state` 指纹。允许 `native`、`multiscale`、`aligned` 和 `region_message`。图支持由各登记后端的原生前向生成，CNN dilation、图范围、区域交换方向和消息模式随代码和回执保留。没有可靠构造字段、张量键或精度不符时拒绝加载。

分类及成本各按授权生成。消费器重新绑定严选评价点顺序、真实保留列、模型和实际观测，报告三头 OA、AA、类别分母和同点变化。徐州与洪湖沿用登记的小类保护以及独立评价每类 20 点门槛。没有登记目标类的场景保留通用逐类保护与描述性估计。可用 `--gpu UUID` 从实体租约做冻结重放。用相同清单捕获的 CPU 结果可通过 `--reference-capture PATH SHA256` 参加设备核验。墙钟、建图、恢复权重及 GPU 分配峰值都绑定此次运行设备。

## 存量和工程验证

`scripts/run_wp12_cross_backend.py fixture --synthetic --kind xuzhou --layout a --seed 64` 可创建工程发布。使用 `--kind honghu --layout b` 切换布局，添加多个 `--seed` 核对登记矩阵。测试发布必须在捕获和消费时显式使用 `--synthetic`。`--cost-only` 发布只核对实际成本路径。正式流程拒绝工程发布替代真实外部资格。

`scripts/replay_wp12_cross_paviau.py --archive wp07` 或 `--archive wp08` 在精确的 `--producer-checkout` 装载旧检查点，补存量汇总缺少的同点变化。用 `scripts/run_wp12_cross_backend.py history` 复核历史，分别以 `--paviau-replay wp07 PATH SHA256` 和 `--paviau-replay wp08 PATH SHA256` 提供两份可选的冻结重放回执。锁定的原训练后端和对照见技术报告。历史开发与独立真实资格分别报告。
