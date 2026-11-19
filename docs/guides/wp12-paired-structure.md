# 缺带与小类保护的跨模型配对训练

`scripts/run_wp_paired_structure.py` 把原生、条件多尺度、对齐与区域消息放在同一个原生释放点运行器。沿用原 Adam 单组、三头损失、几何面积重采样、原像素点损失、同卡计算和追加式检查点。工作目录在授权 LXC 的 checkout 外。所有命令提供 `--runs`、`--campaign` 与该干净 checkout 的 `--expected-commit`。

## 登记字段

- 用 `register --request PATH SHA256 --approval PATH SHA256`。请求 `schema=wp05_wp08_paired_structure_training_v1`，包含 `scope`、`releases` 和 `groups`。真实使用 `external_real_execution`。每个发布项固定已有 `manifest`、持有人 `authority`、`k` 以及可选 `source`。权重来源需要 `weights`、相邻 `.pt.json` 的 `metadata` 及 `state` 指纹，同 Bank 的共享来源绑定。
- [独立 Bank 发布](independent-attribution-package.md)给独立支持、校准和评价块，完整确认类别与已知波段。持有人在清单显式批准 `observation_views` 的 `full`、`central_gap`、`matched_dispersed` 及 `observation_factors`。后两种视图实际去除相同数量的列。支持每类 K 点。评价按已有独立用途与逐类 20 点门槛，空间保护可按可信 E0 分层补充。
- 每组的 `release`、`seed`、`factor`、`steps`、`clock`、`arms`、`pairs`、`device`、`purpose` 共同固定。校准事件包含 0 和终点，选首个最低融合 NLL。曝光按固定种子和轮次安排，每个支持块至少覆盖三条观察视图。每个配对固定基线、候选和参数关系，等参定向对照要求初始张量逐值相同。独立尺度组各自保留同因子基线。
- 单独执行许可绑定请求、精确生产提交与源码哈希、目标目录、设备、总更新数和范围。两类权限由持有人提供。真实设备使用已有物理 UUID 租约和数值资格。没有合法申请、训练批准和可信物理波段的草案可做审查。

## 执行、恢复和确认

1. 用 `arm --group GROUP --arm ARM` 做单例。可用 `--until STEP` 或 SIGTERM 排空，重入保持已登记工作目录、源码、训练算术和设备。用 `group --group GROUP` 逐臂运行同一设备。用 `queue --workers N` 调度已经登记的完整组，跨设备安排时保持每个组的全部臂在同一设备。完整回执与追加链核对后，复调新增更新为 0。
2. `freeze` 在整个队列完成后发布选中的权重、架构 sidecar 和已绑定的校准轨迹。恢复包括模型、Adam、随机状态、曝光历史和选中态。比较清单由同一发布标记指向，每个发布项一份可给上包的冻结消费者。
3. 对各发布项用 `capture --release RELEASE --out OUTPUT.json`。通过现有冻结比较接口，在持有人批准的原倍率或下采样倍率消费实际评价观测。成本使用单独授权及无标签的物理观测接口。每臂保留原生 Q/A0、CNN 尺度、区域消息方向及实际推理、建图和显存样本。可用 `--gpu UUID` 现场做冻结重加载。
4. 用 `scripts/ingest_wp12_evidence.py --paired-structure-review --capture RELEASE PATH SHA256` 重复传入每项捕获，再指定新的 `--out` 目录。它核对全臂收据、选中的历史状态指纹和独立评价分母，复算三头 OA、AA、逐类同点得失及保护门槛。恢复晚于最佳事件的 checkpoint 也以原选中头评价。真实采用要求同一登记对照至少三个种子通过，同因子与跨因子分列。

## 工程与真实草案

`fixture --synthetic --out INPUT` 生成隔离工程发布。可用 `--devices UUID ...` 绑定现有合格卡，工程单卡模式把其余独立组分到 CPU；`--qualification` 限定用于恢复接入的省算格。登记、训练、冻结、捕获、消费都显式带 `--synthetic`。工程批准的场景、标签和数据均为合成，不能进入正式真实工作器。

`draft --evidence REVIEW.json SHA256 --out PENDING.json` 从真实历史预算与计时及本包工程收据生成可审查的三景请求形状。草案经过同一字段设计，填入真实持有人发布和批准前保留空位，运行器拒绝执行。成本窗口是带敏感性范围的资源估算；真实合同、未来观测、独立角色和授权会在实际登记时验签。请以验收报告记录的完整提交和回执为复演入口。
