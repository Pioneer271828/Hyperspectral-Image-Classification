# WP12 真实分类、设备成本与独立开发清单

运行入口使用同一个干净的授权 LXC 检出。环境与源码导入方法见[原生覆盖发布指南](native-coverage-release.md)。输出目录放在已授权的 `runs` 以下。所有读取使用已登记的 SHA-256，产物目录须未创建。

## 汇总原始真实回执

```bash
sha=$(git rev-parse HEAD)
runs=/home/lbx/hsi-pioneer271828/runs
python_bin=/home/lbx/hsi-pioneer271828/envs/hsi/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
"$python_bin" -B scripts/ingest_wp12_evidence.py --native-real \
  --expected-commit "$sha" --runs "$runs" --out "$runs/wp12-real-release-review"
```

- `cases.json` 和 `real-results.csv` 使用已接受的模型、场景、种子、K、数据协议、端点、波段和设备字段。按这些字段精确选取可使用 `evaluation.wp12_release_results.select(rows, scene="xuzhou", seed=64, role="devval_released_focus_only")`。
- `cohorts.json` 和 `real-seed-summary.csv` 在相同协议、波段、端点、设备、工作负载及计时边界内计算均值与样本 SD。单种子或缺测没有 SD。同行多类指标的支持集与模型种子单列。
- `real-per-class.csv` 列三头逐类分母、召回与 F1。`real-class-seed-summary.csv` 提供各组逐类均值与样本 SD。无发布点的类别召回为空。
- 历史 54 例成本由原始计时和既有独立回执重新组装。选中检查点完全一致时才接回历史冻结分类。成本是 RTX 4090 同卡的热态常驻模型前向。输入与图准备、加载和传输在计时边界之外。缺少独立构图计时、冷启动和端到端 I/O。
- PaviaU 原生适配的固定代表块成本与整套开发验证精度分开建账。洪湖五臂只承认已审的同协议组。WP06 逐块单次构图/前向和七层探针开销保留为诊断记录，没有重复测量的尾延迟或峰值显存。各来源的时延单位相同，比较边界可能不同。

旧版 `scripts/ingest_wp12_evidence.py --registration …` 保留。冻结 54 原采集、资源门槛和恢复接口分别见 `scripts/run_wp12_cost_campaign.py` 与 `scripts/measure_wp12_historical_cost.py`。该真实登记已完结。当前汇总只读取已授权历史分类统计、成本与冻结开发诊断，真实模型、原始真值和 query 输入均不执行。

## 两景冻结开发误差归因

从已授权的训练与开发角色读取既有冻结权重。在干净 checkout 用原研究入口 `scripts/run_wp12_retained_evidence.py --real-attribution`，按 `--group honghu_short|honghu_120|xuzhou` 产出逐案例回执。可用 `--shard INDEX COUNT` 分配同一批固定案例。已经落盘的同提交回执可直接复用。洪湖实际误判类的固定对照用 `--group honghu_120 --locked-honghu-rival`，只读取审核内同一 K5 从头训练五臂的选中权重。

用原 WP12 ingest 的 `--real-attribution-review` 读取三个 `--source-shard GROUP DIRECTORY SHA`、`--floor FILE SHA` 及 `--locked-rivals FILE SHA`，各文件来自同一授权 runs 根。消费者独立重算三头误差、固定类别差、角色分母、已有梯度活动和保留位置对照。按发布权限输出描述性精度、保护可评性和正式合格状态。持有人按[最小发布示例](independent-attribution-package.md)提交相容的带表、类表和实际角色清单。

## 审核外部已批准的独立开发清单

发布方先提供封存的 `BankRelease` 和相应 Bank 字节。提交两个已分别固定 SHA 的 JSON 文档。输出只记录摘要、角色/类别计数和每种用途的准入。

- 清单版本为 `wp02_independent_development_intake_v1`，键是 `schema`、`candidate`、`roles`、`minimum_buffer`、`history`。`candidate` 包含 `pin`（`root/sha256/registration`）和按现有 `BankRelease` 序列化的 `release`。每个历史补充项使用相同结构。
- `roles` 为 `support`、`calibration`、`evaluation`。各值为互不重复的已发布块 ID 列表，须覆盖清单中的可读块。支持块来自 Bank 的 `devtrain`，另两类来自 `devval`，各自的父块、样本与缓冲额外分离。
- 审批版本为 `wp02_independent_development_approval_v1`，键为 `schema`、`approved_manifest_sha256`、`historical_bank_sha256`、`uses`、`source_mode`。后者为 `external_real`。`uses` 分别许可 `independent_accuracy`、`cost`、`frozen_diagnostics`。历史 Bank 指纹列表须排序，必须包含正式入口固定的同景发布与所用补充历史。
- 发布审批由数据授权方提供。入口检查源坐标与原带轴的匹配、已发布角色、父块/样本/缓冲交叠及逐类覆盖。未满足独立准确率的开发集仍可在原授权内有单独的成本与冻结诊断资格。成本接口不返回标签。新独立资格不赋予新标签、query、优化或已有模型的独立成绩。

```bash
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$sha" --runs "$runs" --out "$runs/approved-intake" \
  --mode coverage-release-review \
  --manifest "$manifest" "$manifest_sha" --authority "$approval" "$approval_sha"
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$sha" --runs "$runs" --out "$runs/archive-admission-check" \
  --mode coverage-release-history
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$sha" --runs "$runs" --out "$runs/synthetic-admission-check" \
  --mode coverage-release-demo
```

工程演示用两组互斥的合成地块。它在现有原生构图和三头消费路径检验三个已批角色、成本无标签输出和冻结诊断。当前真实历史 Bank 仍是开发/选择材料，其重叠和缺失角色要在准入表逐项保留。

## 跨训练阶段的冻结开发比较

先使用已批准的开发发布登记一个冻结计划。计划固定源和模型、训练预算、支持和验证样本、空间、原带 ID、统计和构图规则。开发发布沿用过去的角色，独立性由原档案决定。中央和分散视图删去相同数量的合法带，保留物理中心和原训练归一化。三头使用同一批开发点。

~~~bash
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$sha" --runs "$runs" --out "$runs/stage-plan.json" \
  --mode frozen-stage-register --manifest "$release" "$release_sha"
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$sha" --runs "$runs" --out "$runs/stage-cpu.json" \
  --mode frozen-stage-run --frozen-plan "$plan" "$plan_sha"
~~~

同一计划可指定唯一物理 GPU UUID 复放，设置 `CUDA_VISIBLE_DEVICES` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。严格资源准入和物理卡锁沿用 GPU 入口。跨设备保留实际概率差和改判计数。历史全54的固定原图使用 `--historical-stage-run`，提供固定的 `--launch`、`--independent`、`--freeze` 路径及 SHA。每景固定种子 64 的无残差 scratch 代表，沿用原图和原 DomainBN 验证。已完成的代表阶段允许重新进入。

~~~bash
"$python_bin" -B scripts/ingest_wp12_evidence.py --frozen-stages \
  --expected-commit "$sha" --runs "$runs" --out "$runs/wp12-stages.json" \
  --plan "$plan" "$plan_sha" --cpu "$cpu_receipt" "$cpu_sha" \
  --launch "$launch" "$launch_sha" --independent "$audit" "$audit_sha" \
  --freeze "$freeze" "$freeze_sha" --honghu-release "$release_hh" "$release_hh_sha" \
  --honghu-review "$review_hh" "$review_hh_sha" --honghu-status "$status_hh" "$status_hh_sha"
~~~

发布可以附上 `--gpu`、`--historical-replay`、`--honghu-alternate` 和 `--synthetic-queue`，各自提供绝对路径和 SHA。旧执行产物只有通过父提交及生产运行代码核对后，才可通过 `--execution-commit` 复用。原生覆盖队列用 `coverage-stage-register` 选择完整的合成五臂，通过相同的 `coverage-run` 进入训练和恢复。阶段端点使用已登记的初始化、同序分段的中间检查点、首次最小验证 NLL 选中态和终态。历史固定预算入口保留原有行为。

## 两景 1,500 步匹配训练与独立复盘

在授权 LXC 的干净检出运行。下列一键入口只登记和训练隔离的合成发布。两景各三个模型种子，每个种子运行 `none` 与 `continuous`。候选会共同使用一套 K5 支持、原始物理带 ID、仅由支持块拟合的归一化和对齐的训练曝光。新登记预计双工作进程约 7–26 分钟，实际时间受设备准入和审计影响。

~~~bash
"$python_bin" -B scripts/run_native_long_coverage.py \
  --runs "$runs" --campaign "$runs/native-matched-1500" --review "$runs/native-matched-1500-review"
~~~

需要 CUDA 时，附至多三个完整的 `--gpu "$gpu_uuid"`。同景同种子的两个臂默认在同一个物理设备，CPU 代表保留跨进程排空和连续参考。计算只通过物理卡租约和资源准入进入。要在独立检出仅重算已冻结的 12 例，可用 `--review-only --producer-checkout "$producer_checkout"` 指定完全匹配的生产检出。既有输出通过原始检查点固定清单复用。若重算使用另一提交，WP12 先核对 Git 祖先及未变化的生产源码。

生产步骤对应 `scripts/run_native_released_points.py` 的 `coverage-long-register`、`coverage-long-run --case "$case"` 和 `coverage-long-freeze`，都使用 `--expected-commit "$sha" --runs "$runs" --out "$campaign"`。运行可用 `--drain-after "$absolute_step"` 或发送 SIGTERM。在当前更新完成后写下唯一可继续的检查点，后续进程按同设备和原始曝光继续。SIGKILL 后只复用最近已落盘的检查点。完成重入返回零更新。单独合成参考为 `coverage-long-reference --case xuzhou-m64-none`，额外 1,500 步、不计入 18,000。

选模只读支持和校准角色，校准在 0／150／…／1,500 的全带点上执行，首次最低融合 NLL 为选中态。冻结十二条轨迹以后，WP12 读取独立确认块，针对同点的全带、中央连续缺带、等量分散缺带分别保留三头。图从各自实际保留的波段重建，历史全带固定图不能用于这个干预。每例审计和统一发布分别使用 `scripts/ingest_wp12_evidence.py --long-stage-case` 和 `--long-stage-publish`，需提供已封存清单的路径和 SHA-256。可附 `--historical-snapshot PATH SHA`复用既有洪湖无权重记录。

### 真实发布方最小交接

真实运行需另行授权。登记命令使用 `coverage-long-register --long-real --manifest REQUEST SHA --authority APPROVAL SHA`。申请与授权都必须在授权运行根目录，输出使用明确的独立运行路径。入口先核对授权和发布才创建训练路径。

- 申请 `schema=wp05_native_matched_1500_v1`。提供 `inputs`，以 `xuzhou` 和 `whu-honghu` 为键，每景包括独立开发 `manifest`、`authority` 和 `source`。前两者各提供绝对路径和 SHA-256，`source` 提供相邻的 `.pt` 与 `.pt.json` 固定清单，其源状态绑定发布的 `shared_state`。
- 提供经预先审核的 WP02 支持、校准、确认三角色，给出原场景父块、空间缓冲及历史祖先。支持种子固定 2048，每类 5 个有效训练点。校准、确认各有逐类支持，校准分数用于选步，确认封存在训练范围外。原波段坐标与物理单位已核验，归一化统计只来自支持数据。
- 申请中的 `physical_devices` 绑定全部 12 个案例到实体 UUID。授权记录提供 `schema`、`decision=approved`、`scope=external_real_execution`、申请固定清单 `request`、精确 `source_commit` 和 `source_hash`、`destination`、`design_sha256`、`optimizer_updates=18000`、`raw_gt_authorized=false`、`query_authorized=false`。授权由负责发布方提供。在 LXC 可从 `research_runtime.code_snapshot()` 和 `digest(transfer.native_long_campaign.DESIGN)` 得到待审核指纹。
- 未获授权、通过旧标签家族冒充独立确认、缺源状态或带轴、设备和目的地错配均拒绝。历史缺权重的案例继续留空。真实 18,000 步及五臂 45,000 步方案均未启动。

## 合成谱信号正对照与上下文审计

在授权 LXC 的干净检出运行以下合成入口。它登记全部工况、预算及验真阈值，先验真支持和校准，再训练与封存，最后才生成确认块。中断后从原检查点继续，已经完成的同设备案例直接复用。可添加至多三个完整的 `--gpu "$uuid"`，每个设备分到完整的三臂配对。

~~~bash
"$python_bin" -B scripts/run_native_positive_control.py \
  --runs "$runs" --campaign "$runs/spectral-positive-control-v2" \
  --review "$runs/spectral-positive-control-v2-review" --workers 8
~~~

训练登记、单例运行和冻结可通过 `scripts/run_native_released_points.py` 的 `positive-control-register`、`positive-control-run --case "$case"`、`positive-control-freeze` 执行。均需精确 SHA 和授权运行根，且拒绝真实许可参数。可在独立检出使用 `--review-only --producer-checkout "$checkout"`，复用已冻结训练。WP12 的 `--positive-control-case` 与 `--positive-control-publish` 分别接收单例和全队列冻结。附上 `--v1 PATH SHA` 可接入旧谱观察审计；旧 v1 归档只读入口为 `--v1-spectral-audit --freeze PATH SHA --review PATH SHA`。这些 WP12 命令使用 `scripts/ingest_wp12_evidence.py`。

实验先比较具有已知谱中心的加载器 oracle、仅支持点质心和带 L2 的线性读出。局部正控若未达到预登记下限，不能据此判定某条网络旁路。位置打乱仅用于确认，对原标签位置和谱轴做身份核对后，重排输入的坐标通道并重建图。真实默认模型和原验收协议继续由现有入口管理。

## 原生分支与空间分层

在授权 LXC 的干净检出运行 `scripts/run_wp12_native_pathways.py`。各动作均提供 `--runs`、`--out`、`--expected-commit`。用 `real --case` 复核已经批准的单个真实端点，`--drain-after` 对总块游标排空；原输出目录可按已落盘块继续。CUDA 通过 `--gpu` 绑定实体 UUID。用 `branches` 输出冻结融合头的 CNN 和图分支投影；`readouts` 复算存档线性读出；`controls` 与 `external` 运行无优化的原生合成对照和空间准入测试。

用 `scripts/ingest_wp12_evidence.py --native-pathways-review` 提供 `--runs`、`--out`、`--expected-commit` 及 `--real-root`、`--branch-root`、`--controls-cpu`、`--controls-gpu`、`--gpu-real`、`--external`、`--readouts`。审核保留固定 Q/A 的 stem 后导数、合成重建 Q/A 的有限差、各真实角色与物理块分母、历史冻结读出及外部三角色权限。可选 E0 只按[独立空间发布](independent-attribution-package.md)的评价几何计算。

## WP05 至 WP08 跨模型配对

[冻结跨模型比较](wp12-cross-backend.md)使用同一授权评价 Bank 消费原生、条件多尺度和区域消息候选，并按旧协议独立复算徐州、PaviaU 的真实开发比较。WP12 验收选项为 `--cross-backend-review`，可匹配同权重 CPU/CUDA 捕获并保留成本与分类的独立资格。

## 缺带与小类保护的配对生产

[原生训练、冻结和确认](wp12-paired-structure.md)在同一释放点优化器接入多尺度、对齐和定向消息。WP12 使用 `--paired-structure-review` 消费完整训练和冻结发布。
