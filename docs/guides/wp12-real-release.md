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
