# WP05/WP06 配对线性头与缺轴回退

## 原生最小协议

两种头都接收当前观测产生的 64 维像素特征，类别索引保持 1 到场景类别数。已有原生三头、输入层、CNN、图、适配器及归一化结构继续由原生前向消费。调整范围是既有融合分类器的权重和偏置。池化臂在同一实际 Q 上把区域聚合结果广播回原格，复用同一个线性头；图头和 CNN 头用于同时核对冻结分支。

| 场景 | 锁定配对 | 支持 | 步数与选模 | 首要确认 |
| --- | --- | --- | --- | --- |
| 徐州 | 全带支持 / 同标签全带、中央缺带、等量分散固定轮换 | K10，固定空间块与采样顺序 | 每臂 128 更新，步 0/32/64/96/128 首个最低全带校准 NLL | 中央缺带里信号保留的同点，类 5/6 单列 |
| 洪湖 | 原融合特征 / 原 Q 聚合广播特征，训练与校准先全带 | K5，固定空间块与采样顺序 | 同一初态，每臂 128 更新和同一全带校准时钟 | 首先全带，类 15/21 单列 |
| 缺轴试验 | 原压缩索引 / 保留交付序 / 等容量反序控制 | K10，三个模型种子及两个支持抽样 | 每臂 48 更新，元数据和波段日程相同，首次最低校准 NLL | 新布局上的实际保留线索、声明/缺失/错位轴及删除负控 |

评估发布中的同点类别计数和空间缓冲是准入的一部分。既有历史开发点不能作为确认。最小真实计划要求独立评估每类至少 20 点。合成统计门槛为每类至少 5 点，分母不足的单元保持不可判定。首要条件要求两目标类各自净改判不下降，至少一项有净收益。总体 AA 差须不低于 −0.005，普通类召回差须不低于 −0.02，洪湖 15/21 各自同点不新增错误。删尽线索的负对照使用实际中央缺带，合成 AA 上限为 25%。预算、角色与保护保持固定，负结果不触发重抽、挑头、延长预算或候选上线。

## 授权容器内命令

`PRODUCER` 和 `CONSUMER` 是对应的完整提交，`RUNS` 为授权根。CPU 下调用以下命令。GPU 另指定单张合格卡的 `CUDA_VISIBLE_DEVICES="$GPU_UUID"`、`CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `--gpu "$GPU_UUID"`，同一已登记的续跑不换物理设备。

```bash
python -B scripts/run_native_released_points.py --runs "$RUNS" --out "$PROGRAM" --expected-commit "$PRODUCER" --mode paired-head-register
python -B scripts/run_native_released_points.py --runs "$RUNS" --out "$PROGRAM" --expected-commit "$PRODUCER" --mode paired-head-run --case xuzhou-s64-k211
python -B scripts/run_native_released_points.py --runs "$RUNS" --out "$PROGRAM" --expected-commit "$PRODUCER" --mode paired-head-run --case honghu-s64-k211
python -B scripts/check_wp12_paired_heads.py capture --runs "$RUNS" --program "$PROGRAM" --out "$CPU" --expected-commit "$CONSUMER" --producer "$PRODUCER" --case xuzhou-s64-k211
python -B scripts/check_wp12_paired_heads.py capture --runs "$RUNS" --program "$PROGRAM" --out "$CPU" --expected-commit "$CONSUMER" --producer "$PRODUCER" --case honghu-s64-k211
```

其余配对组使用 64/512/2048 和 211/613，缺轴组使用例如 `fallback-s64-k211`。每次组执行登记的全部臂，已有完整回执时零更新重入。独立成本、三头、训练状态和全类保护从同一权重和原格公开样本重算。

## 真实只读预检

持有人按[独立角色及光学交接](independent-optical-intake.md)提供独立 Bank 清单和权限，分别确认支持、校准、一次性评估、成本目的及历史曝光。请求 `wp05_wp06_paired_head_read_only_request_v1` 绑定 `kind` 为 `xuzhou` 或 `honghu`，另含 `independent_manifest`、`independent_authority` 两个固定文件、`model_seeds=[64,512,2048]`、`support_seed`、`scope_checkpoint` 的 SHA-256，以及 `comparison` 的 `arms`、`steps=128`、`dimension=64`。臂名分别为 `["full_support","rotating_support"]` 和 `["fusion","pooled"]`。

审批方另行提供可信 `wp05_wp06_paired_head_read_only_authority_v1`，绑定 `approved_request_sha256`、当前消费者 `producer`、`allowed_use="read_only_preflight"` 及 `train_authorized=false`。通过预检仅核对实际发布分母、角色、源模型和头的容量。下面的命令不会发放批准。真实训练需另外登记获批的预算、设备和直接执行授权。

```bash
python -B scripts/check_wp12_paired_heads.py real-preflight --runs "$RUNS" --out "$RESULTS/real-preflight.json" --expected-commit "$CONSUMER" --request "$REQUEST" "$REQUEST_SHA256" --approval "$APPROVAL" "$APPROVAL_SHA256"
```

## 外部发布与优化授权

持有人发布独立 Bank、三项角色的块清单、历史 Bank 和各用途的权限。运行申请 `wp05_wp06_external_paired_campaign_v1` 使用 `external_real_execution`，以字母数字键索引 `releases`。每条登记 `kind`、`k`、`support_seed`、`spatial_repeat`、`manifest`、`authority`、`source`、`model_seeds`、`initialization` 与 `device`。`source` 的两个字段分别是原已接受检查点的 `checkpoint` 和模型状态 `state` 的 SHA-256。`device` 是整张物理卡的 UUID。支持点来自 K1/5/10 的真实发布频数，支持种子可取 64/512/2048。同场景、同 K、同支持种子的空间重复须保持实际父块缓冲。支持和校准的来源也与既有开发历史隔离。

审批方单独给出 `wp05_wp06_external_paired_execution_authority_v1`，包括 `decision="approved"`、同 `scope`、申请文件路径及 SHA-256 的 `request`、完整 `producer`、`source_hash`、绝对 `destination`、整个登记的 `updates`，以及 `raw_gt_authorized=false` 与 `query_authorized=false`。正式外部执行每臂沿用 128 步。原始线性头走 `inherit_head` 且只登记一个模型种子。`seeded_head` 可登记三个种子，每个种子只重新初始化既有融合头，并在配对两臂共享该初态。外部格式工程样本的权限使用独立测试 schema 和 `--test-publication`，正式入口只读取外部提供的授权。

登记通过后使用实际发布键、种子和臂名启动。收到正常终止信号或到达 `--drain-after` 的游标时先落盘。相同身份和设备再入会续跑，完成时再入返回零更新。两景所有登记臂到达预算后，才可发布选中权重并独立消费。分类权限和只读成本权限各自使用独立回执；成本前向从观测接口取得输入。

```bash
python -B scripts/run_native_released_points.py --mode external-paired-register --runs "$RUNS" --out "$PROGRAM" --expected-commit "$PRODUCER" --manifest "$REQUEST" "$REQUEST_SHA256" --authority "$APPROVAL" "$APPROVAL_SHA256"
CUDA_VISIBLE_DEVICES="$GPU_UUID" CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B scripts/run_native_released_points.py --mode external-paired-run --runs "$RUNS" --out "$PROGRAM" --expected-commit "$PRODUCER" --case "$RELEASE" --model-seed "$MODEL_SEED" --head-arm "$ARM" --gpu "$GPU_UUID"
python -B scripts/check_wp12_external_paired.py publish --runs "$RUNS" --program "$PROGRAM" --expected-commit "$PRODUCER"
python -B scripts/check_wp12_external_paired.py capture --runs "$RUNS" --program "$PROGRAM" --out "$CAPTURE" --expected-commit "$PRODUCER"
python -B scripts/check_wp12_external_paired.py review --runs "$RUNS" --program "$PROGRAM" --out "$REVIEW" --expected-commit "$PRODUCER" --capture "$CAPTURE" "$CAPTURE_SHA256"
python -B scripts/ingest_wp12_evidence.py --external-paired-history --runs "$RUNS" --out "$WP12_INDEX" --expected-commit "$CONSUMER" --external-review "$REVIEW" "$REVIEW_SHA256"
```

历史索引固定同点发布和权重，仅按同一历史协议复算得失。洪湖 120 步的已审核混淆边际可比较正确数差，逐点得失留空。全景 Issue 184 和两景外部独立结果保留各自的证据身份。

## 只接收保留观测的交付序回退

`bind_retained_ordinal` 可直接使用带已核身份的保留观测，及来自同一发布的 `DeliveredAxis`。它保留相同的像素、坐标、统计和波段 ID，仅用实际保留列在交付轴上的秩生成 `normalized/index_only` 坐标。没有可信交付序时沿用旧缺轴回退。有声明轴的已存档状态和旧前向保持逐位身份。若已知物理表不可信，应在料源准入中标为缺失。本候选不从输入自动鉴别失配，也不发明实测中心、幅宽或被删列。
