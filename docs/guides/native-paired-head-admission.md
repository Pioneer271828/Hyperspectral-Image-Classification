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

## 只接收保留观测的交付序回退

`bind_retained_ordinal` 可直接使用带已核身份的保留观测，及来自同一发布的 `DeliveredAxis`。它保留相同的像素、坐标、统计和波段 ID，仅用实际保留列在交付轴上的秩生成 `normalized/index_only` 坐标。没有可信交付序时沿用旧缺轴回退。有声明轴的已存档状态和旧前向保持逐位身份。若已知物理表不可信，应在料源准入中标为缺失。本候选不从输入自动鉴别失配，也不发明实测中心、幅宽或被删列。
